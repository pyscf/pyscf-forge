from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jax import lax, tree_util

from ..prop.types import QmcParamsLno
from ..core.ops import MeasOps, k_energy, k_force_bias, o_density_corr, o_rdm1, o_orb_corr
from ..core.system import System
from ..ham.chol import HamChol
from ..meas.uhf import _density_corr_from_greens
from ..trial.rhf import RhfTrial, overlap_g, overlap_r, overlap_u

RhfMeasMemoryMode = Literal["high", "low"]
_RHF_MEAS_CFG_ATTR = "_rhf_meas_cfg"


def _half_green_from_overlap_matrix(w: jax.Array, ovlp_mat: jax.Array) -> jax.Array:
    """
    green_half = (w @ inv(ovlp_mat)).T
    """
    return jnp.linalg.solve(ovlp_mat.T, w.T)


def _trace_last2(x: jax.Array) -> jax.Array:
    """Trace over the last two axes without materializing a full mask."""
    return jnp.sum(jnp.diagonal(x, axis1=-2, axis2=-1), axis=-1)


def _exchange_term(f_g: jax.Array) -> jax.Array:
    return jnp.sum(f_g * f_g.T)


def _exchange_sum_materialized(f: jax.Array) -> jax.Array:
    return jnp.sum(jax.vmap(_exchange_term)(f))


def _two_body_energy_restricted(g_half: jax.Array, meas_ctx: RhfMeasCtx) -> jax.Array:
    memory_mode = meas_ctx.cfg.memory_mode
    if memory_mode == "low":
        zero = jnp.array(0.0, dtype=jnp.result_type(meas_ctx.rot_chol, g_half))

        def scan_over_chol(
            acc: tuple[jax.Array, jax.Array], rot_chol_g: jax.Array
        ) -> tuple[tuple[jax.Array, jax.Array], None]:
            c2_acc, exc_acc = acc
            f_g = rot_chol_g @ g_half.T
            c_g = _trace_last2(f_g)
            return (c2_acc + c_g * c_g, exc_acc + _exchange_term(f_g)), None

        (c2, exc), _ = lax.scan(scan_over_chol, (zero, zero), meas_ctx.rot_chol)
        return 2.0 * c2 - exc

    f = jnp.einsum("gij,jk->gik", meas_ctx.rot_chol, g_half.T, optimize="optimal")
    c = _trace_last2(f)
    exc = _exchange_sum_materialized(f)
    return 2.0 * jnp.sum(c * c) - exc


def _two_body_energy_unrestricted(
    gu: jax.Array,
    gd: jax.Array,
    meas_ctx: RhfMeasCtx,
) -> jax.Array:
    memory_mode = meas_ctx.cfg.memory_mode
    if memory_mode == "low":
        zero = jnp.array(0.0, dtype=jnp.result_type(meas_ctx.rot_chol, gu, gd))

        def scan_over_chol(acc: jax.Array, rot_chol_g: jax.Array) -> tuple[jax.Array, None]:
            f_up_g = rot_chol_g @ gu.T
            f_dn_g = rot_chol_g @ gd.T
            c_up = _trace_last2(f_up_g)
            c_dn = _trace_last2(f_dn_g)
            e2_g = (
                c_up * c_up
                + c_dn * c_dn
                + 2.0 * c_up * c_dn
                - _exchange_term(f_up_g)
                - _exchange_term(f_dn_g)
            ) / 2.0
            return acc + e2_g, None

        e2, _ = lax.scan(scan_over_chol, zero, meas_ctx.rot_chol)
        return e2

    f_up = jnp.einsum("gij,jk->gik", meas_ctx.rot_chol, gu.T, optimize="optimal")
    f_dn = jnp.einsum("gij,jk->gik", meas_ctx.rot_chol, gd.T, optimize="optimal")
    c_up = _trace_last2(f_up)
    c_dn = _trace_last2(f_dn)
    exc_up = _exchange_sum_materialized(f_up)
    exc_dn = _exchange_sum_materialized(f_dn)

    return (
        jnp.sum(c_up * c_up) + jnp.sum(c_dn * c_dn) + 2.0 * jnp.sum(c_up * c_dn) - exc_up - exc_dn
    ) / 2.0


def _validate_memory_mode(memory_mode: str) -> RhfMeasMemoryMode:
    if memory_mode not in ("high", "low"):
        raise ValueError(f"Unsupported RHF measurement memory_mode: {memory_mode!r}")
    return memory_mode


@dataclass(frozen=True)
class RhfMeasCfg:
    memory_mode: RhfMeasMemoryMode = "high"


def get_rhf_meas_cfg(meas_ops: MeasOps) -> RhfMeasCfg | None:
    cfg = getattr(meas_ops, _RHF_MEAS_CFG_ATTR, None)
    if isinstance(cfg, RhfMeasCfg):
        return cfg
    return None


def force_bias_kernel_rw_rh(
    walker: jax.Array, ham_data: Any, meas_ctx: RhfMeasCtx, trial_data: RhfTrial
) -> jax.Array:
    m = trial_data.mo_coeff.conj().T @ walker
    g_half = _half_green_from_overlap_matrix(walker, m)  # (nocc, norb)
    # RHF: factor 2 for (up+dn)
    return 2.0 * (meas_ctx.rot_chol_flat @ g_half.reshape(-1))


def energy_kernel_rw_rh(
    walker: jax.Array, ham_data: HamChol, meas_ctx: RhfMeasCtx, trial_data: RhfTrial
) -> jax.Array:
    m = trial_data.mo_coeff.conj().T @ walker
    g_half = _half_green_from_overlap_matrix(walker, m)  # (nocc, norb)

    e0 = ham_data.h0
    e1 = 2.0 * jnp.sum(g_half * meas_ctx.rot_h1)
    e2 = _two_body_energy_restricted(g_half, meas_ctx)

    return e0 + e1 + e2


def force_bias_kernel_uw_rh(
    walker: tuple[jax.Array, jax.Array],
    ham_data: Any,
    meas_ctx: RhfMeasCtx,
    trial_data: RhfTrial,
) -> jax.Array:
    wu, wd = walker
    mu = trial_data.mo_coeff.conj().T @ wu
    md = trial_data.mo_coeff.conj().T @ wd
    gu = _half_green_from_overlap_matrix(wu, mu)  # (nocc_a, norb)
    gd = _half_green_from_overlap_matrix(wd, md)  # (nocc_b, norb)
    g = gu + gd
    return meas_ctx.rot_chol_flat @ g.reshape(-1)


def energy_kernel_uw_rh(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamChol,
    meas_ctx: RhfMeasCtx,
    trial_data: RhfTrial,
) -> jax.Array:
    wu, wd = walker
    mu = trial_data.mo_coeff.conj().T @ wu
    md = trial_data.mo_coeff.conj().T @ wd
    gu = _half_green_from_overlap_matrix(wu, mu)
    gd = _half_green_from_overlap_matrix(wd, md)

    e0 = ham_data.h0
    e1 = jnp.sum((gu + gd) * meas_ctx.rot_h1)
    e2 = _two_body_energy_unrestricted(gu, gd, meas_ctx)

    return e0 + e1 + e2


def force_bias_kernel_gw_rh(
    walker: jax.Array, ham_data: Any, meas_ctx: RhfMeasCtx, trial_data: RhfTrial
) -> jax.Array:
    norb, nocc = trial_data.norb, trial_data.nocc
    cH = trial_data.mo_coeff.conj().T
    top = cH @ walker[:norb, :]
    bot = cH @ walker[norb:, :]
    m = jnp.vstack([top, bot])  # (2*nocc, 2*nocc)

    g_half = _half_green_from_overlap_matrix(walker, m)  # (2*nocc, 2*norb)
    g_up = g_half[:nocc, :norb]
    g_dn = g_half[nocc:, norb:]
    g = g_up + g_dn
    return meas_ctx.rot_chol_flat @ g.reshape(-1)


def rdm1_kernel_rw(
    walker: jax.Array,
    ham_data: Any,
    meas_ctx: RhfMeasCtx,
    trial_data: RhfTrial,
) -> jax.Array:
    m = trial_data.mo_coeff.conj().T @ walker
    g_half = _half_green_from_overlap_matrix(walker, m)
    g = g_half.T @ trial_data.mo_coeff.conj().T
    return jnp.stack([g, g], axis=0)


def rdm1_kernel_uw(
    walker: tuple[jax.Array, jax.Array],
    ham_data: Any,
    meas_ctx: RhfMeasCtx,
    trial_data: RhfTrial,
) -> jax.Array:
    wu, wd = walker
    mu = trial_data.mo_coeff.conj().T @ wu
    md = trial_data.mo_coeff.conj().T @ wd
    gu = _half_green_from_overlap_matrix(wu, mu)
    gd = _half_green_from_overlap_matrix(wd, md)
    dm_u = gu.T @ trial_data.mo_coeff.conj().T
    dm_d = gd.T @ trial_data.mo_coeff.conj().T
    return jnp.stack([dm_u, dm_d], axis=0)


def rdm1_kernel_gw(
    walker: jax.Array,
    ham_data: Any,
    meas_ctx: RhfMeasCtx,
    trial_data: RhfTrial,
) -> jax.Array:
    norb, nocc = trial_data.norb, trial_data.nocc
    cH = trial_data.mo_coeff.conj().T
    top = cH @ walker[:norb, :]
    bot = cH @ walker[norb:, :]
    m = jnp.vstack([top, bot])

    g_half = _half_green_from_overlap_matrix(walker, m)
    dm_u = g_half[:nocc, :norb].T @ cH
    dm_d = g_half[nocc:, norb:].T @ cH
    return jnp.stack([dm_u, dm_d], axis=0)


def density_corr_kernel_rw(
    walker: jax.Array,
    ham_data: Any,
    meas_ctx: RhfMeasCtx,
    trial_data: RhfTrial,
) -> jax.Array:
    m = trial_data.mo_coeff.conj().T @ walker
    g_half = _half_green_from_overlap_matrix(walker, m)
    g = g_half.T @ trial_data.mo_coeff.conj().T
    return _density_corr_from_greens(g, g)


def density_corr_kernel_uw(
    walker: tuple[jax.Array, jax.Array],
    ham_data: Any,
    meas_ctx: RhfMeasCtx,
    trial_data: RhfTrial,
) -> jax.Array:
    wu, wd = walker
    cH = trial_data.mo_coeff.conj().T
    gu = _half_green_from_overlap_matrix(wu, cH @ wu)
    gd = _half_green_from_overlap_matrix(wd, cH @ wd)
    ga = gu.T @ cH
    gb = gd.T @ cH
    return _density_corr_from_greens(ga, gb)


def density_corr_kernel_gw(
    walker: jax.Array,
    ham_data: Any,
    meas_ctx: RhfMeasCtx,
    trial_data: RhfTrial,
) -> jax.Array:
    norb, nocc = trial_data.norb, trial_data.nocc
    cH = trial_data.mo_coeff.conj().T
    top = cH @ walker[:norb, :]
    bot = cH @ walker[norb:, :]
    m = jnp.vstack([top, bot])

    g_half = _half_green_from_overlap_matrix(walker, m)
    ga = g_half[:nocc, :norb].T @ cH
    gb = g_half[nocc:, norb:].T @ cH
    return _density_corr_from_greens(ga, gb)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class RhfMeasCtx:
    # half-rotated:
    rot_h1: jax.Array  # (nocc, norb)
    rot_chol: jax.Array  # (n_chol, nocc, norb)
    rot_chol_flat: jax.Array  # (n_chol, nocc*norb)
    cfg: RhfMeasCfg = RhfMeasCfg()

    def tree_flatten(self):
        children = (self.rot_h1, self.rot_chol, self.rot_chol_flat)
        aux = (self.cfg,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (cfg,) = aux
        rot_h1, rot_chol, rot_chol_flat = children
        return cls(
            rot_h1=rot_h1,
            rot_chol=rot_chol,
            rot_chol_flat=rot_chol_flat,
            cfg=cfg,
        )


def build_meas_ctx(
    ham_data: HamChol,
    trial_data: RhfTrial,
    cfg: RhfMeasCfg = RhfMeasCfg(),
) -> RhfMeasCtx:
    if ham_data.basis != "restricted":
        raise ValueError("RHF MeasOps currently assumes HamChol.basis == 'restricted'.")
    cH = trial_data.mo_coeff.conj().T  # (nocc, norb)
    rot_h1 = cH @ ham_data.h1  # (nocc, norb)
    rot_chol = jnp.einsum("pi,gij->gpj", cH, ham_data.chol, optimize="optimal")
    rot_chol_flat = rot_chol.reshape(rot_chol.shape[0], -1)
    return RhfMeasCtx(rot_h1=rot_h1, rot_chol=rot_chol, rot_chol_flat=rot_chol_flat, cfg=cfg)


def make_rhf_meas_ops(sys: System, memory_mode: str = "high") -> MeasOps:
    cfg = RhfMeasCfg(memory_mode=_validate_memory_mode(memory_mode))
    wk = sys.walker_kind.lower()
    if wk == "restricted":
        overlap_fn = overlap_r
        build_meas_ctx_fn = lambda ham_data, trial_data: build_meas_ctx(ham_data, trial_data, cfg)
        kernels = {
            k_force_bias: force_bias_kernel_rw_rh,
            k_energy: energy_kernel_rw_rh,
        }
        observables = {
            o_rdm1: rdm1_kernel_rw,
            o_density_corr: density_corr_kernel_rw,
        }
    elif wk == "unrestricted":
        overlap_fn = overlap_u
        build_meas_ctx_fn = lambda ham_data, trial_data: build_meas_ctx(ham_data, trial_data, cfg)
        kernels = {
            k_force_bias: force_bias_kernel_uw_rh,
            k_energy: energy_kernel_uw_rh,
        }
        observables = {
            o_rdm1: rdm1_kernel_uw,
            o_density_corr: density_corr_kernel_uw,
        }
    elif wk == "generalized":
        overlap_fn = overlap_g
        build_meas_ctx_fn = lambda ham_data, trial_data: build_meas_ctx(ham_data, trial_data, cfg)
        kernels = {
            k_force_bias: force_bias_kernel_gw_rh,
        }
        observables = {
            o_rdm1: rdm1_kernel_gw,
            o_density_corr: density_corr_kernel_gw,
        }
    else:
        raise ValueError(f"unknown walker_kind: {sys.walker_kind}")

    meas_ops = MeasOps(
        overlap=overlap_fn,
        build_meas_ctx=build_meas_ctx_fn,
        kernels=kernels,
        observables=observables,
    )
    object.__setattr__(meas_ops, _RHF_MEAS_CFG_ATTR, cfg)
    return meas_ops


def lnoenergy_kernel_rw_rh(
    walker: jax.Array, ham_data: HamChol, meas_ctx: LnoRhfMeasCtx, trial_data: RhfTrial
) -> jax.Array:
    m = trial_data.mo_coeff.conj().T @ walker
    g_half = _half_green_from_overlap_matrix(walker, m)  # (nocc, norb)
    prjlo_mat = jnp.dot(meas_ctx.prjlo.T, meas_ctx.prjlo)
    nocc = trial_data.nocc
    f = jnp.einsum(
        "gij,jk->gik",
        meas_ctx.rot_chol[:, :nocc, nocc:],
        g_half.T[nocc:, :nocc],
        optimize="optimal",
    )
    c = jax.vmap(jnp.trace)(f)
    eneo2Jt = jnp.einsum("Gxk,xk,G->", f, prjlo_mat, c) * 2
    eneo2ext = jnp.einsum("Gxy,Gyk,xk->", f, f, prjlo_mat)

    return eneo2Jt - eneo2ext


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LnoRhfMeasCtx:
    rot_h1: jax.Array
    rot_chol: jax.Array
    rot_chol_flat: jax.Array
    prjlo: jax.Array
    cfg: RhfMeasCfg = RhfMeasCfg()

    def tree_flatten(self):
        children = (self.rot_h1, self.rot_chol, self.rot_chol_flat, self.prjlo)
        aux = (self.cfg,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (cfg,) = aux
        rot_h1, rot_chol, rot_chol_flat, prjlo = children
        return cls(
            rot_h1=rot_h1,
            rot_chol=rot_chol,
            rot_chol_flat=rot_chol_flat,
            prjlo=prjlo,
            cfg=cfg,
        )


def build_lno_meas_ctx(
    ham_data: HamChol,
    trial_data: RhfTrial,
    prjlo: jax.Array,
) -> LnoRhfMeasCtx:
    if ham_data.basis != "restricted":
        raise ValueError("RHF MeasOps currently assumes HamChol.basis == 'restricted'.")
    cH = trial_data.mo_coeff.conj().T  # (nocc, norb)
    rot_h1 = cH @ ham_data.h1  # (nocc, norb)
    rot_chol = jnp.einsum("pi,gij->gpj", cH, ham_data.chol, optimize="optimal")
    rot_chol_flat = rot_chol.reshape(rot_chol.shape[0], -1)
    return LnoRhfMeasCtx(rot_h1=rot_h1, rot_chol=rot_chol, rot_chol_flat=rot_chol_flat, prjlo=prjlo)


def make_build_lno_meas_ctx(prjlo):
    def build_meas_ctx(ham_data: HamChol, trial_data: RhfTrial) -> LnoRhfMeasCtx:
        return build_lno_meas_ctx(
            ham_data=ham_data,
            trial_data=trial_data,
            prjlo=prjlo,
        )

    return build_meas_ctx


def make_lno_rhf_meas_ops(sys: System, params: QmcParamsLno) -> MeasOps:
    wk = sys.walker_kind.lower()
    if wk == "restricted":
        overlap_fn = overlap_r
        build_meas_ctx_fn = make_build_lno_meas_ctx(params.prjlo)
        kernels = {
            k_force_bias: force_bias_kernel_rw_rh,
            k_energy: energy_kernel_rw_rh,
        }
        observables = {
            o_orb_corr: lnoenergy_kernel_rw_rh,
        }
    elif wk == "unrestricted" or wk == "generalized":
        raise NotImplementedError
    else:
        raise ValueError(f"unknown walker_kind: {sys.walker_kind}")

    return MeasOps(
        overlap=overlap_fn,
        build_meas_ctx=build_meas_ctx_fn,
        kernels=kernels,
        observables=observables,
    )
