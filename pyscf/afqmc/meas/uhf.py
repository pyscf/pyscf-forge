from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import tree_util

from ..core.ops import MeasOps, k_energy, k_force_bias, o_density_corr, o_rdm1
from ..core.system import System
from ..ham.chol import HamChol
from ..trial.uhf import UhfTrial, overlap_g, overlap_r, overlap_u


def _half_green_from_overlap_matrix(w: jax.Array, ovlp_mat: jax.Array) -> jax.Array:
    """
    green_half = (w @ inv(ovlp_mat)).T
    """
    return jnp.linalg.solve(ovlp_mat.T, w.T)


def _build_bra_generalized(trial_data: UhfTrial) -> jax.Array:
    Atrial = trial_data.mo_coeff_a
    Btrial = trial_data.mo_coeff_b
    bra = jnp.block([[Atrial, 0 * Btrial], [0 * Atrial, Btrial]])
    return bra


def force_bias_kernel_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: UhfMeasCtx,
    trial_data: UhfTrial,
) -> jax.Array:
    n_elec_0 = trial_data.nocc[0]
    n_elec_1 = trial_data.nocc[1]
    return force_bias_kernel_uw_rh(
        (walker[:, :n_elec_0], walker[:, :n_elec_1]), ham_data, meas_ctx, trial_data
    )


def force_bias_kernel_uw_rh(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamChol,
    meas_ctx: UhfMeasCtx,
    trial_data: UhfTrial,
) -> jax.Array:
    wu, wd = walker
    mu = trial_data.mo_coeff_a.conj().T @ wu
    md = trial_data.mo_coeff_b.conj().T @ wd
    gu = _half_green_from_overlap_matrix(wu, mu)  # (nocc[0], norb)
    gd = _half_green_from_overlap_matrix(wd, md)  # (nocc[1], norb)

    fb_u = jnp.einsum("gij,ij->g", meas_ctx.rot_chol_a, gu, optimize="optimal")
    fb_d = jnp.einsum("gij,ij->g", meas_ctx.rot_chol_b, gd, optimize="optimal")
    return fb_u + fb_d


def force_bias_kernel_gw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: UhfMeasCtx,
    trial_data: UhfTrial,
) -> jax.Array:
    w = walker
    norb = trial_data.norb
    na, _ = trial_data.nocc

    bra = _build_bra_generalized(trial_data)
    g = _half_green_from_overlap_matrix(w, bra.T.conj() @ w)

    g_aa, g_bb = g[:na, :norb], g[na:, norb:]

    rot_chol_aa = meas_ctx.rot_chol_a
    rot_chol_bb = meas_ctx.rot_chol_b

    fb = jnp.einsum("gij,ij->g", rot_chol_aa, g_aa, optimize="optimal")
    fb += jnp.einsum("gij,ij->g", rot_chol_bb, g_bb, optimize="optimal")

    return fb


def rdm1_kernel_rw(
    walker: jax.Array,
    ham_data: Any,
    meas_ctx: UhfMeasCtx,
    trial_data: UhfTrial,
) -> jax.Array:
    n_elec_0 = trial_data.nocc[0]
    n_elec_1 = trial_data.nocc[1]
    return rdm1_kernel_uw(
        (walker[:, :n_elec_0], walker[:, :n_elec_1]), ham_data, meas_ctx, trial_data
    )


def rdm1_kernel_uw(
    walker: tuple[jax.Array, jax.Array],
    ham_data: Any,
    meas_ctx: UhfMeasCtx,
    trial_data: UhfTrial,
) -> jax.Array:
    wu, wd = walker
    mu = trial_data.mo_coeff_a.conj().T @ wu
    md = trial_data.mo_coeff_b.conj().T @ wd
    gu = _half_green_from_overlap_matrix(wu, mu)
    gd = _half_green_from_overlap_matrix(wd, md)
    dm_a = gu.T @ trial_data.mo_coeff_a.conj().T
    dm_b = gd.T @ trial_data.mo_coeff_b.conj().T
    return jnp.stack([dm_a, dm_b], axis=0)


def rdm1_kernel_gw(
    walker: jax.Array,
    ham_data: Any,
    meas_ctx: UhfMeasCtx,
    trial_data: UhfTrial,
) -> jax.Array:
    w = walker
    norb = trial_data.norb
    na, _ = trial_data.nocc
    bra = _build_bra_generalized(trial_data)
    g = _half_green_from_overlap_matrix(w, bra.T.conj() @ w)
    dm_a = g[:na, :norb].T @ trial_data.mo_coeff_a.conj().T
    dm_b = g[na:, norb:].T @ trial_data.mo_coeff_b.conj().T
    return jnp.stack([dm_a, dm_b], axis=0)


def _density_corr_from_greens(ga: jax.Array, gb: jax.Array) -> jax.Array:
    """
    Density correlation from spin-resolved Green's functions ga (norb, norb)
    and gb (norb, norb). Returns (3, norb, norb) array of uu, ud, dd correlations.
    """
    na = jnp.diagonal(ga)
    nb = jnp.diagonal(gb)

    # same-spin: n_i n_j = G_ii G_jj - G_ij G_ji + delta_ij G_ii
    uu = na[:, None] * na[None, :] - ga * ga.T + jnp.diag(na)
    dd = nb[:, None] * nb[None, :] - gb * gb.T + jnp.diag(nb)

    # opposite-spin: n_ia n_jb = G^a_ii G^b_jj  (no exchange)
    ud = na[:, None] * nb[None, :]

    return jnp.stack([uu, ud, dd], axis=0)


def density_corr_kernel_uw(
    walker: tuple[jax.Array, jax.Array],
    ham_data: Any,
    meas_ctx: UhfMeasCtx,
    trial_data: UhfTrial,
) -> jax.Array:
    wu, wd = walker
    mu = trial_data.mo_coeff_a.conj().T @ wu
    md = trial_data.mo_coeff_b.conj().T @ wd
    gu = _half_green_from_overlap_matrix(wu, mu)
    gd = _half_green_from_overlap_matrix(wd, md)
    ga = gu.T @ trial_data.mo_coeff_a.conj().T
    gb = gd.T @ trial_data.mo_coeff_b.conj().T
    return _density_corr_from_greens(ga, gb)


def density_corr_kernel_rw(
    walker: jax.Array,
    ham_data: Any,
    meas_ctx: UhfMeasCtx,
    trial_data: UhfTrial,
) -> jax.Array:
    n_elec_0 = trial_data.nocc[0]
    n_elec_1 = trial_data.nocc[1]
    return density_corr_kernel_uw(
        (walker[:, :n_elec_0], walker[:, :n_elec_1]), ham_data, meas_ctx, trial_data
    )


def density_corr_kernel_gw(
    walker: jax.Array,
    ham_data: Any,
    meas_ctx: UhfMeasCtx,
    trial_data: UhfTrial,
) -> jax.Array:
    w = walker
    norb = trial_data.norb
    na, _ = trial_data.nocc
    bra = _build_bra_generalized(trial_data)
    g = _half_green_from_overlap_matrix(w, bra.T.conj() @ w)
    ga = g[:na, :norb].T @ trial_data.mo_coeff_a.conj().T
    gb = g[na:, norb:].T @ trial_data.mo_coeff_b.conj().T
    return _density_corr_from_greens(ga, gb)


def energy_kernel_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: UhfMeasCtx,
    trial_data: UhfTrial,
) -> jax.Array:
    n_elec_0 = trial_data.nocc[0]
    n_elec_1 = trial_data.nocc[1]
    return energy_kernel_uw_rh(
        (walker[:, :n_elec_0], walker[:, :n_elec_1]), ham_data, meas_ctx, trial_data
    )


def energy_kernel_uw_rh(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamChol,
    meas_ctx: UhfMeasCtx,
    trial_data: UhfTrial,
) -> jax.Array:
    wu, wd = walker
    mu = trial_data.mo_coeff_a.conj().T @ wu
    md = trial_data.mo_coeff_b.conj().T @ wd
    gu = _half_green_from_overlap_matrix(wu, mu)
    gd = _half_green_from_overlap_matrix(wd, md)

    e0 = ham_data.h0
    e1 = jnp.sum(gu * meas_ctx.rot_h1_a) + jnp.sum(gd * meas_ctx.rot_h1_b)

    f_up = jnp.einsum("gij,jk->gik", meas_ctx.rot_chol_a, gu.T, optimize="optimal")
    f_dn = jnp.einsum("gij,jk->gik", meas_ctx.rot_chol_b, gd.T, optimize="optimal")
    c_up = jax.vmap(jnp.trace)(f_up)
    c_dn = jax.vmap(jnp.trace)(f_dn)
    exc_up = jnp.sum(jax.vmap(lambda x: x * x.T)(f_up))
    exc_dn = jnp.sum(jax.vmap(lambda x: x * x.T)(f_dn))

    e2 = (
        jnp.sum(c_up * c_up) + jnp.sum(c_dn * c_dn) + 2.0 * jnp.sum(c_up * c_dn) - exc_up - exc_dn
    ) / 2.0

    return e0 + e1 + e2


def energy_kernel_gw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: UhfMeasCtx,
    trial_data: UhfTrial,
) -> jax.Array:
    w = walker
    norb = trial_data.norb
    na, nb = trial_data.nocc

    bra = _build_bra_generalized(trial_data)
    g = _half_green_from_overlap_matrix(w, bra.T.conj() @ w)

    g_aa = g[:na, :norb]
    g_bb = g[na:, norb:]
    g_ab = g[:na, norb:]
    g_ba = g[na:, :norb]

    rot_h1_a = meas_ctx.rot_h1_a
    rot_h1_b = meas_ctx.rot_h1_b

    rot_chol_a = meas_ctx.rot_chol_a
    rot_chol_b = meas_ctx.rot_chol_b

    e0 = ham_data.h0

    e1 = jnp.sum(g_aa * rot_h1_a) + jnp.sum(g_bb * rot_h1_b)

    f_up = jnp.einsum("gij,jk->gik", rot_chol_a, g_aa.T, optimize="optimal")
    f_dn = jnp.einsum("gij,jk->gik", rot_chol_b, g_bb.T, optimize="optimal")
    c_up = jax.vmap(jnp.trace)(f_up)
    c_dn = jax.vmap(jnp.trace)(f_dn)
    J = jnp.sum(c_up * c_up) + jnp.sum(c_dn * c_dn) + 2.0 * jnp.sum(c_up * c_dn)

    K = jnp.sum(jax.vmap(lambda x: x * x.T)(f_up)) + jnp.sum(jax.vmap(lambda x: x * x.T)(f_dn))

    f_ab = jnp.einsum("gip,pj->gij", rot_chol_a, g_ba.T, optimize="optimal")
    f_ba = jnp.einsum("gip,pj->gij", rot_chol_b, g_ab.T, optimize="optimal")
    K += 2.0 * jnp.sum(jax.vmap(lambda x, y: x * y.T)(f_ab, f_ba))

    return e0 + e1 + (J - K) / 2.0


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class UhfMeasCtx:
    # half-rotated:
    rot_h1_a: jax.Array  # (nocc[0], norb)
    rot_h1_b: jax.Array  # (nocc[1], norb)
    rot_chol_a: jax.Array  # (n_chol, nocc[0], norb)
    rot_chol_b: jax.Array  # (n_chol, nocc[1], norb)
    rot_chol_flat_a: jax.Array  # (n_chol, nocc[0]*norb)
    rot_chol_flat_b: jax.Array  # (n_chol, nocc[1]*norb)

    def tree_flatten(self):
        return (
            self.rot_h1_a,
            self.rot_h1_b,
            self.rot_chol_a,
            self.rot_chol_b,
            self.rot_chol_flat_a,
            self.rot_chol_flat_b,
        ), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        (
            rot_h1_a,
            rot_h1_b,
            rot_chol_a,
            rot_chol_b,
            rot_chol_flat_a,
            rot_chol_flat_b,
        ) = children
        return cls(
            rot_h1_a=rot_h1_a,
            rot_h1_b=rot_h1_b,
            rot_chol_a=rot_chol_a,
            rot_chol_b=rot_chol_b,
            rot_chol_flat_a=rot_chol_flat_a,
            rot_chol_flat_b=rot_chol_flat_b,
        )


def build_meas_ctx(ham_data: HamChol, trial_data: UhfTrial) -> UhfMeasCtx:
    if ham_data.basis != "restricted":
        raise ValueError("UHF MeasOps currently assumes HamChol.basis == 'restricted'.")
    caH = trial_data.mo_coeff_a.conj().T  # (nocc[0], norb)
    cbH = trial_data.mo_coeff_b.conj().T  # (nocc[1], norb)
    rot_h1_a = caH @ ham_data.h1  # (nocc[0], norb)
    rot_h1_b = cbH @ ham_data.h1  # (nocc[1], norb)
    rot_chol_a = jnp.einsum("pi,gij->gpj", caH, ham_data.chol, optimize="optimal")
    rot_chol_b = jnp.einsum("pi,gij->gpj", cbH, ham_data.chol, optimize="optimal")
    rot_chol_flat_a = rot_chol_a.reshape(rot_chol_a.shape[0], -1)
    rot_chol_flat_b = rot_chol_b.reshape(rot_chol_b.shape[0], -1)
    return UhfMeasCtx(
        rot_h1_a=rot_h1_a,
        rot_h1_b=rot_h1_b,
        rot_chol_a=rot_chol_a,
        rot_chol_b=rot_chol_b,
        rot_chol_flat_a=rot_chol_flat_a,
        rot_chol_flat_b=rot_chol_flat_b,
    )


def make_uhf_meas_ops(sys: System) -> MeasOps:
    wk = sys.walker_kind.lower()
    if wk == "restricted":
        overlap_fn = overlap_r
        build_meas_ctx_fn = build_meas_ctx
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
        build_meas_ctx_fn = build_meas_ctx
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
        build_meas_ctx_fn = build_meas_ctx
        kernels = {k_force_bias: force_bias_kernel_gw_rh, k_energy: energy_kernel_gw_rh}
        observables = {
            o_rdm1: rdm1_kernel_gw,
            o_density_corr: density_corr_kernel_gw,
        }
    else:
        raise ValueError(f"unknown walker_kind: {sys.walker_kind}")

    return MeasOps(
        overlap=overlap_fn,
        build_meas_ctx=build_meas_ctx_fn,
        kernels=kernels,
        observables=observables,
    )
