from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import lax, tree_util

from ..core.levels import LevelPack, LevelSpec
from ..core.ops import MeasOps, k_energy, k_force_bias, o_rdm1
from ..core.system import System
from ..ham.chol import HamChol, slice_ham_level
from ..trial.cisd import CisdTrial, slice_trial_level
from ..trial.cisd import overlap_r as cisd_overlap_r

_CISD_MEAS_CFG_ATTR = "_cisd_meas_cfg"


def _greens_restricted(walker: jax.Array, nocc: int) -> jax.Array:
    wocc = walker[:nocc, :]  # (nocc, nocc)
    return jnp.linalg.solve(wocc.T, walker.T)  # (nocc, norb)


def _active_green_blocks(
    green: jax.Array, trial_data: CisdTrial
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Extract active occupied/virtual blocks from the full-space Green's function.

    Returns:
      green_act:  (nocc_act, norb_full)
      green_occ:  (nocc_act, nvir_act)
      greenp_act: (norb_full, nvir_act)
    """
    occ_act = trial_data.occ_act_slice
    vir_act = trial_data.vir_act_slice

    green_act = green[occ_act, :]
    green_occ = green[occ_act, vir_act]

    greenp_act = jnp.zeros((trial_data.norb, trial_data.nvir), dtype=green.dtype)
    greenp_act = greenp_act.at[: trial_data.nocc_full, :].set(green[:, vir_act])
    greenp_act = greenp_act.at[vir_act, :].set(-jnp.eye(trial_data.nvir, dtype=green.dtype))
    return green_act, green_occ, greenp_act


@dataclass(frozen=True)
class CisdMeasCfg:
    memory_mode: str = "low"  # or Literal["low","high"]
    mixed_real_dtype: jnp.dtype = jnp.float64
    mixed_complex_dtype: jnp.dtype = jnp.complex128
    mixed_real_dtype_testing: jnp.dtype = jnp.float32
    mixed_complex_dtype_testing: jnp.dtype = jnp.complex64


def get_cisd_meas_cfg(meas_ops: MeasOps) -> CisdMeasCfg | None:
    cfg = getattr(meas_ops, _CISD_MEAS_CFG_ATTR, None)
    if isinstance(cfg, CisdMeasCfg):
        return cfg
    return None


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CisdMeasCtx:
    rot_chol: jax.Array  # (n_chol, nocc_full, norb_full)
    lci1: jax.Array  # (n_chol, norb_full, nocc_act)
    cfg: CisdMeasCfg  # static

    def tree_flatten(self):
        children = (self.rot_chol, self.lci1)
        aux = (self.cfg,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (cfg,) = aux
        rot_chol, lci1 = children
        return cls(rot_chol=rot_chol, lci1=lci1, cfg=cfg)


def slice_meas_ctx_chol(ctx: CisdMeasCtx, nchol_keep: int | None) -> CisdMeasCtx:
    if nchol_keep is None:
        return ctx
    return CisdMeasCtx(
        rot_chol=ctx.rot_chol[:nchol_keep],
        lci1=ctx.lci1[:nchol_keep],
        cfg=ctx.cfg,
    )


def build_meas_ctx(
    ham_data: HamChol, trial_data: CisdTrial, cfg: CisdMeasCfg = CisdMeasCfg()
) -> CisdMeasCtx:
    if ham_data.basis != "restricted":
        raise ValueError("CISD MeasOps currently assumes HamChol.basis == 'restricted'.")

    chol = ham_data.chol  # (n_chol, norb_full, norb_full)
    nocc_full = trial_data.nocc_full

    rot_chol = chol[:, :nocc_full, :]  # (n_chol, nocc_full, norb_full)

    lci1 = jnp.einsum(
        "git,pt->gip",
        chol[:, :, trial_data.vir_act_slice],
        trial_data.ci1,
        optimize="optimal",
    )  # (n_chol, norb_full, nocc_act)

    return CisdMeasCtx(rot_chol=rot_chol, lci1=lci1, cfg=cfg)


def _force_bias_chol_contract_high_complex(
    chol: jax.Array, mat: jax.Array, cfg: CisdMeasCfg
) -> jax.Array:
    return jnp.einsum(
        "gij,ij->g",
        chol.astype(cfg.mixed_real_dtype),
        mat.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )


def _force_bias_chol_contract_high_realimag(
    chol: jax.Array, mat: jax.Array, cfg: CisdMeasCfg
) -> jax.Array:
    chol_r = chol.astype(cfg.mixed_real_dtype)
    mat_r = jnp.real(mat).astype(cfg.mixed_real_dtype)
    mat_i = jnp.imag(mat).astype(cfg.mixed_real_dtype)
    imag_unit = jnp.asarray(1.0j, dtype=cfg.mixed_complex_dtype)

    real_part = jnp.einsum("gij,ij->g", chol_r, mat_r, optimize="optimal")
    imag_part = jnp.einsum("gij,ij->g", chol_r, mat_i, optimize="optimal")
    return real_part.astype(cfg.mixed_complex_dtype) + imag_unit * imag_part.astype(
        cfg.mixed_complex_dtype
    )


def _force_bias_chol_contract_low(chol: jax.Array, mat: jax.Array, cfg: CisdMeasCfg) -> jax.Array:
    mat_r = jnp.real(mat).astype(cfg.mixed_real_dtype)
    mat_i = jnp.imag(mat).astype(cfg.mixed_real_dtype)
    imag_unit = jnp.asarray(1.0j, dtype=cfg.mixed_complex_dtype)

    def scan_body(_, chol_i: jax.Array) -> tuple[None, jax.Array]:
        chol_i_r = chol_i.astype(cfg.mixed_real_dtype)
        real_part = jnp.einsum("ij,ij->", chol_i_r, mat_r, optimize="optimal")
        imag_part = jnp.einsum("ij,ij->", chol_i_r, mat_i, optimize="optimal")
        y = real_part.astype(cfg.mixed_complex_dtype) + imag_unit * imag_part.astype(
            cfg.mixed_complex_dtype
        )
        return None, y

    _, ys = lax.scan(scan_body, None, chol)
    return ys


def _force_bias_ci2g_low(
    ci2: jax.Array, green_occ: jax.Array, cfg: CisdMeasCfg
) -> tuple[jax.Array, jax.Array]:
    ci2_t = ci2.astype(cfg.mixed_real_dtype)
    green_t = green_occ.astype(cfg.mixed_complex_dtype)
    out_shape = (int(ci2.shape[2]), int(ci2.shape[3]))
    zero_c = jnp.zeros(out_shape, dtype=cfg.mixed_complex_dtype)
    zero_e = jnp.zeros(out_shape, dtype=cfg.mixed_complex_dtype)

    def scan_body(
        carry: tuple[jax.Array, jax.Array], xs: tuple[jax.Array, jax.Array]
    ) -> tuple[tuple[jax.Array, jax.Array], None]:
        acc_c, acc_e = carry
        ci2_p, green_p = xs
        acc_c = acc_c + jnp.einsum("tqu,t->qu", ci2_p, green_p, optimize="optimal")
        acc_e = acc_e + jnp.einsum("tqu,u->qt", ci2_p, green_p, optimize="optimal")
        return (acc_c, acc_e), None

    (ci2g_c, ci2g_e), _ = lax.scan(scan_body, (zero_c, zero_e), (ci2_t, green_t))
    return ci2g_c, ci2g_e


def _force_bias_ci2g_high_realimag(
    ci2: jax.Array, green_occ: jax.Array, cfg: CisdMeasCfg
) -> tuple[jax.Array, jax.Array]:
    ci2_t = ci2.astype(cfg.mixed_real_dtype)
    green_r = jnp.real(green_occ).astype(cfg.mixed_real_dtype)
    green_i = jnp.imag(green_occ).astype(cfg.mixed_real_dtype)
    imag_unit = jnp.asarray(1.0j, dtype=cfg.mixed_complex_dtype)

    ci2g_c_r = jnp.einsum("ptqu,pt->qu", ci2_t, green_r, optimize="optimal")
    ci2g_c_i = jnp.einsum("ptqu,pt->qu", ci2_t, green_i, optimize="optimal")
    ci2g_e_r = jnp.einsum("ptqu,pu->qt", ci2_t, green_r, optimize="optimal")
    ci2g_e_i = jnp.einsum("ptqu,pu->qt", ci2_t, green_i, optimize="optimal")

    ci2g_c = ci2g_c_r.astype(cfg.mixed_complex_dtype) + imag_unit * ci2g_c_i.astype(
        cfg.mixed_complex_dtype
    )
    ci2g_e = ci2g_e_r.astype(cfg.mixed_complex_dtype) + imag_unit * ci2g_e_i.astype(
        cfg.mixed_complex_dtype
    )
    return ci2g_c, ci2g_e


def _ci1gp_low(
    ci1: jax.Array, green: jax.Array, trial_data: CisdTrial, cfg: CisdMeasCfg
) -> jax.Array:
    """
    Build ci1 @ greenp^T without materializing greenp.

    Nonzero columns of the result live only on:
      - occupied_full: green[:, vir_act] projected by ci1
      - active_virtual: -ci1
    """
    nocc_act = trial_data.nocc
    norb = trial_data.norb
    ci1_t = ci1.astype(cfg.mixed_real_dtype)
    green_t = green[:, trial_data.vir_act_slice].astype(cfg.mixed_complex_dtype)
    ci1gp = jnp.zeros((nocc_act, norb), dtype=cfg.mixed_complex_dtype)
    ci1gp = ci1gp.at[:, : trial_data.nocc_full].set(ci1_t @ green_t.T)
    ci1gp = ci1gp.at[:, trial_data.vir_act_slice].set(-ci1_t)
    return ci1gp


def _greenp_times_ci2g_t_low(
    ci2g: jax.Array, green: jax.Array, trial_data: CisdTrial, cfg: CisdMeasCfg
) -> jax.Array:
    """
    Build greenp @ ci2g^T without materializing greenp.

    Nonzero rows of the result live only on:
      - occupied_full: green[:, vir_act] @ ci2g^T
      - active_virtual: -ci2g^T
    """
    green_t = green[:, trial_data.vir_act_slice].astype(cfg.mixed_complex_dtype)
    out = jnp.zeros((trial_data.norb, trial_data.nocc), dtype=cfg.mixed_complex_dtype)
    out = out.at[: trial_data.nocc_full, :].set(green_t @ ci2g.T)
    out = out.at[trial_data.vir_act_slice, :].set(-ci2g.T)
    return out


def _energy_l2ci2_scalar_realimag(
    glgp_i: jax.Array, ci2_t: jax.Array
) -> tuple[jax.Array, jax.Array]:
    x_r = jnp.real(glgp_i).astype(ci2_t.dtype)
    x_i = jnp.imag(glgp_i).astype(ci2_t.dtype)
    imag_unit = jnp.asarray(1.0j, dtype=glgp_i.dtype)

    t1_rr = jnp.einsum("pt,qu,ptqu->", x_r, x_r, ci2_t, optimize="optimal")
    t1_ii = jnp.einsum("pt,qu,ptqu->", x_i, x_i, ci2_t, optimize="optimal")
    t1_ri = jnp.einsum("pt,qu,ptqu->", x_r, x_i, ci2_t, optimize="optimal")
    t1_ir = jnp.einsum("pt,qu,ptqu->", x_i, x_r, ci2_t, optimize="optimal")
    t1 = (t1_rr - t1_ii).astype(glgp_i.dtype) + imag_unit * (t1_ri + t1_ir).astype(glgp_i.dtype)

    t2_rr = jnp.einsum("pu,qt,ptqu->", x_r, x_r, ci2_t, optimize="optimal")
    t2_ii = jnp.einsum("pu,qt,ptqu->", x_i, x_i, ci2_t, optimize="optimal")
    t2_ri = jnp.einsum("pu,qt,ptqu->", x_r, x_i, ci2_t, optimize="optimal")
    t2_ir = jnp.einsum("pu,qt,ptqu->", x_i, x_r, ci2_t, optimize="optimal")
    t2 = (t2_rr - t2_ii).astype(glgp_i.dtype) + imag_unit * (t2_ri + t2_ir).astype(glgp_i.dtype)
    return t1, t2


def _energy_l2ci2_scalar_blocked_realimag(
    glgp_i: jax.Array, ci2_t: jax.Array
) -> tuple[jax.Array, jax.Array]:
    x_r = jnp.real(glgp_i).astype(ci2_t.dtype)
    x_i = jnp.imag(glgp_i).astype(ci2_t.dtype)
    imag_unit = jnp.asarray(1.0j, dtype=glgp_i.dtype)

    zero = jnp.array(0.0, dtype=ci2_t.dtype)

    def scan_body(
        carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        xs: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], None]:
        t1_r, t1_i, t2_r, t2_i = carry
        xp_r, xp_i, ci2_p = xs

        a_r = jnp.einsum("t,tqu->qu", xp_r, ci2_p, optimize="optimal")
        a_i = jnp.einsum("t,tqu->qu", xp_i, ci2_p, optimize="optimal")
        t1_r = (
            t1_r
            + jnp.einsum("qu,qu->", a_r, x_r, optimize="optimal")
            - jnp.einsum("qu,qu->", a_i, x_i, optimize="optimal")
        )
        t1_i = (
            t1_i
            + jnp.einsum("qu,qu->", a_r, x_i, optimize="optimal")
            + jnp.einsum("qu,qu->", a_i, x_r, optimize="optimal")
        )

        b_r = jnp.einsum("u,tqu->qt", xp_r, ci2_p, optimize="optimal")
        b_i = jnp.einsum("u,tqu->qt", xp_i, ci2_p, optimize="optimal")
        t2_r = (
            t2_r
            + jnp.einsum("qt,qt->", b_r, x_r, optimize="optimal")
            - jnp.einsum("qt,qt->", b_i, x_i, optimize="optimal")
        )
        t2_i = (
            t2_i
            + jnp.einsum("qt,qt->", b_r, x_i, optimize="optimal")
            + jnp.einsum("qt,qt->", b_i, x_r, optimize="optimal")
        )
        return (t1_r, t1_i, t2_r, t2_i), None

    (t1_r, t1_i, t2_r, t2_i), _ = lax.scan(
        scan_body,
        (zero, zero, zero, zero),
        (x_r, x_i, ci2_t),
    )
    t1 = t1_r.astype(glgp_i.dtype) + imag_unit * t1_i.astype(glgp_i.dtype)
    t2 = t2_r.astype(glgp_i.dtype) + imag_unit * t2_i.astype(glgp_i.dtype)
    return t1, t2


def _energy_l2ci2_batched_realimag(
    glgp: jax.Array, ci2_t: jax.Array
) -> tuple[jax.Array, jax.Array]:
    x_r = jnp.real(glgp).astype(ci2_t.dtype)
    x_i = jnp.imag(glgp).astype(ci2_t.dtype)
    imag_unit = jnp.asarray(1.0j, dtype=glgp.dtype)

    t1_rr = jnp.einsum("gpt,gqu,ptqu->g", x_r, x_r, ci2_t, optimize="optimal")
    t1_ii = jnp.einsum("gpt,gqu,ptqu->g", x_i, x_i, ci2_t, optimize="optimal")
    t1_ri = jnp.einsum("gpt,gqu,ptqu->g", x_r, x_i, ci2_t, optimize="optimal")
    t1_ir = jnp.einsum("gpt,gqu,ptqu->g", x_i, x_r, ci2_t, optimize="optimal")
    t1 = (t1_rr - t1_ii).astype(glgp.dtype) + imag_unit * (t1_ri + t1_ir).astype(glgp.dtype)

    t2_rr = jnp.einsum("gpu,gqt,ptqu->g", x_r, x_r, ci2_t, optimize="optimal")
    t2_ii = jnp.einsum("gpu,gqt,ptqu->g", x_i, x_i, ci2_t, optimize="optimal")
    t2_ri = jnp.einsum("gpu,gqt,ptqu->g", x_r, x_i, ci2_t, optimize="optimal")
    t2_ir = jnp.einsum("gpu,gqt,ptqu->g", x_i, x_r, ci2_t, optimize="optimal")
    t2 = (t2_rr - t2_ii).astype(glgp.dtype) + imag_unit * (t2_ri + t2_ir).astype(glgp.dtype)
    return t1, t2


def _energy_l2ci2_batched_blocked_realimag(
    glgp: jax.Array, ci2_t: jax.Array
) -> tuple[jax.Array, jax.Array]:
    x_r = jnp.real(glgp).astype(ci2_t.dtype)
    x_i = jnp.imag(glgp).astype(ci2_t.dtype)
    imag_unit = jnp.asarray(1.0j, dtype=glgp.dtype)

    x_r_scan = jnp.swapaxes(x_r, 0, 1)
    x_i_scan = jnp.swapaxes(x_i, 0, 1)
    zero = jnp.zeros((x_r.shape[0],), dtype=ci2_t.dtype)

    def scan_body(
        carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        xs: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], None]:
        t1_r, t1_i, t2_r, t2_i = carry
        xp_r, xp_i, ci2_p = xs

        a_r = jnp.einsum("gt,tqu->gqu", xp_r, ci2_p, optimize="optimal")
        a_i = jnp.einsum("gt,tqu->gqu", xp_i, ci2_p, optimize="optimal")
        t1_r = (
            t1_r
            + jnp.einsum("gqu,gqu->g", a_r, x_r, optimize="optimal")
            - jnp.einsum("gqu,gqu->g", a_i, x_i, optimize="optimal")
        )
        t1_i = (
            t1_i
            + jnp.einsum("gqu,gqu->g", a_r, x_i, optimize="optimal")
            + jnp.einsum("gqu,gqu->g", a_i, x_r, optimize="optimal")
        )

        b_r = jnp.einsum("gu,tqu->gqt", xp_r, ci2_p, optimize="optimal")
        b_i = jnp.einsum("gu,tqu->gqt", xp_i, ci2_p, optimize="optimal")
        t2_r = (
            t2_r
            + jnp.einsum("gqt,gqt->g", b_r, x_r, optimize="optimal")
            - jnp.einsum("gqt,gqt->g", b_i, x_i, optimize="optimal")
        )
        t2_i = (
            t2_i
            + jnp.einsum("gqt,gqt->g", b_r, x_i, optimize="optimal")
            + jnp.einsum("gqt,gqt->g", b_i, x_r, optimize="optimal")
        )
        return (t1_r, t1_i, t2_r, t2_i), None

    (t1_r, t1_i, t2_r, t2_i), _ = lax.scan(
        scan_body,
        (zero, zero, zero, zero),
        (x_r_scan, x_i_scan, ci2_t),
    )
    t1 = t1_r.astype(glgp.dtype) + imag_unit * t1_i.astype(glgp.dtype)
    t2 = t2_r.astype(glgp.dtype) + imag_unit * t2_i.astype(glgp.dtype)
    return t1, t2


def _energy_gl_scalar_realimag(green: jax.Array, chol_i: jax.Array, cfg: CisdMeasCfg) -> jax.Array:
    green_r = jnp.real(green).astype(cfg.mixed_real_dtype)
    green_i = jnp.imag(green).astype(cfg.mixed_real_dtype)
    chol_i_r = chol_i.astype(cfg.mixed_real_dtype)
    imag_unit = jnp.asarray(1.0j, dtype=cfg.mixed_complex_dtype)

    real_part = jnp.einsum("pj,ji->pi", green_r, chol_i_r, optimize="optimal")
    imag_part = jnp.einsum("pj,ji->pi", green_i, chol_i_r, optimize="optimal")
    return real_part.astype(cfg.mixed_complex_dtype) + imag_unit * imag_part.astype(
        cfg.mixed_complex_dtype
    )


def _energy_gl_batched_realimag(green: jax.Array, chol: jax.Array, cfg: CisdMeasCfg) -> jax.Array:
    green_r = jnp.real(green).astype(cfg.mixed_real_dtype)
    green_i = jnp.imag(green).astype(cfg.mixed_real_dtype)
    chol_r = chol.astype(cfg.mixed_real_dtype)
    imag_unit = jnp.asarray(1.0j, dtype=cfg.mixed_complex_dtype)

    real_part = jnp.einsum("pj,gji->gpi", green_r, chol_r, optimize="optimal")
    imag_part = jnp.einsum("pj,gji->gpi", green_i, chol_r, optimize="optimal")
    return real_part.astype(cfg.mixed_complex_dtype) + imag_unit * imag_part.astype(
        cfg.mixed_complex_dtype
    )


def _force_bias_common_terms(
    walker: jax.Array, ham_data: HamChol, meas_ctx: CisdMeasCtx, trial_data: CisdTrial
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    ci1, ci2 = trial_data.ci1, trial_data.ci2
    nocc_full = trial_data.nocc_full

    green = _greens_restricted(walker, nocc_full)  # (nocc_full, norb_full)
    green_act, green_occ, greenp = _active_green_blocks(green, trial_data)

    rot_chol = meas_ctx.rot_chol

    lg = jnp.einsum("gpj,pj->g", rot_chol, green, optimize="optimal")

    ci1g = jnp.einsum("pt,pt->", ci1, green_occ, optimize="optimal")
    ci1gp = jnp.einsum("pt,it->pi", ci1, greenp, optimize="optimal")  # (nocc, norb)
    gci1gp = jnp.einsum("pj,pi->ij", green_act, ci1gp, optimize="optimal")  # (norb, norb)

    ci2g_c = jnp.einsum(
        "ptqu,pt->qu",
        ci2.astype(meas_ctx.cfg.mixed_real_dtype),
        green_occ.astype(meas_ctx.cfg.mixed_complex_dtype),
        optimize="optimal",
    )  # (nocc, nvir)
    ci2g_e = jnp.einsum(
        "ptqu,pu->qt",
        ci2.astype(meas_ctx.cfg.mixed_real_dtype),
        green_occ.astype(meas_ctx.cfg.mixed_complex_dtype),
        optimize="optimal",
    )  # (nocc, nvir)

    cisd_green_c = (greenp @ ci2g_c.T) @ green_act  # (norb, norb)
    cisd_green_e = (greenp @ ci2g_e.T) @ green_act  # (norb, norb)
    cisd_green = -4.0 * cisd_green_c + 2.0 * cisd_green_e

    ci2g = 4.0 * ci2g_c - 2.0 * ci2g_e  # (nocc, nvir)
    gci2g = jnp.einsum("qu,qu->", ci2g, green_occ, optimize="optimal")
    overlap = 1.0 + 2.0 * ci1g + 0.5 * gci2g

    return lg, ci1g, gci1gp, cisd_green, overlap, gci2g


def _force_bias_common_terms_high_realimag(
    walker: jax.Array, ham_data: HamChol, meas_ctx: CisdMeasCtx, trial_data: CisdTrial
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    ci1, ci2 = trial_data.ci1, trial_data.ci2
    nocc_full = trial_data.nocc_full

    green = _greens_restricted(walker, nocc_full)
    green_act, green_occ, greenp = _active_green_blocks(green, trial_data)

    lg = jnp.einsum("gpj,pj->g", meas_ctx.rot_chol, green, optimize="optimal")

    ci1g = jnp.einsum("pt,pt->", ci1, green_occ, optimize="optimal")
    ci1gp = jnp.einsum("pt,it->pi", ci1, greenp, optimize="optimal")
    gci1gp = jnp.einsum("pj,pi->ij", green_act, ci1gp, optimize="optimal")

    ci2g_c, ci2g_e = _force_bias_ci2g_high_realimag(ci2, green_occ, meas_ctx.cfg)
    cisd_green_c = (greenp @ ci2g_c.T) @ green_act
    cisd_green_e = (greenp @ ci2g_e.T) @ green_act
    cisd_green = -4.0 * cisd_green_c + 2.0 * cisd_green_e

    ci2g = 4.0 * ci2g_c - 2.0 * ci2g_e
    gci2g = jnp.einsum("qu,qu->", ci2g, green_occ, optimize="optimal")
    overlap = 1.0 + 2.0 * ci1g + 0.5 * gci2g

    return lg, ci1g, gci1gp, cisd_green, overlap, gci2g


def make_level_pack(
    *,
    ham_data: HamChol,
    trial_data: CisdTrial,
    level: LevelSpec,
    orb_fullchol_ctx: CisdMeasCtx | None = None,
    orb_fullchol_ham: HamChol | None = None,
    memory_mode: str = "low",
    mixed_precision: bool = True,
) -> LevelPack:
    cfg = CisdMeasCfg(
        memory_mode=memory_mode,
        mixed_real_dtype=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed_precision else jnp.complex128,
        mixed_real_dtype_testing=jnp.float32,
        mixed_complex_dtype_testing=jnp.complex64,
    )
    if level.nvir_keep is None:
        trial_orb = trial_data
        norb_keep = None
    else:
        trial_orb = slice_trial_level(trial_data, level.nvir_keep)
        norb_keep = int(trial_data.nocc_full) + int(level.nvir_keep)

    if orb_fullchol_ham is None:
        ham_orb_fullchol = slice_ham_level(ham_data, norb_keep=norb_keep, nchol_keep=None)
    else:
        ham_orb_fullchol = orb_fullchol_ham

    if orb_fullchol_ctx is None:
        ctx_orb_fullchol = build_meas_ctx(ham_orb_fullchol, trial_orb, cfg=cfg)
    else:
        ctx_orb_fullchol = orb_fullchol_ctx

    if level.nchol_keep is None:
        ham_lvl = ham_orb_fullchol
        ctx_lvl = ctx_orb_fullchol
    else:
        ham_lvl = slice_ham_level(ham_orb_fullchol, norb_keep=None, nchol_keep=level.nchol_keep)
        ctx_lvl = slice_meas_ctx_chol(ctx_orb_fullchol, level.nchol_keep)

    return LevelPack(
        level=level,
        ham_data=ham_lvl,
        trial_data=trial_orb,
        meas_ctx=ctx_lvl,
        norb_keep=norb_keep,
    )


def force_bias_kernel_rw_rh_high_complex(
    walker: jax.Array, ham_data: HamChol, meas_ctx: CisdMeasCtx, trial_data: CisdTrial
) -> jax.Array:
    lg, ci1g, gci1gp, cisd_green, overlap, gci2g = _force_bias_common_terms(
        walker, ham_data, meas_ctx, trial_data
    )

    fb_0 = 2.0 * lg
    fb_1_1 = 4.0 * ci1g * lg
    fb_1_2 = -2.0 * _force_bias_chol_contract_high_complex(ham_data.chol, gci1gp, meas_ctx.cfg)
    fb_2_1 = lg * gci2g
    fb_2_2 = _force_bias_chol_contract_high_complex(ham_data.chol, cisd_green, meas_ctx.cfg)

    return (fb_0 + fb_1_1 + fb_1_2 + fb_2_1 + fb_2_2) / overlap


def force_bias_kernel_rw_rh_high(
    walker: jax.Array, ham_data: HamChol, meas_ctx: CisdMeasCtx, trial_data: CisdTrial
) -> jax.Array:
    lg, ci1g, gci1gp, cisd_green, overlap, gci2g = _force_bias_common_terms_high_realimag(
        walker, ham_data, meas_ctx, trial_data
    )

    fb_0 = 2.0 * lg
    fb_1_1 = 4.0 * ci1g * lg
    fb_2_1 = lg * gci2g
    fb_corr = _force_bias_chol_contract_high_realimag(
        ham_data.chol,
        cisd_green - 2.0 * gci1gp,
        meas_ctx.cfg,
    )

    return (fb_0 + fb_1_1 + fb_2_1 + fb_corr) / overlap


force_bias_kernel_rw_rh_high_realimag = force_bias_kernel_rw_rh_high


def force_bias_kernel_rw_rh_low(
    walker: jax.Array, ham_data: HamChol, meas_ctx: CisdMeasCtx, trial_data: CisdTrial
) -> jax.Array:
    ci1 = trial_data.ci1
    nocc_full = trial_data.nocc_full

    green = _greens_restricted(walker, nocc_full)
    green_act = green[trial_data.occ_act_slice, :]
    green_occ = green[trial_data.occ_act_slice, trial_data.vir_act_slice]
    green_act_t = green_act.astype(meas_ctx.cfg.mixed_complex_dtype)

    lg = jnp.einsum("gpj,pj->g", meas_ctx.rot_chol, green, optimize="optimal")
    ci1g = jnp.einsum("pt,pt->", ci1, green_occ, optimize="optimal")
    ci1gp = _ci1gp_low(ci1, green, trial_data, meas_ctx.cfg)

    ci2g_c, ci2g_e = _force_bias_ci2g_low(trial_data.ci2, green_occ, meas_ctx.cfg)
    ci2g = 4.0 * ci2g_c - 2.0 * ci2g_e
    gci2g = jnp.einsum("qu,qu->", ci2g, green_occ, optimize="optimal")
    overlap = 1.0 + 2.0 * ci1g + 0.5 * gci2g

    fb_left = -_greenp_times_ci2g_t_low(ci2g, green, trial_data, meas_ctx.cfg) - 2.0 * ci1gp.T
    fb_corr_mat = fb_left @ green_act_t
    fb_corr = _force_bias_chol_contract_low(ham_data.chol, fb_corr_mat, meas_ctx.cfg)

    fb_0 = 2.0 * lg
    fb_1_1 = 4.0 * ci1g * lg
    fb_2_1 = lg * gci2g
    return (fb_0 + fb_1_1 + fb_2_1 + fb_corr) / overlap


def force_bias_kernel_rw_rh(
    walker: jax.Array, ham_data: HamChol, meas_ctx: CisdMeasCtx, trial_data: CisdTrial
) -> jax.Array:
    if meas_ctx.cfg.memory_mode == "low":
        return force_bias_kernel_rw_rh_low(walker, ham_data, meas_ctx, trial_data)
    return force_bias_kernel_rw_rh_high(walker, ham_data, meas_ctx, trial_data)


def energy_kernel_rw_rh(
    walker: jax.Array, ham_data: HamChol, meas_ctx: CisdMeasCtx, trial_data: CisdTrial
) -> jax.Array:
    ci1, ci2 = trial_data.ci1, trial_data.ci2
    nocc_full = trial_data.nocc_full

    green = _greens_restricted(walker, nocc_full)  # (nocc_full, norb_full)
    green_act, green_occ, greenp = _active_green_blocks(green, trial_data)

    h1 = ham_data.h1
    chol = ham_data.chol
    rot_chol = meas_ctx.rot_chol

    # 0 body
    e0 = ham_data.h0

    # 1 body
    hg = jnp.einsum("pj,pj->", h1[:nocc_full, :], green, optimize="optimal")
    e1_0 = 2.0 * hg

    # singles
    ci1g = jnp.einsum("pt,pt->", ci1, green_occ, optimize="optimal")
    e1_1_1 = 4.0 * ci1g * hg
    gpci1 = greenp @ ci1.T  # (norb, nocc)
    ci1_green = gpci1 @ green_act  # (norb, norb)
    e1_1_2 = -2.0 * jnp.einsum("ij,ij->", h1, ci1_green, optimize="optimal")
    e1_1 = e1_1_1 + e1_1_2

    # doubles
    ci2g_c, ci2g_e = _force_bias_ci2g_high_realimag(ci2, green_occ, meas_ctx.cfg)
    ci2_green_c = (greenp @ ci2g_c.T) @ green_act
    ci2_green_e = (greenp @ ci2g_e.T) @ green_act
    ci2_green = 2.0 * ci2_green_c - 1.0 * ci2_green_e

    ci2g = 2.0 * ci2g_c - 1.0 * ci2g_e
    gci2g = jnp.einsum("qu,qu->", ci2g, green_occ, optimize="optimal")

    e1_2_1 = 2.0 * hg * gci2g
    e1_2_2 = -2.0 * jnp.einsum("ij,ij->", h1, ci2_green, optimize="optimal")
    e1_2 = e1_2_1 + e1_2_2

    e1 = e1_0 + e1_1 + e1_2

    # 2 body
    lg = jnp.einsum("gpj,pj->g", rot_chol, green, optimize="optimal")  # (n_chol,)
    e2_0_1 = 2.0 * (lg @ lg)

    e2_1_3_1 = jnp.array(0.0, dtype=jnp.result_type(walker, ci1, ci2))
    e2_1_3_2 = jnp.array(0.0, dtype=jnp.result_type(walker, ci1, ci2))
    if meas_ctx.cfg.memory_mode == "low":
        ci1g1 = ci1 @ green[:, trial_data.vir_act_slice].T  # (nocc_act, nocc_full)
        dtype_acc = jnp.result_type(walker, ci1, ci2)
        zero = jnp.array(0.0, dtype=dtype_acc)

        def scan_lg1_branch(
            carry: tuple[jax.Array, jax.Array, jax.Array],
            xs: tuple[jax.Array, jax.Array],
        ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], None]:
            e20_acc, e2131_acc, e2132_acc = carry
            rot_chol_i, lci1_i = xs  # (nocc_full, norb_full), (norb_full, nocc_act)

            lg1_i = jnp.einsum("pj,qj->pq", rot_chol_i, green, optimize="optimal")
            e20_acc = e20_acc - jnp.sum(lg1_i * jnp.swapaxes(lg1_i, -1, -2))
            e2131_acc = e2131_acc + jnp.einsum(
                "pq,qa,ap->",
                lg1_i,
                lg1_i[:, trial_data.occ_act_slice],
                ci1g1,
                optimize="optimal",
            )
            lci1g_mat_i = jnp.einsum("ia,qi->aq", lci1_i, green, optimize="optimal")
            e2132_acc = e2132_acc - jnp.einsum(
                "aq,qa->",
                lci1g_mat_i,
                lg1_i[:, trial_data.occ_act_slice],
                optimize="optimal",
            )
            return (e20_acc, e2131_acc, e2132_acc), None

        (e2_0_2, e2_1_3_1, e2_1_3_2), _ = lax.scan(
            scan_lg1_branch,
            (zero, zero, zero),
            (rot_chol, meas_ctx.lci1),
        )
    else:
        lg1 = jnp.einsum("gpj,qj->gpq", rot_chol, green, optimize="optimal")  # (n_chol,nocc,nocc)
        e2_0_2 = -jnp.sum(lg1 * jnp.swapaxes(lg1, -1, -2))
    e2_0 = e2_0_1 + e2_0_2

    # singles
    e2_1_1 = 2.0 * e2_0 * ci1g
    lci1g = _force_bias_chol_contract_high_realimag(chol, ci1_green, meas_ctx.cfg)
    e2_1_2 = -2.0 * (lci1g @ lg)

    if meas_ctx.cfg.memory_mode != "low":
        ci1g1 = ci1 @ green[:, trial_data.vir_act_slice].T  # (nocc_act, nocc_full)
        e2_1_3_1 = jnp.einsum(
            "gpq,gqa,ap->",
            lg1,
            lg1[:, :, trial_data.occ_act_slice],
            ci1g1,
            optimize="optimal",
        )
        lci1g_mat = jnp.einsum("gia,qi->gaq", meas_ctx.lci1, green, optimize="optimal")
        e2_1_3_2 = -jnp.einsum(
            "gaq,gqa->",
            lci1g_mat,
            lg1[:, :, trial_data.occ_act_slice],
            optimize="optimal",
        )
    e2_1_3 = e2_1_3_1 + e2_1_3_2

    e2_1 = e2_1_1 + 2.0 * (e2_1_2 + e2_1_3)

    # doubles
    e2_2_1 = e2_0 * gci2g
    lci2g = _force_bias_chol_contract_high_realimag(chol, ci2_green, meas_ctx.cfg)
    e2_2_2_1 = -(lci2g @ lg)

    if meas_ctx.cfg.memory_mode == "low":
        ci2_t = ci2.astype(meas_ctx.cfg.mixed_real_dtype)

        def scan_over_chol(carry, x):
            e22_acc, e23_acc = carry
            chol_i, rot_chol_i = x  # (norb,norb), (nocc,norb)

            gl_i = _energy_gl_scalar_realimag(green, chol_i, meas_ctx.cfg)
            lci2_green_i = jnp.einsum("pi,ji->pj", rot_chol_i, ci2_green, optimize="optimal")

            e22_acc = e22_acc + 0.5 * jnp.einsum("pi,pi->", gl_i, lci2_green_i, optimize="optimal")

            glgp_i = jnp.einsum("pi,it->pt", gl_i, greenp, optimize="optimal").astype(
                meas_ctx.cfg.mixed_complex_dtype_testing
            )
            glgp_i = glgp_i[trial_data.occ_act_slice, :]
            l2ci2_1, l2ci2_2 = _energy_l2ci2_scalar_realimag(glgp_i, ci2_t)
            e23_acc = e23_acc + (2.0 * l2ci2_1 - l2ci2_2)

            return (e22_acc, e23_acc), zero

        (e2_2_2_2, e2_2_3), _ = lax.scan(scan_over_chol, (zero, zero), (chol, rot_chol))
    else:
        lci2_green = jnp.einsum("gpi,ji->gpj", rot_chol, ci2_green, optimize="optimal")
        gl = _energy_gl_batched_realimag(green, chol, meas_ctx.cfg)
        e2_2_2_2 = 0.5 * jnp.einsum("gpi,gpi->", gl, lci2_green, optimize="optimal")

        glgp = jnp.einsum("gpi,it->gpt", gl, greenp, optimize="optimal").astype(
            meas_ctx.cfg.mixed_complex_dtype_testing
        )
        glgp = glgp[:, trial_data.occ_act_slice, :]
        l2ci2_1, l2ci2_2 = _energy_l2ci2_batched_realimag(
            glgp, ci2.astype(meas_ctx.cfg.mixed_real_dtype_testing)
        )
        e2_2_3 = 2.0 * l2ci2_1.sum() - l2ci2_2.sum()

    e2_2_2 = 4.0 * (e2_2_2_1 + e2_2_2_2)
    e2_2 = e2_2_1 + e2_2_2 + e2_2_3

    e2 = e2_0 + e2_1 + e2_2

    overlap = 1.0 + 2.0 * ci1g + gci2g
    return (e1 + e2) / overlap + e0


def rdm1_kernel_rw(
    walker: jax.Array, ham_data: HamChol, meas_ctx: CisdMeasCtx, trial_data: CisdTrial
) -> jax.Array:
    """
    Mixed estimator 1RDM for CISD trial with restricted walker.

    gamma = G^0 - (ci1_green + ci2_green) / Omega

    Returns (2, norb, norb): [alpha, beta] (identical for restricted).
    """
    ci1, ci2 = trial_data.ci1, trial_data.ci2
    nocc_full = trial_data.nocc_full
    norb = trial_data.norb

    green = _greens_restricted(walker, nocc_full)  # (nocc_full, norb_full)
    green_act, green_occ, greenp = _active_green_blocks(green, trial_data)

    # HF Green's function: G^0[:, :nocc_full] = green.T, G^0[:, nocc_full:] = 0
    g0 = jnp.zeros((norb, norb), dtype=green.dtype)
    g0 = g0.at[:, :nocc_full].set(green.T)

    # overlap components
    ci1g = jnp.einsum("pt,pt->", ci1, green_occ, optimize="optimal")

    # singles correction
    ci1_green = (greenp @ ci1.T) @ green_act  # (norb, norb)

    # doubles correction
    ci2g_c = jnp.einsum(
        "ptqu,pt->qu",
        ci2.astype(meas_ctx.cfg.mixed_real_dtype),
        green_occ.astype(meas_ctx.cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    ci2g_e = jnp.einsum(
        "ptqu,pu->qt",
        ci2.astype(meas_ctx.cfg.mixed_real_dtype),
        green_occ.astype(meas_ctx.cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    ci2_green = 2.0 * (greenp @ ci2g_c.T) @ green_act - (greenp @ ci2g_e.T) @ green_act

    ci2g = 2.0 * ci2g_c - ci2g_e
    gci2g = jnp.einsum("qu,qu->", ci2g, green_occ, optimize="optimal")

    overlap = 1.0 + 2.0 * ci1g + gci2g
    gamma = g0 - (ci1_green + ci2_green) / overlap

    return jnp.stack([gamma, gamma], axis=0)


def make_cisd_meas_ops(
    sys: System,
    memory_mode: str = "high",
    mixed_precision: bool = True,
    testing: bool = False,
) -> MeasOps:
    if sys.walker_kind.lower() != "restricted":
        raise ValueError(
            f"CISD MeasOps currently supports only restricted walkers, got: {sys.walker_kind}"
        )

    cfg = CisdMeasCfg(
        memory_mode=memory_mode,
        mixed_real_dtype=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed_precision else jnp.complex128,
        mixed_real_dtype_testing=jnp.float64 if testing else jnp.float32,
        mixed_complex_dtype_testing=jnp.complex128 if testing else jnp.complex64,
    )

    force_bias_kernel = (
        force_bias_kernel_rw_rh_high if memory_mode == "low" else force_bias_kernel_rw_rh
    )

    meas_ops = MeasOps(
        overlap=cisd_overlap_r,
        build_meas_ctx=lambda ham_data, trial_data: build_meas_ctx(ham_data, trial_data, cfg),
        kernels={k_force_bias: force_bias_kernel, k_energy: energy_kernel_rw_rh},
        observables={o_rdm1: rdm1_kernel_rw},
    )
    object.__setattr__(meas_ops, _CISD_MEAS_CFG_ATTR, cfg)
    return meas_ops
