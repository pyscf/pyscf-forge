from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
from jax import lax

from ..core.ops import MeasOps, k_energy, k_force_bias
from ..core.system import System
from ..ham.chol import HamChol
from ..trial.cisd import CisdTrial
from ..trial.eom_cisd import EomCisdTrial, overlap_r
from . import cisd
from .cisd import CisdMeasCtx, _greens_restricted


def _build_meas_ctx_eom(
    ham_data: HamChol, trial_data: EomCisdTrial, cfg: cisd.CisdMeasCfg
) -> CisdMeasCtx:
    # EOM CISD trial carries the same CI tensors needed by CISD measurement context.
    return cisd.build_meas_ctx(ham_data, cast(CisdTrial, trial_data), cfg)


def force_bias_kernel_rw_rh(
    walker: jax.Array, ham_data: HamChol, meas_ctx: CisdMeasCtx, trial_data: EomCisdTrial
) -> jax.Array:
    c1 = trial_data.ci1
    c2 = trial_data.ci2
    r1 = trial_data.r1
    r2 = trial_data.r2

    nocc = trial_data.nocc
    nvir = trial_data.nvir

    green = _greens_restricted(walker, nocc)
    green_occ = green[:, nocc:].copy()
    greenp = jnp.vstack((green_occ, -jnp.eye(nvir)))

    chol = ham_data.chol
    rot_chol = meas_ctx.rot_chol

    # r1
    # 2: spin
    r1g = 2 * jnp.einsum("pt,pt", r1, green_occ)
    # 2: spin
    lg = 2.0 * jnp.einsum("gpj,pj->g", rot_chol, green, optimize="optimal")
    fb_r1_1 = r1g * lg
    g_r1_gp = green.T @ (r1 @ greenp.T)
    # 2: spin
    fb_r1_2 = -2 * jnp.einsum("gij,ji", chol, g_r1_gp)
    fb_r1 = fb_r1_1 + fb_r1_2

    # r1 c1
    # 2: spin
    c1g = 2 * jnp.einsum("pt,pt", c1, green_occ)
    r1c1_c = r1g * c1g
    r1_g = r1 @ green_occ.T
    c1_g = c1 @ green_occ.T
    # 2: spin
    r1c1_e = -2 * jnp.einsum("pq,qp", r1_g, c1_g)
    r1c1 = r1c1_c + r1c1_e
    fb_r1c1_1 = r1c1 * lg
    r1_c1 = r1 * c1g + r1g * c1 - r1_g @ c1 - c1_g @ r1
    g_r1_c1_gp = green.T @ (r1_c1 @ greenp.T)
    # 2: spin
    fb_r1c1_2 = -2 * jnp.einsum("gij,ji", chol, g_r1_c1_gp)
    fb_r1c1 = fb_r1c1_1 + fb_r1c1_2

    # r2
    # 2: spin, 0.5: r2, 2: permutation
    r2g_c = 2.0 * jnp.einsum("ptqu,pt->qu", r2, green_occ)
    # 0.5: r2, 2: permutation
    r2g_e = jnp.einsum("ptqu,pu->qt", r2, green_occ)
    r2g = r2g_c - r2g_e
    # 2: spin, 0.5: no permuation
    r2g2 = jnp.einsum("qu,qu", r2g, green_occ, optimize="optimal")
    fb_r2_1 = lg * r2g2
    g_r2g_gp = green.T @ (r2g @ greenp.T)
    # 2: spin
    fb_r2_2 = -2.0 * jnp.einsum("gij,ji", chol, g_r2g_gp)
    fb_r2 = fb_r2_1 + fb_r2_2

    # r2 c1
    r2c1_c = r2g2 * c1g
    r2g_g = r2g @ green_occ.T
    # 2: spin
    r2c1_e = -2.0 * jnp.einsum("pq,qp", r2g_g, c1_g, optimize="optimal")
    r2c1 = r2c1_c + r2c1_e
    fb_r2c1_1 = lg * r2c1
    r2_c1 = r2g * c1g + r2g2 * c1 - r2g_g @ c1 - c1_g @ r2g
    g_c1_g = green_occ.T @ c1_g
    # 2: spin, 2: permutation, 0.5: r2
    r2_c1 -= 2.0 * jnp.einsum("tp,ptqu->qu", g_c1_g, r2, optimize="optimal")
    # 2: permutation, 0.5: r2
    r2_c1 += jnp.einsum("tq,ptqu->pu", g_c1_g, r2, optimize="optimal")
    g_r2_c1_gp = green.T @ (r2_c1 @ greenp.T)
    # 2: spin
    fb_r2c1_2 = -2.0 * jnp.einsum("gij,ji", chol, g_r2_c1_gp)
    fb_r2c1 = fb_r2c1_1 + fb_r2c1_2

    # r1 c2
    # 2: spin, 0.5: c2, 2: permutation
    c2g_c = 2.0 * jnp.einsum("ptqu,pt->qu", c2, green_occ)
    # 0.5: c2, 2: permutation
    c2g_e = jnp.einsum("ptqu,pu->qt", c2, green_occ)
    c2g = c2g_c - c2g_e
    # 2: spin, 0.5: no permuation
    c2g2 = jnp.einsum("qu,qu", c2g, green_occ, optimize="optimal")
    r1c2_c = r1g * c2g2
    c2g_g = c2g @ green_occ.T
    # 2: spin
    r1c2_e = -2.0 * jnp.einsum("pq,qp", r1_g, c2g_g, optimize="optimal")
    r1c2 = r1c2_c + r1c2_e
    fb_r1c2_1 = lg * r1c2
    r1_c2 = r1 * c2g2 + r1g * c2g - r1_g @ c2g - c2g_g @ r1
    g_r1_g = green_occ.T @ r1_g
    # 2: spin, 2: permutation, 0.5: c2
    r1_c2 -= 2.0 * jnp.einsum("tp,ptqu->qu", g_r1_g, c2, optimize="optimal")
    # 2: permutation, 0.5: c2
    r1_c2 += jnp.einsum("tq,ptqu->pu", g_r1_g, c2, optimize="optimal")
    g_r1_c2_gp = green.T @ (r1_c2 @ greenp.T)
    # 2: spin
    fb_r1c2_2 = -2.0 * jnp.einsum("gij,ji", chol, g_r1_c2_gp)
    fb_r1c2 = fb_r1c2_1 + fb_r1c2_2

    # r2 c2
    r2c2_c = r2g2 * c2g2
    # 2: spin
    r2c2_e_1 = -2.0 * jnp.einsum("pq,qp", r2g_g, c2g_g, optimize="optimal")
    # 0.5: r2
    r2_g = 0.5 * jnp.einsum("ptqu,rt->prqu", r2, green_occ)
    r2_g_g = jnp.einsum("prqu,su->prqs", r2_g, green_occ)
    # del r2_g
    # 0.5: c2
    c2_g = 0.5 * jnp.einsum("ptqu,rt->prqu", c2, green_occ)
    c2_g_g = jnp.einsum("prqu,su->prqs", c2_g, green_occ)
    # del c2_g
    # 4: spin, 2: permutation
    r2c2_e_2_c = 8.0 * jnp.einsum("prqs,rpsq", r2_g_g, c2_g_g, optimize="optimal")
    # 2: spin, 2: permutation
    r2c2_e_2_e = -4.0 * jnp.einsum("prqs,rqsp", r2_g_g, c2_g_g, optimize="optimal")
    r2c2_e_2 = r2c2_e_2_c + r2c2_e_2_e
    r2c2 = r2c2_c + r2c2_e_1 + r2c2_e_2
    fb_r2c2_1 = lg * r2c2
    r2_c2 = r2g2 * c2g + r2g * c2g2 - r2g_g @ c2g - c2g_g @ r2g
    # 2: spin, 2: permutation
    r2_c2 -= 4.0 * jnp.einsum("pr,rpqu->qu", r2g_g, c2_g, optimize="optimal")
    r2_c2 -= 4.0 * jnp.einsum("pr,rpqu->qu", c2g_g, r2_g, optimize="optimal")
    # 2: permutation
    r2_c2 += 2.0 * jnp.einsum("pr,qpru->qu", r2g_g, c2_g, optimize="optimal")
    r2_c2 += 2.0 * jnp.einsum("pr,qpru->qu", c2g_g, r2_g, optimize="optimal")
    # 2: spin, 4: permutation
    r2_c2 += 8.0 * jnp.einsum("pqrs,qpst->rt", r2_g_g, c2_g, optimize="optimal")
    r2_c2 += 8.0 * jnp.einsum("pqrs,qpst->rt", c2_g_g, r2_g, optimize="optimal")
    # 4: permutation
    r2_c2 -= 4.0 * jnp.einsum("pqrs,spqt->rt", r2_g_g, c2_g, optimize="optimal")
    r2_c2 -= 4.0 * jnp.einsum("pqrs,spqt->rt", c2_g_g, r2_g, optimize="optimal")
    g_r2_c2_gp = green.T @ (r2_c2 @ greenp.T)
    # 2: spin
    fb_r2c2_2 = -2.0 * jnp.einsum("gij,ji", chol, g_r2_c2_gp)
    fb_r2c2 = fb_r2c2_1 + fb_r2c2_2

    overlap = r1g + r1c1 + r2g2 + r2c1 + r1c2 + r2c2
    fb = (fb_r1 + fb_r1c1 + fb_r2 + fb_r2c1 + fb_r1c2 + fb_r2c2) / overlap
    return fb


def energy_kernel_rw_rh(
    walker: jax.Array, ham_data: HamChol, meas_ctx: CisdMeasCtx, trial_data: EomCisdTrial
) -> jax.Array:
    c1 = trial_data.ci1
    c2 = trial_data.ci2
    r1 = trial_data.r1
    r2 = trial_data.r2

    nocc = trial_data.nocc
    nvir = trial_data.nvir

    green = _greens_restricted(walker, nocc)
    green_occ = green[:, nocc:].copy()
    greenp = jnp.vstack((green_occ, -jnp.eye(nvir)))

    cfg = meas_ctx.cfg

    h1 = ham_data.h1
    chol = ham_data.chol
    rot_chol = meas_ctx.rot_chol

    # 0 body energy
    e0 = ham_data.h0

    # 1 body energy
    # r1
    # 2: spin
    r1g = 2 * jnp.einsum("pt,pt", r1, green_occ)
    # 2: spin
    h1g = 2 * jnp.einsum("pt,pt", h1[:nocc, :], green)
    e1_r1_1 = r1g * h1g
    gp_h_g = greenp.T @ (h1 @ green.T)
    # 2: spin
    e1_r1_2 = -2 * jnp.einsum("tp,pt", gp_h_g, r1)
    e1_r1 = e1_r1_1 + e1_r1_2

    # r1 c1
    # 2: spin
    c1g = 2 * jnp.einsum("pt,pt", c1, green_occ)
    r1c1_c = r1g * c1g
    r1_g = r1 @ green_occ.T
    c1_g = c1 @ green_occ.T
    # 2: spin
    r1c1_e = -2 * jnp.einsum("pq,qp", r1_g, c1_g)
    r1c1 = r1c1_c + r1c1_e
    e1_r1c1_1 = r1c1 * h1g

    r1_c1 = r1 * c1g + r1g * c1 - r1_g @ c1 - c1_g @ r1
    # 2: spin
    e1_r1c1_2 = -2 * jnp.einsum("tp,pt", gp_h_g, r1_c1)
    e1_r1c1 = e1_r1c1_1 + e1_r1c1_2

    # r2
    # 2: spin, 0.5: r2, 2: permutation
    r2g_c = 2.0 * jnp.einsum("ptqu,pt->qu", r2, green_occ)
    # 0.5: r2, 2: permutation
    r2g_e = jnp.einsum("ptqu,pu->qt", r2, green_occ)
    r2g = r2g_c - r2g_e
    # 2: spin, 0.5: no permuation
    r2g2 = jnp.einsum("qu,qu", r2g, green_occ, optimize="optimal")
    e1_r2_1 = h1g * r2g2
    # 2: spin
    e1_r2_2 = -2.0 * jnp.einsum("pt,tp", r2g, gp_h_g, optimize="optimal")
    e1_r2 = e1_r2_1 + e1_r2_2

    # r2 c1
    r2c1_c = r2g2 * c1g
    r2g_g = r2g @ green_occ.T
    # 2: spin
    r2c1_e = -2.0 * jnp.einsum("pq,qp", r2g_g, c1_g, optimize="optimal")
    r2c1 = r2c1_c + r2c1_e
    e1_r2c1_1 = h1g * r2c1
    r2_c1 = r2g * c1g + r2g2 * c1 - r2g_g @ c1 - c1_g @ r2g
    g_c1_g = green_occ.T @ c1_g
    # 2: spin, 2: permutation, 0.5: r2
    r2_c1 -= 2.0 * jnp.einsum("tp,ptqu->qu", g_c1_g, r2, optimize="optimal")
    # 2: permutation, 0.5: r2
    r2_c1 += jnp.einsum("tq,ptqu->pu", g_c1_g, r2, optimize="optimal")
    # 2: spin
    e1_r2c1_2 = -2.0 * jnp.einsum("tp,pt", gp_h_g, r2_c1, optimize="optimal")
    e1_r2c1 = e1_r2c1_1 + e1_r2c1_2

    # r1 c2
    # 2: spin, 0.5: c2, 2: permutation
    c2g_c = 2.0 * jnp.einsum("ptqu,pt->qu", c2, green_occ)
    # 0.5: c2, 2: permutation
    c2g_e = jnp.einsum("ptqu,pu->qt", c2, green_occ)
    c2g = c2g_c - c2g_e
    # 2: spin, 0.5: no permuation
    c2g2 = jnp.einsum("qu,qu", c2g, green_occ, optimize="optimal")
    r1c2_c = r1g * c2g2
    c2g_g = c2g @ green_occ.T
    # 2: spin
    r1c2_e = -2.0 * jnp.einsum("pq,qp", r1_g, c2g_g, optimize="optimal")
    r1c2 = r1c2_c + r1c2_e
    e1_r1c2_1 = h1g * r1c2
    r1_c2 = r1 * c2g2 + r1g * c2g - r1_g @ c2g - c2g_g @ r1
    g_r1_g = green_occ.T @ r1_g
    # 2: spin, 2: permutation, 0.5: c2
    r1_c2 -= 2.0 * jnp.einsum("tp,ptqu->qu", g_r1_g, c2, optimize="optimal")
    # 2: permutation, 0.5: c2
    r1_c2 += jnp.einsum("tq,ptqu->pu", g_r1_g, c2, optimize="optimal")
    # 2: spin
    e1_r1c2_2 = -2.0 * jnp.einsum("tp,pt", gp_h_g, r1_c2, optimize="optimal")
    e1_r1c2 = e1_r1c2_1 + e1_r1c2_2

    # r2 c2
    r2c2_c = r2g2 * c2g2
    # 2: spin
    r2c2_e_1 = -2.0 * jnp.einsum("pq,qp", r2g_g, c2g_g, optimize="optimal")
    # 0.5: r2
    r2_g = 0.5 * jnp.einsum("ptqu,rt->prqu", r2, green_occ)
    r2_g_g = jnp.einsum("prqu,su->prqs", r2_g, green_occ)
    # 0.5: c2
    c2_g = 0.5 * jnp.einsum("ptqu,rt->prqu", c2, green_occ, optimize="optimal")
    c2_g_g = jnp.einsum("prqu,su->prqs", c2_g, green_occ, optimize="optimal")
    # 4: spin, 2: permutation
    r2c2_e_2_c = 8.0 * jnp.einsum("prqs,rpsq", r2_g_g, c2_g_g, optimize="optimal")
    # 2: spin, 2: permutation
    r2c2_e_2_e = -4.0 * jnp.einsum("prqs,rqsp", r2_g_g, c2_g_g, optimize="optimal")
    r2c2_e_2 = r2c2_e_2_c + r2c2_e_2_e
    r2c2 = r2c2_c + r2c2_e_1 + r2c2_e_2
    e1_r2c2_1 = h1g * r2c2
    r2_c2 = r2g2 * c2g + r2g * c2g2 - r2g_g @ c2g - c2g_g @ r2g
    # 2: spin, 2: permutation
    r2_c2 -= 4.0 * jnp.einsum("pr,rpqu->qu", r2g_g, c2_g, optimize="optimal")
    r2_c2 -= 4.0 * jnp.einsum("pr,rpqu->qu", c2g_g, r2_g, optimize="optimal")
    # 2: permutation
    r2_c2 += 2.0 * jnp.einsum("pr,qpru->qu", r2g_g, c2_g, optimize="optimal")
    r2_c2 += 2.0 * jnp.einsum("pr,qpru->qu", c2g_g, r2_g, optimize="optimal")
    # 2: spin, 4: permutation
    r2_c2 += 8.0 * jnp.einsum("pqrs,qpst->rt", r2_g_g, c2_g, optimize="optimal")
    r2_c2 += 8.0 * jnp.einsum("pqrs,qpst->rt", c2_g_g, r2_g, optimize="optimal")
    # 4: permutation
    r2_c2 -= 4.0 * jnp.einsum("pqrs,spqt->rt", r2_g_g, c2_g, optimize="optimal")
    r2_c2 -= 4.0 * jnp.einsum("pqrs,spqt->rt", c2_g_g, r2_g, optimize="optimal")
    e1_r2c2_2 = -2.0 * jnp.einsum("tp,pt", gp_h_g, r2_c2, optimize="optimal")
    e1_r2c2 = e1_r2c2_1 + e1_r2c2_2

    e1 = e1_r1 + e1_r1c1 + e1_r2 + e1_r2c1 + e1_r1c2 + e1_r2c2

    # 2 body energy
    # 2: spin
    lg = 2.0 * jnp.einsum("gpj,pj->g", rot_chol, green, optimize="optimal")
    l_g = jnp.einsum("gpj,qj->gpq", rot_chol, green, optimize="optimal")
    # 0.5: coulomb
    l2g2_c = 0.5 * (lg @ lg)
    l2g2_e = -jnp.sum(jax.vmap(lambda x: x * x.T)(l_g))
    l2g2 = l2g2_c + l2g2_e

    # doing this first to build intermediates
    # r2
    e2_r2_1 = l2g2 * r2g2

    # carry: [e2_r2_, e2_c2_, e2_r1c1_, e2_r2c1_, e2_r1c2_, e2_r2c2_, l2g]
    def loop_over_chol(carry, x):
        chol_i, lg_i, l_g_i = x
        # build intermediate
        gp_l_g_i = greenp.T @ (chol_i @ green.T)
        # 0.5: coulomb, 2: permutation
        l2g_i_c = gp_l_g_i * lg_i
        l2g_i_e = gp_l_g_i @ l_g_i
        l2g_i = l2g_i_c - l2g_i_e
        carry[6] += l2g_i

        gp_l_g_i = gp_l_g_i.astype(cfg.mixed_complex_dtype)
        # evaluate energy
        # r2
        # 4: spin, 2: permutation, 0.5: r2, 0.5: coulomb
        l2r2_c = 2.0 * jnp.einsum(
            "tp,uq,ptqu",
            gp_l_g_i,
            gp_l_g_i,
            r2.astype(cfg.mixed_real_dtype),
            optimize="optimal",
        )
        # 2: spin, 2: permutation, 0.5: r2, 0.5: coulomb
        l2r2_e = jnp.einsum(
            "up,tq,ptqu",
            gp_l_g_i,
            gp_l_g_i,
            r2.astype(cfg.mixed_real_dtype),
            optimize="optimal",
        )
        l2r2 = l2r2_c - l2r2_e
        carry[0] += l2r2

        # c2
        # 4: spin, 2: permutation, 0.5: c2, 0.5: coulomb
        l2c2_c = 2.0 * jnp.einsum(
            "tp,uq,ptqu",
            gp_l_g_i,
            gp_l_g_i,
            c2.astype(cfg.mixed_real_dtype),
            optimize="optimal",
        )
        # 2: spin, 2: permutation, 0.5: c2, 0.5: coulomb
        l2c2_e = jnp.einsum(
            "up,tq,ptqu",
            gp_l_g_i,
            gp_l_g_i,
            c2.astype(cfg.mixed_real_dtype),
            optimize="optimal",
        )
        l2c2 = l2c2_c - l2c2_e
        carry[1] += l2c2

        # r1 c1
        # 2: spin
        lr1 = -2.0 * jnp.einsum("tp,pt", gp_l_g_i, r1, optimize="optimal")
        lc1 = -2.0 * jnp.einsum("tp,pt", gp_l_g_i, c1, optimize="optimal")
        # 2: permutation, 0.5: coulomb
        l2r1c1_c = lr1 * lc1
        # 2: spin, 2: permutation, 0.5: coulomb
        l2r1c1_e = 2.0 * jnp.einsum(
            "up,tq,pt,qu",
            gp_l_g_i,
            gp_l_g_i,
            r1.astype(cfg.mixed_real_dtype),
            c1.astype(cfg.mixed_real_dtype),
            optimize="optimal",
        )
        l2r1c1 = l2r1c1_c - l2r1c1_e
        carry[2] += l2r1c1

        # r2 c1
        # 2: spin
        lr2g = -2.0 * jnp.einsum("tp,pt", gp_l_g_i, r2g, optimize="optimal")
        # 2: permutation, 0.5: coulomb
        l2r2c1_1_c = lc1 * lr2g
        # 2: spin, 2: permutation, 0.5: coulomb
        l2r2c1_1_e = 2.0 * jnp.einsum(
            "up,tq,pt,qu",
            gp_l_g_i,
            gp_l_g_i,
            r2g.astype(cfg.mixed_complex_dtype),
            c1.astype(cfg.mixed_real_dtype),
            optimize="optimal",
        )
        l2r2c1_1 = l2r2c1_1_c - l2r2c1_1_e

        # 2: spin, 0.5: r2, 2: permutation
        lr2_c = -2.0 * jnp.einsum(
            "tp,ptqu->qu",
            gp_l_g_i,
            r2.astype(cfg.mixed_real_dtype),
            optimize="optimal",
        )
        # 0.5: r2, 2: permutation
        lr2_e = jnp.einsum(
            "tp,puqt->qu",
            gp_l_g_i,
            r2.astype(cfg.mixed_real_dtype),
            optimize="optimal",
        )
        lr2 = lr2_c + lr2_e
        lr2_c1 = (lr2 @ green_occ.T) @ c1
        c1_lr2 = c1_g @ lr2
        # 2: spin, 2: permutation, 0.5: coulomb
        l2r2c1_2 = 2.0 * jnp.einsum("tp,pt", gp_l_g_i, lr2_c1 + c1_lr2, optimize="optimal")
        l2r2c1 = l2r2c1_1 + l2r2c1_2
        carry[3] += l2r2c1

        # r1 c2
        # 2: spin
        lc2g = -2.0 * jnp.einsum("tp,pt", gp_l_g_i, c2g, optimize="optimal")
        # 2: permutation, 0.5: coulomb
        l2r1c2_1_c = lr1 * lc2g
        # 2: spin, 2: permutation, 0.5: coulomb
        l2r1c2_1_e = 2.0 * jnp.einsum(
            "up,tq,pt,qu",
            gp_l_g_i,
            gp_l_g_i,
            r1.astype(cfg.mixed_real_dtype),
            c2g.astype(cfg.mixed_complex_dtype),
            optimize="optimal",
        )
        l2r1c2_1 = l2r1c2_1_c - l2r1c2_1_e

        # 2: spin, 0.5: c2, 2: permutation
        lc2_c = -2.0 * jnp.einsum(
            "tp,ptqu->qu",
            gp_l_g_i,
            c2.astype(cfg.mixed_real_dtype),
            optimize="optimal",
        )
        # 0.5: c2, 2: permutation
        lc2_e = jnp.einsum(
            "tp,puqt->qu",
            gp_l_g_i,
            c2.astype(cfg.mixed_real_dtype),
            optimize="optimal",
        )
        lc2 = lc2_c + lc2_e
        lc2_r1 = (lc2 @ green_occ.T) @ r1
        r1_lc2 = r1_g @ lc2
        # 2: spin, 2: permutation, 0.5: coulomb
        l2r1c2_2 = 2.0 * jnp.einsum("tp,pt", gp_l_g_i, lc2_r1 + r1_lc2, optimize="optimal")
        l2r1c2 = l2r1c2_1 + l2r1c2_2
        carry[4] += l2r1c2

        # r2 c2
        # 2: permutation, 0.5: coulomb
        l2r2c2_1_c = lr2g * lc2g
        # 2: spin, 2: permutation, 0.5: coulomb
        l2r2c2_1_e = -2.0 * jnp.einsum(
            "up,tq,pt,qu",
            gp_l_g_i,
            gp_l_g_i,
            r2g.astype(cfg.mixed_complex_dtype),
            c2g.astype(cfg.mixed_complex_dtype),
            optimize="optimal",
        )
        l2r2c2_1 = l2r2c2_1_c + l2r2c2_1_e

        lr2_g = lr2 @ green_occ.T
        lr2_c2g = lr2_g @ c2g
        # 2: spin, 2: permutaion, 0.5: coulomb
        l2r2c2_2_1 = 2.0 * jnp.einsum("tp,pt", gp_l_g_i, lr2_c2g, optimize="optimal")
        c2g_lr2 = c2g_g @ lr2
        # 2: spin, 2: permutation, 0.5: coulomb
        l2r2c2_2_2 = 2.0 * jnp.einsum("tp,pt", gp_l_g_i, c2g_lr2, optimize="optimal")
        lc2_g = lc2 @ green_occ.T
        lc2_r2g = lc2_g @ r2g
        # 2: spin, 2: permutation, 0.5: coulomb
        l2r2c2_2_3 = 2.0 * jnp.einsum("tp,pt", gp_l_g_i, lc2_r2g, optimize="optimal")
        r2g_lc2 = r2g_g @ lc2
        # 2: spin, 2: permutation, 0.5: coulomb
        l2r2c2_2_4 = 2.0 * jnp.einsum("tp,pt", gp_l_g_i, r2g_lc2, optimize="optimal")
        l2r2c2_2 = l2r2c2_2_1 + l2r2c2_2_2 + l2r2c2_2_3 + l2r2c2_2_4

        # 4: spin, 0.5: coulomb, 0.5: r2, 4: permutation
        l2r2c2_3_1_1_c = 4.0 * jnp.einsum(
            "vp,wq,rvsw,prqs",
            gp_l_g_i,
            gp_l_g_i,
            r2.astype(cfg.mixed_real_dtype),
            c2_g_g.astype(cfg.mixed_complex_dtype),
            optimize="optimal",
        )
        # 2: spin, 0.5: coulomb, 0.5; r2, 4: permutation
        l2r2c2_3_1_1_e = -2.0 * jnp.einsum(
            "vp,wq,svrw,prqs",
            gp_l_g_i,
            gp_l_g_i,
            r2.astype(cfg.mixed_real_dtype),
            c2_g_g.astype(cfg.mixed_complex_dtype),
            optimize="optimal",
        )
        l2r2c2_3_1_1 = l2r2c2_3_1_1_c + l2r2c2_3_1_1_e
        # 4: spin, 0.5: coulomb, 0.5: c2, 4: permutation
        l2r2c2_3_1_2_c = 4.0 * jnp.einsum(
            "vp,wq,rvsw,prqs",
            gp_l_g_i,
            gp_l_g_i,
            c2.astype(cfg.mixed_real_dtype),
            r2_g_g.astype(cfg.mixed_complex_dtype),
            optimize="optimal",
        )
        # 2: spin, 0.5: coulomb, 0.5: c2, 4: permutation
        l2r2c2_3_1_2_e = -2.0 * jnp.einsum(
            "vp,wq,svrw,prqs",
            gp_l_g_i,
            gp_l_g_i,
            c2.astype(cfg.mixed_real_dtype),
            r2_g_g.astype(cfg.mixed_complex_dtype),
            optimize="optimal",
        )
        l2r2c2_3_1_2 = l2r2c2_3_1_2_c + l2r2c2_3_1_2_e
        l2r2c2_3_1 = l2r2c2_3_1_1 + l2r2c2_3_1_2

        # 4: spin, 8: permutation, 0.5: coulomb
        l2r2c2_3_2 = 16.0 * jnp.einsum(
            "vq,us,rpsv,prqu",
            gp_l_g_i,
            gp_l_g_i,
            r2_g.astype(cfg.mixed_complex_dtype),
            c2_g.astype(cfg.mixed_complex_dtype),
            optimize="optimal",
        )

        # 2: spin, 8: permutation, 0.5: coulomb
        l2r2c2_3_3_1 = -8.0 * jnp.einsum(
            "vq,us,sprv,prqu",
            gp_l_g_i,
            gp_l_g_i,
            r2_g.astype(cfg.mixed_complex_dtype),
            c2_g.astype(cfg.mixed_complex_dtype),
            optimize="optimal",
        )
        l2r2c2_3_3_2 = -8.0 * jnp.einsum(
            "vq,us,sprv,prqu",
            gp_l_g_i,
            gp_l_g_i,
            c2_g.astype(cfg.mixed_complex_dtype),
            r2_g.astype(cfg.mixed_complex_dtype),
            optimize="optimal",
        )
        l2r2c2_3_3 = l2r2c2_3_3_1 + l2r2c2_3_3_2

        # 4: spin, 8: permutation, 0.5: coulomb
        l2r2c2_3_4 = 16.0 * jnp.einsum(
            "vp,us,sqrv,prqu",
            gp_l_g_i,
            gp_l_g_i,
            r2_g.astype(cfg.mixed_complex_dtype),
            c2_g.astype(cfg.mixed_complex_dtype),
            optimize="optimal",
        )

        l2r2c2_3 = l2r2c2_3_1 + l2r2c2_3_2 + l2r2c2_3_3 + l2r2c2_3_4

        # 2: spin, 2: permutation, 0.5: coulomb
        l2r2c2_4 = -2.0 * jnp.einsum("pq,qp", lr2_g, lc2_g, optimize="optimal")

        l2r2c2 = l2r2c2_1 + l2r2c2_2 + l2r2c2_3 + l2r2c2_4
        carry[5] += l2r2c2

        return carry, 0.0

    l2g = jnp.zeros((nvir, nocc)) + 0.0j
    [e2_r2_3, e2_c2_3, e2_r1c1_3, e2_r2c1_3, e2_r1c2_3, e2_r2c2_3, l2g], _ = lax.scan(
        loop_over_chol,
        [0.0j, 0.0j, 0.0j, 0.0j, 0.0j, 0.0j, l2g],
        (chol, lg, l_g),
    )
    e2_r2_2 = -2.0 * jnp.einsum("tp,pt->", l2g, r2g, optimize="optimal")
    e2_r2 = e2_r2_1 + e2_r2_2 + e2_r2_3

    # r1
    e2_r1_1 = l2g2 * r1g
    # 2: spin
    e2_r1_2 = -2.0 * jnp.einsum("tp,pt->", l2g, r1, optimize="optimal")
    e2_r1 = e2_r1_1 + e2_r1_2 + e2_r1c1_3

    # r1 c1
    e2_r1c1_1 = l2g2 * r1c1
    e2_r1c1_2 = -2.0 * jnp.einsum("tp,pt", l2g, r1_c1, optimize="optimal")
    e2_r1c1 = e2_r1c1_1 + e2_r1c1_2

    # r2 c1
    e2_r2c1_1 = l2g2 * r2c1
    e2_r2c1_2 = -2.0 * jnp.einsum("tp,pt", l2g, r2_c1, optimize="optimal")
    e2_r2c1_3 += e2_r2_3 * c1g
    e2_r2c1 = e2_r2c1_1 + e2_r2c1_2 + e2_r2c1_3

    # r1 c2
    e2_r1c2_1 = l2g2 * r1c2
    e2_r1c2_2 = -2.0 * jnp.einsum("tp,pt", l2g, r1_c2, optimize="optimal")
    e2_r1c2_3 += r1g * e2_c2_3
    e2_r1c2 = e2_r1c2_1 + e2_r1c2_2 + e2_r1c2_3

    # r2 c2
    e2_r2c2_1 = l2g2 * r2c2
    e2_r2c2_2 = -2.0 * jnp.einsum("tp,pt", l2g, r2_c2, optimize="optimal")
    e2_r2c2_3 += r2g2 * e2_c2_3
    e2_r2c2_3 += c2g2 * e2_r2_3
    e2_r2c2 = e2_r2c2_1 + e2_r2c2_2 + e2_r2c2_3

    e2 = e2_r1 + e2_r2 + e2_r1c1 + e2_r2c1 + e2_r1c2 + e2_r2c2

    overlap = r1g + r1c1 + r2g2 + r2c1 + r1c2 + r2c2
    return (e1 + e2) / overlap + e0


def make_eom_cisd_meas_ops(
    sys: System,
    mixed_precision: bool = True,
    testing: bool = False,
) -> MeasOps:
    if sys.walker_kind.lower() != "restricted":
        raise ValueError(
            f"EOM CISD MeasOps currently supports only restricted walkers, got: {sys.walker_kind}"
        )

    cfg = cisd.CisdMeasCfg(
        memory_mode="low",
        mixed_real_dtype=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed_precision else jnp.complex128,
        mixed_real_dtype_testing=jnp.float64 if testing else jnp.float32,
        mixed_complex_dtype_testing=jnp.complex128 if testing else jnp.complex64,
    )

    return MeasOps(
        overlap=overlap_r,
        build_meas_ctx=lambda ham_data, trial_data: _build_meas_ctx_eom(ham_data, trial_data, cfg),
        kernels={k_force_bias: force_bias_kernel_rw_rh, k_energy: energy_kernel_rw_rh},
    )
