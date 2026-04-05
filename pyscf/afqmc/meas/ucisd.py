from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import lax, tree_util, vmap

from ..core.ops import MeasOps, k_energy, k_force_bias
from ..core.system import System
from ..ham.chol import HamChol
from ..trial.ucisd import UcisdTrial, overlap_g, overlap_r, overlap_u


def _half_green_from_overlap_matrix(w: jax.Array, ovlp_mat: jax.Array) -> jax.Array:
    """
    green_half = (w @ inv(ovlp_mat)).T
    """
    return jnp.linalg.solve(ovlp_mat.T, w.T)


def _build_bra_generalized(trial_data: UcisdTrial) -> jax.Array:
    n_oa, n_ob = trial_data.nocc
    c_a = trial_data.mo_coeff_a
    c_b = trial_data.mo_coeff_b

    Atrial, Btrial = (
        c_a[:, :n_oa],
        c_b[:, :n_ob],
    )

    bra = jnp.block(
        [
            [Atrial, 0 * Btrial],
            [
                0 * Atrial,
                (c_b.T @ c_b)[:, :n_ob],
            ],
        ]
    )

    return bra


def _get_generalized_walker_in_alpha_basis(walker: jax.Array, trial_data: UcisdTrial) -> jax.Array:
    norb = trial_data.norb
    c_b = trial_data.mo_coeff_b

    w = jnp.vstack(
        [walker[:norb], c_b.T @ walker[norb:, :]]
    )  # put walker_dn in the basis of alpha reference

    return w


@dataclass(frozen=True)
class UcisdMeasCfg:
    memory_mode: str = "low"  # or Literal["low","high"]
    mixed_real_dtype: jnp.dtype = jnp.float64
    mixed_complex_dtype: jnp.dtype = jnp.complex128
    mixed_real_dtype_testing: jnp.dtype = jnp.float32
    mixed_complex_dtype_testing: jnp.dtype = jnp.complex64


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class UcisdMeasCtx:
    h1_b: jax.Array  # (norb, norb)
    chol_b: jax.Array  # (n_chol, norb, norb)

    # half-rotated:
    rot_h1_a: jax.Array  # (nocc[0], norb)
    rot_h1_b: jax.Array  # (nocc[1], norb)
    rot_chol_a: jax.Array  # (n_chol, nocc[0], norb)
    rot_chol_b: jax.Array  # (n_chol, nocc[1], norb)
    rot_chol_flat_a: jax.Array  # (n_chol, nocc[0]*norb)
    rot_chol_flat_b: jax.Array  # (n_chol, nocc[1]*norb)

    lci1_a: jax.Array  # (n_chol, norb, nocc[0])
    lci1_b: jax.Array  # (n_chol, norb, nocc[1])

    cfg: UcisdMeasCfg

    def tree_flatten(self):
        children = (
            self.h1_b,
            self.chol_b,
            self.rot_h1_a,
            self.rot_h1_b,
            self.rot_chol_a,
            self.rot_chol_b,
            self.rot_chol_flat_a,
            self.rot_chol_flat_b,
            self.lci1_a,
            self.lci1_b,
        )
        aux = (self.cfg,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (cfg,) = aux
        (
            h1_b,
            chol_b,
            rot_h1_a,
            rot_h1_b,
            rot_chol_a,
            rot_chol_b,
            rot_chol_flat_a,
            rot_chol_flat_b,
            lci1_a,
            lci1_b,
        ) = children
        return cls(
            h1_b=h1_b,
            chol_b=chol_b,
            rot_h1_a=rot_h1_a,
            rot_h1_b=rot_h1_b,
            rot_chol_a=rot_chol_a,
            rot_chol_b=rot_chol_b,
            rot_chol_flat_a=rot_chol_flat_a,
            rot_chol_flat_b=rot_chol_flat_b,
            lci1_a=lci1_a,
            lci1_b=lci1_b,
            cfg=cfg,
        )


def force_bias_kernel_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: UcisdMeasCtx,
    trial_data: UcisdTrial,
) -> jax.Array:
    """Calculates force bias < psi_T | chol_gamma | walker > / < psi_T | walker >"""
    n_elec_0 = trial_data.nocc[0]
    n_elec_1 = trial_data.nocc[1]
    return force_bias_kernel_uw_rh(
        (walker[:, :n_elec_0], walker[:, :n_elec_1]), ham_data, meas_ctx, trial_data
    )


def force_bias_kernel_uw_rh(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamChol,
    meas_ctx: UcisdMeasCtx,
    trial_data: UcisdTrial,
) -> jax.Array:
    """Calculates force bias < psi_T | chol_gamma | walker > / < psi_T | walker >"""
    wa, wb = walker
    n_oa, n_ob = trial_data.nocc
    n_va, n_vb = trial_data.nvir
    c1a = trial_data.c1a
    c1b = trial_data.c1b
    c2aa = trial_data.c2aa
    c2ab = trial_data.c2ab
    c2bb = trial_data.c2bb
    c_b = trial_data.mo_coeff_b

    cfg = meas_ctx.cfg

    wb = c_b.T @ wb[:, :n_ob]
    woa = wa[:n_oa, :]  # (n_oa, n_oa)
    wob = wb[:n_ob, :]  # (n_ob, n_ob)

    green_a = _half_green_from_overlap_matrix(wa, woa)  # (n_oa, norb)
    green_b = _half_green_from_overlap_matrix(wb, wob)  # (n_ob, norb)

    green_occ_a = green_a[:, n_oa:].copy()
    green_occ_b = green_b[:, n_ob:].copy()
    greenp_a = jnp.vstack((green_occ_a, -jnp.eye(n_va)))
    greenp_b = jnp.vstack((green_occ_b, -jnp.eye(n_vb)))

    chol_a = ham_data.chol
    chol_b = meas_ctx.chol_b
    rot_chol_a = meas_ctx.rot_chol_a
    rot_chol_b = meas_ctx.rot_chol_b
    lg_a = jnp.einsum("gpj,pj->g", rot_chol_a, green_a, optimize="optimal")
    lg_b = jnp.einsum("gpj,pj->g", rot_chol_b, green_b, optimize="optimal")
    lg = lg_a + lg_b

    # ref
    fb_0 = lg_a + lg_b

    # single excitations
    ci1g_a = jnp.einsum("pt,pt->", c1a, green_occ_a, optimize="optimal")
    ci1g_b = jnp.einsum("pt,pt->", c1b, green_occ_b, optimize="optimal")
    ci1g = ci1g_a + ci1g_b
    fb_1_1 = ci1g * lg
    ci1gp_a = jnp.einsum("pt,it->pi", c1a, greenp_a, optimize="optimal")
    ci1gp_b = jnp.einsum("pt,it->pi", c1b, greenp_b, optimize="optimal")
    gci1gp_a = jnp.einsum("pj,pi->ij", green_a, ci1gp_a, optimize="optimal")
    gci1gp_b = jnp.einsum("pj,pi->ij", green_b, ci1gp_b, optimize="optimal")
    fb_1_2 = -jnp.einsum(
        "gij,ij->g",
        chol_a.astype(cfg.mixed_real_dtype),
        gci1gp_a.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    ) - jnp.einsum(
        "gij,ij->g",
        chol_b.astype(cfg.mixed_real_dtype),
        gci1gp_b.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    fb_1 = fb_1_1 + fb_1_2

    # double excitations
    ci2g_a = jnp.einsum(
        "ptqu,pt->qu",
        c2aa.astype(cfg.mixed_real_dtype),
        green_occ_a.astype(cfg.mixed_complex_dtype),
    )
    ci2g_b = jnp.einsum(
        "ptqu,pt->qu",
        c2bb.astype(cfg.mixed_real_dtype),
        green_occ_b.astype(cfg.mixed_complex_dtype),
    )
    ci2g_ab_a = jnp.einsum(
        "ptqu,qu->pt",
        c2ab.astype(cfg.mixed_real_dtype),
        green_occ_b.astype(cfg.mixed_complex_dtype),
    )
    ci2g_ab_b = jnp.einsum(
        "ptqu,pt->qu",
        c2ab.astype(cfg.mixed_real_dtype),
        green_occ_a.astype(cfg.mixed_complex_dtype),
    )
    gci2g_a = 0.5 * jnp.einsum("qu,qu->", ci2g_a, green_occ_a, optimize="optimal")
    gci2g_b = 0.5 * jnp.einsum("qu,qu->", ci2g_b, green_occ_b, optimize="optimal")
    gci2g_ab = jnp.einsum("pt,pt->", ci2g_ab_a, green_occ_a, optimize="optimal")
    gci2g = gci2g_a + gci2g_b + gci2g_ab
    fb_2_1 = lg * gci2g
    ci2_green_a = (greenp_a @ (ci2g_a + ci2g_ab_a).T) @ green_a
    ci2_green_b = (greenp_b @ (ci2g_b + ci2g_ab_b).T) @ green_b
    fb_2_2_a = -jnp.einsum(
        "gij,ij->g",
        chol_a.astype(cfg.mixed_real_dtype),
        ci2_green_a.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    fb_2_2_b = -jnp.einsum(
        "gij,ij->g",
        chol_b.astype(cfg.mixed_real_dtype),
        ci2_green_b.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    fb_2_2 = fb_2_2_a + fb_2_2_b
    fb_2 = fb_2_1 + fb_2_2

    # overlap
    overlap_1 = ci1g
    overlap_2 = gci2g
    overlap = 1.0 + overlap_1 + overlap_2

    return (fb_0 + fb_1 + fb_2) / overlap


def force_bias_kernel_gw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: UcisdMeasCtx,
    trial_data: UcisdTrial,
) -> jax.Array:
    c1a = trial_data.c1a
    c1b = trial_data.c1b
    c2aa = trial_data.c2aa
    c2ab = trial_data.c2ab
    c2bb = trial_data.c2bb
    norb = trial_data.norb
    n_oa, n_ob = trial_data.nocc
    n_va, n_vb = trial_data.nvir

    w = _get_generalized_walker_in_alpha_basis(walker, trial_data)
    bra = _build_bra_generalized(trial_data)

    # Half green function (U (V^\dag U)^{-1})^T
    #        n_oa n_va n_ob n_vb
    # n_oa (  1    2    3    4  )
    # n_ob (  5    6    7    8  )
    #
    green = _half_green_from_overlap_matrix(w, bra.T.conj() @ w)

    # (1, 2)
    green_aa = green[:n_oa, :norb]
    # (7, 8)
    green_bb = green[n_oa:, norb:]
    # (3, 4)
    green_ab = green[:n_oa, norb:]
    # (5, 6)
    green_ba = green[n_oa:, :norb]

    # (2)
    green_occ_aa = green_aa[:, n_oa:]
    # (8)
    green_occ_bb = green_bb[:, n_ob:]
    # (4)
    green_occ_ab = green_ab[:, n_ob:]
    # (6)
    green_occ_ba = green_ba[:, n_oa:]

    greenp_aa = jnp.vstack((green_occ_aa, -jnp.eye(n_va)))
    greenp_bb = jnp.vstack((green_occ_bb, -jnp.eye(n_vb)))
    greenp_ab = jnp.vstack((green_occ_ab, -jnp.zeros((n_va, n_vb))))
    greenp_ba = jnp.vstack((green_occ_ba, -jnp.zeros((n_vb, n_va))))

    chol_aa = ham_data.chol
    chol_bb = meas_ctx.chol_b

    rot_chol_aa = meas_ctx.rot_chol_a
    rot_chol_bb = meas_ctx.rot_chol_b

    # Ref
    # nu0 = jnp.einsum("gpq,pq->g", chol[:, :nocc, :], green)
    nu0 = jnp.einsum("gpq,pq->g", rot_chol_aa, green_aa)
    nu0 += jnp.einsum("gpq,pq->g", rot_chol_bb, green_bb)

    # Single excitations
    # nu1 = jnp.einsum("gpq,ia,pq,ia->g", rot_chol, ci1.conj(), green, green_occ)
    nu1 = jnp.einsum("gpq,ia,pq,ia->g", rot_chol_aa, c1a.conj(), green_aa, green_occ_aa)
    nu1 += jnp.einsum("gpq,ia,pq,ia->g", rot_chol_aa, c1b.conj(), green_aa, green_occ_bb)
    nu1 += jnp.einsum("gpq,ia,pq,ia->g", rot_chol_bb, c1a.conj(), green_bb, green_occ_aa)
    nu1 += jnp.einsum("gpq,ia,pq,ia->g", rot_chol_bb, c1b.conj(), green_bb, green_occ_bb)

    # nu1 -= jnp.einsum("gpq,ia,iq,pa->g", chol, ci1.conj(), green, greenp)
    nu1 -= jnp.einsum("gpq,ia,iq,pa->g", chol_aa, c1a.conj(), green_aa, greenp_aa)
    nu1 -= jnp.einsum("gpq,ia,iq,pa->g", chol_aa, c1b.conj(), green_ba, greenp_ab)
    nu1 -= jnp.einsum("gpq,ia,iq,pa->g", chol_bb, c1a.conj(), green_ab, greenp_ba)
    nu1 -= jnp.einsum("gpq,ia,iq,pa->g", chol_bb, c1b.conj(), green_bb, greenp_bb)

    # Double excitations
    # nu2 = 2.0 * jnp.einsum("gpq,iajb,pq,ia,jb->g", rot_chol, ci2.conj(), green, green_occ, green_occ)
    nu2 = 2.0 * jnp.einsum(
        "gpq,iajb,pq,ia,jb->g",
        rot_chol_aa,
        c2aa.conj(),
        green_aa,
        green_occ_aa,
        green_occ_aa,
    )
    nu2 += 2.0 * jnp.einsum(
        "gpq,iajb,pq,ia,jb->g",
        rot_chol_aa,
        c2bb.conj(),
        green_aa,
        green_occ_bb,
        green_occ_bb,
    )
    nu2 += 4.0 * jnp.einsum(
        "gpq,iajb,pq,ia,jb->g",
        rot_chol_aa,
        c2ab.conj(),
        green_aa,
        green_occ_aa,
        green_occ_bb,
    )
    nu2 -= 4.0 * jnp.einsum(
        "gpq,iajb,pq,ib,ja->g",
        rot_chol_aa,
        c2ab.conj(),
        green_aa,
        green_occ_ab,
        green_occ_ba,
    )
    nu2 += 2.0 * jnp.einsum(
        "gpq,iajb,pq,ia,jb->g",
        rot_chol_bb,
        c2aa.conj(),
        green_bb,
        green_occ_aa,
        green_occ_aa,
    )
    nu2 += 2.0 * jnp.einsum(
        "gpq,iajb,pq,ia,jb->g",
        rot_chol_bb,
        c2bb.conj(),
        green_bb,
        green_occ_bb,
        green_occ_bb,
    )
    nu2 += 4.0 * jnp.einsum(
        "gpq,iajb,pq,ia,jb->g",
        rot_chol_bb,
        c2ab.conj(),
        green_bb,
        green_occ_aa,
        green_occ_bb,
    )
    nu2 -= 4.0 * jnp.einsum(
        "gpq,iajb,pq,ib,ja->g",
        rot_chol_bb,
        c2ab.conj(),
        green_bb,
        green_occ_ab,
        green_occ_ba,
    )

    # nu2 -= 4.0 * jnp.einsum("gpq,iajb,pa,iq,jb->g", chol, ci2.conj(), greenp, green, green_occ)
    nu2 -= 4.0 * jnp.einsum(
        "gpq,iajb,pa,iq,jb->g", chol_aa, c2aa.conj(), greenp_aa, green_aa, green_occ_aa
    )
    nu2 -= 4.0 * jnp.einsum(
        "gpq,iajb,pa,iq,jb->g", chol_aa, c2bb.conj(), greenp_ab, green_ba, green_occ_bb
    )
    nu2 -= 4.0 * jnp.einsum(
        "gpq,iajb,pa,iq,jb->g", chol_aa, c2ab.conj(), greenp_aa, green_aa, green_occ_bb
    )
    nu2 += 4.0 * jnp.einsum(
        "gpq,iajb,pa,jq,ib->g", chol_aa, c2ab.conj(), greenp_aa, green_ba, green_occ_ab
    )
    nu2 += 4.0 * jnp.einsum(
        "gpq,iajb,pb,iq,ja->g", chol_aa, c2ab.conj(), greenp_ab, green_aa, green_occ_ba
    )
    nu2 -= 4.0 * jnp.einsum(
        "gpq,iajb,pb,jq,ia->g", chol_aa, c2ab.conj(), greenp_ab, green_ba, green_occ_aa
    )
    nu2 -= 4.0 * jnp.einsum(
        "gpq,iajb,pa,iq,jb->g", chol_bb, c2aa.conj(), greenp_ba, green_ab, green_occ_aa
    )
    nu2 -= 4.0 * jnp.einsum(
        "gpq,iajb,pa,iq,jb->g", chol_bb, c2bb.conj(), greenp_bb, green_bb, green_occ_bb
    )
    nu2 -= 4.0 * jnp.einsum(
        "gpq,iajb,pa,iq,jb->g", chol_bb, c2ab.conj(), greenp_ba, green_ab, green_occ_bb
    )
    nu2 += 4.0 * jnp.einsum(
        "gpq,iajb,pa,jq,ib->g", chol_bb, c2ab.conj(), greenp_ba, green_bb, green_occ_ab
    )
    nu2 += 4.0 * jnp.einsum(
        "gpq,iajb,pb,iq,ja->g", chol_bb, c2ab.conj(), greenp_bb, green_ab, green_occ_ba
    )
    nu2 -= 4.0 * jnp.einsum(
        "gpq,iajb,pb,jq,ia->g", chol_bb, c2ab.conj(), greenp_bb, green_bb, green_occ_aa
    )

    nu2 *= 0.25

    nu = nu0 + nu1 + nu2

    # o1 = jnp.einsum("ia,ia->", ci1.conj(), green_occ)
    o1 = jnp.einsum("ia,ia->", c1a.conj(), green_occ_aa)
    o1 += jnp.einsum("ia,ia->", c1b.conj(), green_occ_bb)

    # o2 = 0.5 * jnp.einsum("iajb, ia, jb->", ci2.conj(), green_occ, green_occ)
    o2 = jnp.einsum("iajb, ia, jb", c2aa, green_occ_aa, green_occ_aa)
    o2 += jnp.einsum("iajb, ia, jb", c2bb, green_occ_bb, green_occ_bb)
    o2 += 2.0 * jnp.einsum("iajb, ia, jb", c2ab, green_occ_aa, green_occ_bb)
    o2 -= 2.0 * jnp.einsum("iajb, ib, ja", c2ab, green_occ_ab, green_occ_ba)
    o2 = 0.5 * o2

    overlap = 1.0 + o1 + o2
    nu = nu / overlap

    return nu


def energy_kernel_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: UcisdMeasCtx,
    trial_data: UcisdTrial,
) -> jax.Array:
    n_elec_0 = trial_data.nocc[0]
    n_elec_1 = trial_data.nocc[1]
    return energy_kernel_uw_rh(
        (walker[:, :n_elec_0], walker[:, :n_elec_1]), ham_data, meas_ctx, trial_data
    )


def energy_kernel_uw_rh(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamChol,
    meas_ctx: UcisdMeasCtx,
    trial_data: UcisdTrial,
) -> jax.Array:
    wa, wb = walker
    n_oa, n_ob = trial_data.nocc
    n_va, n_vb = trial_data.nvir
    c1a = trial_data.c1a
    c1b = trial_data.c1b
    c2aa = trial_data.c2aa
    c2ab = trial_data.c2ab
    c2bb = trial_data.c2bb
    c_b = trial_data.mo_coeff_b

    cfg = meas_ctx.cfg

    wb = c_b.T @ wb[:, :n_ob]
    woa = wa[:n_oa, :]  # (n_oa, n_oa)
    wob = wb[:n_ob, :]  # (n_ob, n_ob)

    green_a = _half_green_from_overlap_matrix(wa, woa)  # (n_oa, norb)
    green_b = _half_green_from_overlap_matrix(wb, wob)  # (n_ob, norb)

    green_occ_a = green_a[:, n_oa:].copy()
    green_occ_b = green_b[:, n_ob:].copy()
    greenp_a = jnp.vstack((green_occ_a, -jnp.eye(n_va)))
    greenp_b = jnp.vstack((green_occ_b, -jnp.eye(n_vb)))

    lci1_a = meas_ctx.lci1_a
    lci1_b = meas_ctx.lci1_b

    chol_a = ham_data.chol
    chol_b = meas_ctx.chol_b
    rot_chol_a = meas_ctx.rot_chol_a
    rot_chol_b = meas_ctx.rot_chol_b

    h1_a = (ham_data.h1 + ham_data.h1.T) / 2.0
    h1_b = meas_ctx.h1_b
    hg_a = jnp.einsum("pj,pj->", h1_a[:n_oa, :], green_a)
    hg_b = jnp.einsum("pj,pj->", h1_b[:n_ob, :], green_b)
    hg = hg_a + hg_b

    # 0 body energy
    e0 = ham_data.h0

    # 1 body energy
    # ref
    e1_0 = hg

    # single excitations
    ci1g_a = jnp.einsum("pt,pt->", c1a, green_occ_a, optimize="optimal")
    ci1g_b = jnp.einsum("pt,pt->", c1b, green_occ_b, optimize="optimal")
    ci1g = ci1g_a + ci1g_b
    e1_1_1 = ci1g * hg
    gpc1a = greenp_a @ c1a.T
    gpc1b = greenp_b @ c1b.T
    ci1_green_a = gpc1a @ green_a
    ci1_green_b = gpc1b @ green_b
    e1_1_2 = -(
        jnp.einsum("ij,ij->", h1_a, ci1_green_a, optimize="optimal")
        + jnp.einsum("ij,ij->", h1_b, ci1_green_b, optimize="optimal")
    )
    e1_1 = e1_1_1 + e1_1_2

    # double excitations
    ci2g_a = (
        jnp.einsum(
            "ptqu,pt->qu",
            c2aa.astype(cfg.mixed_real_dtype),
            green_occ_a.astype(cfg.mixed_complex_dtype),
        )
        / 4
    )
    ci2g_b = (
        jnp.einsum(
            "ptqu,pt->qu",
            c2bb.astype(cfg.mixed_real_dtype),
            green_occ_b.astype(cfg.mixed_complex_dtype),
        )
        / 4
    )
    ci2g_ab_a = jnp.einsum(
        "ptqu,qu->pt",
        c2ab.astype(cfg.mixed_real_dtype),
        green_occ_b.astype(cfg.mixed_complex_dtype),
    )
    ci2g_ab_b = jnp.einsum(
        "ptqu,pt->qu",
        c2ab.astype(cfg.mixed_real_dtype),
        green_occ_a.astype(cfg.mixed_complex_dtype),
    )
    gci2g_a = jnp.einsum("qu,qu->", ci2g_a, green_occ_a, optimize="optimal")
    gci2g_b = jnp.einsum("qu,qu->", ci2g_b, green_occ_b, optimize="optimal")
    gci2g_ab = jnp.einsum("pt,pt->", ci2g_ab_a, green_occ_a, optimize="optimal")
    gci2g = 2 * (gci2g_a + gci2g_b) + gci2g_ab
    e1_2_1 = hg * gci2g
    ci2_green_a = (greenp_a @ ci2g_a.T) @ green_a
    ci2_green_ab_a = (greenp_a @ ci2g_ab_a.T) @ green_a
    ci2_green_b = (greenp_b @ ci2g_b.T) @ green_b
    ci2_green_ab_b = (greenp_b @ ci2g_ab_b.T) @ green_b
    e1_2_2_a = -jnp.einsum("ij,ij->", h1_a, 4 * ci2_green_a + ci2_green_ab_a, optimize="optimal")
    e1_2_2_b = -jnp.einsum("ij,ij->", h1_b, 4 * ci2_green_b + ci2_green_ab_b, optimize="optimal")
    e1_2_2 = e1_2_2_a + e1_2_2_b
    e1_2 = e1_2_1 + e1_2_2

    e1 = e1_0 + e1_1 + e1_2

    # two body energy
    # ref
    lg_a = jnp.einsum("gpj,pj->g", rot_chol_a, green_a, optimize="optimal")
    lg_b = jnp.einsum("gpj,pj->g", rot_chol_b, green_b, optimize="optimal")
    e2_0_1 = ((lg_a + lg_b) @ (lg_a + lg_b)) / 2.0
    lg1_a = jnp.einsum("gpj,qj->gpq", rot_chol_a, green_a, optimize="optimal")
    lg1_b = jnp.einsum("gpj,qj->gpq", rot_chol_b, green_b, optimize="optimal")
    e2_0_2 = (
        -(jnp.sum(jax.vmap(lambda x: x * x.T)(lg1_a)) + jnp.sum(jax.vmap(lambda x: x * x.T)(lg1_b)))
        / 2.0
    )
    e2_0 = e2_0_1 + e2_0_2

    # single excitations
    e2_1_1 = e2_0 * ci1g
    lci1g_a = jnp.einsum(
        "gij,ij->g",
        chol_a.astype(cfg.mixed_real_dtype),
        ci1_green_a.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    lci1g_b = jnp.einsum(
        "gij,ij->g",
        chol_b.astype(cfg.mixed_real_dtype),
        ci1_green_b.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    e2_1_2 = -((lci1g_a + lci1g_b) @ (lg_a + lg_b))
    ci1g1_a = c1a @ green_occ_a.T
    ci1g1_b = c1b @ green_occ_b.T
    e2_1_3_1 = jnp.einsum("gpq,gqr,rp->", lg1_a, lg1_a, ci1g1_a, optimize="optimal") + jnp.einsum(
        "gpq,gqr,rp->", lg1_b, lg1_b, ci1g1_b, optimize="optimal"
    )
    lci1g_a = jnp.einsum("gip,qi->gpq", lci1_a, green_a, optimize="optimal")
    lci1g_b = jnp.einsum("gip,qi->gpq", lci1_b, green_b, optimize="optimal")
    e2_1_3_2 = -jnp.einsum("gpq,gqp->", lci1g_a, lg1_a, optimize="optimal") - jnp.einsum(
        "gpq,gqp->", lci1g_b, lg1_b, optimize="optimal"
    )
    e2_1_3 = e2_1_3_1 + e2_1_3_2
    e2_1 = e2_1_1 + e2_1_2 + e2_1_3

    # double excitations
    e2_2_1 = e2_0 * gci2g
    lci2g_a = jnp.einsum(
        "gij,ij->g",
        chol_a.astype(cfg.mixed_real_dtype),
        8 * ci2_green_a.astype(cfg.mixed_complex_dtype)
        + 2 * ci2_green_ab_a.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    lci2g_b = jnp.einsum(
        "gij,ij->g",
        chol_b.astype(cfg.mixed_real_dtype),
        8 * ci2_green_b.astype(cfg.mixed_complex_dtype)
        + 2 * ci2_green_ab_b.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    e2_2_2_1 = -((lci2g_a + lci2g_b) @ (lg_a + lg_b)) / 2.0

    if cfg.memory_mode == "low":

        def scan_over_chol(carry, x):
            chol_a_i, rot_chol_a_i, chol_b_i, rot_chol_b_i = x
            gl_a_i = jnp.einsum("pj,ji->pi", green_a, chol_a_i, optimize="optimal")
            gl_b_i = jnp.einsum("pj,ji->pi", green_b, chol_b_i, optimize="optimal")
            lci2_green_a_i = jnp.einsum(
                "pi,ji->pj",
                rot_chol_a_i,
                8 * ci2_green_a + 2 * ci2_green_ab_a,
                optimize="optimal",
            )
            lci2_green_b_i = jnp.einsum(
                "pi,ji->pj",
                rot_chol_b_i,
                8 * ci2_green_b + 2 * ci2_green_ab_b,
                optimize="optimal",
            )
            carry[0] += 0.5 * (
                jnp.einsum("pi,pi->", gl_a_i, lci2_green_a_i, optimize="optimal")
                + jnp.einsum("pi,pi->", gl_b_i, lci2_green_b_i, optimize="optimal")
            )
            glgp_a_i = jnp.einsum("pi,it->pt", gl_a_i, greenp_a, optimize="optimal").astype(
                cfg.mixed_complex_dtype_testing
            )
            glgp_b_i = jnp.einsum("pi,it->pt", gl_b_i, greenp_b, optimize="optimal").astype(
                cfg.mixed_complex_dtype_testing
            )
            l2ci2_a = 0.5 * jnp.einsum(
                "pt,qu,ptqu->",
                glgp_a_i,
                glgp_a_i,
                c2aa.astype(cfg.mixed_real_dtype_testing),
                optimize="optimal",
            )
            l2ci2_b = 0.5 * jnp.einsum(
                "pt,qu,ptqu->",
                glgp_b_i,
                glgp_b_i,
                c2bb.astype(cfg.mixed_real_dtype_testing),
                optimize="optimal",
            )
            l2c2ab = jnp.einsum(
                "pt,qu,ptqu->",
                glgp_a_i,
                glgp_b_i,
                c2ab.astype(cfg.mixed_real_dtype_testing),
                optimize="optimal",
            )
            carry[1] += l2ci2_a + l2ci2_b + l2c2ab
            return carry, 0.0

        [e2_2_2_2, e2_2_3], _ = jax.lax.scan(
            scan_over_chol, [0.0, 0.0], (chol_a, rot_chol_a, chol_b, rot_chol_b)
        )
    else:
        gl_a = jnp.einsum(
            "pj,gji->gpi",
            green_a.astype(cfg.mixed_complex_dtype),
            chol_a.astype(cfg.mixed_real_dtype),
            optimize="optimal",
        )
        gl_b = jnp.einsum(
            "pj,gji->gpi",
            green_b.astype(cfg.mixed_complex_dtype),
            chol_b.astype(cfg.mixed_real_dtype),
            optimize="optimal",
        )
        lci2_green_a = jnp.einsum(
            "gpi,ji->gpj",
            rot_chol_a,
            8 * ci2_green_a + 2 * ci2_green_ab_a,
            optimize="optimal",
        )
        lci2_green_b = jnp.einsum(
            "gpi,ji->gpj",
            rot_chol_b,
            8 * ci2_green_b + 2 * ci2_green_ab_b,
            optimize="optimal",
        )
        e2_2_2_2 = 0.5 * (
            jnp.einsum("gpi,gpi->", gl_a, lci2_green_a, optimize="optimal")
            + jnp.einsum("gpi,gpi->", gl_b, lci2_green_b, optimize="optimal")
        )
        glgp_a = jnp.einsum("gpi,it->gpt", gl_a, greenp_a, optimize="optimal").astype(
            cfg.mixed_complex_dtype_testing
        )
        glgp_b = jnp.einsum("gpi,it->gpt", gl_b, greenp_b, optimize="optimal").astype(
            cfg.mixed_complex_dtype_testing
        )
        l2ci2_a = 0.5 * jnp.einsum(
            "gpt,gqu,ptqu->g",
            glgp_a,
            glgp_a,
            c2aa.astype(cfg.mixed_real_dtype_testing),
            optimize="optimal",
        )
        l2ci2_b = 0.5 * jnp.einsum(
            "gpt,gqu,ptqu->g",
            glgp_b,
            glgp_b,
            c2bb.astype(cfg.mixed_real_dtype_testing),
            optimize="optimal",
        )
        l2c2ab = jnp.einsum(
            "gpt,gqu,ptqu->g",
            glgp_a,
            glgp_b,
            c2ab.astype(cfg.mixed_real_dtype_testing),
            optimize="optimal",
        )
        e2_2_3 = l2ci2_a.sum() + l2ci2_b.sum() + l2c2ab.sum()

    e2_2_2 = e2_2_2_1 + e2_2_2_2
    e2_2 = e2_2_1 + e2_2_2 + e2_2_3

    e2 = e2_0 + e2_1 + e2_2

    # overlap
    overlap_1 = ci1g  # jnp.einsum("ia,ia", ci1, green_occ)
    overlap_2 = gci2g
    overlap = 1.0 + overlap_1 + overlap_2
    return (e1 + e2) / overlap + e0


def energy_kernel_gw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: UcisdMeasCtx,
    trial_data: UcisdTrial,
) -> jax.Array:
    norb = trial_data.norb
    n_oa, n_ob = trial_data.nocc
    n_va, n_vb = trial_data.nvir
    c1a = trial_data.c1a
    c1b = trial_data.c1b
    c2aa = trial_data.c2aa
    c2ab = trial_data.c2ab
    c2bb = trial_data.c2bb

    cfg = meas_ctx.cfg

    w = _get_generalized_walker_in_alpha_basis(walker, trial_data)
    bra = _build_bra_generalized(trial_data)

    # Half green function (U (V^\dag U)^{-1})^T
    #        n_oa n_va n_ob n_vb
    # n_oa (  1    2    3    4  )
    # n_ob (  5    6    7    8  )
    #
    green = _half_green_from_overlap_matrix(w, bra.T.conj() @ w)

    # (1, 2)
    green_aa = green[:n_oa, :norb]
    # (7, 8)
    green_bb = green[n_oa:, norb:]
    # (3, 4)
    green_ab = green[:n_oa, norb:]
    # (5, 6)
    green_ba = green[n_oa:, :norb]

    # (2)
    green_occ_aa = green_aa[:, n_oa:]
    # (8)
    green_occ_bb = green_bb[:, n_ob:]
    # (4)
    green_occ_ab = green_ab[:, n_ob:]
    # (6)
    green_occ_ba = green_ba[:, n_oa:]

    green_occ = jnp.block([[green_occ_aa, green_occ_ab], [green_occ_ba, green_occ_bb]])

    greenp_aa = jnp.vstack((green_occ_aa, -jnp.eye(n_va)))
    greenp_bb = jnp.vstack((green_occ_bb, -jnp.eye(n_vb)))
    greenp_ab = jnp.vstack((green_occ_ab, -jnp.zeros((n_va, n_vb))))
    greenp_ba = jnp.vstack((green_occ_ba, -jnp.zeros((n_vb, n_va))))

    greenp = jnp.block([[greenp_aa, greenp_ab], [greenp_ba, greenp_bb]])

    h1_aa = ham_data.h1
    h1_bb = meas_ctx.h1_b
    # h1 = la.block_diag(h1_aa, h1_bb)

    rot_h1_aa = h1_aa[:n_oa, :]
    rot_h1_bb = h1_bb[:n_ob, :]
    # rot_h1 = la.block_diag(rot_h1_aa, rot_h1_bb)

    chol_aa = ham_data.chol
    chol_bb = meas_ctx.chol_b
    nchol = jnp.shape(chol_aa)[0]

    # def chol_block(i):
    #    return la.block_diag(chol_aa[i], chol_bb[i])

    # chol = jax.vmap(chol_block)(jnp.arange(nchol))

    rot_chol_aa = chol_aa[:, :n_oa, :]
    rot_chol_bb = chol_bb[:, :n_ob, :]

    # def rot_chol_block(i):
    #    return la.block_diag(rot_chol_aa[i], rot_chol_bb[i])

    # rot_chol = jax.vmap(rot_chol_block)(jnp.arange(nchol))

    # ci1 = la.block_diag(c1a, c1b)

    # ci2 = jnp.zeros((n_oa + n_ob, n_va + n_vb, n_oa + n_ob, n_va + n_vb))
    # ci2 = lax.dynamic_update_slice(ci2, c2aa, (0, 0, 0, 0))
    # ci2 = lax.dynamic_update_slice(ci2, c2bb, (n_oa, n_va, n_oa, n_va))
    # ci2 = lax.dynamic_update_slice(ci2, c2ab, (0, 0, n_oa, n_va))
    # ci2 = lax.dynamic_update_slice(
    #    ci2, -jnp.einsum("iajb->jaib", c2ab), (n_oa, 0, 0, n_va)
    # )
    # ci2 = lax.dynamic_update_slice(
    #    ci2, -jnp.einsum("iajb->ibja", c2ab), (0, n_va, n_oa, 0)
    # )
    # ci2 = lax.dynamic_update_slice(
    #    ci2, jnp.einsum("iajb->jbia", c2ab), (n_oa, n_va, 0, 0)
    # )

    GRCaa = jnp.einsum("ir,gpr->igp", green[:, :norb], rot_chol_aa, optimize="optimal")
    GRCbb = jnp.einsum("ir,gpr->igp", green[:, norb:], rot_chol_bb, optimize="optimal")
    GRC = jnp.concatenate((GRCaa, GRCbb), axis=2)

    GCaa = jnp.einsum("ps,gqs->pgq", green[:, :norb], chol_aa, optimize="optimal")
    GCbb = jnp.einsum("ps,gqs->pgq", green[:, norb:], chol_bb, optimize="optimal")
    GC = jnp.concatenate((GCaa, GCbb), axis=2)

    GC_GRC = jnp.einsum("pgq,igp->qi", GC, GRC, optimize="optimal")

    GRC_gaa = jnp.einsum("qs,gqs->g", green[:n_oa, :norb], rot_chol_aa, optimize="optimal")
    GRC_gbb = jnp.einsum("qs,gqs->g", green[n_oa:, norb:], rot_chol_bb, optimize="optimal")
    GRC_g = GRC_gaa + GRC_gbb

    GpC1aa = jnp.einsum("qa,ia->qi", greenp[:, :n_va], c1a, optimize="optimal")
    GpC1bb = jnp.einsum("qa,ia->qi", greenp[:, n_va:], c1b, optimize="optimal")
    GpC1 = jnp.concatenate((GpC1aa, GpC1bb), axis=1)

    GoC1aa = jnp.einsum("ia,ia->", green_occ[:n_oa, :n_va], c1a, optimize="optimal")
    GoC1bb = jnp.einsum("ia,ia->", green_occ[n_oa:, n_va:], c1b, optimize="optimal")
    GoC1 = GoC1aa + GoC1bb

    GpC1G = jnp.einsum("qi,is->qs", GpC1, green, optimize="optimal")
    GpC1GCaa = jnp.einsum("qs,gqs->g", GpC1G[:norb, :norb], chol_aa, optimize="optimal")
    GpC1GCbb = jnp.einsum("qs,gqs->g", GpC1G[norb:, norb:], chol_bb, optimize="optimal")
    GpC1GC = GpC1GCaa + GpC1GCbb

    Caaaa = jnp.einsum(
        "ja,iajb->ib",
        green_occ[:n_oa, :n_va],
        c2aa,
        optimize="optimal",
    )
    Caabb = jnp.einsum(
        "ja,iajb->ib",
        green_occ[n_oa:, :n_va],
        c2ab,
        optimize="optimal",
    )
    Cbbaa = jnp.einsum(
        "ja,iajb->ib",
        green_occ[:n_oa, n_va:],
        jnp.einsum("iajb->jbia", c2ab),
        optimize="optimal",
    )
    Cabba = jnp.einsum(
        "ja,iajb->ib",
        green_occ[n_oa:, n_va:],
        -jnp.einsum("iajb->ibja", c2ab),
        optimize="optimal",
    )
    Cbaab = jnp.einsum(
        "ja,iajb->ib",
        green_occ[:n_oa, :n_va],
        -jnp.einsum("iajb->jaib", c2ab),
        optimize="optimal",
    )
    Cbbbb = jnp.einsum("ja,iajb->ib", green_occ[n_oa:, n_va:], c2bb, optimize="optimal")

    CGo = jnp.zeros((n_oa + n_ob, n_va + n_vb), dtype=Caaaa.dtype)
    CGo = lax.dynamic_update_slice(CGo, Caaaa + Cabba, (0, 0))
    CGo = lax.dynamic_update_slice(CGo, Caabb, (0, n_va))
    CGo = lax.dynamic_update_slice(CGo, Cbbaa, (n_oa, 0))
    CGo = lax.dynamic_update_slice(CGo, Cbbbb + Cbaab, (n_oa, n_va))

    # 0 body energy
    e0 = ham_data.h0

    # 1 body energy
    # ref
    e1_0aa = jnp.einsum("pq,pq->", rot_h1_aa, green[:n_oa, :norb], optimize="optimal")
    e1_0bb = jnp.einsum("pq,pq->", rot_h1_bb, green[n_oa:, norb:], optimize="optimal")
    e1_0 = e1_0aa + e1_0bb

    # single excitations
    # e1_1 = jnp.einsum("pq,ia,pq,ia->", rot_h1, ci1.conj(), green, green_occ)
    # e1_1_0 = jnp.einsum("pq,ia,pq,ia->", rot_h1, ci1, green, green_occ)

    Aaa = jnp.einsum("pq,pq->", rot_h1_aa, green[:n_oa, :norb], optimize="optimal")
    Abb = jnp.einsum("pq,pq->", rot_h1_bb, green[n_oa:, norb:], optimize="optimal")
    A = Aaa + Abb

    Baa = jnp.einsum("ia,ia->", c1a, green_occ[:n_oa, :n_va], optimize="optimal")
    Bbb = jnp.einsum("ia,ia->", c1b, green_occ[n_oa:, n_va:], optimize="optimal")
    B = Baa + Bbb
    e1_1_0 = A * B

    # e1_1 -= jnp.einsum("pq,ia,iq,pa->", h1, ci1.conj(), green, greenp)
    # e1_1_1 = -jnp.einsum("pq,ia,iq,pa->", h1, ci1, green, greenp)

    Aaa = jnp.einsum("iq,pq->ip", green[:, :norb], h1_aa, optimize="optimal")
    Abb = jnp.einsum("iq,pq->ip", green[:, norb:], h1_bb, optimize="optimal")
    A = jnp.concatenate((Aaa, Abb), axis=1)
    B = jnp.einsum("ip,pa->ia", A, greenp)
    e1_1_1aa = -jnp.einsum("ia,ia->", B[:n_oa, :n_va], c1a, optimize="optimal")
    e1_1_1bb = -jnp.einsum("ia,ia->", B[n_oa:, n_va:], c1b, optimize="optimal")
    e1_1_1 = e1_1_1aa + e1_1_1bb

    e1_1 = e1_1_0 + e1_1_1

    ## double excitations
    # e1_2 = 2.0 * jnp.einsum("rq,rq,iajb,ia,jb", rot_h1, green, ci2.conj(), green_occ, green_occ)
    # e1_2_0 = 2.0 * jnp.einsum(
    #    "rq,rq,iajb,ia,jb", rot_h1, green, ci2, green_occ, green_occ
    # )

    Aaa = jnp.einsum("rq,rq->", rot_h1_aa, green[:n_oa, :norb], optimize="optimal")
    Abb = jnp.einsum("rq,rq->", rot_h1_bb, green[n_oa:, norb:], optimize="optimal")
    A = Aaa + Abb

    e1_2_0 = 2.0 * A * jnp.einsum("jb,jb", -CGo, green_occ)

    # e1i_2 -= 4.0 * jnp.einsum("pq,iajb,pa,iq,jb", h1, ci2.conj(), greenp, green, green_occ)
    # e1_2_1 = -4.0 * jnp.einsum(
    #    "pq,iajb,pa,iq,jb", h1, ci2, greenp, green, green_occ
    # )

    Aaa = jnp.einsum("iq,pq->ip", green[:, :norb], h1_aa, optimize="optimal")
    Abb = jnp.einsum("iq,pq->ip", green[:, norb:], h1_bb, optimize="optimal")
    A = jnp.concatenate((Aaa, Abb), axis=1)

    e1_2_1 = -4.0 * jnp.einsum("ip,ia,pa", A, -CGo, greenp, optimize="optimal")

    e1_2 = e1_2_0 + e1_2_1
    e1_2 *= 0.25

    # 2 body energy
    # ref
    # f = jnp.einsum("gij,jk->gik", rot_chol, green.T, optimize="optimal")
    faa = jnp.einsum("gij,jk->gik", rot_chol_aa, green[:, :norb].T, optimize="optimal")
    fbb = jnp.einsum("gij,jk->gik", rot_chol_bb, green[:, norb:].T, optimize="optimal")
    f = jnp.concatenate((faa, fbb), axis=1)
    c = vmap(jnp.trace)(f)
    exc = jnp.sum(vmap(lambda x: x * x.T)(f))
    e2_0 = (jnp.sum(c * c) - exc) / 2.0

    # single excitations
    # e2_1 = jnp.einsum( "gpr,gqs,ia,ir,ps,qa->", chol[:, :nocc, :], chol[:, :, :], ci1.conj(), green, green, greenp)
    # e2_1 = jnp.einsum(
    #    "gpr,gqs,ia,ir,ps,qa->", rot_chol, chol, ci1, green, green, greenp
    # )

    A = jnp.einsum("qi,qa->ia", GC_GRC, greenp, optimize="optimal")

    e2_1aa = jnp.einsum("ia,ia->", A[:n_oa, :n_va], c1a, optimize="optimal")
    e2_1bb = jnp.einsum("ia,ia->", A[n_oa:, n_va:], c1b, optimize="optimal")

    e2_1 = e2_1aa + e2_1bb

    # e2_1 -= jnp.einsum( "gpr,gqs,ia,pr,is,qa->", chol[:, :nocc, :], chol[:, :, :], ci1.conj(), green, green, greenp)
    # e2_1 -= jnp.einsum(
    #    "gpr,gqs,ia,pr,is,qa->", rot_chol, chol, ci1, green, green, greenp
    # )

    e2_1 -= jnp.einsum("g,g->", GRC_g, GpC1GC, optimize="optimal")

    # e2_1 -= jnp.einsum( "gpr,gqs,ia,ir,pa,qs->", chol[:, :, :], chol[:, :nocc, :], ci1.conj(), green, greenp, green)
    # e2_1 -= jnp.einsum(
    #    "gpr,gqs,ia,ir,pa,qs->", chol, rot_chol, ci1, green, greenp, green
    # )

    e2_1 -= jnp.einsum("g,g->", GpC1GC, GRC_g, optimize="optimal")

    # e2_1 += jnp.einsum( "gpr,gqs,ia,qr,is,pa->", chol[:, :, :], chol[:, :nocc, :], ci1.conj(), green, green, greenp)
    # e2_1 += jnp.einsum(
    #    "gpr,gqs,ia,qr,is,pa->", chol, rot_chol, ci1, green, green, greenp
    # )

    e2_1 += e2_1aa + e2_1bb

    # e2_1 += jnp.einsum( "gpr,gqs,ia,pr,ia,qs->", chol[:, :nocc, :], chol[:, :nocc, :], ci1.conj(), green, green_occ, green)
    # e2_1 += jnp.einsum(
    #    "gpr,gqs,ia,pr,ia,qs->", rot_chol, rot_chol, ci1, green, green_occ, green
    # )

    e2_1 += GoC1 * jnp.einsum("g,g->", GRC_g, GRC_g, optimize="optimal")

    # e2_1 -= jnp.einsum( "gpr,gqs,ia,qr,ia,ps->", chol[:, :nocc, :], chol[:, :nocc, :], ci1.conj(), green, green_occ, green)
    # e2_1 -= jnp.einsum(
    #    "gpr,gqs,ia,qr,ia,ps->", rot_chol, rot_chol, ci1, green, green_occ, green
    # )

    Aaa = jnp.einsum("qgp,gqs->ps", GRC[:n_oa, :, :], rot_chol_aa, optimize="optimal")
    Abb = jnp.einsum("qgp,gqs->ps", GRC[n_oa:, :, :], rot_chol_bb, optimize="optimal")
    A = jnp.concatenate((Aaa, Abb), axis=1)

    e2_1 -= GoC1 * jnp.einsum("ps,ps->", A, green, optimize="optimal")

    e2_1 *= 0.5

    ## double excitations
    # e2_2 = 2.0 * jnp.einsum("gpr,gqs,iajb,ir,js,pa,qb->", chol, chol, ci2.conj(), green, green, greenp, greenp)
    # e2_2 = 2.0 * jnp.einsum(
    #    "gpr,gqs,iajb,ir,js,pa,qb->", chol, chol, ci2, green, green, greenp, greenp
    # )

    B = jnp.einsum(
        "igp,pa->iga",
        GC,
        greenp,
        optimize="optimal",
    ).astype(cfg.mixed_complex_dtype_testing)

    Caaaa = jnp.einsum(
        "iga,iajb->gjb",
        B[:n_oa, :, :n_va],
        c2aa.astype(cfg.mixed_complex_dtype_testing),
        optimize="optimal",
    )
    Caabb = jnp.einsum(
        "iga,iajb->gjb",
        B[:n_oa, :, :n_va],
        c2ab.astype(cfg.mixed_complex_dtype_testing),
        optimize="optimal",
    )
    Cbbaa = jnp.einsum(
        "iga,iajb->gjb",
        B[n_oa:, :, n_va:],
        jnp.einsum("iajb->jbia", c2ab.astype(cfg.mixed_complex_dtype_testing)),
        optimize="optimal",
    )
    Cabba = jnp.einsum(
        "iga,iajb->gjb",
        B[:n_oa, :, n_va:],
        -jnp.einsum("iajb->ibja", c2ab.astype(cfg.mixed_complex_dtype_testing)),
        optimize="optimal",
    )
    Cbaab = jnp.einsum(
        "iga,iajb->gjb",
        B[n_oa:, :, :n_va],
        -jnp.einsum("iajb->jaib", c2ab.astype(cfg.mixed_complex_dtype_testing)),
        optimize="optimal",
    )
    Cbbbb = jnp.einsum(
        "iga,iajb->gjb",
        B[n_oa:, :, n_va:],
        c2bb.astype(cfg.mixed_complex_dtype_testing),
        optimize="optimal",
    )

    C = jnp.zeros((nchol, n_oa + n_ob, n_va + n_vb), dtype=Caaaa.dtype)
    C = lax.dynamic_update_slice(C, Caaaa + Cbbaa, (0, 0, 0))
    C = lax.dynamic_update_slice(C, Cabba, (0, n_oa, 0))
    C = lax.dynamic_update_slice(C, Cbaab, (0, 0, n_va))
    C = lax.dynamic_update_slice(C, Cbbbb + Caabb, (0, n_oa, n_va))

    D = jnp.einsum("gjb,qb->gjq", C, greenp, optimize="optimal")

    Eaa = jnp.einsum("gjq,gqs->js", D[:, :, :norb], chol_aa, optimize="optimal")
    Ebb = jnp.einsum("gjq,gqs->js", D[:, :, norb:], chol_bb, optimize="optimal")
    E = jnp.concatenate((Eaa, Ebb), axis=1)

    e2_2 = 2.0 * jnp.einsum("js,js->", E, green, optimize="optimal")

    # e2_2 -= 2.0 * jnp.einsum("gpr,gqs,iajb,ir,ps,ja,qb->", chol[:, :nocc, :], chol, ci2.conj(), green, green, green_occ, greenp)
    # e2_2 -= 2.0 * jnp.einsum(
    #    "gpr,gqs,iajb,ir,ps,ja,qb->",
    #    rot_chol,
    #    chol,
    #    ci2,
    #    green,
    #    green,
    #    green_occ,
    #    greenp,
    # )

    e2_2 -= 2.0 * jnp.einsum("igp,pgq,ib,qb->", GRC, GC, CGo, greenp, optimize="optimal")

    # e2_2 += 2.0 * jnp.einsum("gpr,gqs,iajb,ir,qs,ja,pb->", chol, chol[:, :nocc, :], ci2.conj(), green, green, green_occ, greenp)
    # e2_2 += 2.0 * jnp.einsum(
    #    "gpr,gqs,iajb,ir,qs,ja,pb->",
    #    chol,
    #    rot_chol,
    #    ci2,
    #    green,
    #    green,
    #    green_occ,
    #    greenp,
    # )

    B = jnp.einsum("ib,pb->ip", CGo, greenp, optimize="optimal")
    C = jnp.einsum("ip,ir->pr", B, green, optimize="optimal")

    Iaa = jnp.einsum("pr,gpr->g", C[:norb, :norb], chol_aa, optimize="optimal")
    Ibb = jnp.einsum("pr,gpr->g", C[norb:, norb:], chol_bb, optimize="optimal")
    I = Iaa + Ibb

    e2_2 += 2.0 * jnp.einsum("g,g->", I, GRC_g, optimize="optimal")

    ## P_ij
    e2_2 *= 2.0

    # e2_2 += 4.0 * jnp.einsum("gpr,gqs,iajb,pr,is,ja,qb->", chol[:, :nocc, :], chol, ci2.conj(), green, green, green_occ, greenp)
    # e2_2 += 4.0 * jnp.einsum(
    #    "gpr,gqs,iajb,pr,is,ja,qb->",
    #    rot_chol,
    #    chol,
    #    ci2,
    #    green,
    #    green,
    #    green_occ,
    #    greenp,
    # )

    e2_2 += 4.0 * jnp.einsum(
        "g,g->",
        GRC_g,
        I,
        optimize="optimal",
    )

    # e2_2 += 2.0 * jnp.einsum("gpr,gqs,iajb,pr,qs,ia,jb->", chol[:, :nocc, :], chol[:, :nocc, :], ci2.conj(), green, green, green_occ, green_occ)
    # e2_2 += 2.0 * jnp.einsum(
    #    "gpr,gqs,iajb,pr,qs,ia,jb->",
    #    rot_chol,
    #    rot_chol,
    #    ci2,
    #    green,
    #    green,
    #    green_occ,
    #    green_occ,
    # )

    e2_2 += 2.0 * jnp.einsum(
        "g,g,jb,jb->",
        GRC_g,
        GRC_g,
        -CGo,  # - because ia,iajb->jb instead of ja,iajb->ib
        green_occ,
        optimize="optimal",
    )

    ## P_pq
    # e2_2 -= 4.0 * jnp.einsum("gpr,gqs,iajb,qr,is,ja,pb->", chol, chol[:, :nocc, :], ci2.conj(), green, green, green_occ, greenp)
    # e2_2 -= 4.0 * jnp.einsum(
    #    "gpr,gqs,iajb,qr,is,ja,pb->",
    #    chol,
    #    rot_chol,
    #    ci2,
    #    green,
    #    green,
    #    green_occ,
    #    greenp,
    # )

    e2_2 -= 4.0 * jnp.einsum(
        "pi,ib,pb->",
        GC_GRC,
        CGo,
        greenp,
        optimize="optimal",
    )

    # e2_2 -= 2.0 * jnp.einsum("gpr,gqs,iajb,qr,ps,ia,jb->", chol[:, :nocc, :], chol[:, :nocc, :], ci2.conj(), green, green, green_occ, green_occ)
    # e2_2 -= 2.0 * jnp.einsum(
    #    "gpr,gqs,iajb,qr,ps,ia,jb->",
    #    rot_chol,
    #    rot_chol,
    #    ci2,
    #    green,
    #    green,
    #    green_occ,
    #    green_occ,
    # )

    Aaa = jnp.einsum("qgp,gqs->ps", GRC[:n_oa, :, :], rot_chol_aa, optimize="optimal")
    Abb = jnp.einsum("qgp,gqs->ps", GRC[n_oa:, :, :], rot_chol_bb, optimize="optimal")
    A = jnp.concatenate((Aaa, Abb), axis=1)

    e2_2 -= 2.0 * jnp.einsum(
        "ps,jb,ps,jb->",
        A,
        -CGo,
        green,
        green_occ,
        optimize="optimal",
    )

    e2_2 *= 0.5 * 0.25

    e = e1_0 + e1_1 + e1_2 + e2_0 + e2_1 + e2_2
    o1 = jnp.einsum("ia,ia", c1a, green_occ_aa) + jnp.einsum("ia,ia", c1b, green_occ_bb)
    o2 = jnp.einsum("iajb, ia, jb", c2aa, green_occ_aa, green_occ_aa)
    o2 += jnp.einsum("iajb, ia, jb", c2bb, green_occ_bb, green_occ_bb)
    o2 += 2.0 * jnp.einsum("iajb, ia, jb", c2ab, green_occ_aa, green_occ_bb)
    o2 -= 2.0 * jnp.einsum("iajb, ib, ja", c2ab, green_occ_ab, green_occ_ba)
    overlap = 1.0 + o1 + 0.5 * o2
    e = e / overlap

    return e + e0


def build_meas_ctx(
    ham_data: HamChol, trial_data: UcisdTrial, cfg: UcisdMeasCfg = UcisdMeasCfg()
) -> UcisdMeasCtx:
    if ham_data.basis != "restricted":
        raise ValueError("UCISD MeasOps currently assumes HamChol.basis == 'restricted'.")
    n_oa, n_ob = trial_data.nocc
    cb = trial_data.mo_coeff_b  # (norb, nocc[1])
    cbH = trial_data.mo_coeff_b.conj().T  # (nocc[1], norb)
    h1_b = 0.5 * (cbH @ (ham_data.h1 + ham_data.h1.T) @ cb)
    chol_b = jnp.einsum("pi,gij,jq->gpq", cbH, ham_data.chol, cb)
    rot_h1_a = ham_data.h1[:n_oa, :]  # (nocc[0], norb)
    rot_h1_b = ham_data.h1[:n_ob, :]  # (nocc[1], norb)
    rot_chol_a = ham_data.chol[:, :n_oa, :]
    rot_chol_b = chol_b[:, :n_ob, :]
    rot_chol_flat_a = rot_chol_a.reshape(rot_chol_a.shape[0], -1)
    rot_chol_flat_b = rot_chol_b.reshape(rot_chol_b.shape[0], -1)

    lci1_a = jnp.einsum(
        "git,pt->gip",
        ham_data.chol[:, :, n_oa:],
        trial_data.c1a,
        optimize="optimal",
    )
    lci1_b = jnp.einsum(
        "git,pt->gip",
        chol_b[:, :, n_ob:],
        trial_data.c1b,
        optimize="optimal",
    )
    return UcisdMeasCtx(
        h1_b=h1_b,
        chol_b=chol_b,
        rot_h1_a=rot_h1_a,
        rot_h1_b=rot_h1_b,
        rot_chol_a=rot_chol_a,
        rot_chol_b=rot_chol_b,
        rot_chol_flat_a=rot_chol_flat_a,
        rot_chol_flat_b=rot_chol_flat_b,
        lci1_a=lci1_a,
        lci1_b=lci1_b,
        cfg=cfg,
    )


def make_ucisd_meas_ops(
    sys: System,
    memory_mode: str = "high",
    mixed_precision: bool = True,
    testing: bool = False,
) -> MeasOps:
    wk = sys.walker_kind.lower()

    cfg = UcisdMeasCfg(
        memory_mode=memory_mode,
        mixed_real_dtype=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed_precision else jnp.complex128,
        mixed_real_dtype_testing=jnp.float64 if testing else jnp.float32,
        mixed_complex_dtype_testing=jnp.complex128 if testing else jnp.complex64,
    )

    if wk == "restricted":
        overlap_fn = overlap_r
        kernels = {
            k_force_bias: force_bias_kernel_rw_rh,
            k_energy: energy_kernel_rw_rh,
        }
    elif wk == "unrestricted":
        overlap_fn = overlap_u
        kernels = {
            k_force_bias: force_bias_kernel_uw_rh,
            k_energy: energy_kernel_uw_rh,
        }
    elif wk == "generalized":
        overlap_fn = overlap_g
        kernels = {
            k_force_bias: force_bias_kernel_gw_rh,
            k_energy: energy_kernel_gw_rh,
        }
    else:
        raise ValueError(f"unknown walker_kind: {sys.walker_kind}")

    return MeasOps(
        overlap=overlap_fn,
        build_meas_ctx=lambda ham_data, trial_data: build_meas_ctx(ham_data, trial_data, cfg),
        kernels=kernels,
    )
