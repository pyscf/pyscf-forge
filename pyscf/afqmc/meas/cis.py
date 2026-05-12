from __future__ import annotations


import jax
import jax.numpy as jnp

from ..core.ops import MeasOps, k_energy, k_force_bias
from ..core.system import System
from ..ham.chol import HamChol
from ..trial.cis import CisTrial, overlap_r
from . import cisd
from .cisd import CisdMeasCtx


def force_bias_kernel_rw_rh(
    walker: jax.Array, ham_data: HamChol, meas_ctx: CisdMeasCtx, trial_data: CisTrial
) -> jax.Array:
    """Calculates force bias < psi_T | chol_gamma | walker > / < psi_T | walker >"""
    ci1 = trial_data.ci1
    nocc = trial_data.nocc
    norb = trial_data.norb
    green = (walker @ jnp.linalg.inv(walker[:nocc, :])).T
    green_occ = green[:, nocc:].copy()
    greenp = jnp.vstack((green_occ, -jnp.eye(norb - nocc)))

    chol = ham_data.chol
    rot_chol = meas_ctx.rot_chol

    lg = jnp.einsum("gpj,pj->g", rot_chol, green, optimize="optimal")

    ci1g = jnp.einsum("pt,pt->", ci1, green_occ, optimize="optimal")
    ci1gp = jnp.einsum("pt,it->pi", ci1, greenp, optimize="optimal")
    gci1gp = jnp.einsum("pj,pi->ij", green, ci1gp, optimize="optimal")
    fb_1_1 = 4 * ci1g * lg
    fb_1_2 = -2 * jnp.einsum("gij,ij->g", chol, gci1gp, optimize="optimal")
    fb_1 = fb_1_1 + fb_1_2

    # overlap
    overlap_1 = 2 * ci1g
    overlap = overlap_1

    return fb_1 / overlap


def energy_kernel_rw_rh(
    walker: jax.Array, ham_data: HamChol, meas_ctx: CisdMeasCtx, trial_data: CisTrial
) -> jax.Array:
    ci1 = trial_data.ci1
    nocc = trial_data.nocc
    norb = trial_data.norb
    green = (walker @ jnp.linalg.inv(walker[:nocc, :])).T
    green_occ = green[:, nocc:].copy()
    greenp = jnp.vstack((green_occ, -jnp.eye(norb - nocc)))

    chol = ham_data.chol
    rot_chol = meas_ctx.rot_chol
    h1 = ham_data.h1
    hg = jnp.einsum("pj,pj->", h1[:nocc, :], green)

    lci1 = meas_ctx.lci1

    # 0 body energy
    e0 = ham_data.h0

    # 1 body energy
    ci1g = jnp.einsum("pt,pt->", ci1, green_occ, optimize="optimal")
    e1_1_1 = 4 * ci1g * hg
    gpci1 = greenp @ ci1.T
    ci1_green = gpci1 @ green
    e1_1_2 = -2 * jnp.einsum("ij,ij->", h1, ci1_green, optimize="optimal")
    e1_1 = e1_1_1 + e1_1_2

    e1 = e1_1

    # two body energy
    lg = jnp.einsum("gpj,pj->g", rot_chol, green, optimize="optimal")
    # lg1 = jnp.einsum("gpj,pk->gjk", rot_chol, green, optimize="optimal")
    lg1 = jnp.einsum("gpj,qj->gpq", rot_chol, green, optimize="optimal")
    e2_0_1 = 2 * lg @ lg
    e2_0_2 = -jnp.sum(jax.vmap(lambda x: x * x.T)(lg1))
    e2_0 = e2_0_1 + e2_0_2

    # single excitations
    e2_1_1 = 2 * e2_0 * ci1g
    lci1g = jnp.einsum("gij,ij->g", chol, ci1_green, optimize="optimal")
    e2_1_2 = -2 * (lci1g @ lg)
    # lci1g1 = jnp.einsum("gij,jk->gik", chol, ci1_green, optimize="optimal")
    # glgpci1 = jnp.einsum(("gpi,iq->gpq"), gl, gpci1, optimize="optimal")
    ci1g1 = ci1 @ green_occ.T
    # e2_1_3 = jnp.einsum("gpq,gpq->", glgpci1, lg1, optimize="optimal")
    e2_1_3_1 = jnp.einsum("gpq,gqr,rp->", lg1, lg1, ci1g1, optimize="optimal")
    lci1g = jnp.einsum("gip,qi->gpq", lci1, green, optimize="optimal")
    e2_1_3_2 = -jnp.einsum("gpq,gqp->", lci1g, lg1, optimize="optimal")
    e2_1_3 = e2_1_3_1 + e2_1_3_2
    e2_1 = e2_1_1 + 2 * (e2_1_2 + e2_1_3)

    e2 = e2_1

    # overlap
    overlap_1 = 2 * ci1g  # jnp.einsum("ia,ia", ci1, green_occ)
    overlap = overlap_1
    return (e1 + e2) / overlap + e0


def make_cis_meas_ops(sys: System) -> MeasOps:
    if sys.walker_kind.lower() != "restricted":
        raise ValueError(
            f"CIS MeasOps currently supports only restricted walkers, got: {sys.walker_kind}"
        )

    return MeasOps(
        overlap=overlap_r,
        build_meas_ctx=lambda ham_data, trial_data: cisd.build_meas_ctx(ham_data, trial_data),
        kernels={k_force_bias: force_bias_kernel_rw_rh, k_energy: energy_kernel_rw_rh},
    )
