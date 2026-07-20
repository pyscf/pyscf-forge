from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import tree_util

from ..core.ops import MeasOps, k_energy, k_force_bias, o_density_corr, o_rdm1
from ..core.system import System
from ..ham.chol import HamChol
from ..ham.hubbard import HamHubbard
from ..trial.ghf import (
    GhfTrial,
    calc_green_g,
    calc_green_u,
    overlap_g,
    overlap_r,
    overlap_u,
)


def _rdm1_from_green(g: jax.Array, norb: int) -> jax.Array:
    return jnp.stack([g[:norb, :norb], g[norb:, norb:]], axis=0)


def _density_corr_from_green(g: jax.Array, norb: int) -> jax.Array:
    """Density correlation from a full (2norb, 2norb) Green's function.
    Returns (3, norb, norb): uu, ud, dd."""
    guu = g[:norb, :norb]
    gdd = g[norb:, norb:]
    gud = g[:norb, norb:]
    gdu = g[norb:, :norb]

    nu = jnp.diagonal(guu)
    nd = jnp.diagonal(gdd)

    uu = nu[:, None] * nu[None, :] - guu * guu.T + jnp.diag(nu)
    dd = nd[:, None] * nd[None, :] - gdd * gdd.T + jnp.diag(nd)
    ud = nu[:, None] * nd[None, :] - gud * gdu.T

    return jnp.stack([uu, ud, dd], axis=0)


def rdm1_kernel_rw(
    walker: jax.Array,
    ham_data: Any,
    meas_ctx: Any,
    trial_data: GhfTrial,
) -> jax.Array:
    g = calc_green_u((walker, walker), trial_data)
    return _rdm1_from_green(g, trial_data.norb)


def rdm1_kernel_uw(
    walker: tuple[jax.Array, jax.Array],
    ham_data: Any,
    meas_ctx: Any,
    trial_data: GhfTrial,
) -> jax.Array:
    g = calc_green_u(walker, trial_data)
    return _rdm1_from_green(g, trial_data.norb)


def rdm1_kernel_gw(
    walker: jax.Array,
    ham_data: Any,
    meas_ctx: Any,
    trial_data: GhfTrial,
) -> jax.Array:
    g = calc_green_g(walker, trial_data)
    return _rdm1_from_green(g, trial_data.norb)


def density_corr_kernel_rw(
    walker: jax.Array,
    ham_data: Any,
    meas_ctx: Any,
    trial_data: GhfTrial,
) -> jax.Array:
    g = calc_green_u((walker, walker), trial_data)
    return _density_corr_from_green(g, trial_data.norb)


def density_corr_kernel_uw(
    walker: tuple[jax.Array, jax.Array],
    ham_data: Any,
    meas_ctx: Any,
    trial_data: GhfTrial,
) -> jax.Array:
    g = calc_green_u(walker, trial_data)
    return _density_corr_from_green(g, trial_data.norb)


def density_corr_kernel_gw(
    walker: jax.Array,
    ham_data: Any,
    meas_ctx: Any,
    trial_data: GhfTrial,
) -> jax.Array:
    g = calc_green_g(walker, trial_data)
    return _density_corr_from_green(g, trial_data.norb)


# ---------------------
# chol
# ---------------------


def _green_half_u(wu: jax.Array, wd: jax.Array, trial_data: GhfTrial) -> jax.Array:
    """
    Mixed half Green for unrestricted walker, returned as (nelec_total, 2*norb).
    """
    norb = trial_data.norb
    cH = trial_data.mo_coeff.conj().T  # (ne,2n)
    top = cH[:, :norb] @ wu  # (ne,nup)
    bot = cH[:, norb:] @ wd  # (ne,ndn)
    ovlp = jnp.hstack([top, bot])  # (ne,ne)
    inv = jnp.linalg.inv(ovlp)

    nup = wu.shape[1]
    gT = jnp.vstack([wu @ inv[:nup], wd @ inv[nup:]])  # (2n,ne)
    return gT.T  # (ne,2n)


def _green_half_r(w: jax.Array, trial_data: GhfTrial) -> jax.Array:
    """
    Mixed half Green for restricted walker, returned as (nelec_total, 2*norb).
    """
    norb = trial_data.norb
    cH = trial_data.mo_coeff.conj().T
    top = cH[:, :norb] @ w
    bot = cH[:, norb:] @ w
    ovlp = jnp.hstack([top, bot])  # (ne,ne)
    inv = jnp.linalg.inv(ovlp)

    nocc = w.shape[1]
    gT = jnp.vstack([w @ inv[:nocc], w @ inv[nocc:]])  # (2n,ne)
    return gT.T  # (ne,2n)


def _green_half_g(w: jax.Array, trial_data: GhfTrial) -> jax.Array:
    """
    Mixed half Green for generalized walker, returned as (nelec_total, 2*norb).
    """
    ovlp = trial_data.mo_coeff.conj().T @ w  # (ne,ne)
    inv = jnp.linalg.inv(ovlp)
    return (w @ inv).T  # (ne,2n)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GhfCholMeasCtx:
    """
    Half-rotated intermediates for GHF estimators with cholesky hamiltonian.

    rot_h1: (ne, ns) where ns = 2*norb
    rot_chol: (nchol, ne, ns)
    rot_chol_flat: (nchol, ne*ns)
    """

    rot_h1: jax.Array
    rot_chol: jax.Array
    rot_chol_flat: jax.Array

    def tree_flatten(self):
        return (self.rot_h1, self.rot_chol, self.rot_chol_flat), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        rot_h1, rot_chol, rot_chol_flat = children
        return cls(
            rot_h1=rot_h1,
            rot_chol=rot_chol,
            rot_chol_flat=rot_chol_flat,
        )


def build_meas_ctx_chol(ham_data: HamChol, trial_data: GhfTrial) -> GhfCholMeasCtx:
    cH = trial_data.mo_coeff.conj().T  # (ne, 2n)
    norb = trial_data.norb

    if ham_data.basis == "restricted":
        z = jnp.zeros_like(ham_data.h1)
        h1_so = jnp.block([[ham_data.h1, z], [z, ham_data.h1]])  # (2n,2n)
        rot_h1 = cH @ h1_so  # (ne,2n)

        chol_sp = ham_data.chol.reshape(ham_data.chol.shape[0], norb, norb)

        def _rot_one(x):
            left = cH[:, :norb] @ x
            right = cH[:, norb:] @ x
            return jnp.concatenate([left, right], axis=1)

        rot_chol = jax.vmap(_rot_one, in_axes=0)(chol_sp)  # (nchol,ne,2n)

    else:
        rot_h1 = cH @ ham_data.h1  # (ne,ns)
        rot_chol = jax.vmap(lambda x: cH @ x, in_axes=0)(ham_data.chol)  # (nchol,ne,ns)

    rot_chol_flat = rot_chol.reshape(rot_chol.shape[0], -1)
    return GhfCholMeasCtx(rot_h1=rot_h1, rot_chol=rot_chol, rot_chol_flat=rot_chol_flat)


def force_bias_kernel_from_green(g_half: jax.Array, meas_ctx: GhfCholMeasCtx) -> jax.Array:
    return jnp.einsum("gij,ij->g", meas_ctx.rot_chol, g_half, optimize="optimal")


def energy_kernel_from_green(
    g_half: jax.Array, ham_data: HamChol, meas_ctx: GhfCholMeasCtx
) -> jax.Array:
    ene0 = ham_data.h0
    ene1 = jnp.sum(g_half * meas_ctx.rot_h1)
    f = jnp.einsum("gij,jk->gik", meas_ctx.rot_chol, g_half.T, optimize="optimal")  # (nchol,ne,ne)
    coul = jnp.trace(f, axis1=1, axis2=2)  # (nchol,)
    exc = jnp.sum(f * jnp.swapaxes(f, 1, 2))
    ene2 = 0.5 * (jnp.sum(coul * coul) - exc)

    return ene0 + ene1 + ene2


def force_bias_kernel_rw_rh(
    walker: jax.Array, ham_data: Any, meas_ctx: GhfCholMeasCtx, trial_data: GhfTrial
) -> jax.Array:
    g = _green_half_r(walker, trial_data)
    return force_bias_kernel_from_green(g, meas_ctx)


force_bias_kernel_rw_gh = force_bias_kernel_rw_rh


def force_bias_kernel_uw_rh(
    walker: tuple[jax.Array, jax.Array],
    ham_data: Any,
    meas_ctx: GhfCholMeasCtx,
    trial_data: GhfTrial,
) -> jax.Array:
    wu, wd = walker
    g = _green_half_u(wu, wd, trial_data)
    return force_bias_kernel_from_green(g, meas_ctx)


force_bias_kernel_uw_gh = force_bias_kernel_uw_rh


def force_bias_kernel_gw_rh(
    walker: jax.Array, ham_data: Any, meas_ctx: GhfCholMeasCtx, trial_data: GhfTrial
) -> jax.Array:
    g = _green_half_g(walker, trial_data)
    return force_bias_kernel_from_green(g, meas_ctx)


force_bias_kernel_gw_gh = force_bias_kernel_gw_rh


def energy_kernel_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: GhfCholMeasCtx,
    trial_data: GhfTrial,
) -> jax.Array:
    g = _green_half_r(walker, trial_data)
    return energy_kernel_from_green(g, ham_data, meas_ctx)


energy_kernel_rw_gh = energy_kernel_rw_rh


def energy_kernel_uw_rh(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamChol,
    meas_ctx: GhfCholMeasCtx,
    trial_data: GhfTrial,
) -> jax.Array:
    wu, wd = walker
    g = _green_half_u(wu, wd, trial_data)
    return energy_kernel_from_green(g, ham_data, meas_ctx)


energy_kernel_uw_gh = energy_kernel_uw_rh


def energy_kernel_gw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: GhfCholMeasCtx,
    trial_data: GhfTrial,
) -> jax.Array:
    g = _green_half_g(walker, trial_data)
    return energy_kernel_from_green(g, ham_data, meas_ctx)


energy_kernel_gw_gh = energy_kernel_gw_rh


def make_ghf_meas_ops_chol(sys: System) -> MeasOps:
    """
    GHF measurement ops for Cholesky Hamiltonians
    """
    wk = sys.walker_kind.lower()

    if wk == "restricted":
        overlap_fn = overlap_r
        build_meas_ctx_fn = build_meas_ctx_chol
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
        build_meas_ctx_fn = build_meas_ctx_chol
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
        build_meas_ctx_fn = build_meas_ctx_chol
        kernels = {
            k_force_bias: force_bias_kernel_gw_rh,
            k_energy: energy_kernel_gw_rh,
        }
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


# ---------------------
# hubbard
# ---------------------


def _energy_from_full_green(G: jax.Array, ham_data: HamHubbard, norb: int) -> jax.Array:
    h1 = ham_data.h1
    u = ham_data.u

    e1 = jnp.sum(G[:norb, :norb] * h1) + jnp.sum(G[norb:, norb:] * h1)

    g_uu = jnp.diagonal(G[:norb, :norb])
    g_dd = jnp.diagonal(G[norb:, norb:])
    g_ud = jnp.diagonal(G[:norb, norb:])
    g_du = jnp.diagonal(G[norb:, :norb])

    e2 = u * (jnp.sum(g_uu * g_dd) - jnp.sum(g_ud * g_du))
    return e1 + e2


def energy_kernel_hubbard_u(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamHubbard,
    meas_ctx: Any,
    trial_data: GhfTrial,
) -> jax.Array:
    g = calc_green_u(walker, trial_data)
    norb = trial_data.norb
    return _energy_from_full_green(g, ham_data, norb)


def energy_kernel_hubbard_g(
    walker: jax.Array,
    ham_data: HamHubbard,
    meas_ctx: Any,
    trial_data: GhfTrial,
) -> jax.Array:
    g = calc_green_g(walker, trial_data)
    norb = trial_data.norb
    return _energy_from_full_green(g, ham_data, norb)


def make_ghf_meas_ops_hubbard(sys: System) -> MeasOps:
    """
    GHF measurement ops for hubbard hamiltonian
    """
    wk = sys.walker_kind.lower()

    if wk == "unrestricted":
        overlap_fn = overlap_u
        kernels = {
            k_energy: energy_kernel_hubbard_u,
        }
        observables = {
            o_rdm1: rdm1_kernel_uw,
            o_density_corr: density_corr_kernel_uw,
        }
    elif wk == "generalized":
        overlap_fn = overlap_g
        kernels = {
            k_energy: energy_kernel_hubbard_g,
        }
        observables = {
            o_rdm1: rdm1_kernel_gw,
            o_density_corr: density_corr_kernel_gw,
        }
    else:
        raise ValueError(
            f"hubbard GHF meas only implemented for unrestricted/generalized, got walker_kind={sys.walker_kind}"
        )
    return MeasOps(
        overlap=overlap_fn,
        kernels=kernels,
        observables=observables,
    )
