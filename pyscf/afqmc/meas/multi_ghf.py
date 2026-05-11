from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from ..core.ops import MeasOps, k_energy, o_density_corr, o_rdm1
from ..core.system import System
from ..ham.hubbard import HamHubbard
from ..trial.multi_ghf import (
    MultiGhfTrial,
    calc_green_g,
    calc_green_u,
    overlap_g,
    overlap_u,
)
from .ghf import _energy_from_full_green

# ---------------------
# hubbard
# ---------------------


def energy_kernel_hubbard_u(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamHubbard,
    meas_ctx: Any,
    trial_data: MultiGhfTrial,
) -> jax.Array:
    """
    Multi-det local energy
    """
    greens = calc_green_u(walker, trial_data)
    G_states = greens["G"]  # (nd,2n,2n)
    w_states = greens["w"]  # (nd,)

    norb = trial_data.norb
    E_k = jax.vmap(lambda G: _energy_from_full_green(G, ham_data, norb), in_axes=0)(G_states)

    num = jnp.sum(w_states * E_k)
    den = jnp.sum(w_states)

    E = jnp.where(jnp.abs(den) < 1.0e-16, 0.0 + 0.0j, num / den)
    return jnp.real(E)


def energy_kernel_hubbard_g(
    walker: jax.Array,
    ham_data: HamHubbard,
    meas_ctx: Any,
    trial_data: MultiGhfTrial,
) -> jax.Array:
    greens = calc_green_g(walker, trial_data)
    G_states = greens["G"]
    w_states = greens["w"]

    norb = trial_data.norb
    E_k = jax.vmap(lambda G: _energy_from_full_green(G, ham_data, norb), in_axes=0)(G_states)

    num = jnp.sum(w_states * E_k)
    den = jnp.sum(w_states)

    E = jnp.where(jnp.abs(den) < 1.0e-16, 0.0 + 0.0j, num / den)
    return jnp.real(E)


def _rdm1_from_weighted_greens(g_states: jax.Array, w_states: jax.Array, norb: int) -> jax.Array:
    den = jnp.sum(w_states)
    g_mix = jnp.where(
        jnp.abs(den) < 1.0e-16,
        jnp.zeros_like(g_states[0]),
        jnp.sum(w_states[:, None, None] * g_states, axis=0) / den,
    )
    return jnp.stack([g_mix[:norb, :norb], g_mix[norb:, norb:]], axis=0)


def rdm1_kernel_uw(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamHubbard,
    meas_ctx: Any,
    trial_data: MultiGhfTrial,
) -> jax.Array:
    greens = calc_green_u(walker, trial_data)
    return _rdm1_from_weighted_greens(greens["G"], greens["w"], trial_data.norb)


def rdm1_kernel_gw(
    walker: jax.Array,
    ham_data: HamHubbard,
    meas_ctx: Any,
    trial_data: MultiGhfTrial,
) -> jax.Array:
    greens = calc_green_g(walker, trial_data)
    return _rdm1_from_weighted_greens(greens["G"], greens["w"], trial_data.norb)


def _density_corr_from_weighted_greens(
    g_states: jax.Array, w_states: jax.Array, norb: int
) -> jax.Array:
    den = jnp.sum(w_states)
    g_mix = jnp.where(
        jnp.abs(den) < 1.0e-16,
        jnp.zeros_like(g_states[0]),
        jnp.sum(w_states[:, None, None] * g_states, axis=0) / den,
    )
    guu = g_mix[:norb, :norb]
    gdd = g_mix[norb:, norb:]
    gud = g_mix[:norb, norb:]
    gdu = g_mix[norb:, :norb]

    nu = jnp.diagonal(guu)
    nd = jnp.diagonal(gdd)

    uu = nu[:, None] * nu[None, :] - guu * guu.T + jnp.diag(nu)
    dd = nd[:, None] * nd[None, :] - gdd * gdd.T + jnp.diag(nd)
    ud = nu[:, None] * nd[None, :] - gud * gdu.T

    return jnp.stack([uu, ud, dd], axis=0)


def density_corr_kernel_uw(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamHubbard,
    meas_ctx: Any,
    trial_data: MultiGhfTrial,
) -> jax.Array:
    greens = calc_green_u(walker, trial_data)
    return _density_corr_from_weighted_greens(greens["G"], greens["w"], trial_data.norb)


def density_corr_kernel_gw(
    walker: jax.Array,
    ham_data: HamHubbard,
    meas_ctx: Any,
    trial_data: MultiGhfTrial,
) -> jax.Array:
    greens = calc_green_g(walker, trial_data)
    return _density_corr_from_weighted_greens(greens["G"], greens["w"], trial_data.norb)


def make_multi_ghf_meas_ops_hubbard(sys: System) -> MeasOps:
    # meant to be used with CPMC, which requires real overlaps
    wk = sys.walker_kind.lower()

    if wk == "unrestricted":

        def real_overlap_u(
            walker: tuple[jax.Array, jax.Array], trial_data: MultiGhfTrial
        ) -> jax.Array:
            ov = overlap_u(walker, trial_data)
            return jnp.real(ov)

        return MeasOps(
            overlap=real_overlap_u,
            kernels={k_energy: energy_kernel_hubbard_u},
            observables={
                o_rdm1: rdm1_kernel_uw,
                o_density_corr: density_corr_kernel_uw,
            },
        )

    if wk == "generalized":

        def real_overlap_g(walker: jax.Array, trial_data: MultiGhfTrial) -> jax.Array:
            ov = overlap_g(walker, trial_data)
            return jnp.real(ov)

        return MeasOps(
            overlap=real_overlap_g,
            kernels={k_energy: energy_kernel_hubbard_g},
            observables={
                o_rdm1: rdm1_kernel_gw,
                o_density_corr: density_corr_kernel_gw,
            },
        )

    raise ValueError(
        f"multi-GHF Hubbard meas only implemented for unrestricted/generalized; got walker_kind={sys.walker_kind}"
    )
