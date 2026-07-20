from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .. import walkers as wk
from ..core.ops import MeasOps, TrialOps
from ..core.system import System
from ..ham.chol import HamChol
from .afqmc import init_prop_state
from .chol_afqmc_ops import TrotterOps, make_trotter_ops
from .chol_afqmc_ops_fp import FpCholAfqmcCtx, _build_prop_ctx_fp
from .types import PropOps, PropState, QmcParamsBase, QmcParamsFp


def afqmc_step_fp(
    state: PropState,
    sys: System,
    *,
    params: QmcParamsBase,
    ham_data: HamChol,
    trial_data: Any,
    meas_ops: MeasOps,
    trotter_ops: TrotterOps,
    prop_ctx: FpCholAfqmcCtx,
    meas_ctx: Any,
) -> PropState:
    key, subkey = jax.random.split(state.rng_key)
    nw = wk.n_walkers(state.walkers)
    fields = jax.random.normal(subkey, (nw, prop_ctx.chol_flat.shape[0]))
    wk_kind = sys.walker_kind.lower()

    shift_term = jnp.einsum("wg,g->w", fields, prop_ctx.mf_shifts)
    constants = jnp.exp(-prop_ctx.sqrt_dt * shift_term + prop_ctx.dt * prop_ctx.h0_prop)

    walkers_new = wk.vmap_chunked(
        trotter_ops.apply_trotter, n_chunks=params.n_chunks, in_axes=(0, 0, None, None)
    )(state.walkers, fields, prop_ctx, 10)

    walkers_new = wk.multiply_constants(walkers_new, constants, wk_kind)
    q, norms = wk.orthogonalize(walkers_new, wk_kind)
    weights_new = state.weights * norms.real
    key, subkey = jax.random.split(key)
    zeta = jax.random.uniform(subkey)
    walker_sr, weight_sr = wk.stochastic_reconfiguration(q, weights_new, zeta, wk_kind)

    return PropState(
        walkers=walker_sr,
        weights=weight_sr,
        overlaps=state.overlaps,
        rng_key=key,
        pop_control_ene_shift=state.pop_control_ene_shift,
        e_estimate=state.e_estimate,
        node_encounters=state.node_encounters,
    )


def make_prop_ops_fp(
    ham_basis: str, walker_kind: str, sys: System, mixed_precision=False
) -> PropOps:
    trotter_ops = make_trotter_ops(ham_basis, walker_kind, mixed_precision=mixed_precision)

    def step_fp(
        state: PropState,
        *,
        params: QmcParamsBase,
        ham_data: Any,
        trial_data: Any,
        trial_ops: TrialOps,
        meas_ops: MeasOps,
        meas_ctx: Any,
        prop_ctx: Any,
    ) -> PropState:
        return afqmc_step_fp(
            state,
            sys,
            params=params,
            ham_data=ham_data,
            trial_data=trial_data,
            meas_ops=meas_ops,
            meas_ctx=meas_ctx,
            prop_ctx=prop_ctx,
            trotter_ops=trotter_ops,
        )

    def build_prop_ctx_fp(ham_data: Any, rdm1: jax.Array, params: QmcParamsBase) -> FpCholAfqmcCtx:
        assert isinstance(params, QmcParamsFp)
        assert params.ene0 is not None, "ene0 must be set for FP propagation"
        return _build_prop_ctx_fp(
            ham_data,
            rdm1,
            params.dt,
            params.ene0,
            chol_flat_precision=jnp.float32 if mixed_precision else jnp.float64,
        )

    return PropOps(init_prop_state=init_prop_state, build_prop_ctx=build_prop_ctx_fp, step=step_fp)
