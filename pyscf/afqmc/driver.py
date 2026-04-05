from __future__ import annotations

import dataclasses
import time
from functools import partial
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from .core.ops import MeasOps, TrialOps
from .core.system import System
from .prop.blocks import BlockFn, MixedBlockFn
from .prop.types import PropOps, PropState, QmcParamsBase, QmcParams, QmcParamsFp
from .stat_utils import (
    blocking_analysis_ratio,
    jackknife_ratios,
    rebin_observable,
    reject_outliers,
    pt2ccsd_blocking,
    clean_pt2ccsd,
)
from .walkers import stochastic_reconfiguration
from .meas.pt2ccsd import get_init_pt2trial_energy

print = partial(print, flush=True)


class QmcResult(NamedTuple):
    mean_energy: jax.Array
    stderr_energy: jax.Array
    block_energies: jax.Array
    block_weights: jax.Array
    block_observables: dict[str, jax.Array]
    observable_means: dict[str, jax.Array]
    observable_stderrs: dict[str, jax.Array]


class MixedQmcResult(NamedTuple):
    # currently only support pt2CCSD
    guide_mean_energy: jax.Array
    guide_stderr_energy: jax.Array
    guide_block_energies: jax.Array
    guide_block_weights: jax.Array
    trial_mean_energy: jax.Array
    trial_stderr_energy: jax.Array
    trial_block_weights: jax.Array
    trial_block_t2s: jax.Array
    trial_block_e0s: jax.Array
    trial_block_e1s: jax.Array


def _weighted_block_mean(values: jax.Array, weights: jax.Array) -> jax.Array:
    w_sum = jnp.sum(weights)
    w_shape = (weights.shape[0],) + (1,) * max(values.ndim - 1, 0)
    num = jnp.sum(weights.reshape(w_shape) * values, axis=0)
    zero = jnp.zeros_like(num)
    return jnp.where(w_sum == 0, zero, num / w_sum)


def make_run_blocks(
    *,
    block_fn: BlockFn,
    sys: System,
    params: QmcParamsBase,
    trial_ops: TrialOps,
    meas_ops: MeasOps,
    prop_ops: PropOps,
    observable_names: tuple[str, ...] = (),
) -> Callable:
    """
    Build a jitted run_blocks.
    We keep ham_data, trial_data, meas_ctx, prop_ctx as arguments to
    improve compilation, as these objects can be large.
    """

    @partial(jax.jit, static_argnames=("n_blocks",))
    def run_blocks(
        state0,
        *,
        ham_data,
        trial_data,
        meas_ctx,
        prop_ctx,
        n_blocks: int,
    ):
        def one_block(state, _):
            state, obs = block_fn(
                state,
                sys=sys,
                params=params,
                ham_data=ham_data,
                trial_data=trial_data,
                trial_ops=trial_ops,
                meas_ops=meas_ops,
                meas_ctx=meas_ctx,
                prop_ops=prop_ops,
                prop_ctx=prop_ctx,
                observable_names=observable_names,
            )
            obs_tuple = tuple(obs.observables[name] for name in observable_names)
            return state, (obs.scalars, obs_tuple)

        stateN, (scalars, obs) = lax.scan(one_block, state0, xs=None, length=n_blocks)
        return stateN, scalars, obs

    return run_blocks


def make_run_mixed_blocks(
    *,
    mixed_block_fn: MixedBlockFn,
    sys: System,
    params: QmcParams,
    guide_ops: TrialOps,
    guide_meas_ops: MeasOps,
    guide_prop_ops: PropOps,
    trial_meas_ops: MeasOps,
    observable_names: tuple[str, ...] = (),
) -> Callable:
    """
    Build a jitted run_blocks for mixed sampling (Trial =! Guide).
    We keep ham_data, trial_data, meas_ctx, prop_ctx as arguments to
    improve compilation, as these objects can be large.
    """

    @partial(jax.jit, static_argnames=("n_blocks",))
    def run_mixed_blocks(
        state0,
        *,
        ham_data,
        guide_data,
        guide_meas_ctx,
        guide_prop_ctx,
        trial_data,
        trial_meas_ctx,
        n_blocks: int,
    ):
        def one_block(state, _):
            state, obs = mixed_block_fn(
                state,
                sys=sys,
                params=params,
                ham_data=ham_data,
                guide_data=guide_data,
                guide_ops=guide_ops,
                guide_meas_ops=guide_meas_ops,
                guide_meas_ctx=guide_meas_ctx,
                guide_prop_ops=guide_prop_ops,
                guide_prop_ctx=guide_prop_ctx,
                trial_data=trial_data,
                trial_meas_ops=trial_meas_ops,
                trial_meas_ctx=trial_meas_ctx,
                observable_names=observable_names,
            )
            obs_tuple = tuple(obs.observables[name] for name in observable_names)
            return state, (obs.scalars, obs_tuple)

        stateN, (scalars, obs) = lax.scan(one_block, state0, xs=None, length=n_blocks)
        return stateN, scalars, obs

    return run_mixed_blocks


def run_qmc(
    *,
    sys: System,
    params: QmcParams,
    ham_data: Any,
    trial_data: Any,
    meas_ops: MeasOps,
    trial_ops: TrialOps,
    prop_ops: PropOps,
    block_fn: BlockFn,
    state: PropState | None = None,
    meas_ctx: Any | None = None,
    prop_ctx: Any | None = None,
    target_error: float | None = None,
    mesh: Mesh | None = None,
    observable_names: tuple[str, ...] = (),
) -> QmcResult:
    """
    equilibration blocks then sampling blocks.

    Returns:
      QmcResult with energy statistics plus block-level observable estimates.
    """
    for name in observable_names:
        meas_ops.require_observable(name)

    # build ctx
    if prop_ctx is None:
        prop_ctx = prop_ops.build_prop_ctx(ham_data, trial_ops.get_rdm1(trial_data), params)
    if meas_ctx is None:
        meas_ctx = meas_ops.build_meas_ctx(ham_data, trial_data)
    if state is None:
        state = prop_ops.init_prop_state(
            sys=sys,
            ham_data=ham_data,
            trial_ops=trial_ops,
            trial_data=trial_data,
            meas_ops=meas_ops,
            params=params,
            mesh=mesh,
        )

    if mesh is None or mesh.size == 1:
        block_fn_sr = block_fn
    else:
        data_sh = NamedSharding(mesh, P("data"))
        sr_sharded = partial(stochastic_reconfiguration, data_sharding=data_sh)
        block_fn_sr = partial(block_fn, sr_fn=sr_sharded)

    run_blocks = make_run_blocks(
        block_fn=block_fn_sr,
        sys=sys,
        params=params,
        trial_ops=trial_ops,
        meas_ops=meas_ops,
        prop_ops=prop_ops,
        observable_names=observable_names,
    )

    t0 = time.perf_counter()
    t_mark = t0

    print_every = params.n_eql_blocks // 5 if params.n_eql_blocks >= 5 else 0
    block_e_eq = []
    block_w_eq = []
    block_obs_eq = {name: [] for name in observable_names}
    block_e_eq.append(state.e_estimate)
    block_w_eq.append(jnp.sum(state.weights))
    print("\nEquilibration:\n")
    if print_every:
        print(
            f"{'':4s}"
            f"{'block':>9s}  "
            f"{'E_blk':>14s}  "
            f"{'W':>12s}   "
            f"{'nodes':>10s}  "
            f"{'t[s]':>8s}"
        )
    print(
        f"[eql {0:4d}/{params.n_eql_blocks}]  "
        f"{float(state.e_estimate):14.10f}  "
        f"{float(jnp.sum(state.weights)):12.6e}  "
        f"{int(state.node_encounters):10d}  "
        f"{0.0:8.1f}"
    )
    chunk = print_every if print_every > 0 else 1
    for start in range(0, params.n_eql_blocks, chunk):
        n = min(chunk, params.n_eql_blocks - start)
        state, scalars_chunk, obs_chunk = run_blocks(
            state,
            ham_data=ham_data,
            trial_data=trial_data,
            meas_ctx=meas_ctx,
            prop_ctx=prop_ctx,
            n_blocks=n,
        )
        block_e_eq.extend(scalars_chunk["energy"].tolist())
        block_w_eq.extend(scalars_chunk["weight"].tolist())
        for i, name in enumerate(observable_names):
            block_obs_eq[name].append(obs_chunk[i])
        e_chunk = scalars_chunk["energy"]
        w_chunk = scalars_chunk["weight"]
        w_chunk_avg = jnp.mean(w_chunk)
        e_chunk_avg = jnp.mean(e_chunk * w_chunk) / w_chunk_avg
        elapsed = time.perf_counter() - t0
        print(
            f"[eql {start + n:4d}/{params.n_eql_blocks}]  "
            f"{float(e_chunk_avg):14.10f}  "
            f"{float(w_chunk_avg):12.6e}  "
            f"{int(state.node_encounters):10d}  "
            f"{elapsed:8.1f}"
        )
    block_e_eq = jnp.asarray(block_e_eq)
    block_w_eq = jnp.asarray(block_w_eq)
    block_obs_eq = {
        name: (jnp.concatenate(block_obs_eq[name], axis=0) if len(block_obs_eq[name]) > 0 else None)
        for name in observable_names
    }

    # sampling
    print("\nSampling:\n")
    if target_error is None:
        target_error = 0.0
    print_every = params.n_blocks // 10 if params.n_blocks >= 10 else 0
    block_e_s = []
    block_w_s = []
    block_obs_s = {name: [] for name in observable_names}
    if print_every:
        print(
            f"{'':4s}{'block':>9s}  {'E_avg':>14s}  {'E_err':>10s}  {'E_block':>14s}  "
            f"{'W':>12s}    {'nodes':>10s}  {'dt[s/bl]':>10s}  {'t[s]':>7s}"
        )

    chunk = print_every if print_every > 0 else 1
    for start in range(0, params.n_blocks, chunk):
        n = min(chunk, params.n_blocks - start)
        state, scalars_chunk, obs_chunk = run_blocks(
            state,
            ham_data=ham_data,
            trial_data=trial_data,
            meas_ctx=meas_ctx,
            prop_ctx=prop_ctx,
            n_blocks=n,
        )
        e_chunk = scalars_chunk["energy"]
        w_chunk = scalars_chunk["weight"]
        block_e_s.extend(e_chunk.tolist())
        block_w_s.extend(w_chunk.tolist())
        for i, name in enumerate(observable_names):
            block_obs_s[name].append(obs_chunk[i])
        w_chunk_avg = jnp.mean(w_chunk)
        e_chunk_avg = jnp.mean(e_chunk * w_chunk) / w_chunk_avg
        elapsed = time.perf_counter() - t0
        dt_per_block = (time.perf_counter() - t_mark) / float(n)
        t_mark = time.perf_counter()
        stats = blocking_analysis_ratio(
            jnp.asarray(block_e_s), jnp.asarray(block_w_s), print_q=False
        )
        mu = stats["mu"]
        se = stats["se_star"]
        nodes = int(state.node_encounters)
        print(
            f"[blk {start + n:4d}/{params.n_blocks}]  "
            f"{mu:14.10f}  "
            f"{(f'{se:10.3e}' if se is not None else ' ' * 10)}  "
            f"{float(e_chunk_avg):16.10f}  "
            f"{float(w_chunk_avg):12.6e}  "
            f"{nodes:10d}  "
            f"{dt_per_block:9.3f}  "
            f"{elapsed:8.1f}"
        )
        if se is not None and se <= target_error and target_error > 0.0:
            print(f"\nTarget error {target_error:.3e} reached at block {start + n}.")
            break
    block_e_s = jnp.asarray(block_e_s)
    block_w_s = jnp.asarray(block_w_s)
    block_obs_s = {
        name: (jnp.concatenate(block_obs_s[name], axis=0) if len(block_obs_s[name]) > 0 else None)
        for name in observable_names
    }

    data_clean, keep_mask = reject_outliers(jnp.column_stack((block_e_s, block_w_s)), obs=0)
    print(f"\nRejected {block_e_s.shape[0] - data_clean.shape[0]} outlier blocks.")
    block_e_s = jnp.asarray(data_clean[:, 0])
    block_w_s = jnp.asarray(data_clean[:, 1])
    keep_mask = jnp.asarray(keep_mask)
    block_obs_s = {
        name: (arr[keep_mask] if arr is not None else None) for name, arr in block_obs_s.items()
    }
    print("\nFinal blocking analysis:")
    stats = blocking_analysis_ratio(block_e_s, block_w_s, print_q=True)
    mean, err = stats["mu"], stats["se_star"]

    block_e_all = jnp.concatenate([block_e_eq, block_e_s])
    block_w_all = jnp.concatenate([block_w_eq, block_w_s])
    block_obs_all: dict[str, jax.Array] = {}
    for name in observable_names:
        arr_eq = block_obs_eq[name]
        arr_s = block_obs_s[name]
        if arr_eq is None and arr_s is None:
            block_obs_all[name] = jnp.zeros((0,))
            continue
        if arr_eq is None:
            arr_eq = arr_s[:0]
        if arr_s is None:
            arr_s = arr_eq[:0]
        block_obs_all[name] = jnp.concatenate([arr_eq, arr_s], axis=0)

    obs_means: dict[str, jax.Array] = {}
    obs_stderrs: dict[str, jax.Array] = {}
    b_star = stats.get("B_star")
    for name in observable_names:
        arr = block_obs_s[name]
        if arr is None:
            obs_means[name] = jnp.zeros((0,))
            obs_stderrs[name] = jnp.zeros((0,))
            continue
        obs_means[name] = _weighted_block_mean(arr, block_w_s)
        if b_star is not None and b_star >= 1:
            import numpy as np

            num, denom = rebin_observable(np.asarray(arr), np.asarray(block_w_s), b_star)
            if num.shape[0] >= 2:
                _, se = jackknife_ratios(num, denom)
                obs_stderrs[name] = jnp.asarray(se)
            else:
                obs_stderrs[name] = jnp.full(arr.shape[1:], jnp.nan)
        else:
            obs_stderrs[name] = jnp.full(arr.shape[1:], jnp.nan)

    return QmcResult(
        mean_energy=mean,
        stderr_energy=err,
        block_energies=block_e_all,
        block_weights=block_w_all,
        block_observables=block_obs_all,
        observable_means=obs_means,
        observable_stderrs=obs_stderrs,
    )


def run_mixed_qmc(
    *,
    sys: System,
    params: QmcParams,
    ham_data: Any,
    guide_data: Any,
    guide_ops: TrialOps,
    guide_prop_ops: PropOps,
    guide_meas_ops: MeasOps,
    trial_data: Any,
    trial_meas_ops: MeasOps,
    mix_block_fn: MixedBlockFn,
    state: PropState | None = None,
    guide_meas_ctx: Any | None = None,
    trial_meas_ctx: Any | None = None,
    target_error: float | None = None,
    mesh: Mesh | None = None,
    observable_names: tuple[str, ...] = (),
) -> MixedQmcResult:
    """
    equilibration blocks then sampling blocks.
    Guide != Trial
    Currently only support pt2CCSD trial without observables

    Returns:
      MixedQmcResult with energy statistics plus block-level observable estimates.
    """

    for name in observable_names:
        guide_meas_ops.require_observable(name)

    # build ctx
    guide_prop_ctx = guide_prop_ops.build_prop_ctx(ham_data, guide_ops.get_rdm1(guide_data), params)
    if guide_meas_ctx is None:
        guide_meas_ctx = guide_meas_ops.build_meas_ctx(ham_data, guide_data)

    if trial_meas_ctx is None:
        trial_meas_ctx = trial_meas_ops.build_meas_ctx(ham_data, trial_data)

    if state is None:
        state = guide_prop_ops.init_prop_state(
            sys=sys,
            ham_data=ham_data,
            trial_ops=guide_ops,
            trial_data=guide_data,
            meas_ops=guide_meas_ops,
            params=params,
            mesh=mesh,
        )

    assert trial_meas_ctx is not None
    trial_energy0, trial_weights0 = get_init_pt2trial_energy(
        init_state=state,
        ham_data=ham_data,
        trial_data=trial_data,
        trial_meas_ops=trial_meas_ops,
        trial_meas_ctx=trial_meas_ctx,
        params=params,
    )

    if mesh is None or mesh.size == 1:
        mix_block_fn_sr = mix_block_fn
    else:
        data_sh = NamedSharding(mesh, P("data"))
        sr_sharded = partial(stochastic_reconfiguration, data_sharding=data_sh)
        mix_block_fn_sr = partial(mix_block_fn, sr_fn=sr_sharded)

    run_blocks = make_run_mixed_blocks(
        mixed_block_fn=mix_block_fn_sr,
        sys=sys,
        params=params,
        guide_ops=guide_ops,
        guide_meas_ops=guide_meas_ops,
        guide_prop_ops=guide_prop_ops,
        trial_meas_ops=trial_meas_ops,
        observable_names=observable_names,
    )

    t0 = time.perf_counter()
    t_mark = t0

    block_time = params.dt * params.n_prop_steps
    print_every = params.n_eql_blocks // 5 if params.n_eql_blocks >= 5 else 0
    guide_block_e_eq = []
    guide_block_w_eq = []
    trial_block_w_eq = []
    trial_block_t2_eq = []
    trial_block_e0_eq = []
    trial_block_e1_eq = []
    # guide_block_obs_eq = {name: [] for name in observable_names}
    guide_block_e_eq.append(state.e_estimate)
    guide_block_w_eq.append(jnp.sum(state.weights))
    # trial_block_e_eq.append(trial_energy0)
    # trial_block_w_eq.append(jnp.sum(trial_weights0))
    print("\nEquilibration:\n")
    if print_every:
        print(
            f"{'':4s}"
            f"{'block':>9s}  "
            f"{'1/Tmp':>6s}  "
            f"{'Guide_E_blk':>14s}  "
            f"{'Guide_W_blk':>12s}   "
            f"{'Trial_E_blk':>14s}  "
            f"{'Trial_W_blk':>12s}   "
            f"{'nodes':>10s}  "
            f"{'t[s]':>8s}"
        )
    print(
        f"[eql {0:4d}/{params.n_eql_blocks}]  "
        f"{0 * block_time:6.2f}  "
        f"{float(guide_block_e_eq[0].real):14.10f}  "
        f"{float(guide_block_w_eq[0].real):12.6e}  "
        f"{float(trial_energy0.real):14.10f}  "
        f"{float(trial_weights0.real):12.6e}  "
        f"{int(state.node_encounters):10d}  "
        f"{0.0:8.1f}"
    )
    chunk = print_every if print_every > 0 else 1
    for start in range(0, params.n_eql_blocks, chunk):
        n = min(chunk, params.n_eql_blocks - start)
        state, scalars_chunk, obs_chunk = run_blocks(
            state,
            ham_data=ham_data,
            guide_data=guide_data,
            guide_meas_ctx=guide_meas_ctx,
            guide_prop_ctx=guide_prop_ctx,
            trial_data=trial_data,
            trial_meas_ctx=trial_meas_ctx,
            n_blocks=n,
        )
        # guide
        guide_block_e_eq.extend(scalars_chunk["guide_energy"].tolist())
        guide_block_w_eq.extend(scalars_chunk["guide_weight"].tolist())
        guide_e_chunk = scalars_chunk["guide_energy"]
        guide_w_chunk = scalars_chunk["guide_weight"]
        guide_w_chunk_avg = jnp.mean(guide_w_chunk)
        guide_e_chunk_avg = jnp.mean(guide_e_chunk * guide_w_chunk) / guide_w_chunk_avg
        # trial
        trial_block_w_eq.extend(scalars_chunk["trial_weight"].tolist())
        trial_block_t2_eq.extend(scalars_chunk["trial_t2"].tolist())
        trial_block_e0_eq.extend(scalars_chunk["trial_e0"].tolist())
        trial_block_e1_eq.extend(scalars_chunk["trial_e1"].tolist())
        # for i, name in enumerate(observable_names):
        #     block_obs_eq[name].append(obs_chunk[i])
        trial_w_chunk = scalars_chunk["trial_weight"]
        trial_t2_chunk = scalars_chunk["trial_t2"]
        trial_e0_chunk = scalars_chunk["trial_e0"]
        trial_e1_chunk = scalars_chunk["trial_e1"]
        trial_w_chunk_avg = jnp.mean(trial_w_chunk)
        trial_t2_chunk_avg = jnp.mean(trial_w_chunk * trial_t2_chunk) / trial_w_chunk_avg
        trial_e0_chunk_avg = jnp.mean(trial_w_chunk * trial_e0_chunk) / trial_w_chunk_avg
        trial_e1_chunk_avg = jnp.mean(trial_w_chunk * trial_e1_chunk) / trial_w_chunk_avg
        pt2trial_energy_avg = (
            ham_data.h0
            + trial_e0_chunk_avg
            + trial_e1_chunk_avg
            - trial_t2_chunk_avg * trial_e0_chunk_avg
        )
        elapsed = time.perf_counter() - t0
        print(
            f"[eql {start + n:4d}/{params.n_eql_blocks}]  "
            f"{(start + n) * block_time:6.2f}  "
            f"{float(guide_e_chunk_avg):14.10f}  "
            f"{float(guide_w_chunk_avg):12.6e}  "
            f"{float(pt2trial_energy_avg.real):14.10f}  "
            f"{float(trial_w_chunk_avg.real):12.6e}  "
            f"{int(state.node_encounters):10d}  "
            f"{elapsed:8.1f}"
        )

    guide_block_w_eq = jnp.asarray(guide_block_w_eq)
    guide_block_e_eq = jnp.asarray(guide_block_e_eq)

    trial_block_w_eq = jnp.asarray(trial_block_w_eq)
    trial_block_t2_eq = jnp.asarray(trial_block_t2_eq)
    trial_block_e0_eq = jnp.asarray(trial_block_e0_eq)
    trial_block_e1_eq = jnp.asarray(trial_block_e1_eq)

    # block_obs_eq = {
    #     name: (jnp.concatenate(block_obs_eq[name], axis=0) if len(block_obs_eq[name]) > 0 else None)
    #     for name in observable_names
    # }

    # sampling
    print("\nSampling:\n")
    if target_error is None:
        target_error = 0.0
    print_every = params.n_blocks // 10 if params.n_blocks >= 10 else 0

    guide_block_w_sp = []
    guide_block_e_sp = []
    trial_block_w_sp = []
    trial_block_t2_sp = []
    trial_block_e0_sp = []
    trial_block_e1_sp = []

    # block_obs_sp = {name: [] for name in observable_names}
    if print_every:
        print(
            f"{'':4s}{'block':>9s}  {'Guide_E_avg':>14s}  {'Guide_E_err':>10s}  {'Guide_W':>12s}  "
            f"{'Trial_E_avg':>14s}  {'Trial_E_err':>10s}  {'nodes':>10s}  {'dt[s/bl]':>10s}  {'t[s]':>7s}"
        )

    chunk = print_every if print_every > 0 else 1
    for start in range(0, params.n_blocks, chunk):
        n = min(chunk, params.n_blocks - start)
        state, scalars_chunk, obs_chunk = run_blocks(
            state,
            ham_data=ham_data,
            guide_data=guide_data,
            guide_meas_ctx=guide_meas_ctx,
            guide_prop_ctx=guide_prop_ctx,
            trial_data=trial_data,
            trial_meas_ctx=trial_meas_ctx,
            n_blocks=n,
        )
        # guide
        # guide_w_chunk = scalars_chunk["guide_weight"]
        # guide_e_chunk = scalars_chunk["guide_energy"]
        guide_block_w_sp.extend(scalars_chunk["guide_weight"].tolist())
        guide_block_e_sp.extend(scalars_chunk["guide_energy"].tolist())
        # for i, name in enumerate(observable_names):
        #     block_obs_sp[name].append(obs_chunk[i])
        guide_w_avg = jnp.mean(jnp.asarray(guide_block_w_sp))
        # guide_e_avg = jnp.mean(jnp.asarray(guide_block_w_sp) * jnp.asarray(guide_block_e_sp)) / guide_w_avg
        elapsed = time.perf_counter() - t0
        dt_per_block = (time.perf_counter() - t_mark) / float(n)
        t_mark = time.perf_counter()
        stats = blocking_analysis_ratio(
            jnp.asarray(guide_block_e_sp), jnp.asarray(guide_block_w_sp), print_q=False
        )
        guide_mu = stats["mu"]
        guide_se = stats["se_star"]
        nodes = int(state.node_encounters)
        # trial
        trial_block_w_sp.extend(scalars_chunk["trial_weight"].tolist())
        trial_block_t2_sp.extend(scalars_chunk["trial_t2"].tolist())
        trial_block_e0_sp.extend(scalars_chunk["trial_e0"].tolist())
        trial_block_e1_sp.extend(scalars_chunk["trial_e1"].tolist())
        # trial_w_avg = jnp.mean(jnp.asarray(trial_block_w_sp))
        # trial_t2_avg = jnp.mean(jnp.asarray(trial_block_w_sp) * jnp.asarray(trial_block_t2_sp)) / trial_w_avg
        # trial_e0_avg = jnp.mean(jnp.asarray(trial_block_w_sp) * jnp.asarray(trial_block_e0_sp)) / trial_w_avg
        # trial_e1_avg = jnp.mean(jnp.asarray(trial_block_w_sp) * jnp.asarray(trial_block_e1_sp)) / trial_w_avg
        # trial_e_avg = ham_data.h0 + trial_e0_avg + trial_e1_avg - trial_t2_avg * trial_e0_avg
        trial_e_avg, trial_error = pt2ccsd_blocking(
            ham_data.h0,
            jnp.asarray(trial_block_w_sp),
            jnp.asarray(trial_block_t2_sp),
            jnp.asarray(trial_block_e0_sp),
            jnp.asarray(trial_block_e1_sp),
            printQ=False,
        )

        print(
            f"[blk {start + n:4d}/{params.n_blocks}]  "
            f"{guide_mu:14.10f}  "
            f"{(f'{guide_se:10.3e}' if guide_se is not None else ' ' * 10)}  "
            # f"{float(guide_e_avg):16.10f}  "
            f"{float(guide_w_avg):12.6e}  "
            f"{float(trial_e_avg.real):14.10f}  "
            f"{float(trial_error.real):10.3e}  "
            f"{nodes:10d}  "
            f"{dt_per_block:9.3f}  "
            f"{elapsed:8.1f}"
        )
        if guide_se is not None and guide_se <= target_error and target_error > 0.0:
            print(f"\nTarget error {target_error:.3e} reached at block {start + n}.")
            break

    # Guide
    guide_block_w_sp = jnp.asarray(guide_block_w_sp)
    guide_block_e_sp = jnp.asarray(guide_block_e_sp)
    guide_block_w_all = jnp.concatenate(
        [guide_block_w_eq, guide_block_w_sp]
    )  # return both eql and sampling
    guide_block_e_all = jnp.concatenate([guide_block_e_eq, guide_block_e_sp])
    guide_data_clean, guide_keep_mask = reject_outliers(
        jnp.column_stack((guide_block_e_sp, guide_block_w_sp)), obs=0
    )
    print(
        f"\nRejected {guide_block_e_sp.shape[0] - guide_data_clean.shape[0]} guiding outlier blocks."
    )
    guide_block_e_sp = jnp.asarray(guide_data_clean[:, 0])
    guide_block_w_sp = jnp.asarray(guide_data_clean[:, 1])
    guide_keep_mask = jnp.asarray(guide_keep_mask)

    # Trial
    trial_block_w_sp = jnp.asarray(trial_block_w_sp)
    trial_block_t2_sp = jnp.asarray(trial_block_t2_sp)
    trial_block_e0_sp = jnp.asarray(trial_block_e0_sp)
    trial_block_e1_sp = jnp.asarray(trial_block_e1_sp)
    trial_block_w_all = jnp.concatenate([trial_block_w_eq, trial_block_w_sp])
    trial_block_t2_all = jnp.concatenate([trial_block_t2_eq, trial_block_t2_sp])
    trial_block_e0_all = jnp.concatenate([trial_block_e0_eq, trial_block_e0_sp])
    trial_block_e1_all = jnp.concatenate([trial_block_e1_eq, trial_block_e1_sp])
    # esmitate of pt2ccsd energy sample, biased
    trial_block_e_sp = (
        ham_data.h0 + trial_block_e0_sp + trial_block_e1_sp - trial_block_t2_sp * trial_block_e0_sp
    )
    # trial_data_clean, trial_keep_mask = reject_outliers(
    #     jnp.column_stack((trial_block_e_sp.real,
    #                       trial_block_w_sp,
    #                       trial_block_t2_sp,
    #                       trial_block_e0_sp,
    #                       trial_block_e1_sp)),
    #                       obs=0)
    # trial_block_e_sp = trial_data_clean[:, 0]
    # trial_block_w_sp = trial_data_clean[:, 1]
    # trial_block_t2_sp = trial_data_clean[:, 2]
    # trial_block_e0_sp = trial_data_clean[:, 3]
    # trial_block_e1_sp = trial_data_clean[:, 4]
    print("Clean AFQMC/pt2CCSD Observation...")
    (
        trial_block_w_sp_clean,
        trial_block_t2_sp_clean,
        trial_block_e0_sp_clean,
        trial_block_e1_sp_clean,
    ) = clean_pt2ccsd(
        trial_block_e_sp.real,
        trial_block_w_sp,
        trial_block_t2_sp,
        trial_block_e0_sp,
        trial_block_e1_sp,
        zeta=20,
    )

    print(
        f"\nRejected {len(trial_block_w_sp) - len(trial_block_w_sp_clean)} AFQMC/pt2CCSD outlier blocks."
    )

    print("\nFinal blocking analysis:")
    guide_stats = blocking_analysis_ratio(guide_block_e_sp, guide_block_w_sp, print_q=True)
    guide_e_mean, guide_e_err = guide_stats["mu"], guide_stats["se_star"]

    trial_e_mean, trial_e_err = pt2ccsd_blocking(
        ham_data.h0,
        trial_block_w_sp_clean,
        trial_block_t2_sp_clean,
        trial_block_e0_sp_clean,
        trial_block_e1_sp_clean,
        printQ=True,
    )

    print(f"AFQMC/pt2CCSD energy = {trial_e_mean.real:.6f} +/- {trial_e_err.real:.6f} (1-sigma)")

    return MixedQmcResult(
        guide_mean_energy=guide_e_mean,
        guide_stderr_energy=guide_e_err,
        guide_block_energies=guide_block_e_all,
        guide_block_weights=guide_block_w_all,
        trial_mean_energy=trial_e_mean,
        trial_stderr_energy=trial_e_err,
        trial_block_weights=trial_block_w_all,
        trial_block_t2s=trial_block_t2_all,
        trial_block_e0s=trial_block_e0_all,
        trial_block_e1s=trial_block_e1_all,
        # block_observables=block_obs_all,
        # observable_means=obs_means,
        # observable_stderrs=obs_stderrs,
    )


def run_qmc_energy(
    *,
    sys: System,
    params: QmcParams,
    ham_data: Any,
    trial_data: Any,
    meas_ops: MeasOps,
    trial_ops: TrialOps,
    prop_ops: PropOps,
    block_fn: BlockFn,
    state: PropState | None = None,
    meas_ctx: Any | None = None,
    prop_ctx: Any | None = None,
    target_error: float | None = None,
    mesh: Mesh | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    out = run_qmc(
        sys=sys,
        params=params,
        ham_data=ham_data,
        trial_data=trial_data,
        meas_ops=meas_ops,
        trial_ops=trial_ops,
        prop_ops=prop_ops,
        block_fn=block_fn,
        state=state,
        meas_ctx=meas_ctx,
        prop_ctx=prop_ctx,
        target_error=target_error,
        mesh=mesh,
        observable_names=(),
    )
    return out.mean_energy, out.stderr_energy, out.block_energies, out.block_weights


def run_qmc_fp(
    *,
    sys: System,
    params: QmcParamsFp,
    ham_data: Any,
    trial_data: Any,
    meas_ops: MeasOps,
    trial_ops: TrialOps,
    prop_ops: PropOps,
    block_fn: BlockFn,
    state: PropState | None = None,
    meas_ctx: Any | None = None,
    prop_ctx: Any | None = None,
    target_error: float | None = None,
    mesh: Mesh | None = None,
) -> QmcResult:
    """
    Returns:
      (mean_energy, stderr, block_energies, block_weights)
    """
    # build ctx
    if prop_ctx is None:
        prop_ctx = prop_ops.build_prop_ctx(ham_data, trial_ops.get_rdm1(trial_data), params)
    if meas_ctx is None:
        meas_ctx = meas_ops.build_meas_ctx(ham_data, trial_data)
    if state is None:
        state = prop_ops.init_prop_state(
            sys=sys,
            ham_data=ham_data,
            trial_ops=trial_ops,
            trial_data=trial_data,
            meas_ops=meas_ops,
            params=params,
        )

    block_fn_sr = block_fn

    run_blocks = make_run_blocks(
        block_fn=block_fn_sr,
        sys=sys,
        params=params,
        trial_ops=trial_ops,
        meas_ops=meas_ops,
        prop_ops=prop_ops,
    )

    t0 = time.perf_counter()

    # sampling
    print("\nSampling:\n")
    # print_every = params.n_blocks // 10 if params.n_blocks >= 10 else 1
    print_every = 1
    block_e_all = jnp.zeros((params.n_traj, params.n_blocks + 1)) + 0.0j
    block_w_all = jnp.zeros((params.n_traj, params.n_blocks + 1)) + 0.0j
    total_sign = jnp.ones((params.n_traj, params.n_blocks + 1)) + 0.0j
    block_e_all = block_e_all.at[:, 0].set(jnp.array(state.e_estimate))
    block_w_all = block_w_all.at[:, 0].set(jnp.sum(state.weights))
    total_sign = total_sign.at[:, 0].set(
        jnp.sum(state.overlaps) / (jnp.sum(jnp.abs(state.overlaps)))
    )
    chunk = print_every
    for i in range(params.n_traj):
        print("Trajectory count", i + 1)
        print(f"{'tau':^12s}    " f"{'E_avg':^14s}  " f"{'E_err':^13s}  " f"{'sign':>6s}")
        if i > 0:
            params = dataclasses.replace(params, seed=params.seed + i)
            state = prop_ops.init_prop_state(
                sys=sys,
                ham_data=ham_data,
                trial_ops=trial_ops,
                trial_data=trial_data,
                meas_ops=meas_ops,
                params=params,
            )

        n = params.n_blocks
        state, scalars_chunk, _obs = run_blocks(
            state,
            ham_data=ham_data,
            trial_data=trial_data,
            meas_ctx=meas_ctx,
            prop_ctx=prop_ctx,
            n_blocks=n,
        )
        block_e_s = scalars_chunk["energy"]
        block_w_s = scalars_chunk["weight"]
        block_ov_s = scalars_chunk["overlap"]
        block_abs_ov_s = scalars_chunk["abs_overlap"]

        block_e_all = block_e_all.at[i, 1:].set(block_e_s)
        block_w_all = block_w_all.at[i, 1:].set(block_w_s)
        sign = block_ov_s / block_abs_ov_s
        total_sign = total_sign.at[i, 1:].set(sign)
        mean = jnp.sum(block_e_all[: i + 1] * block_w_all[: i + 1], axis=0) / jnp.sum(
            block_w_all[: i + 1], axis=0
        )
        sign = jnp.sum(total_sign[: i + 1] * block_w_all[: i + 1], axis=0) / jnp.sum(
            block_w_all[: i + 1], axis=0
        )
        if i == 0:
            err = jnp.zeros_like(mean)
        else:
            err = jnp.std(block_e_all[: i + 1], axis=0) / jnp.sqrt(i)

        timer = params.dt * params.n_prop_steps * jnp.arange(params.n_blocks + 1)
        for j in range(0, params.n_blocks + 1, chunk):
            print(
                f"{(timer[j]):12.4f}    "
                f"{(mean[j].real):14.10f}  "
                f"{(err[j].real):13.7e}  "
                f"{(sign[j].real):6.2f}"
            )

        # not implemented in free projection yet
        block_obs_all: dict[str, jax.Array] = {}
        obs_means: dict[str, jax.Array] = {}
        obs_stderrs: dict[str, jax.Array] = {}

        elapsed = time.perf_counter() - t0

        print(f"Wall time :{elapsed:12.1f} s\n")

    return QmcResult(
        mean_energy=mean,
        stderr_energy=err,
        block_energies=block_e_all,
        block_weights=block_w_all,
        block_observables=block_obs_all,
        observable_means=obs_means,
        observable_stderrs=obs_stderrs,
    )


def run_qmc_energy_fp(
    *,
    sys: System,
    params: QmcParamsFp,
    ham_data: Any,
    trial_data: Any,
    meas_ops: MeasOps,
    trial_ops: TrialOps,
    prop_ops: PropOps,
    block_fn: BlockFn,
    state: PropState | None = None,
    meas_ctx: Any | None = None,
    prop_ctx: Any | None = None,
    target_error: float | None = None,
    mesh: Mesh | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    out = run_qmc_fp(
        sys=sys,
        params=params,
        ham_data=ham_data,
        trial_data=trial_data,
        meas_ops=meas_ops,
        trial_ops=trial_ops,
        prop_ops=prop_ops,
        block_fn=block_fn,
        state=state,
        meas_ctx=meas_ctx,
        prop_ctx=prop_ctx,
        target_error=target_error,
    )
    return out.mean_energy, out.stderr_energy, out.block_energies, out.block_weights
