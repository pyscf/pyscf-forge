from __future__ import annotations

from typing import Any, Callable, NamedTuple, Protocol

import jax
import jax.numpy as jnp
from jax import lax, tree_util

from .. import walkers as wk
from ..core.levels import LevelPack
from ..core.ops import MeasOps, TrialOps, k_energy
from ..core.system import System
from ..walkers import SrFn
from .types import PropOps, PropState, QmcParams, QmcParamsFp


class BlockFn(Protocol):
    def __call__(
        self,
        state: PropState,
        *,
        sys: System,
        params: Any,
        ham_data: Any,
        trial_data: Any,
        trial_ops: TrialOps,
        meas_ops: MeasOps,
        meas_ctx: Any,
        prop_ops: Any,
        prop_ctx: Any,
        sr_fn: SrFn = wk.stochastic_reconfiguration,
        observable_names: tuple[str, ...] = (),
    ) -> tuple[PropState, BlockObs]: ...


class MixedBlockFn(Protocol):
    def __call__(
        self,
        state: PropState,
        *,
        sys: System,
        params: QmcParams,
        ham_data: Any,
        guide_data: Any,
        guide_ops: TrialOps,
        guide_meas_ops: MeasOps,
        guide_meas_ctx: Any,
        guide_prop_ops: PropOps,
        guide_prop_ctx: Any,
        trial_data: Any,
        trial_meas_ops: MeasOps,
        trial_meas_ctx: Any,
        observable_names: tuple[str, ...] = (),
        sr_fn: Callable = wk.stochastic_reconfiguration,
    ) -> tuple[PropState, BlockObs]: ...


class BlockObs(NamedTuple):
    scalars: dict[str, jax.Array]
    observables: dict[str, jax.Array]


def block(
    state: PropState,
    *,
    sys: System,
    params: QmcParams,
    ham_data: Any,
    trial_data: Any,
    trial_ops: TrialOps,
    meas_ops: MeasOps,
    meas_ctx: Any,
    prop_ops: PropOps,
    prop_ctx: Any,
    sr_fn: Callable = wk.stochastic_reconfiguration,
    observable_names: tuple[str, ...] = (),
) -> tuple[PropState, BlockObs]:
    """
    propagation + measurement
    """
    step = lambda st: prop_ops.step(
        st,
        params=params,
        ham_data=ham_data,
        trial_data=trial_data,
        trial_ops=trial_ops,
        meas_ops=meas_ops,
        prop_ctx=prop_ctx,
        meas_ctx=meas_ctx,
    )

    def _scan_step(carry: PropState, _x: Any):
        carry = step(carry)
        return carry, None

    state, _ = lax.scan(_scan_step, state, xs=None, length=params.n_prop_steps)

    walkers_new = wk.orthonormalize(state.walkers, sys.walker_kind)
    overlaps_new = wk.vmap_chunked(meas_ops.overlap, n_chunks=params.n_chunks, in_axes=(0, None))(
        walkers_new, trial_data
    )
    state = state._replace(walkers=walkers_new, overlaps=overlaps_new)

    e_kernel = meas_ops.require_kernel(k_energy)
    e_samples = wk.vmap_chunked(e_kernel, n_chunks=params.n_chunks, in_axes=(0, None, None, None))(
        state.walkers, ham_data, meas_ctx, trial_data
    )
    e_samples = jnp.real(e_samples)

    thresh = jnp.sqrt(2.0 / jnp.asarray(params.dt))
    e_ref = state.e_estimate
    is_nan = ~jnp.isfinite(e_samples)
    e_samples = jnp.where(is_nan | (jnp.abs(e_samples - e_ref) > thresh), e_ref, e_samples)

    weights = jnp.where(is_nan, 0.0, state.weights)
    w_sum = jnp.sum(weights)
    w_sum_safe = jnp.where(w_sum == 0, 1.0, w_sum)
    e_block = jnp.sum(weights * e_samples) / w_sum_safe
    e_block = jnp.where(w_sum == 0, e_ref, e_block)

    alpha = jnp.asarray(params.shift_ema, dtype=jnp.result_type(e_block))
    state = state._replace(
        weights=weights,
        e_estimate=(1.0 - alpha) * state.e_estimate + alpha * e_block,
    )

    obs_samples: dict[str, jax.Array] = {}
    for name in observable_names:
        kernel = meas_ops.require_observable(name)
        samples = wk.vmap_chunked(kernel, n_chunks=params.n_chunks, in_axes=(0, None, None, None))(
            state.walkers, ham_data, meas_ctx, trial_data
        )
        w_shape = (weights.shape[0],) + (1,) * max(samples.ndim - 1, 0)
        num = jnp.sum(weights.reshape(w_shape) * samples, axis=0)
        zero = jnp.zeros_like(num)
        obs_samples[name] = jnp.where(w_sum == 0, zero, num / w_sum_safe)

    key, subkey = jax.random.split(state.rng_key)
    zeta = jax.random.uniform(subkey)
    w_sr, weights_sr = sr_fn(state.walkers, state.weights, zeta, sys.walker_kind)
    overlaps_sr = wk.vmap_chunked(meas_ops.overlap, n_chunks=params.n_chunks, in_axes=(0, None))(
        w_sr, trial_data
    )
    state = state._replace(
        walkers=w_sr,
        weights=weights_sr,
        overlaps=overlaps_sr,
        rng_key=key,
    )

    obs = BlockObs(
        scalars={"energy": e_block, "weight": w_sum},
        observables=obs_samples,
    )
    return state, obs


def block_mixed(
    state: PropState,
    *,
    sys: System,
    params: QmcParams,
    ham_data: Any,
    guide_data: Any,
    guide_ops: TrialOps,
    guide_meas_ops: MeasOps,
    guide_meas_ctx: Any,
    guide_prop_ops: PropOps,
    guide_prop_ctx: Any,
    trial_data: Any,
    trial_meas_ops: MeasOps,
    trial_meas_ctx: Any,
    observable_names: tuple[str, ...] = (),
    sr_fn: Callable = wk.stochastic_reconfiguration,
) -> tuple[PropState, BlockObs]:
    """
    Block function for mixed sampling -- Trial =! Guide
    propagation(Guide) + measurement(Trial)
    currently only support pt2CCSD trial
    TODO generalize the output of trial kernel to multiple variable
         without saving each term according to their names
    """

    # propagation is guided with the guiding wavefunction
    step = lambda st: guide_prop_ops.step(
        st,
        params=params,
        ham_data=ham_data,
        trial_data=guide_data,
        trial_ops=guide_ops,
        meas_ops=guide_meas_ops,
        prop_ctx=guide_prop_ctx,
        meas_ctx=guide_meas_ctx,
    )

    def _scan_step(carry: PropState, _x: Any):
        carry = step(carry)
        return carry, None

    state, _ = lax.scan(_scan_step, state, xs=None, length=params.n_prop_steps)

    walkers_new = wk.orthonormalize(state.walkers, sys.walker_kind)
    guide_overlaps = wk.vmap_chunked(
        guide_meas_ops.overlap, n_chunks=params.n_chunks, in_axes=(0, None)
    )(walkers_new, guide_data)
    state = state._replace(walkers=walkers_new, overlaps=guide_overlaps)

    # some measurements with the guiding wavefunction if necessary
    guide_e_kernel = guide_meas_ops.require_kernel(k_energy)
    guide_e_samples = wk.vmap_chunked(
        guide_e_kernel, n_chunks=params.n_chunks, in_axes=(0, None, None, None)
    )(
        state.walkers, ham_data, guide_meas_ctx, guide_data
    )  # local energy with respect to the guiding wavefunction = <guide|H|walker>/<guide|walker>
    guide_e_samples = jnp.real(guide_e_samples)

    thresh = jnp.sqrt(2.0 / jnp.asarray(params.dt))
    e_ref = state.e_estimate
    is_nan = ~jnp.isfinite(guide_e_samples)
    guide_e_samples = jnp.where(
        is_nan | (jnp.abs(guide_e_samples - e_ref) > thresh), e_ref, guide_e_samples
    )

    guide_weights = jnp.where(is_nan, 0.0, state.weights)
    guide_w_block = jnp.sum(guide_weights)
    guide_w_block_safe = jnp.where(guide_w_block == 0, 1.0, guide_w_block)
    guide_e_block = jnp.sum(guide_weights * guide_e_samples) / guide_w_block_safe
    guide_e_block = jnp.where(guide_w_block == 0, e_ref, guide_e_block)

    alpha = jnp.asarray(params.shift_ema, dtype=jnp.result_type(guide_e_block))
    state = state._replace(
        weights=guide_weights,
        e_estimate=(1.0 - alpha) * state.e_estimate + alpha * guide_e_block,
    )

    # measuing with respect to trial
    trial_e_kernel = trial_meas_ops.require_kernel(k_energy)
    pt2results = wk.vmap_chunked(
        trial_e_kernel, n_chunks=params.n_chunks, in_axes=(0, None, None, None)
    )(state.walkers, ham_data, trial_meas_ctx, trial_data)
    trial_t2s, trial_e0s, trial_e1s = pt2results[:, 0], pt2results[:, 1], pt2results[:, 2]
    trial_overlaps = wk.vmap_chunked(
        trial_meas_ops.overlap, n_chunks=params.n_chunks, in_axes=(0, None)
    )(walkers_new, trial_data)
    trial_weights = (
        guide_weights * trial_overlaps / guide_overlaps
    )  # w_trial = w_guide * <G|walker>/<T|walker>
    trial_w_block = jnp.sum(trial_weights)
    trial_t2_block = jnp.sum(trial_weights * trial_t2s) / trial_w_block
    trial_e0_block = jnp.sum(trial_weights * trial_e0s) / trial_w_block
    trial_e1_block = jnp.sum(trial_weights * trial_e1s) / trial_w_block

    obs_samples: dict[str, jax.Array] = {}

    # performing SR at the end of Block propagation and measurement (Guide)
    key, subkey = jax.random.split(state.rng_key)
    zeta = jax.random.uniform(subkey)
    w_sr, weights_sr = sr_fn(state.walkers, state.weights, zeta, sys.walker_kind)
    overlaps_sr = wk.vmap_chunked(
        guide_meas_ops.overlap, n_chunks=params.n_chunks, in_axes=(0, None)
    )(w_sr, guide_data)
    state = state._replace(
        walkers=w_sr,
        weights=weights_sr,
        overlaps=overlaps_sr,
        rng_key=key,
    )

    obs = BlockObs(
        scalars={
            "guide_weight": guide_w_block,
            "guide_energy": guide_e_block,
            "trial_weight": trial_w_block,
            "trial_t2": trial_t2_block,
            "trial_e0": trial_e0_block,
            "trial_e1": trial_e1_block,
        },
        observables=obs_samples,
    )
    return state, obs


def block_fp(
    state: PropState,
    *,
    sys: System,
    params: QmcParamsFp,
    ham_data: Any,
    trial_data: Any,
    trial_ops: TrialOps,
    meas_ops: MeasOps,
    meas_ctx: Any,
    prop_ops: PropOps,
    prop_ctx: Any,
    sr_fn: Callable = wk.stochastic_reconfiguration,
    observable_names: tuple[str, ...] = (),
) -> tuple[PropState, BlockObs]:
    """
    propagation + measurement
    """
    step_fp = lambda st: prop_ops.step(
        st,
        params=params,
        ham_data=ham_data,
        trial_data=trial_data,
        trial_ops=trial_ops,
        meas_ops=meas_ops,
        prop_ctx=prop_ctx,
        meas_ctx=meas_ctx,
    )

    def _scan_step(carry: PropState, _x: Any):
        carry = step_fp(carry)
        return carry, None

    state, _ = lax.scan(_scan_step, state, xs=None, length=params.n_prop_steps)
    overlaps_new = wk.vmap_chunked(meas_ops.overlap, n_chunks=params.n_chunks, in_axes=(0, None))(
        state.walkers, trial_data
    )
    state = state._replace(overlaps=overlaps_new)
    e_kernel = meas_ops.require_kernel(k_energy)
    e_samples = wk.vmap_chunked(e_kernel, n_chunks=params.n_chunks, in_axes=(0, None, None, None))(
        state.walkers, ham_data, meas_ctx, trial_data
    )

    thresh = jnp.sqrt(2.0 / jnp.asarray(params.dt))

    e_samples = jnp.where(jnp.abs(e_samples - params.ene0) > thresh, params.ene0, e_samples)
    e_samples = jnp.array(e_samples)

    weights = state.weights
    overlaps = state.overlaps
    w_sum = jnp.sum(weights * overlaps)
    e_block = jnp.sum(weights * overlaps * e_samples) / w_sum
    ov = jnp.sum(overlaps)
    abs_ov = jnp.sum(jnp.abs(overlaps))

    obs = BlockObs(
        scalars={"energy": e_block, "weight": w_sum, "overlap": ov, "abs_overlap": abs_ov},
        observables={},
    )
    return state, obs


@tree_util.register_pytree_node_class
class MlmcMeasCtx(NamedTuple):
    """
    prop_meas_ctx:
      The meas_ctx used by prop_ops.step (force bias etc). Typically "full" (untruncated).

    packs:
      Tuple of LevelPack's, ordered low to high, used only for energy evaluation.

    m_deltas:
      Tuple of fixed subsample sizes for each increment delta_l = E_l - E_{l-1}.
      Must have length len(packs) - 1.
      Each m must be int and <= n_walkers.
    """

    prop_meas_ctx: Any
    packs: tuple[LevelPack, ...]
    m_deltas: tuple[int, ...]

    def tree_flatten(self):
        # children: pytrees with arrays
        children = (self.prop_meas_ctx, self.packs)
        # aux: static python metadata
        aux = (self.m_deltas,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (m_deltas,) = aux
        prop_meas_ctx, packs = children
        return cls(prop_meas_ctx=prop_meas_ctx, packs=packs, m_deltas=m_deltas)


def block_mlmc(
    state: PropState,
    *,
    sys: System,
    params: QmcParams,
    ham_data: Any,
    trial_data: Any,
    trial_ops: TrialOps,
    meas_ops: MeasOps,
    meas_ctx: MlmcMeasCtx,
    prop_ops: PropOps,
    prop_ctx: Any,
    sr_fun: Callable = wk.stochastic_reconfiguration,
    observable_names: tuple[str, ...] = (),
) -> tuple[PropState, BlockObs]:
    """
    propagation + MLMC measurement

    sum_i w_i E0(i) + sum_{l>=1} N/m_l sum_{i in S_l} w_i [E_l(i) - E_{l-1}(i)] / sum_i w_i
    """
    prop_meas_ctx = meas_ctx.prop_meas_ctx

    step = lambda st: prop_ops.step(
        st,
        params=params,
        ham_data=ham_data,
        trial_data=trial_data,
        trial_ops=trial_ops,
        meas_ops=meas_ops,
        prop_ctx=prop_ctx,
        meas_ctx=prop_meas_ctx,
    )

    def _scan_step(carry: PropState, _x: Any):
        carry = step(carry)
        return carry, None

    state, _ = lax.scan(_scan_step, state, xs=None, length=params.n_prop_steps)

    walkers_new = wk.orthonormalize(state.walkers, sys.walker_kind)
    overlaps_new = wk.vmap_chunked(meas_ops.overlap, n_chunks=params.n_chunks, in_axes=(0, None))(
        walkers_new, trial_data
    )
    state = state._replace(walkers=walkers_new, overlaps=overlaps_new)

    # --- MLMC measurement ---
    packs = meas_ctx.packs
    m_deltas = meas_ctx.m_deltas
    if len(packs) < 1:
        raise ValueError("MlmcMeasCtx.packs must contain at least one level pack.")
    if len(m_deltas) != max(0, len(packs) - 1):
        raise ValueError("MlmcMeasCtx.m_deltas must have length len(packs)-1.")

    e_kernel = meas_ops.require_kernel(k_energy)
    energy_vmapped = wk.vmap_chunked(
        e_kernel, n_chunks=params.n_chunks, in_axes=(0, None, None, None)
    )

    e_ref = state.e_estimate

    # baseline level (0): evaluate on all walkers
    p0 = packs[0]
    walkers0 = wk.slice_walkers(state.walkers, sys.walker_kind, p0.norb_keep)
    e0 = energy_vmapped(walkers0, p0.ham_data, p0.meas_ctx, p0.trial_data)
    e0 = jnp.real(e0)
    thresh = jnp.sqrt(2.0 / jnp.asarray(params.dt))
    is_nan = ~jnp.isfinite(e0)
    e0 = jnp.where(is_nan | (jnp.abs(e0 - e_ref) > thresh), e_ref, e0)

    weights = jnp.where(is_nan, 0.0, state.weights)
    n_walkers = int(weights.shape[0])
    w_sum = jnp.sum(weights)
    w_sum_safe = jnp.where(w_sum == 0, 1.0, w_sum)

    num0 = jnp.sum(weights * e0)
    e0_block = jnp.where(w_sum == 0, e_ref, num0 / w_sum_safe)

    # corrections: telescope with fixed-size subsamples
    # independent subsets per level increment
    key_next, key_sel, key_sr = jax.random.split(state.rng_key, 3)
    if len(m_deltas) > 0:
        keys = jax.random.split(key_sel, len(m_deltas))
    else:
        keys = jnp.zeros((0, 2), dtype=key_sel.dtype)  # unused

    num_corr = jnp.array(0.0, dtype=jnp.result_type(num0))
    delta_blocks = []  # for optional diagnostics

    for ell in range(1, len(packs)):
        m = int(m_deltas[ell - 1])
        if m <= 0:
            delta_blocks.append(jnp.array(0.0, dtype=jnp.result_type(num0)))
            continue

        # sample subset indices
        idx = jax.random.choice(keys[ell - 1], n_walkers, shape=(m,), replace=False)

        # evaluate E_ell and E_{ell-1} on same subset
        phi = wk.take_walkers(state.walkers, idx)
        w_sub = weights[idx]

        phi_hi = wk.slice_walkers(phi, sys.walker_kind, packs[ell].norb_keep)
        phi_lo = wk.slice_walkers(phi, sys.walker_kind, packs[ell - 1].norb_keep)

        e_hi = energy_vmapped(
            phi_hi,
            packs[ell].ham_data,
            packs[ell].meas_ctx,
            packs[ell].trial_data,
        )
        e_lo = energy_vmapped(
            phi_lo,
            packs[ell - 1].ham_data,
            packs[ell - 1].meas_ctx,
            packs[ell - 1].trial_data,
        )

        thresh = jnp.sqrt(2.0 / jnp.asarray(params.dt))
        e0 = jnp.where(~jnp.isfinite(e0) | (jnp.abs(e0 - e_ref) > thresh), e_ref, e0)
        e_hi = jnp.where(~jnp.isfinite(e_hi) | (jnp.abs(e_hi - e_ref) > thresh), e_ref, e_hi)
        e_lo = jnp.where(~jnp.isfinite(e_lo) | (jnp.abs(e_lo - e_ref) > thresh), e_ref, e_lo)

        delta = jnp.array(e_hi) - jnp.array(e_lo)

        # unbiased estimator of sum_i w_i delta_i via Horvitz–Thompson scaling
        scale = jnp.asarray(n_walkers / m, dtype=jnp.result_type(num0))
        num_inc_hat = scale * jnp.sum(w_sub * delta)

        num_corr = num_corr + jnp.real(num_inc_hat)
        delta_blocks.append(jnp.where(w_sum == 0, 0.0, jnp.real(num_inc_hat) / w_sum_safe))

    num_total = num0 + num_corr
    e_mlmc_block = jnp.where(w_sum == 0, e_ref, num_total / w_sum_safe)

    alpha = jnp.asarray(params.shift_ema, dtype=jnp.result_type(e_mlmc_block))
    state = state._replace(
        weights=weights,
        e_estimate=(1.0 - alpha) * state.e_estimate + alpha * e_mlmc_block,
    )

    zeta = jax.random.uniform(key_sr)
    w_sr, weights_sr = sr_fun(state.walkers, state.weights, zeta, sys.walker_kind)
    overlaps_sr = wk.vmap_chunked(meas_ops.overlap, n_chunks=params.n_chunks, in_axes=(0, None))(
        w_sr, trial_data
    )
    state = state._replace(
        walkers=w_sr,
        weights=weights_sr,
        overlaps=overlaps_sr,
        rng_key=key_next,
    )

    obs = BlockObs(
        scalars={
            "energy": e_mlmc_block,
            "energy_base": e0_block,  # optional: for debugging/monitoring
            "weight": w_sum,
            "mlmc_delta": (
                jnp.asarray(delta_blocks)
                if delta_blocks
                else jnp.zeros((0,), dtype=jnp.result_type(num0))
            ),
        },
        observables={},
    )
    return state, obs
