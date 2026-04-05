from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, NamedTuple, Protocol, cast

import jax

from .typing import ham_data, trial_data

# Using Protocols for public APIs, Callables for internal helper APIs.

# trial


class OverlapFn(Protocol):
    def __call__(self, walker: Any, trial_data: Any) -> jax.Array: ...


class Rdm1Fn(Protocol):
    def __call__(self, trial_data: Any) -> jax.Array: ...


class GreensFn(Protocol):
    # returns the CPMC cache for one walker
    #   - single det: Array (n,n)
    #   - multi det:  dict like {"G": (nd,n,n), "w": (nd,)}

    def __call__(self, walker: Any, trial_data: Any) -> Any: ...


class OverlapRatioFn(Protocol):
    def __call__(
        self,
        greens: Any,
        update_indices: jax.Array,
        update_constants: jax.Array,
    ) -> jax.Array: ...


class UpdateGreenFn(Protocol):
    def __call__(
        self,
        greens: Any,
        update_indices: jax.Array,
        update_constants: jax.Array,
    ) -> Any: ...


class TrialOps(NamedTuple):
    """
    Trial operations.
      - overlap: overlap for a single walker
      - get_rdm1: trial rdm1
      Optional fast update functions (mainly for CPMC):
      - calc_green: compute the greens function
      - calc_overlap_ratio: compute overlap ratio for updates
      - update_green: update greens function after walker update
    """

    overlap: OverlapFn  # (walker, trial_data) -> overlap
    get_rdm1: Rdm1Fn  # (trial_data) -> rdm1
    calc_green: GreensFn | None = None  # (walker, trial_data) -> greens
    calc_overlap_ratio: OverlapRatioFn | None = (
        None  # (greens, update_indices, update_constants) -> ratio
    )
    update_green: UpdateGreenFn | None = (
        None  # (greens, update_indices, update_constants) -> new_greens
    )


@dataclass(frozen=True)
class CpmcTrialFns:
    calc_green: GreensFn
    calc_overlap_ratio: OverlapRatioFn
    update_green: UpdateGreenFn


def require_cpmc_trial_ops(trial_ops: TrialOps) -> CpmcTrialFns:
    if trial_ops.calc_green is None:
        raise ValueError("CPMC requires trial_ops.calc_green")
    if trial_ops.calc_overlap_ratio is None:
        raise ValueError("CPMC requires trial_ops.calc_overlap_ratio")
    if trial_ops.update_green is None:
        raise ValueError("CPMC requires trial_ops.update_green")

    # cast narrows for Pylance
    return CpmcTrialFns(
        calc_green=cast(GreensFn, trial_ops.calc_green),
        calc_overlap_ratio=cast(OverlapRatioFn, trial_ops.calc_overlap_ratio),
        update_green=cast(UpdateGreenFn, trial_ops.update_green),
    )


# hamiltonian


class HamOps(NamedTuple):
    """
    Hamiltonian (would probably be helpful when adding different Hamiltonians).
    """

    n_fields: Callable[[ham_data], int]


# measurements


class MeasKernel(Protocol):
    """
    Measurement kernel protocol.
    """

    def __call__(self, walker: Any, ham_data: Any, meas_ctx: Any, trial_data: Any) -> jax.Array: ...


# usual kernel names
k_energy = "energy"
k_force_bias = "force_bias"
o_rdm1 = "rdm1"
o_density_corr = "density_corr"
o_orb_corr = "orb_corr"


@dataclass(frozen=True)
class MeasOps:
    """
    Measurement ops: trial + ham estimators + optional observables.
    """

    # same as TrialOps.overlap
    overlap: OverlapFn  # (walker, trial_data) -> overlap

    # intermediates for measurements
    build_meas_ctx: Callable[[ham_data, trial_data], Any] = lambda ham_data, trial_data: None

    # algorithm kernels (e.g. "energy", "force_bias")
    kernels: Mapping[str, MeasKernel] = field(default_factory=dict)

    # optional observables (e.g. "rdm1", "density_corr", ...)
    observables: Mapping[str, MeasKernel] = field(default_factory=dict)

    def has_kernel(self, name: str) -> bool:
        return name in self.kernels

    def has_observable(self, name: str) -> bool:
        return name in self.observables

    def require_kernel(self, name: str) -> MeasKernel:
        try:
            return self.kernels[name]
        except KeyError as e:
            avail = ", ".join(sorted(self.kernels.keys()))
            raise KeyError(f"missing required kernel '{name}'. available: [{avail}]") from e

    def require_observable(self, name: str) -> MeasKernel:
        try:
            return self.observables[name]
        except KeyError as e:
            avail = ", ".join(sorted(self.observables.keys()))
            raise KeyError(f"missing requested observable '{name}'. available: [{avail}]") from e

    def available_kernels(self) -> tuple[str, ...]:
        return tuple(sorted(self.kernels.keys()))

    def available_observables(self) -> tuple[str, ...]:
        return tuple(sorted(self.observables.keys()))
