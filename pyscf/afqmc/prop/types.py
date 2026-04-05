from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple, Protocol

import jax
import numpy as np
from jax.sharding import Mesh

from numpy.typing import NDArray

from ..core.ops import MeasOps, TrialOps
from ..core.system import System


def _random_seed() -> int:
    return int(np.random.randint(0, int(1e6)))


class PropState(NamedTuple):
    walkers: Any
    weights: jax.Array
    overlaps: jax.Array
    rng_key: jax.Array
    pop_control_ene_shift: jax.Array
    e_estimate: jax.Array
    node_encounters: jax.Array


@dataclass(frozen=True)
class QmcParamsBase:
    dt: float = 0.005
    n_chunks: int = 1
    n_exp_terms: int = 6
    n_prop_steps: int = 50
    n_blocks: int = 200
    n_walkers: int = 200
    seed: int = field(default_factory=_random_seed)


@dataclass(frozen=True)
class QmcParams(QmcParamsBase):
    pop_control_damping: float = 0.1
    weight_floor: float = 1.0e-3
    weight_cap: float = 100.0
    shift_ema: float = 0.1
    n_eql_blocks: int = 20


@dataclass(frozen=True)
class QmcParamsLno(QmcParams):
    prjlo: NDArray | None = None


@dataclass(frozen=True)
class QmcParamsFp(QmcParamsBase):
    dt: float = 0.05
    n_prop_steps: int = 20
    n_blocks: int = 5
    ene0: float | None = None
    n_traj: int = 10


class StepKernel(Protocol):

    def __call__(
        self,
        state: PropState,
        *,
        params: Any,
        ham_data: Any,
        trial_data: Any,
        trial_ops: TrialOps,
        meas_ops: MeasOps,
        meas_ctx: Any,
        prop_ctx: Any,
    ) -> PropState: ...


class InitPropState(Protocol):

    def __call__(
        self,
        *,
        sys: System,
        ham_data: Any,
        trial_ops: TrialOps,
        trial_data: Any,
        meas_ops: MeasOps,
        params: Any,
        initial_walkers: Any | None = None,
        initial_e_estimate: jax.Array | None = None,
        rdm1: jax.Array | None = None,
        mesh: Mesh | None = None,
    ) -> PropState: ...


@dataclass(frozen=True)
class PropOps:
    init_prop_state: InitPropState
    build_prop_ctx: Callable[[Any, jax.Array, Any], Any]  # (ham_data, rdm1, params) -> prop_ctx
    step: StepKernel
