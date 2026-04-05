from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, Callable, ClassVar, Union, cast

import numpy as np
from jax.sharding import Mesh

print = partial(print, flush=True)

from . import driver
from .driver import QmcResult
from .core.ops import MeasOps, TrialOps
from .core.system import System, WalkerKind
from .ham.chol import HamChol
from .prop.afqmc import make_prop_ops
from .prop.blocks import block as default_block
from .prop.types import PropOps, PropState, QmcParams, QmcParamsBase
from .runtime_layout import RuntimeLayout, make_runtime_layout
from .staging import StagedInputs, _resolve_stage_frozen_arg, load, stage


def _setup_begin(message: str) -> float:
    print(f"[setup] {message}...")
    return time.time()


def _setup_end(start: float, message: str, *, details: str | None = None) -> None:
    suffix = f" | {details}" if details else ""
    print(f"[setup] {message} in {time.time() - start:.2f}s{suffix}")


def _filter_kwargs_for(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Filter kwargs to only those accepted by callable_obj's signature.
    """
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs
    params = sig.parameters
    return {k: v for k, v in kwargs.items() if k in params}


def _make_dataclass_params(
    params_type: type[QmcParamsBase],
    *,
    params: QmcParamsBase | None = None,
    seed: int | None = None,
    **params_kwargs: Any,
) -> QmcParamsBase:
    base = params or params_type()

    if seed is None and params is None:
        seed = int(np.random.randint(0, int(1e9)))

    merged = dict(params_kwargs)
    if seed is not None:
        merged["seed"] = int(seed)

    merged = _filter_kwargs_for(params_type, merged)

    return replace(base, **merged)


def _make_params(
    *,
    params: QmcParams | None = None,
    n_eql_blocks: int | None = None,
    n_blocks: int | None = None,
    seed: int | None = None,
    dt: float | None = None,
    n_walkers: int | None = None,
    **params_kwargs: Any,
) -> QmcParams:
    explicit: dict[str, Any] = {}
    if n_eql_blocks is not None:
        explicit["n_eql_blocks"] = int(n_eql_blocks)
    if n_blocks is not None:
        explicit["n_blocks"] = int(n_blocks)
    if dt is not None:
        explicit["dt"] = float(dt)
    if n_walkers is not None:
        explicit["n_walkers"] = int(n_walkers)
    return cast(
        QmcParams,
        _make_dataclass_params(
            QmcParams,
            params=params,
            seed=seed,
            **params_kwargs,
            **explicit,
        ),
    )


def _make_prop(
    ham_data: HamChol,
    walker_kind: str,
    sys: System | None = None,
    *,
    mixed_precision: bool,
) -> Any:
    return make_prop_ops(
        ham_data.basis,
        walker_kind,
        mixed_precision=mixed_precision,
    )


def _resolve_staged(
    obj_or_staged: Union[Any, StagedInputs, str, Path],
    *,
    norb_frozen_core: int | None,
    chol_cut: float,
    cache: Union[str, Path] | None,
    overwrite: bool,
    verbose: bool,
) -> StagedInputs:
    staged: StagedInputs
    if isinstance(obj_or_staged, StagedInputs):
        staged = obj_or_staged
        return staged

    p = (
        Path(obj_or_staged).expanduser().resolve()
        if isinstance(obj_or_staged, (str, Path))
        else None
    )
    if p is not None and p.exists():
        staged = load(p)
        return staged

    staged = stage(
        obj_or_staged,
        norb_frozen_core=norb_frozen_core,
        chol_cut=chol_cut,
        cache=cache,
        overwrite=overwrite,
        verbose=verbose,
    )
    return staged


def _make_trial_bundle(
    sys: System, staged: StagedInputs, mixed_precision: bool
) -> tuple[Any, Any, Any]:
    """
    Return (trial_data, trial_ops, meas_ops)
    """
    tr = staged.trial
    data = tr.data

    kind = tr.kind.lower()
    t_bundle = _setup_begin(f"building trial bundle ({kind})")

    if kind == "rhf":
        from .meas.rhf import make_rhf_meas_ops
        from .trial.rhf import make_rhf_trial_data, make_rhf_trial_ops

        trial_data = make_rhf_trial_data(data, sys)
        trial_ops = make_rhf_trial_ops(sys=sys)
        meas_ops = make_rhf_meas_ops(sys=sys)
        _setup_end(t_bundle, "trial bundle ready", details=f"kind={kind}")
        return trial_data, trial_ops, meas_ops

    if kind == "uhf":
        from .meas.uhf import make_uhf_meas_ops
        from .trial.uhf import make_uhf_trial_data, make_uhf_trial_ops

        trial_data = make_uhf_trial_data(data, sys)
        trial_ops = make_uhf_trial_ops(sys=sys)
        meas_ops = make_uhf_meas_ops(sys=sys)
        _setup_end(t_bundle, "trial bundle ready", details=f"kind={kind}")
        return trial_data, trial_ops, meas_ops

    if kind == "ghf":
        from .meas.ghf import make_ghf_meas_ops_chol
        from .trial.ghf import make_ghf_trial_data, make_ghf_trial_ops

        trial_data = make_ghf_trial_data(data, sys=sys)
        trial_ops = make_ghf_trial_ops(sys=sys)
        meas_ops = make_ghf_meas_ops_chol(sys=sys)
        _setup_end(t_bundle, "trial bundle ready", details=f"kind={kind}")
        return trial_data, trial_ops, meas_ops

    if kind == "cisd":
        from .meas.cisd import make_cisd_meas_ops
        from .trial.cisd import make_cisd_trial_data, make_cisd_trial_ops

        trial_data = make_cisd_trial_data(data, sys)
        trial_ops = make_cisd_trial_ops(sys=sys)
        meas_ops = make_cisd_meas_ops(sys=sys, mixed_precision=mixed_precision)
        _setup_end(t_bundle, "trial bundle ready", details=f"kind={kind}")
        return trial_data, trial_ops, meas_ops

    if kind == "ucisd":
        from .meas.ucisd import make_ucisd_meas_ops
        from .trial.ucisd import make_ucisd_trial_data, make_ucisd_trial_ops

        trial_data = make_ucisd_trial_data(data, sys)
        trial_ops = make_ucisd_trial_ops(sys=sys)
        meas_ops = make_ucisd_meas_ops(sys=sys, mixed_precision=mixed_precision)
        _setup_end(t_bundle, "trial bundle ready", details=f"kind={kind}")
        return trial_data, trial_ops, meas_ops

    if kind == "gcisd":
        from .meas.gcisd import make_gcisd_meas_ops
        from .trial.gcisd import make_gcisd_trial_data, make_gcisd_trial_ops

        trial_data = make_gcisd_trial_data(data, sys)
        trial_ops = make_gcisd_trial_ops(sys=sys)
        meas_ops = make_gcisd_meas_ops(sys=sys)
        _setup_end(t_bundle, "trial bundle ready", details=f"kind={kind}")
        return trial_data, trial_ops, meas_ops

    raise ValueError(f"Unsupported TrialInput.kind: {tr.kind!r}")


def _resolve_default_walker_kind(ham: Any, walker_kind: WalkerKind | None) -> WalkerKind:
    if walker_kind is None:
        return cast(WalkerKind, ham.basis)
    return walker_kind


@dataclass
class Job:
    """
    A fully assembled AFQMC run bundle.
    """

    staged: StagedInputs
    sys: System
    params: QmcParamsBase
    ham_data: HamChol
    trial_data: object
    trial_ops: TrialOps
    meas_ops: MeasOps
    prop_ops: PropOps
    block_fn: Callable[..., Any]
    runtime_layout: RuntimeLayout
    mesh: Mesh | None = None
    _runtime_prop_ctx: object | None = field(default=None, init=False, repr=False)
    _runtime_meas_ctx: object | None = field(default=None, init=False, repr=False)
    _runtime_state: PropState | None = field(default=None, init=False, repr=False)

    params_cls: ClassVar[type[QmcParamsBase]] = QmcParams
    driver_fn: ClassVar[Callable[..., Any]] = staticmethod(driver.run_qmc)

    def _prepare_runtime(
        self,
        *,
        state: PropState | None = None,
        meas_ctx: object | None = None,
        prop_ctx: object | None = None,
    ) -> tuple[PropState, object, object]:
        if prop_ctx is None:
            prop_ctx = self._runtime_prop_ctx
        if meas_ctx is None:
            meas_ctx = self._runtime_meas_ctx
        if state is None:
            state = self._runtime_state

        if prop_ctx is not None and meas_ctx is not None and state is not None:
            return state, meas_ctx, prop_ctx

        prepared = self.runtime_layout.prepare(
            self,
            state=state,
            meas_ctx=meas_ctx,
            prop_ctx=prop_ctx,
        )
        self.ham_data = prepared.ham_data

        self._runtime_prop_ctx = prepared.prop_ctx
        self._runtime_meas_ctx = prepared.meas_ctx
        self._runtime_state = prepared.state
        return prepared.state, prepared.meas_ctx, prepared.prop_ctx

    def kernel(self, **driver_kwargs: Any) -> QmcResult:
        """
        Run AFQMC energy driver.
        Extra kwargs are forwarded to driver.run_qmc_energy (e.g. state=..., meas_ctx=...).
        """
        assert isinstance(self.params, self.params_cls)
        state, meas_ctx, prop_ctx = self._prepare_runtime(
            state=driver_kwargs.get("state"),
            meas_ctx=driver_kwargs.get("meas_ctx"),
            prop_ctx=driver_kwargs.get("prop_ctx"),
        )
        driver_kwargs["state"] = state
        driver_kwargs["meas_ctx"] = meas_ctx
        driver_kwargs["prop_ctx"] = prop_ctx
        driver_kwargs.setdefault("mesh", self.mesh)
        out = self.driver_fn(
            sys=self.sys,
            params=self.params,
            ham_data=self.ham_data,
            trial_ops=self.trial_ops,
            trial_data=self.trial_data,
            meas_ops=self.meas_ops,
            prop_ops=self.prop_ops,
            block_fn=self.block_fn,
            **driver_kwargs,
        )
        return out


def _assemble_job(
    obj_or_staged: Union[Any, StagedInputs, str, Path],
    *,
    norb_frozen_core: int | None = None,
    norb_frozen: int | None = None,
    chol_cut: float = 1e-5,
    cache: Union[str, Path] | None = None,
    overwrite: bool = False,
    verbose: bool = False,
    walker_kind: WalkerKind | None = None,
    mesh: Mesh | None = None,
    mixed_precision: bool = True,
    params: QmcParamsBase | None = None,
    trial_data: Any = None,
    trial_ops: Any = None,
    meas_ops: Any = None,
    prop_ops: Any = None,
    block_fn: Callable[..., Any] | None = None,
    params_kwargs: dict[str, Any] | None = None,
    prop_kwargs: dict[str, Any] | None = None,
    params_builder: Callable[..., QmcParamsBase],
    prop_builder: Callable[..., Any],
    default_block_fn: Callable[..., Any],
    job_cls: type[Job],
    walker_kind_resolver: Callable[[Any, WalkerKind | None], WalkerKind],
) -> Job:
    trial_data_override = trial_data
    trial_ops_override = trial_ops
    meas_ops_override = meas_ops
    prop_ops_override = prop_ops

    resolved_norb_frozen_core = cast(
        int | None,
        _resolve_stage_frozen_arg(norb_frozen_core, norb_frozen, None),
    )

    staged = _resolve_staged(
        obj_or_staged,
        norb_frozen_core=resolved_norb_frozen_core,
        chol_cut=chol_cut,
        cache=cache,
        overwrite=overwrite,
        verbose=verbose,
    )
    ham = staged.ham

    resolved_walker_kind = walker_kind_resolver(ham, walker_kind)
    sys = System(norb=int(ham.norb), nelec=ham.nelec, walker_kind=resolved_walker_kind)

    qmc_params = params_builder(params=params, **(params_kwargs or {}))

    if trial_data is None or trial_ops is None or meas_ops is None:
        td, to, mo = _make_trial_bundle(sys, staged, mixed_precision)
        trial_data = td if trial_data is None else trial_data
        trial_ops = to if trial_ops is None else trial_ops
        meas_ops = mo if meas_ops is None else meas_ops

    runtime_layout = make_runtime_layout(
        staged=staged,
        allow_host_rhf=job_cls is Job,
        trial_data_override=trial_data_override,
        trial_ops_override=trial_ops_override,
        meas_ops_override=meas_ops_override,
        prop_ops_override=prop_ops_override,
        mixed_precision=mixed_precision,
    )
    t_ham_runtime = _setup_begin("preparing runtime Hamiltonian")
    ham_data = runtime_layout.make_initial_ham_data(ham, mesh)
    _setup_end(t_ham_runtime, "runtime Hamiltonian ready")

    if prop_ops is None:
        prop_ops = prop_builder(
            ham_data,
            sys.walker_kind,
            sys=sys,
            mixed_precision=mixed_precision,
            **(prop_kwargs or {}),
        )

    if block_fn is None:
        block_fn = default_block_fn

    return job_cls(
        staged=staged,
        sys=sys,
        params=qmc_params,
        ham_data=ham_data,
        trial_data=trial_data,
        trial_ops=trial_ops,
        meas_ops=meas_ops,
        prop_ops=prop_ops,
        block_fn=block_fn,
        runtime_layout=runtime_layout,
        mesh=mesh,
    )


def setup(
    obj_or_staged: Union[Any, StagedInputs, str, Path],
    *,
    # staging options (used only if we need to stage)
    norb_frozen_core: int | None = None,
    norb_frozen: int | None = None,
    chol_cut: float = 1e-5,
    cache: Union[str, Path] | None = None,
    overwrite: bool = False,
    verbose: bool = False,
    # system/prop options
    walker_kind: WalkerKind | None = None,
    mesh: Mesh | None = None,
    mixed_precision: bool = True,
    # params options
    params: QmcParams | None = None,
    # overrides for customized runs
    trial_data: Any = None,
    trial_ops: Any = None,
    meas_ops: Any = None,
    prop_ops: Any = None,
    block_fn: Callable[..., Any] | None = None,
    # extra kwargs
    params_kwargs: dict[str, Any] | None = None,
    prop_kwargs: dict[str, Any] | None = None,
) -> Job:
    """
    Assemble a runnable AFQMC Job from either:
      - a pyscf mf/cc object,
      - StagedInputs,
      - or a path to a staged .h5 cache file.

    Basic usage:
        job = setup(mf)
        job.kernel()

    Advanced usage:
        staged = stage(cc, cache="afqmc.h5")
        job = setup(staged, walker_kind="restricted", mixed_precision=False, params=myparams)
        job.kernel()
    """
    return _assemble_job(
        obj_or_staged,
        norb_frozen_core=norb_frozen_core,
        norb_frozen=norb_frozen,
        chol_cut=chol_cut,
        cache=cache,
        overwrite=overwrite,
        verbose=verbose,
        walker_kind=walker_kind,
        mesh=mesh,
        mixed_precision=mixed_precision,
        params=params,
        trial_data=trial_data,
        trial_ops=trial_ops,
        meas_ops=meas_ops,
        prop_ops=prop_ops,
        block_fn=block_fn,
        params_kwargs=params_kwargs,
        prop_kwargs=prop_kwargs,
        params_builder=_make_params,
        prop_builder=_make_prop,
        default_block_fn=default_block,
        job_cls=Job,
        walker_kind_resolver=_resolve_default_walker_kind,
    )
