from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, ClassVar, Union, cast

from jax.sharding import Mesh

from . import driver
from .core.system import System, WalkerKind
from .ham.chol import HamChol
from .prop.afqmc_fp import make_prop_ops_fp
from .prop.blocks import block_fp as default_block
from .prop.types import QmcParamsBase, QmcParamsFp
from .setup import Job, _assemble_job, _make_dataclass_params
from .staging import StagedInputs


def _make_params_fp(
    *,
    params: QmcParamsFp | None = None,
    n_traj: int | None = None,
    ene0: float | None = None,
    n_blocks: int | None = None,
    seed: int | None = None,
    dt: float | None = None,
    n_walkers: int | None = None,
    **params_kwargs: Any,
) -> QmcParamsFp:
    explicit: dict[str, Any] = {}
    if n_blocks is not None:
        explicit["n_blocks"] = int(n_blocks)
    if dt is not None:
        explicit["dt"] = float(dt)
    if n_walkers is not None:
        explicit["n_walkers"] = int(n_walkers)
    if n_traj is not None:
        explicit["n_traj"] = int(n_traj)
    if ene0 is not None:
        explicit["ene0"] = float(ene0)

    return cast(
        QmcParamsFp,
        _make_dataclass_params(
            QmcParamsFp,
            params=params,
            seed=seed,
            **params_kwargs,
            **explicit,
        ),
    )


def _make_prop_fp(
    ham_data: HamChol,
    walker_kind: str,
    sys: System,
    *,
    mixed_precision: bool,
) -> Any:
    return make_prop_ops_fp(
        ham_data.basis,
        walker_kind,
        sys,
        mixed_precision=mixed_precision,
    )


def _resolve_fp_walker_kind(ham: Any, walker_kind: WalkerKind | None) -> WalkerKind:
    if walker_kind is not None:
        return walker_kind

    if ham.basis == "restricted":
        if ham.nelec[0] == ham.nelec[1]:
            return "restricted"
        return "unrestricted"

    if ham.basis == "generalized":
        return "generalized"

    raise ValueError(f"Unsupported ham.basis: {ham.basis!r}")


class JobFp(Job):
    """
    A fully assembled FP-AFQMC run bundle.
    """

    params_cls: ClassVar[type[QmcParamsBase]] = QmcParamsFp
    driver_fn: ClassVar[Callable[..., Any]] = staticmethod(driver.run_qmc_fp)


def setup_fp(
    obj_or_staged: Union[Any, StagedInputs, str, Path],
    *,
    # staging options (used only if we need to stage)
    norb_frozen_core: int | None = None,
    norb_frozen: int | None = None,
    chol_cut: float = 1e-5,
    cache: Union[str, Path] | None = None,
    overwrite: bool = False,
    verbose: bool = False,
    log: Any | None = None,
    # system/prop options
    walker_kind: WalkerKind | None = None,
    mesh: Mesh | None = None,
    mixed_precision: bool = True,
    # params options
    params: QmcParamsFp | None = None,
    # overrides for customized runs
    trial_data: Any = None,
    trial_ops: Any = None,
    meas_ops: Any = None,
    prop_ops: Any = None,
    block_fn: Callable[..., Any] | None = None,
    # extra kwargs
    params_kwargs: dict[str, Any] | None = None,
    prop_kwargs: dict[str, Any] | None = None,
) -> JobFp:
    """
    Assemble a runnable AFQMC Job from either:
      - a pyscf mf/cc object,
      - StagedInputs,
      - or a path to a staged .h5 cache file.

    Basic usage:
        job = setup_fp(mf)
        job.kernel()

    Advanced usage:
        staged = stage(cc, cache="afqmc.h5")
        job = setup_fp(staged, walker_kind="restricted", mixed_precision=False, params=myparams)
        job.kernel()
    """
    return cast(
        JobFp,
        _assemble_job(
            obj_or_staged,
            norb_frozen_core=norb_frozen_core,
            norb_frozen=norb_frozen,
            chol_cut=chol_cut,
            cache=cache,
            overwrite=overwrite,
            verbose=verbose,
            log=log,
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
            params_builder=_make_params_fp,
            prop_builder=_make_prop_fp,
            default_block_fn=default_block,
            job_cls=JobFp,
            walker_kind_resolver=_resolve_fp_walker_kind,
        ),
    )
