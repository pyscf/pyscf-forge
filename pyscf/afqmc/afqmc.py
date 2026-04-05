from __future__ import annotations

from .config import configure_once

configure_once()

import copy
import dataclasses
import sys
from pathlib import Path
from typing import Any, Callable, Union, cast

import numpy as np
from jax.sharding import Mesh
from numpy.typing import ArrayLike, NDArray

from pyscf import lib
from pyscf.lib import logger

from . import staging
from .core.system import WalkerKind
from .driver import QmcResult
from .prop.types import QmcParams, QmcParamsBase, QmcParamsFp, QmcParamsLno
from .runtime_provenance import dump_runtime_provenance
from .setup import Job
from .setup import setup as setup_job
from .setup_fp import JobFp
from .setup_fp import setup_fp as setup_job_fp
from .staging import StagedInputs, _is_cc_like
from .staging import dump as dump_staged
from .staging import load as load_staged
from .staging import stage as stage_inputs


def banner_afqmc() -> str:
    return r"""
    ████████╗██████╗  ██████╗ ████████╗
    ╚══██╔══╝██╔══██╗██╔═══██╗╚══██╔══╝
       ██║   ██████╔╝██║   ██║   ██║
       ██║   ██╔══██╗██║   ██║   ██║
       ██║   ██║  ██║╚██████╔╝   ██║
       ╚═╝   ╚═╝  ╚═╝ ╚═════╝    ╚═╝
  Trotter-propagated Random Orbital Trajectories
differentiable auxiliary-field quantum Monte Carlo
"""


def _frozen_cache_key(frozen: int | ArrayLike | None) -> int | tuple[int, ...] | None:
    if isinstance(frozen, np.ndarray):
        arr = np.asarray(frozen, dtype=np.int64).reshape(-1)
        return tuple(int(x) for x in arr)
    if isinstance(frozen, (list, tuple)):
        arr = np.asarray(frozen, dtype=np.int64).reshape(-1)
        return tuple(int(x) for x in arr)
    if frozen is None:
        return None
    if isinstance(frozen, (int, np.integer)):
        return int(frozen)
    raise TypeError(f"Unsupported frozen type for cache key: {type(frozen)}")


def _default_stream_config(obj: Any) -> tuple[int, Any]:
    if obj is not None:
        mol = getattr(obj, "mol", None)
        if mol is not None:
            return int(getattr(mol, "verbose", logger.NOTE)), getattr(
                mol, "stdout", sys.stdout
            )
        if hasattr(obj, "verbose") or hasattr(obj, "stdout"):
            return int(getattr(obj, "verbose", logger.NOTE)), getattr(
                obj, "stdout", sys.stdout
            )
    return logger.NOTE, sys.stdout


class Afqmc(lib.StreamObject):
    """
    AFQMC driver object.

    Parameters
    ----------
    mf_or_cc : Any
        Mean-field or coupled-cluster object from which to build Hamiltonian and trial wavefunction.
    norb_frozen_core : int, optional
        Preferred name for the number of lowest occupied core orbitals removed from the AFQMC
        Hamiltonian.
    norb_frozen : int, optional
        Backward-compatible alias for ``norb_frozen_core``. For CC objects with integer
        ``cc.frozen``, this is inferred from ``cc.frozen``. For restricted CCSD objects with
        list-valued ``cc.frozen``, the trial-space frozen occupied/virtual blocks are inferred
        from ``cc.frozen`` while ``norb_frozen_core``/``norb_frozen`` control the occupied core
        orbitals removed from the AFQMC Hamiltonian.
    chol_cut : float, optional
        Cholesky decomposition cutoff, by default 1e-5
    cache : Union[str, Path], optional
        Path to cache file for staged inputs, by default None
    n_eql_blocks : int, optional
        Number of equilibration blocks if params is not provided, by default 20
    n_blocks : int, optional
        Number of production blocks if params is not provided, by default 200
    seed : int | None, optional
        Random seed if params is not provided, by default None
    dt : float | None, optional
        Time step if params is not provided, by default None
    n_walkers : int | None, optional
        Number of walkers if params is not provided, by default None
    n_chunk : int | None, optional
        Number of chunks if params is not provided, by default 1
    """

    params_cls = QmcParams
    job_cls = Job
    setup_fn = staticmethod(setup_job)

    def __init__(
        self,
        mf_or_cc: Any,
        *,
        norb_frozen_core: int | None = None,
        norb_frozen: int | None = None,
        chol_cut: float = 1e-5,
        cache: Union[str, Path] | None = None,
        n_eql_blocks: int | None = None,
        n_blocks: int | None = None,
        seed: int | None = None,
        dt: float | None = None,
        n_walkers: int | None = None,
        n_chunks: int | None = None,
    ):
        self._obj = mf_or_cc
        self._cc: Any = None
        if _is_cc_like(mf_or_cc):
            self._cc = mf_or_cc
            self._scf = mf_or_cc._scf
            self.source_kind = "cc"
        else:
            self._scf = mf_or_cc
            self.source_kind = "mf"

        resolved_norb_frozen = staging._resolve_stage_frozen_arg(
            norb_frozen_core, norb_frozen, None
        )
        assert resolved_norb_frozen is None or isinstance(resolved_norb_frozen, int)
        self.norb_frozen_core = resolved_norb_frozen
        self.norb_frozen = resolved_norb_frozen
        self.chol_cut = float(chol_cut)
        self.cache = Path(cache).expanduser().resolve() if cache is not None else None
        self.overwrite_cache = False
        self.verbose, self.stdout = _default_stream_config(self._scf)
        self.max_memory = (
            getattr(self._scf, "max_memory", 0) if self._scf is not None else 0
        )

        self.walker_kind: WalkerKind | None = None  # resolved in kernel
        self.mixed_precision = True

        self.params: QmcParamsBase | None = None  # resolved in kernel
        defaults = self.params_cls()
        self.dt = defaults.dt if dt is None else dt
        self.n_walkers = defaults.n_walkers if n_walkers is None else n_walkers
        self.n_blocks = defaults.n_blocks if n_blocks is None else n_blocks
        self.seed = defaults.seed if seed is None else seed
        self.n_chunks = defaults.n_chunks if n_chunks is None else n_chunks
        if hasattr(defaults, "n_eql_blocks"):
            self.n_eql_blocks = (
                defaults.n_eql_blocks if n_eql_blocks is None else n_eql_blocks
            )

        self._staged: StagedInputs | None = None
        self._job: Job | None = None
        self._cache_key: tuple | None = None

        self.e_tot: Any = None
        self.e_err: Any = None
        self.block_energies: Any = None
        self.block_weights: Any = None

    @property
    def staged(self) -> StagedInputs | None:
        return self._staged

    @property
    def job(self) -> Job | None:
        return self._job

    def _debug_prints_enabled(self) -> bool:
        return self.verbose >= logger.DEBUG

    def _dump_params(self, params: QmcParamsBase, log: Any) -> None:
        fields = dataclasses.fields(params)
        width = len(max(fields, key=lambda f: len(f.name)).name)
        log.info(" %s:", type(params).__name__)
        for field in fields:
            log.info("  %s = %s", f"{field.name:<{width}}", getattr(params, field.name))

    def _resolve_meas_cfg(self, job: Job) -> object | None:
        from .meas.cisd import get_cisd_meas_cfg
        from .meas.rhf import get_rhf_meas_cfg

        for getter in (get_cisd_meas_cfg, get_rhf_meas_cfg):
            cfg = getter(job.meas_ops)
            if cfg is not None:
                return cfg
        return None

    def _dump_cfg(self, name: str, cfg: object, log: Any) -> None:
        if not dataclasses.is_dataclass(cfg):
            log.info(" %s = %s", f"{name:<15}", cfg)
            return

        log.info(" %s = %s", f"{name:<15}", type(cfg).__name__)
        fields = dataclasses.fields(cfg)
        width = len(max(fields, key=lambda f: len(f.name)).name)
        for field in fields:
            value = getattr(cfg, field.name)
            if isinstance(value, type):
                value_str = value.__name__
            else:
                value_str = str(value)
            log.info("  %s = %s", f"{field.name:<{width}}", value_str)

    def dump_flags(self, job: Job, verbose: int | None = None) -> None:
        self._dump_flags_helper(job, logger.new_logger(self, verbose))

    def _dump_flags_helper(self, job: Job, log: Any) -> None:
        meta = job.staged.meta
        src = meta["source_kind"]
        chol_cut = meta["chol_cut"]
        sys = job.sys
        nchol = job.ham_data.nchol
        params = job.params
        trial = job.staged.trial
        log.info("")
        log.info("******** AFQMC ********")
        log.info(" norb            = %s", sys.norb)
        log.info(" nelec_up        = %s", sys.nelec[0])
        log.info(" nelec_dn        = %s", sys.nelec[1])
        log.info(" nchol           = %s", nchol)
        log.info(" source_kind     = %s", src)
        log.info(" trial_kind      = %s", trial.kind)
        log.info(" chol_cut        = %g", chol_cut)
        log.info(" cache           = %s", str(self.cache) if self.cache else None)
        log.info(" walker_kind     = %s", sys.walker_kind)
        log.info(" mixed_precision = %s", self.mixed_precision)
        meas_cfg = self._resolve_meas_cfg(job)
        if meas_cfg is not None:
            log.info("")
            self._dump_cfg("meas_cfg", meas_cfg, log)
        log.info("")
        self._dump_params(params, log)

    def _key(self) -> tuple:
        """Key for determining whether staged/job caches are still valid."""
        cache_mtime = None
        if self.cache is not None and self.cache.exists():
            cache_mtime = self.cache.stat().st_mtime
        return (
            self.source_kind,
            _frozen_cache_key(self.norb_frozen_core),
            float(self.chol_cut),
            str(self.cache) if self.cache is not None else None,
            bool(self.overwrite_cache),
            cache_mtime,
        )

    def stage(self, *, force: bool = False, log: Any | None = None) -> StagedInputs:
        """
        Compute or load HamInput/TrialInput.
        If cache is set and exists, loads unless overwrite_cache=True.
        """
        if isinstance(self.norb_frozen_core, (list, tuple, np.ndarray)):
            raise TypeError(
                "Array-valued frozen orbitals are not supported by the public "
                "Afqmc API. Pass an integer number of frozen core orbitals."
            )

        key = self._key()
        if self._staged is not None and self._cache_key == key and not force:
            return self._staged

        staged = stage_inputs(
            self._obj,
            norb_frozen_core=(
                int(self.norb_frozen_core)
                if self.norb_frozen_core is not None
                else None
            ),
            chol_cut=self.chol_cut,
            cache=self.cache,
            overwrite=self.overwrite_cache if self.cache is not None else False,
            verbose=self._debug_prints_enabled(),
            log=log,
        )
        self._staged = staged
        self._cache_key = key
        self._job = None
        return staged

    def save_staged(self, path: Union[str, Path]) -> None:
        """Write current staged inputs to a single file cache."""
        staged = self.stage()
        dump_staged(staged, path)

    # def load_staged(self, path: Union[str, Path]): -> StagedInputs:
    #    """Load staged inputs from a cache file and attach them to this object."""
    #    staged = load_staged(path)
    #    self._staged = staged
    #    self._cache_key = None
    #    self._job = None
    #    return staged

    def _validate_params(self, params: QmcParamsBase) -> QmcParamsBase:
        return params

    def _make_params(self) -> QmcParamsBase:
        """
        Create QmcParams if user didn't provide one.
        """
        params_cls = self.params_cls

        if self.params is not None and isinstance(self.params, params_cls):
            params = self.params
        elif self.params is not None and not isinstance(self.params, params_cls):
            raise TypeError(
                f"Expected type {params_cls.__name__} for self.params, but received '{type(self.params)}'"
            )
        else:
            kwargs: dict[str, Any] = {}
            for field in dataclasses.fields(params_cls):
                if hasattr(self, field.name):
                    val = getattr(self, field.name)
                    if val is not None:
                        kwargs[field.name] = val

            params = params_cls(**kwargs)

        return self._validate_params(params)

    def build_job(
        self,
        *,
        force: bool = False,
        trial_data: Any = None,
        trial_ops: Any = None,
        meas_ops: Any = None,
        prop_ops: Any = None,
        block_fn: Callable[..., Any] | None = None,
        prop_kwargs: dict[str, Any] | None = None,
        mesh: Mesh | None = None,
        log: Any | None = None,
    ) -> Job:
        """
        Assemble a runnable Job from current settings and staged inputs.
        """
        if (
            self._job is not None
            and not force
            and (mesh is None or self._job.mesh is mesh)
        ):
            return self._job

        staged = self.stage(log=log)
        qmc_params = self._make_params()
        self.params = qmc_params

        job = self.setup_fn(
            staged,
            walker_kind=self.walker_kind,
            mesh=mesh,
            mixed_precision=self.mixed_precision,
            params=cast(Any, qmc_params),
            trial_data=trial_data,
            trial_ops=trial_ops,
            meas_ops=meas_ops,
            prop_ops=prop_ops,
            block_fn=block_fn,
            prop_kwargs=prop_kwargs,
            verbose=self._debug_prints_enabled(),
            log=log,
        )
        self._job = job
        return job

    def _coerce_result(self, value: Any) -> Any:
        return float(value)

    def kernel(self, **driver_kwargs: Any) -> tuple[Any, Any]:
        """
        Runs AFQMC, returns (e_tot, e_err), and stores samples.
        """
        log = logger.new_logger(self)
        log.info("%s", banner_afqmc().rstrip())
        dump_runtime_provenance(log)
        mesh = driver_kwargs.get("mesh")
        job = self.build_job(mesh=mesh, log=log)
        self.dump_flags(job, verbose=log.verbose)

        qmc_result = job.kernel(log=log, **driver_kwargs)

        e_tot = float(qmc_result.mean_energy)
        e_err = (
            float(qmc_result.stderr_energy)
            if qmc_result.stderr_energy is not None
            else float("nan")
        )

        self.qmc_result = qmc_result
        self.e_tot = e_tot
        self.e_err = e_err
        self.block_energies = qmc_result.block_energies
        self.block_weights = qmc_result.block_weights

        return e_tot, e_err

    run = kernel

    @classmethod
    def _from_staged_common(cls, path: Union[str, Path], **kwargs: Any):
        staged = load_staged(path)
        meta = staged.meta

        af = cls(
            None,
            norb_frozen_core=meta["frozen"],
            chol_cut=meta["chol_cut"],
            **kwargs,
        )
        af._staged = staged
        af.source_kind = meta["source_kind"]
        af._cache_key = af._key()
        return af

    @classmethod
    def from_staged(
        cls,
        path: Union[str, Path],
        *,
        n_eql_blocks: int | None = None,
        n_blocks: int | None = None,
        seed: int | None = None,
        dt: float | None = None,
        n_walkers: int | None = None,
        n_chunks: int = 1,
    ) -> Afqmc:
        """
        Returns a new AFQMC object from a previously staged calculations
        (using save_staged method). The number of frozen core orbitals, norb_frozen_core
        (legacy alias ``norb_frozen``),
        and the cholesky decomposition threshold, chol_cut, cannot be changed.
        Parameters
        ----------
        path: str, pathlib.Path
        The other parameters are identical to the ones in the AFQMC class.
        """
        return cls._from_staged_common(
            path,
            n_eql_blocks=n_eql_blocks,
            n_blocks=n_blocks,
            seed=seed,
            dt=dt,
            n_walkers=n_walkers,
            n_chunks=n_chunks,
        )


class AfqmcFp(Afqmc):
    params_cls = QmcParamsFp
    job_cls = JobFp
    setup_fn = staticmethod(setup_job_fp)

    def __init__(
        self,
        mf_or_cc: Any,
        *,
        norb_frozen_core: int | None = None,
        norb_frozen: int | None = None,
        chol_cut: float = 1e-5,
        cache: Union[str, Path] | None = None,
        n_blocks: int | None = None,
        seed: int | None = None,
        dt: float | None = None,
        n_prop_steps: int | None = None,
        n_walkers: int | None = None,
        n_chunks: int = 1,
        ene0: float | None = None,
        n_traj: int | None = None,
    ):
        super().__init__(
            mf_or_cc,
            norb_frozen_core=norb_frozen_core,
            norb_frozen=norb_frozen,
            chol_cut=chol_cut,
            cache=cache,
            n_eql_blocks=None,
            n_blocks=n_blocks,
            seed=seed,
            dt=dt,
            n_walkers=n_walkers,
            n_chunks=n_chunks,
        )
        defaults = self.params_cls()
        self.n_prop_steps = (
            defaults.n_prop_steps if n_prop_steps is None else n_prop_steps
        )
        self.n_traj = defaults.n_traj if n_traj is None else n_traj
        self.ene0 = ene0

    def _validate_params(self, params: QmcParamsBase) -> QmcParamsBase:
        assert isinstance(params, QmcParamsFp)
        if params.ene0 is None:
            raise ValueError(
                "The value of the parameter 'ene0' must be set, typically with SCF or CC energy."
            )
        return params

    def _coerce_result(self, value: Any) -> Any:
        return value

    def kernel(self, **driver_kwargs: Any) -> tuple[Any, Any]:
        """
        Runs AFQMC, returns (e_tot, e_err), and stores samples.
        """
        log = logger.new_logger(self)
        log.info("%s", banner_afqmc().rstrip())
        dump_runtime_provenance(log)
        mesh = driver_kwargs.get("mesh")
        job = self.build_job(mesh=mesh, log=log)
        self.dump_flags(job, verbose=log.verbose)

        qmc_result = job.kernel(log=log, **driver_kwargs)

        e_tot = qmc_result.mean_energy
        e_err = qmc_result.stderr_energy

        self.qmc_result = qmc_result
        self.e_tot = e_tot
        self.e_err = e_err
        self.block_energies = qmc_result.block_energies
        self.block_weights = qmc_result.block_weights

        return e_tot, e_err

    run_fp = kernel

    @classmethod
    def from_staged(
        cls,
        path: Union[str, Path],
        *,
        n_blocks: int | None = None,
        seed: int | None = None,
        dt: float | None = None,
        n_prop_steps: int | None = None,
        n_walkers: int | None = None,
        n_chunks: int = 1,
    ) -> AfqmcFp:
        """
        Returns a new AFQMC object from a previously staged calculations
        (using save_staged method). The number of frozen core orbitals, norb_frozen_core
        (legacy alias ``norb_frozen``),
        and the choliesky decomposition threshold, chol_cut, cannot be changed.
        Parameters
        ----------
        path: str, pathlib.Path
        The other parameters are identical to the ones in the AFQMC class.
        """
        return cls._from_staged_common(
            path,
            n_blocks=n_blocks,
            seed=seed,
            dt=dt,
            n_prop_steps=n_prop_steps,
            n_walkers=n_walkers,
            n_chunks=n_chunks,
        )


class AfqmcLnoFrag(Afqmc):
    params_cls = QmcParamsLno

    def __init__(
        self,
        mf_or_cc: Any,
        *,
        frozen_orbitals: ArrayLike | None = None,
        chol_cut: float = 1e-5,
        cache: Union[str, Path] | None = None,
        n_eql_blocks: int | None = None,
        n_blocks: int | None = None,
        seed: int | None = None,
        dt: float | None = None,
        n_walkers: int | None = None,
        n_chunks: int | None = None,
        prjlo: NDArray | None = None,
    ):
        super().__init__(
            mf_or_cc,
            norb_frozen_core=0,
            chol_cut=chol_cut,
            cache=cache,
            n_eql_blocks=n_eql_blocks,
            n_blocks=n_blocks,
            seed=seed,
            dt=dt,
            n_walkers=n_walkers,
            n_chunks=n_chunks,
        )

        self.mixed_precision = False
        self.prjlo = prjlo
        self.frozen_orbitals = frozen_orbitals

    def stage(self, *, force: bool = False, log: Any | None = None) -> StagedInputs:
        """
        Compute or load HamInput/TrialInput.
        If cache is set and exists, loads unless overwrite_cache=True.
        """
        key = self._key()
        if self._staged is not None and self._cache_key == key and not force:
            return self._staged

        frozen_orbitals = self.frozen_orbitals
        if frozen_orbitals is None:
            frozen_orbitals = np.zeros((0,), dtype=np.int64)

        ham = staging.build_ham_lno(
            self._obj,
            frozen_orbitals=frozen_orbitals,
            chol_cut=self.chol_cut,
        )

        staged = stage_inputs(
            self._obj,
            frozen_orbitals=frozen_orbitals,
            chol_cut=self.chol_cut,
            cache=self.cache,
            overwrite=self.overwrite_cache if self.cache is not None else False,
            verbose=self._debug_prints_enabled(),
            log=log,
            ham=ham,
            trial=None,
        )

        self._staged = staged
        self._cache_key = key
        self._job = None
        return staged

    def _key(self) -> tuple:
        base = super()._key()
        return base + (_frozen_cache_key(self.frozen_orbitals),)

    @classmethod
    def from_staged(
        cls,
        path: Union[str, Path],
        *,
        n_eql_blocks: int | None = None,
        n_blocks: int | None = None,
        seed: int | None = None,
        dt: float | None = None,
        n_walkers: int | None = None,
        n_chunks: int = 1,
        prjlo: NDArray | None = None,
    ) -> "AfqmcLnoFrag":
        staged = load_staged(path)
        meta = staged.meta
        frozen_orbitals = meta["frozen"]
        if frozen_orbitals is not None and not isinstance(frozen_orbitals, np.ndarray):
            frozen_orbitals = np.asarray(frozen_orbitals, dtype=np.int64)

        af = cls(
            None,
            frozen_orbitals=frozen_orbitals,
            chol_cut=meta["chol_cut"],
            n_eql_blocks=n_eql_blocks,
            n_blocks=n_blocks,
            seed=seed,
            dt=dt,
            n_walkers=n_walkers,
            n_chunks=n_chunks,
            prjlo=prjlo,
        )
        af._staged = staged
        af.source_kind = meta["source_kind"]
        af._cache_key = af._key()
        return af

    def build_job(
        self,
        *,
        force: bool = False,
        trial_data: Any = None,
        trial_ops: Any = None,
        meas_ops: Any = None,
        prop_ops: Any = None,
        block_fn: Callable[..., Any] | None = None,
        prop_kwargs: dict[str, Any] | None = None,
        log: Any | None = None,
    ) -> Job:
        """
        Assemble a runnable Job from current settings and staged inputs.
        """
        from .core.system import System
        from .meas.rhf import make_lno_rhf_meas_ops

        if self._job is not None and not force:
            return self._job

        if meas_ops is not None:
            raise ValueError("meas_ops must be None as we overwrite it.")

        staged = self.stage(log=log)
        params = self._make_params()
        assert isinstance(params, QmcParamsLno)
        self.params = params

        ham = staged.ham
        walker_kind = ham.basis
        sys = System(norb=int(ham.norb), nelec=ham.nelec, walker_kind=walker_kind)
        meas_ops = make_lno_rhf_meas_ops(sys=sys, params=params)

        job = setup_job(
            staged,
            walker_kind=self.walker_kind,
            mixed_precision=self.mixed_precision,
            params=params,
            trial_data=trial_data,
            trial_ops=trial_ops,
            meas_ops=meas_ops,
            prop_ops=prop_ops,
            block_fn=block_fn,
            prop_kwargs=prop_kwargs,
            verbose=self._debug_prints_enabled(),
            log=log,
        )
        self._job = job
        return job

    def kernel(self, **driver_kwargs: Any) -> tuple[NDArray, NDArray]:
        """
        Runs AFQMC, returns (e_tot, e_err), and stores samples.
        """
        log = logger.new_logger(self)
        log.info("%s", banner_afqmc().rstrip())
        job = self.build_job(log=log)
        self.dump_flags(job, verbose=log.verbose)

        obs = driver_kwargs.get("observable_names", ())
        if "orb_corr" not in obs:
            driver_kwargs["observable_names"] = obs + ("orb_corr",)

        qmc_result = job.kernel(log=log, **driver_kwargs)

        if not isinstance(qmc_result, QmcResult):
            raise TypeError(
                f"Unexpected return from Job.kernel(), expected QmcResult but received {type(qmc_result)}."
            )

        orb_corr = np.array(qmc_result.observable_means["orb_corr"].real)
        orb_corr_stderr = np.array(qmc_result.observable_stderrs["orb_corr"])

        self.qmc_result = qmc_result

        return orb_corr, orb_corr_stderr


def run_afqmc_lno_helper(
    mf: Any,
    norb_act=None,
    nelec_act=None,
    mo_coeff=None,
    frozen_orbitals: ArrayLike | None = None,
    chol_cut: float = 1e-5,
    seed: int | None = None,
    dt: float = 0.005,
    n_walkers: int = 5,
    nblocks: int = 1000,
    target_error: float = 1e-4,
    prjlo: NDArray | None = None,
    n_eql: int = 2,
):
    from pyscf import scf

    # choose the orbital basis
    if mo_coeff is None:
        if isinstance(mf, scf.uhf.UHF):
            mo_coeff = mf.mo_coeff[0]
        elif isinstance(mf, scf.rhf.RHF):
            mo_coeff = mf.mo_coeff
        else:
            raise Exception("# Invalid mean field object!")

    mf2 = copy.deepcopy(mf)
    mf2.mo_coeff = mo_coeff

    myafqmc = AfqmcLnoFrag(
        mf2,
        frozen_orbitals=frozen_orbitals,
        chol_cut=chol_cut,
        n_eql_blocks=n_eql,
        n_blocks=nblocks,
        seed=seed,
        dt=dt,
        n_walkers=n_walkers,
        prjlo=prjlo,
    )
    mean_ecorr, err_ecorr = myafqmc.kernel(target_error=target_error)

    return mean_ecorr, err_ecorr


# Backward-compatible aliases
AFQMC = Afqmc
AFQMCFP = AfqmcFp
AFQMCFp = AfqmcFp
