from __future__ import annotations

import time
from dataclasses import dataclass
from functools import partial
from typing import Any, ClassVar, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from .core.ops import MeasOps, TrialOps, k_energy
from .core.system import System
from .ham.chol import HamChol
from .meas.cisd import CisdMeasCfg, CisdMeasCtx, get_cisd_meas_cfg
from .meas.rhf import RhfMeasCfg, RhfMeasCtx, get_rhf_meas_cfg
from .prop.chol_afqmc_ops import CholAfqmcCtx
from .prop.types import PropOps, PropState, QmcParams, QmcParamsBase
from .sharding import has_model_axis, replicate, shard_model_axis
from .staging import HamInput, StagedInputs
from .trial.cisd import CisdTrial
from .trial.rhf import RhfTrial

print = partial(print, flush=True)

_HOST_CHOL_BLOCK_SIZE = 16


def _setup_begin(message: str) -> float:
    print(f"[setup] {message}...")
    return time.time()


def _setup_end(start: float, message: str, *, details: str | None = None) -> None:
    suffix = f" | {details}" if details else ""
    print(f"[setup] {message} in {time.time() - start:.2f}s{suffix}")


@dataclass(frozen=True)
class PreparedRuntime:
    ham_data: HamChol
    state: PropState
    meas_ctx: object
    prop_ctx: object


class RuntimeJob(Protocol):
    staged: StagedInputs
    sys: System
    params: QmcParamsBase
    params_cls: ClassVar[type[QmcParamsBase]]
    ham_data: HamChol
    trial_data: object
    trial_ops: TrialOps
    meas_ops: MeasOps
    prop_ops: PropOps
    mesh: Mesh | None


class RuntimeLayout(Protocol):
    def make_initial_ham_data(self, ham: HamInput | HamChol, mesh: Mesh | None) -> HamChol: ...

    def prepare(
        self,
        job: RuntimeJob,
        *,
        state: PropState | None = None,
        meas_ctx: object | None = None,
        prop_ctx: object | None = None,
    ) -> PreparedRuntime: ...


def _model_axis_size(mesh: Mesh) -> int:
    return int(dict(zip(mesh.axis_names, mesh.devices.shape, strict=True))["model"])


def _padded_model_length(length: int, mesh: Mesh | None) -> int:
    if mesh is None or mesh.size <= 1 or not has_model_axis(mesh):
        return length

    n_model = _model_axis_size(mesh)
    remainder = length % n_model
    if remainder == 0:
        return length
    return length + (n_model - remainder)


def _make_ham_data(ham: HamInput | HamChol, mesh: Mesh | None, *, compact_chol: bool) -> HamChol:
    chol = ham.chol
    runtime_n_chol = _padded_model_length(int(chol.shape[0]), mesh)
    if compact_chol:
        n_chol = int(chol.shape[0])
        if runtime_n_chol != n_chol:
            assert mesh is not None
            print(
                f"[shard] padding chol from {n_chol} to {runtime_n_chol} "
                f"to shard evenly over n_model={_model_axis_size(mesh)}.",
                flush=True,
            )
        chol = np.zeros((0, 0, 0), dtype=np.asarray(chol).dtype)

    if mesh is not None and mesh.size > 1 and has_model_axis(mesh):
        return HamChol(
            replicate(jnp.asarray(ham.h0), mesh),
            replicate(ham.h1, mesh),
            shard_model_axis(chol, mesh),
            basis=ham.basis,
            nchol=runtime_n_chol if compact_chol else None,
        )

    return HamChol(
        jnp.asarray(ham.h0),
        jnp.asarray(ham.h1),
        jnp.asarray(chol),
        basis=ham.basis,
        nchol=runtime_n_chol,
    )


def _as_host_array(x: Any) -> np.ndarray:
    return np.asarray(jax.device_get(x))


def _build_restricted_prop_ctx_from_host(
    staged: StagedInputs,
    *,
    trial_rdm1: jax.Array,
    dt: float,
    mixed_precision: bool,
    mesh: Mesh | None,
) -> CholAfqmcCtx:
    from .prop.chol_afqmc_ops import CholAfqmcCtx

    ham = staged.ham
    chol = np.asarray(ham.chol)
    h1 = np.asarray(ham.h1)
    dm = _as_host_array(trial_rdm1)
    if dm.ndim == 3 and dm.shape[0] == 2:
        dm = dm[0] + dm[1]

    n_chol = int(chol.shape[0])
    norb = int(chol.shape[1])
    mf = np.empty(n_chol, dtype=np.result_type(chol.dtype, dm.dtype, np.complex128))
    v0m = np.zeros((norb, norb), dtype=np.result_type(h1.dtype, chol.dtype))
    v1m = np.zeros((norb, norb), dtype=np.result_type(h1.dtype, chol.dtype))

    for start in range(0, n_chol, _HOST_CHOL_BLOCK_SIZE):
        stop = min(start + _HOST_CHOL_BLOCK_SIZE, n_chol)
        chol_blk = chol[start:stop]
        mf_blk = 1.0j * np.einsum("gij,ji->g", chol_blk, dm, optimize="optimal")
        mf[start:stop] = mf_blk
        v0m += 0.5 * np.einsum("gik,gkj->ij", chol_blk, chol_blk, optimize="optimal")
        v1m += np.einsum(
            "g,gik->ik",
            np.real(1.0j * mf_blk),
            chol_blk,
            optimize="optimal",
        )

    h0_prop = -ham.h0 - 0.5 * np.sum(mf**2)
    h1_eff = h1 - v0m - v1m
    exp_h1_half_np = np.asarray(
        jax.device_get(jax.scipy.linalg.expm(-0.5 * jnp.asarray(dt) * jnp.asarray(h1_eff)))
    )
    chol_flat = chol.reshape(n_chol, -1)
    chol_flat_dtype = np.float32 if mixed_precision else chol_flat.dtype

    if mesh is not None and mesh.size > 1 and has_model_axis(mesh):
        dt_a = replicate(np.asarray(dt), mesh)
        sqrt_dt = replicate(np.asarray(np.sqrt(dt)), mesh)
        exp_h1_half = replicate(exp_h1_half_np, mesh)
        mf_shifts = shard_model_axis(mf, mesh, announce_padding=False)
        chol_flat_a = shard_model_axis(
            chol_flat,
            mesh,
            dtype=chol_flat_dtype,
            announce_padding=False,
        )
        h0_prop_a = replicate(np.asarray(h0_prop), mesh)
    else:
        dt_a = jnp.asarray(dt)
        sqrt_dt = jnp.asarray(np.sqrt(dt))
        exp_h1_half = jnp.asarray(exp_h1_half_np)
        mf_shifts = jnp.asarray(mf)
        chol_flat_a = jnp.asarray(chol_flat, dtype=chol_flat_dtype)
        h0_prop_a = jnp.asarray(h0_prop)

    return CholAfqmcCtx(
        dt=dt_a,
        sqrt_dt=sqrt_dt,
        exp_h1_half=exp_h1_half,
        mf_shifts=mf_shifts,
        h0_prop=h0_prop_a,
        chol_flat=chol_flat_a,
        norb=norb,
    )


def _build_rhf_meas_ctx_from_host(
    staged: StagedInputs,
    trial_data: RhfTrial,
    *,
    cfg: RhfMeasCfg,
    mesh: Mesh | None,
) -> RhfMeasCtx:
    from .meas.rhf import RhfMeasCtx

    ham = staged.ham
    chol = np.asarray(ham.chol)
    h1 = np.asarray(ham.h1)
    mo_coeff = _as_host_array(trial_data.mo_coeff)
    c_h = mo_coeff.conj().T

    rot_h1 = c_h @ h1
    n_chol = int(chol.shape[0])
    rot_chol = np.empty(
        (n_chol, c_h.shape[0], chol.shape[1]),
        dtype=np.result_type(c_h.dtype, chol.dtype),
    )
    for start in range(0, n_chol, _HOST_CHOL_BLOCK_SIZE):
        stop = min(start + _HOST_CHOL_BLOCK_SIZE, n_chol)
        rot_chol[start:stop] = np.einsum(
            "pi,gij->gpj",
            c_h,
            chol[start:stop],
            optimize="optimal",
        )
    rot_chol_flat = rot_chol.reshape(n_chol, -1)

    if mesh is not None and mesh.size > 1 and has_model_axis(mesh):
        rot_h1_a = replicate(rot_h1, mesh)
        rot_chol_a = shard_model_axis(rot_chol, mesh, announce_padding=False)
        rot_chol_flat_a = shard_model_axis(rot_chol_flat, mesh, announce_padding=False)
    else:
        rot_h1_a = jnp.asarray(rot_h1)
        rot_chol_a = jnp.asarray(rot_chol)
        rot_chol_flat_a = jnp.asarray(rot_chol_flat)

    return RhfMeasCtx(
        rot_h1=rot_h1_a,
        rot_chol=rot_chol_a,
        rot_chol_flat=rot_chol_flat_a,
        cfg=cfg,
    )


def _build_cisd_meas_ctx_from_host(
    staged: StagedInputs,
    trial_data: CisdTrial,
    *,
    cfg: CisdMeasCfg,
    mesh: Mesh | None,
) -> CisdMeasCtx:
    ham = staged.ham
    chol = np.asarray(ham.chol)
    ci1 = _as_host_array(trial_data.ci1)

    n_chol = int(chol.shape[0])
    norb = int(chol.shape[1])
    nocc_full = int(trial_data.nocc_full)
    nocc_act = int(trial_data.nocc)

    rot_chol = np.empty(
        (n_chol, nocc_full, norb),
        dtype=np.result_type(chol.dtype),
    )
    lci1 = np.empty(
        (n_chol, norb, nocc_act),
        dtype=np.result_type(chol.dtype, ci1.dtype),
    )
    vir_act_slice = trial_data.vir_act_slice

    for start in range(0, n_chol, _HOST_CHOL_BLOCK_SIZE):
        stop = min(start + _HOST_CHOL_BLOCK_SIZE, n_chol)
        chol_blk = chol[start:stop]
        rot_chol[start:stop] = chol_blk[:, :nocc_full, :]
        lci1[start:stop] = np.einsum(
            "git,pt->gip",
            chol_blk[:, :, vir_act_slice],
            ci1,
            optimize="optimal",
        )

    if mesh is not None and mesh.size > 1 and has_model_axis(mesh):
        rot_chol_a = shard_model_axis(rot_chol, mesh, announce_padding=False)
        lci1_a = shard_model_axis(lci1, mesh, announce_padding=False)
    else:
        rot_chol_a = jnp.asarray(rot_chol)
        lci1_a = jnp.asarray(lci1)

    return CisdMeasCtx(rot_chol=rot_chol_a, lci1=lci1_a, cfg=cfg)


def _init_state_from_prebuilt_ctx(
    job: RuntimeJob,
    *,
    trial_data: object,
    trial_rdm1: jax.Array,
    ham_data_runtime: HamChol,
    meas_ctx: object,
) -> PropState:
    from . import walkers as wk

    initial_walkers = wk.init_walkers(
        sys=job.sys,
        rdm1=trial_rdm1,
        n_walkers=job.params.n_walkers,
    )
    e_kernel = job.meas_ops.require_kernel(k_energy)
    walker_0 = wk.take_walkers(initial_walkers, jnp.array([0]))
    e_samples = jnp.real(
        wk.vmap_chunked(e_kernel, n_chunks=1, in_axes=(0, None, None, None))(
            walker_0,
            ham_data_runtime,
            meas_ctx,
            trial_data,
        )
    )
    return job.prop_ops.init_prop_state(
        sys=job.sys,
        ham_data=ham_data_runtime,
        trial_ops=job.trial_ops,
        trial_data=trial_data,
        meas_ops=job.meas_ops,
        params=job.params,
        mesh=job.mesh,
        initial_walkers=initial_walkers,
        initial_e_estimate=jnp.mean(e_samples),
        rdm1=trial_rdm1,
    )


def _compact_ham_data_for_runtime(ham_data: Any, meas_ctx: Any) -> Any:
    if not isinstance(ham_data, HamChol):
        return ham_data

    from .meas.ghf import GhfCholMeasCtx
    from .meas.rhf import RhfMeasCtx
    from .meas.uhf import UhfMeasCtx

    if isinstance(meas_ctx, (RhfMeasCtx, UhfMeasCtx, GhfCholMeasCtx)):
        chol = ham_data.chol
        if isinstance(chol, jax.Array):
            compact_chol = jax.device_put(
                jnp.zeros((0, 0, 0), dtype=chol.dtype),
                chol.sharding,
            )
        else:
            compact_chol = jnp.asarray(np.zeros((0, 0, 0), dtype=np.asarray(chol).dtype))
        return HamChol(
            h0=ham_data.h0,
            h1=ham_data.h1,
            chol=compact_chol,
            basis=ham_data.basis,
            nchol=ham_data.nchol,
        )

    return ham_data


@dataclass(frozen=True)
class DefaultRuntimeLayout:
    def make_initial_ham_data(self, ham: HamInput | HamChol, mesh: Mesh | None) -> HamChol:
        return _make_ham_data(ham, mesh, compact_chol=False)

    def prepare(
        self,
        job: RuntimeJob,
        *,
        state: PropState | None = None,
        meas_ctx: object | None = None,
        prop_ctx: object | None = None,
    ) -> PreparedRuntime:
        ham_data_runtime = job.ham_data

        if prop_ctx is None:
            t_prop = _setup_begin("building propagation context")
            prop_ctx = job.prop_ops.build_prop_ctx(
                ham_data_runtime,
                job.trial_ops.get_rdm1(job.trial_data),
                job.params,
            )
            _setup_end(t_prop, "propagation context ready")
        if meas_ctx is None:
            t_meas = _setup_begin("building measurement context")
            meas_ctx = job.meas_ops.build_meas_ctx(ham_data_runtime, job.trial_data)
            _setup_end(t_meas, "measurement context ready")

        if state is None:
            t_state = _setup_begin("initializing propagation state")
            state = job.prop_ops.init_prop_state(
                sys=job.sys,
                ham_data=ham_data_runtime,
                trial_ops=job.trial_ops,
                trial_data=job.trial_data,
                meas_ops=job.meas_ops,
                params=job.params,
                mesh=job.mesh,
            )
            _setup_end(t_state, "propagation state ready")

        if job.params_cls is QmcParams:
            ham_data_runtime = _compact_ham_data_for_runtime(ham_data_runtime, meas_ctx)

        return PreparedRuntime(
            ham_data=ham_data_runtime,
            state=state,
            meas_ctx=meas_ctx,
            prop_ctx=prop_ctx,
        )


@dataclass(frozen=True)
class RhfHostRuntimeLayout:
    mixed_precision: bool = True
    rhf_meas_cfg: RhfMeasCfg = RhfMeasCfg()

    def make_initial_ham_data(self, ham: HamInput | HamChol, mesh: Mesh | None) -> HamChol:
        return _make_ham_data(ham, mesh, compact_chol=True)

    def prepare(
        self,
        job: RuntimeJob,
        *,
        state: PropState | None = None,
        meas_ctx: object | None = None,
        prop_ctx: object | None = None,
    ) -> PreparedRuntime:
        from .trial.rhf import RhfTrial

        ham_data_runtime = job.ham_data
        trial_data = job.trial_data
        assert isinstance(trial_data, RhfTrial)
        trial_rdm1 = job.trial_ops.get_rdm1(trial_data)

        if prop_ctx is None:
            t_prop = _setup_begin("building propagation context")
            prop_ctx = _build_restricted_prop_ctx_from_host(
                job.staged,
                trial_rdm1=trial_rdm1,
                dt=job.params.dt,
                mixed_precision=self.mixed_precision,
                mesh=job.mesh,
            )
            _setup_end(t_prop, "propagation context ready")
        if meas_ctx is None:
            t_meas = _setup_begin("building measurement context")
            meas_ctx = _build_rhf_meas_ctx_from_host(
                job.staged,
                trial_data,
                cfg=self.rhf_meas_cfg,
                mesh=job.mesh,
            )
            _setup_end(t_meas, "measurement context ready")

        if state is None:
            t_state = _setup_begin("initializing propagation state")
            state = _init_state_from_prebuilt_ctx(
                job,
                trial_data=trial_data,
                trial_rdm1=trial_rdm1,
                ham_data_runtime=ham_data_runtime,
                meas_ctx=meas_ctx,
            )
            _setup_end(t_state, "propagation state ready")

        return PreparedRuntime(
            ham_data=ham_data_runtime,
            state=state,
            meas_ctx=meas_ctx,
            prop_ctx=prop_ctx,
        )


@dataclass(frozen=True)
class CisdHostRuntimeLayout:
    mixed_precision: bool = True

    def make_initial_ham_data(self, ham: HamInput | HamChol, mesh: Mesh | None) -> HamChol:
        return _make_ham_data(ham, mesh, compact_chol=False)

    def prepare(
        self,
        job: RuntimeJob,
        *,
        state: PropState | None = None,
        meas_ctx: object | None = None,
        prop_ctx: object | None = None,
    ) -> PreparedRuntime:
        ham_data_runtime = job.ham_data
        trial_data = job.trial_data
        assert isinstance(trial_data, CisdTrial)
        trial_rdm1 = job.trial_ops.get_rdm1(trial_data)

        if prop_ctx is None:
            t_prop = _setup_begin("building propagation context")
            prop_ctx = _build_restricted_prop_ctx_from_host(
                job.staged,
                trial_rdm1=trial_rdm1,
                dt=job.params.dt,
                mixed_precision=self.mixed_precision,
                mesh=job.mesh,
            )
            _setup_end(t_prop, "propagation context ready")
        if meas_ctx is None:
            t_meas = _setup_begin("building measurement context")
            cfg = get_cisd_meas_cfg(job.meas_ops)
            if cfg is None:
                raise TypeError("CISD host runtime layout requires CisdMeasCfg-aware MeasOps.")
            meas_ctx = _build_cisd_meas_ctx_from_host(
                job.staged,
                trial_data,
                cfg=cfg,
                mesh=job.mesh,
            )
            _setup_end(t_meas, "measurement context ready")

        if state is None:
            t_state = _setup_begin("initializing propagation state")
            state = _init_state_from_prebuilt_ctx(
                job,
                trial_data=trial_data,
                trial_rdm1=trial_rdm1,
                ham_data_runtime=ham_data_runtime,
                meas_ctx=meas_ctx,
            )
            _setup_end(t_state, "propagation state ready")

        return PreparedRuntime(
            ham_data=ham_data_runtime,
            state=state,
            meas_ctx=meas_ctx,
            prop_ctx=prop_ctx,
        )


def make_runtime_layout(
    *,
    staged: StagedInputs,
    allow_host_rhf: bool,
    trial_data_override: Any,
    trial_ops_override: Any,
    meas_ops_override: Any,
    prop_ops_override: Any,
    mixed_precision: bool,
) -> RuntimeLayout:
    rhf_meas_cfg = None
    if isinstance(meas_ops_override, MeasOps):
        rhf_meas_cfg = get_rhf_meas_cfg(meas_ops_override)
    use_host_rhf = (
        allow_host_rhf
        and staged.trial.kind.lower() == "rhf"
        and staged.ham.basis == "restricted"
        and trial_data_override is None
        and trial_ops_override is None
        and (meas_ops_override is None or rhf_meas_cfg is not None)
        and prop_ops_override is None
    )
    if use_host_rhf:
        return RhfHostRuntimeLayout(
            mixed_precision=mixed_precision,
            rhf_meas_cfg=rhf_meas_cfg or RhfMeasCfg(),
        )

    cisd_meas_cfg = None
    if isinstance(meas_ops_override, MeasOps):
        cisd_meas_cfg = get_cisd_meas_cfg(meas_ops_override)
    use_host_cisd = (
        allow_host_rhf
        and staged.trial.kind.lower() == "cisd"
        and staged.ham.basis == "restricted"
        and trial_data_override is None
        and trial_ops_override is None
        and (meas_ops_override is None or cisd_meas_cfg is not None)
        and prop_ops_override is None
    )
    if use_host_cisd:
        return CisdHostRuntimeLayout(mixed_precision=mixed_precision)
    return DefaultRuntimeLayout()
