from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Tuple, TypeAlias, Union, cast

import h5py
import numpy as np
from numpy.typing import ArrayLike, NDArray

print = partial(print, flush=True)

from .ham.chol import HamBasis

# This file contains staging utilities to convert pyscf mf/cc objects
# into serializable data classes representing Hamiltonian and trial
# wavefunction inputs which can be used for building AFQMC objects.

Array: TypeAlias = NDArray[Any]

# to keep track of format versions when loading/saving staged inputs
STAGE_FORMAT_VERSION = 1


def _stage_begin(message: str, *, log: Any | None = None) -> float:
    if log is None:
        print(f"[stage] {message}...")
    else:
        log.info("[stage] %s...", message)
    return time.time()


def _stage_end(
    start: float,
    message: str,
    *,
    details: str | None = None,
    log: Any | None = None,
) -> None:
    suffix = f" | {details}" if details else ""
    text = f"[stage] {message} in {time.time() - start:.2f}s{suffix}"
    if log is None:
        print(text)
    else:
        log.info("%s", text)


def _normalize_frozen_list(frozen: Any, *, nmo: int) -> NDArray:
    frozen_arr = np.asarray(frozen, dtype=np.int64)
    if frozen_arr.ndim != 1:
        raise ValueError("cc.frozen must be a one-dimensional list of MO indices.")
    if frozen_arr.size == 0:
        return frozen_arr

    frozen_arr = np.sort(frozen_arr)
    if np.unique(frozen_arr).size != frozen_arr.size:
        raise ValueError("cc.frozen contains duplicate MO indices.")
    if frozen_arr[0] < 0 or frozen_arr[-1] >= nmo:
        raise ValueError(f"cc.frozen indices must lie in [0, {nmo}).")

    return frozen_arr


def _mo_coeff_signature(mo_coeff: Any) -> tuple[tuple[int, ...], ...]:
    if isinstance(mo_coeff, (tuple, list)):
        return tuple(tuple(int(dim) for dim in np.asarray(block).shape) for block in mo_coeff)
    return (tuple(int(dim) for dim in np.asarray(mo_coeff).shape),)


def _copy_scf_with_cc_mo_coeff(cc: Any, mf: Any) -> Any:
    if not hasattr(cc, "mo_coeff") or cc.mo_coeff is None:
        return mf

    cc_sig = _mo_coeff_signature(cc.mo_coeff)
    mf_sig = _mo_coeff_signature(mf.mo_coeff)
    if cc_sig != mf_sig:
        raise ValueError(
            "CC object mo_coeff shape does not match the underlying SCF mo_coeff shape: "
            f"{cc_sig} != {mf_sig}."
        )

    mf_copy = copy.copy(mf)
    mf_copy.mo_coeff = cc.mo_coeff
    return mf_copy


def _infer_restricted_trial_freeze_from_cc(
    *,
    cc_frozen: Any,
    nmo_full: int,
    nocc_full: int,
    norb_frozen: int,
    t1_shape: tuple[int, int],
) -> tuple[int, int]:
    frozen = _normalize_frozen_list(cc_frozen, nmo=nmo_full)
    occ_frozen = frozen[frozen < nocc_full]
    vir_frozen = frozen[frozen >= nocc_full]

    nocc_cc_frozen = int(occ_frozen.size)
    nvir_cc_frozen = int(vir_frozen.size)

    if occ_frozen.size and not np.array_equal(
        occ_frozen, np.arange(nocc_cc_frozen, dtype=np.int64)
    ):
        raise ValueError(
            "Occupied orbitals in list-valued cc.frozen must form a contiguous prefix."
        )

    vir_expected = np.arange(nmo_full - nvir_cc_frozen, nmo_full, dtype=np.int64)
    if vir_frozen.size and not np.array_equal(vir_frozen, vir_expected):
        raise ValueError("Virtual orbitals in list-valued cc.frozen must form a contiguous suffix.")

    if norb_frozen > nocc_cc_frozen:
        raise ValueError("norb_frozen cannot exceed the number of occupied orbitals frozen in CC.")

    nocc_act_expected = nocc_full - nocc_cc_frozen
    nvir_act_expected = (nmo_full - nocc_full) - nvir_cc_frozen
    if t1_shape != (nocc_act_expected, nvir_act_expected):
        raise ValueError(
            "cc.frozen is inconsistent with the CC amplitudes in the restricted CISD trial."
        )

    nocc_t_core = nocc_cc_frozen - norb_frozen
    nvir_t_outer = nvir_cc_frozen
    return nocc_t_core, nvir_t_outer


def modified_cholesky(
    mat: Array,
    max_error: float = 1e-6,
) -> Array:
    """Modified cholesky decomposition for a given matrix.

    Args:
        mat (Array): Matrix to decompose.
        max_error (float, optional): Maximum error allowed. Defaults to 1e-6.

    Returns:
        Array: Cholesky vectors.
    """
    diag = mat.diagonal()
    norb = int(((-1 + (1 + 8 * mat.shape[0]) ** 0.5) / 2))
    size = mat.shape[0]
    nchol_max = size
    chol_vecs = np.zeros((nchol_max, nchol_max))
    # ndiag = 0
    nu = np.argmax(diag)
    delta_max = diag[nu]
    Mapprox = np.zeros(size)
    chol_vecs[0] = np.copy(mat[nu]) / delta_max**0.5

    nchol = 0
    while abs(delta_max) > max_error and (nchol + 1) < nchol_max:
        Mapprox += chol_vecs[nchol] * chol_vecs[nchol]
        delta = diag - Mapprox
        nu = np.argmax(np.abs(delta))
        delta_max = np.abs(delta[nu])
        R = np.dot(chol_vecs[: nchol + 1, nu], chol_vecs[: nchol + 1, :])
        chol_vecs[nchol + 1] = (mat[nu] - R) / (delta_max + 1e-10) ** 0.5
        nchol += 1

    chol0 = chol_vecs[:nchol]
    nchol = chol0.shape[0]
    chol = np.zeros((nchol, norb, norb))
    for i in range(nchol):
        for m in range(norb):
            for n in range(m + 1):
                triind = m * (m + 1) // 2 + n
                chol[i, m, n] = chol0[i, triind]
                chol[i, n, m] = chol0[i, triind]
    return chol


def chunked_cholesky(mol, max_error=1e-6, verbose=False, cmax=10) -> NDArray:
    """Modified cholesky decomposition from pyscf eris.

    See, e.g. [Motta17]_

    Only works for molecular systems. (copied from pauxy)

    Parameters
    ----------
    mol : :class:`pyscf.mol`
        pyscf mol object.
    orthoAO: :class:`numpy.ndarray`
        Orthogonalising matrix for AOs. (e.g., mo_coeff).
    delta : float
        Accuracy desired.
    verbose : bool
        If true print out convergence progress.
    cmax : int
        nchol = cmax * M, where M is the number of basis functions.
        Controls buffer size for cholesky vectors.

    Returns
    -------
    chol_vecs : :class:`numpy.ndarray`
        Matrix of cholesky vectors in AO basis.
    """
    nao = mol.nao_nr()
    diag = np.zeros(nao * nao)
    nchol_max = cmax * nao
    chol_vecs = np.zeros((nchol_max, nao * nao))
    ndiag = 0
    dims = [0]
    nao_per_i = 0
    for i in range(0, mol.nbas):
        l = mol.bas_angular(i)
        nc = mol.bas_nctr(i)
        nao_per_i += (2 * l + 1) * nc
        dims.append(nao_per_i)
    # print (dims)
    for i in range(0, mol.nbas):
        shls = (i, i + 1, 0, mol.nbas, i, i + 1, 0, mol.nbas)
        buf = mol.intor("int2e_sph", shls_slice=shls)
        di, dk, dj, dl = buf.shape
        diag[ndiag : ndiag + di * nao] = buf.reshape(di * nao, di * nao).diagonal()
        ndiag += di * nao
    nu = np.argmax(diag)
    delta_max = diag[nu]
    if verbose:
        print("# Generating Cholesky decomposition of ERIs.")
        print("# max number of cholesky vectors = %d" % nchol_max)
        print("# iteration %5d: delta_max = %f" % (0, delta_max))
    j = nu // nao
    l = nu % nao
    sj = np.searchsorted(dims, j)
    sl = np.searchsorted(dims, l)
    if dims[sj] != j and j != 0:
        sj -= 1
    if dims[sl] != l and l != 0:
        sl -= 1
    Mapprox = np.zeros(nao * nao)
    # ERI[:,jl]
    eri_col = mol.intor("int2e_sph", shls_slice=(0, mol.nbas, 0, mol.nbas, sj, sj + 1, sl, sl + 1))
    cj, cl = max(j - dims[sj], 0), max(l - dims[sl], 0)
    chol_vecs[0] = np.copy(eri_col[:, :, cj, cl].reshape(nao * nao)) / delta_max**0.5

    nchol = 0
    while abs(delta_max) > max_error:
        # Update cholesky vector
        start = time.time()
        # M'_ii = L_i^x L_i^x
        Mapprox += chol_vecs[nchol] * chol_vecs[nchol]
        # D_ii = M_ii - M'_ii
        delta = diag - Mapprox
        nu = np.argmax(np.abs(delta))
        delta_max = np.abs(delta[nu])
        # Compute ERI chunk.
        # shls_slice computes shells of integrals as determined by the angular
        # momentum of the basis function and the number of contraction
        # coefficients. Need to search for AO index within this shell indexing
        # scheme.
        # AO index.
        j = nu // nao
        l = nu % nao
        # Associated shell index.
        sj = np.searchsorted(dims, j)
        sl = np.searchsorted(dims, l)
        if dims[sj] != j and j != 0:
            sj -= 1
        if dims[sl] != l and l != 0:
            sl -= 1
        # Compute ERI chunk.
        eri_col = mol.intor(
            "int2e_sph", shls_slice=(0, mol.nbas, 0, mol.nbas, sj, sj + 1, sl, sl + 1)
        )
        # Select correct ERI chunk from shell.
        cj, cl = max(j - dims[sj], 0), max(l - dims[sl], 0)
        Munu0 = eri_col[:, :, cj, cl].reshape(nao * nao)
        # Updated residual = \sum_x L_i^x L_nu^x
        R = np.dot(chol_vecs[: nchol + 1, nu], chol_vecs[: nchol + 1, :])
        chol_vecs[nchol + 1] = (Munu0 - R) / (delta_max) ** 0.5
        nchol += 1
        if verbose:
            step_time = time.time() - start
            info = (nchol, delta_max, step_time)
            print("# iteration %5d: delta_max = %13.8e: time = %13.8e" % info)

    return chol_vecs[:nchol]


def _rotate_chol_to_mo(chol_vec: Array, basis_coeff: Array) -> Array:
    """Rotate AO-space Cholesky into an MO basis."""
    C = np.asarray(basis_coeff)
    nao, norb = C.shape
    nchol = int(chol_vec.shape[0])
    out_dtype = np.result_type(chol_vec.dtype, C.dtype)

    reuse_storage = nao == norb and out_dtype == chol_vec.dtype
    if reuse_storage:
        chol = chol_vec.reshape(nchol, nao, nao)
    else:
        chol = np.empty((nchol, norb, norb), dtype=out_dtype)

    Cdag = np.asarray(C.conj().T)
    tmp = np.empty((nao, norb), dtype=out_dtype)
    for i in range(nchol):
        chol_i_ao = chol_vec[i].reshape(nao, nao)
        np.dot(chol_i_ao, C, out=tmp)
        np.dot(Cdag, tmp, out=chol[i])

    return chol


def _rotate_chol_to_ghf_mo(chol_vec: Array, basis_coeff: Array) -> Array:
    """Rotate spatial AO Cholesky factors into a generalized-spin MO basis."""
    C = np.asarray(basis_coeff)
    nao2, nmo = C.shape
    if nao2 % 2 != 0:
        raise ValueError(f"Expected even GHF AO dimension, got {nao2}")

    nao = nao2 // 2
    nchol = int(chol_vec.shape[0])
    out_dtype = np.result_type(chol_vec.dtype, C.dtype)
    chol = np.empty((nchol, nmo, nmo), dtype=out_dtype)

    Cdag = np.asarray(C.conj().T)
    chol_i_full = np.zeros((nao2, nao2), dtype=out_dtype)
    tmp = np.empty((nao2, nmo), dtype=out_dtype)
    for i in range(nchol):
        chol_i = chol_vec[i].reshape(nao, nao)
        chol_i_full.fill(0)
        chol_i_full[:nao, :nao] = chol_i
        chol_i_full[nao:, nao:] = chol_i
        np.dot(chol_i_full, C, out=tmp)
        np.dot(Cdag, tmp, out=chol[i])

    return chol


def _stage_frozen(frozen: int | ArrayLike | None) -> int | NDArray | None:
    if isinstance(frozen, (list, tuple, np.ndarray)):
        frozen = np.asarray(frozen, dtype=int)
    elif frozen is None:
        frozen = None
    elif isinstance(frozen, int):
        frozen = int(frozen)
    else:
        raise TypeError(f"Unsupported type '{type(frozen)}'.")

    return frozen


def _resolve_stage_frozen_arg(
    norb_frozen_core: int | None,
    norb_frozen: int | None,
    frozen_orbitals: ArrayLike | None,
) -> int | ArrayLike | None:
    if norb_frozen_core is not None and norb_frozen is not None and norb_frozen_core != norb_frozen:
        raise ValueError("norb_frozen_core and norb_frozen must match when both are passed.")
    core_frozen = norb_frozen_core if norb_frozen_core is not None else norb_frozen
    if core_frozen is not None and frozen_orbitals is not None:
        raise ValueError("Pass only one of norb_frozen_core/norb_frozen or frozen_orbitals.")
    if frozen_orbitals is not None:
        return frozen_orbitals
    return core_frozen


def _freeze_meta_value(frozen: int | NDArray | None) -> int | list[int] | None:
    if isinstance(frozen, np.ndarray):
        arr = np.asarray(frozen, dtype=np.int64).reshape(-1)
        return [int(x) for x in arr]
    if isinstance(frozen, int):
        return frozen
    if isinstance(frozen, np.integer):
        return int(frozen.item())
    if frozen is None:
        return None
    raise TypeError(f"Unsupported frozen metadata type: {type(frozen)}")


def _freeze_from_meta_value(frozen: Any) -> int | NDArray | None:
    if frozen is None:
        return None
    if isinstance(frozen, list):
        return np.asarray(frozen, dtype=np.int64)
    if isinstance(frozen, np.ndarray):
        return np.asarray(frozen, dtype=np.int64)
    if isinstance(frozen, (int, np.integer)):
        return int(frozen)
    raise TypeError(f"Unsupported frozen metadata type: {type(frozen)}")


def _dump_frozen(group: h5py.Group, frozen: int | NDArray, *, attr_name: str = "frozen") -> None:
    if isinstance(frozen, np.ndarray):
        group.create_dataset(attr_name, data=np.asarray(frozen, dtype=np.int64))
    else:
        group.attrs[attr_name] = int(frozen)


def _load_frozen(group: h5py.Group, *, attr_name: str = "frozen") -> int | NDArray:
    if attr_name in group:
        dataset = group[attr_name]
        if not isinstance(dataset, h5py.Dataset):
            raise TypeError(f"Expected dataset '{attr_name}', got {type(dataset)}")
        return np.asarray(dataset[...], dtype=np.int64)

    attr_value = group.attrs[attr_name]
    if isinstance(attr_value, np.ndarray):
        return int(np.asarray(attr_value).item())
    if isinstance(attr_value, (int, np.integer)):
        return int(attr_value)
    raise TypeError(f"Unsupported frozen attribute type: {type(attr_value)}")


@dataclass(frozen=True, slots=True)
class HamInput:
    """ham inputs in the chosen orthonormal one particle basis"""

    h0: float
    h1: Array  # (norb, norb)
    chol: Array  # (nchol, norb, norb)
    nelec: Tuple[int, int]
    norb: int
    chol_cut: float
    frozen: int | NDArray
    source_kind: str  # "mf" or "cc"
    basis: HamBasis  # "restricted" or "generalized"


@dataclass(frozen=True, slots=True)
class TrialInput:
    """trial inputs used to construct an afqmc trial object"""

    kind: str  # "slater", "cisd", "ucisd"
    data: Dict[str, Array]
    frozen: int | NDArray
    source_kind: str  # "mf" or "cc"


@dataclass(frozen=True, slots=True)
class StagedInputs:
    ham: HamInput
    trial: TrialInput
    meta: Dict[str, Any]


@dataclass(frozen=True, slots=True)
class StagedCc:
    """Wrapper ensuring the validity of the CC object"""

    _delegate = {"t1", "t2", "_scf", "frozen"}
    kind: str  # "ccsd", "uccsd", "gccsd"
    cc: Any
    mf: Any
    trial_frozen: int | NDArray
    afqmc_frozen: int | NDArray

    def __init__(self, cc: Any, frozen: int | ArrayLike | None):
        from pyscf.cc.ccsd import CCSD
        from pyscf.cc.gccsd import GCCSD
        from pyscf.cc.uccsd import UCCSD

        if not isinstance(cc, (CCSD, UCCSD, GCCSD)):
            raise TypeError(f"Unsupported object type: {type(cc)}")

        if not hasattr(cc, "_scf"):
            raise TypeError("CC-like object missing _scf reference to underlying scf object.")
        else:
            mf = _copy_scf_with_cc_mo_coeff(cc, cc._scf)

        if not hasattr(cc, "t1") or not hasattr(cc, "t2"):
            raise ValueError("CC amplitudes not found; did you run cc.kernel()?")

        if isinstance(cc, CCSD):
            kind = "ccsd"
        elif isinstance(cc, UCCSD):
            kind = "uccsd"
        elif isinstance(cc, GCCSD):
            kind = "gccsd"

        frozen = _stage_frozen(frozen)
        cc_frozen = _stage_frozen(cc.frozen)

        if cc_frozen is None:
            if frozen is not None and not (isinstance(frozen, int) and frozen == 0):
                raise ValueError(
                    "Explicit AFQMC frozen orbitals are unsupported for CC objects without cc.frozen."
                )
            afqmc_frozen = 0
            trial_frozen = 0
        elif isinstance(cc_frozen, np.ndarray):
            if kind != "ccsd":
                raise NotImplementedError(
                    "List-valued cc.frozen is currently supported only for restricted CCSD staging."
                )
            trial_frozen = _normalize_frozen_list(cc_frozen, nmo=mf.mo_coeff.shape[-1])
            if frozen is None:
                afqmc_frozen = 0
            elif isinstance(frozen, int):
                afqmc_frozen = frozen
            else:
                raise TypeError(
                    "List-valued cc.frozen requires an integer AFQMC frozen-core count."
                )
        else:
            if frozen is None:
                afqmc_frozen = int(cc_frozen)
            elif isinstance(frozen, int):
                if int(cc_frozen) != frozen:
                    raise ValueError("cc.frozen and staging frozen must be equal.")
                afqmc_frozen = frozen
            else:
                raise TypeError(
                    "Integer cc.frozen is incompatible with list-valued staging frozen."
                )
            trial_frozen = int(cc_frozen)

        if isinstance(trial_frozen, np.ndarray) and kind != "ccsd":
            raise NotImplementedError(
                "List-valued cc.frozen is currently supported only for restricted CCSD staging."
            )

        mf = StagedMf(mf, afqmc_frozen)

        object.__setattr__(self, "cc", cc)
        object.__setattr__(self, "mf", mf)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "afqmc_frozen", afqmc_frozen)
        object.__setattr__(self, "trial_frozen", trial_frozen)

    def __getattr__(self, name):
        if name in StagedCc._delegate:
            return getattr(self.cc, name)
        elif hasattr(self.cc, name):
            raise AttributeError(
                f"Attribute '{name}' exists in the CC object but not in this wrapper."
            )
        elif hasattr(self.mf, name):
            raise AttributeError(
                f"Attribute '{name}' exists in the SCF object but not in this wrapper."
            )
        else:
            raise AttributeError(
                f"Attribute '{name}' does not exist in the SCF and CC objects or in this wrapper."
            )


@dataclass(frozen=True, slots=True)
class StagedMf:
    """Wrapper ensuring the validity of the SCF object"""

    _delegate = {"mo_coeff", "mol", "nelec", "get_ovlp", "energy_nuc", "get_hcore"}
    kind: str  # "rhf", "rohf", "uhf", ghf
    mf: Any  # Python SCF object
    trial_frozen: int | NDArray
    afqmc_frozen: int | NDArray

    def __init__(self, mf: Any, frozen: int | ArrayLike | None):
        from pyscf.scf.ghf import GHF
        from pyscf.scf.hf import RHF
        from pyscf.scf.rohf import ROHF
        from pyscf.scf.uhf import UHF

        if not isinstance(mf, (RHF, ROHF, UHF, GHF)):
            raise TypeError(f"Unsupported object type: {type(mf)}")

        # if not hasattr(mf, "mol"):
        #   raise TypeError("SCF-like object missing mol reference to underlying mol object.")
        # else:
        #   mol = mf.mol

        if not hasattr(mf, "mo_coeff"):
            raise ValueError("MO coefficients not found; did you run mf.kernel()?")

        if isinstance(mf, RHF):
            kind = "rhf"
        elif isinstance(mf, ROHF):
            kind = "rohf"
        elif isinstance(mf, UHF):
            kind = "uhf"
        elif isinstance(mf, GHF):
            kind = "ghf"

        frozen = _stage_frozen(frozen)

        if isinstance(frozen, np.ndarray):
            frozen_arr = _normalize_frozen_list(frozen, nmo=mf.mo_coeff.shape[-1])
            if frozen_arr.size > 0:
                assert frozen_arr.size < mf.mo_coeff.shape[-1]
            frozen = frozen_arr
        elif isinstance(frozen, int):
            assert frozen < mf.mo_coeff.shape[-1]
            assert frozen >= 0
        elif frozen is None:
            frozen = 0
        else:
            raise TypeError(
                f"Expected a type int | np.ndarray | None, but received '{type(frozen)}'."
            )

        object.__setattr__(self, "mf", mf)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "afqmc_frozen", frozen)
        object.__setattr__(self, "trial_frozen", 0)

    @property
    def norb(self) -> int:
        return self.mf.mo_coeff.shape[-1]

    def __getattr__(self, name: str):
        if name in StagedMf._delegate:
            return getattr(self.mf, name)
        elif hasattr(self.mf, name):
            raise AttributeError(
                f"Attribute '{name}' exists in the SCF object but not in this wrapper."
            )
        else:
            raise AttributeError(
                f"Attribute '{name}' does not exist in the SCF object or in this wrapper."
            )


@dataclass(frozen=True, slots=True)
class StagedMfOrCc:
    """Wrapper ensuring the validity of the SCF/CC object"""

    _delegate_mf = StagedMf._delegate
    _delegate_cc = StagedCc._delegate
    kind: str  # StageCc.kind or StagedMf.kind
    source: str  # "cc", "mf"
    mf_or_cc: Any  # StagedMf or StagedCc
    mf: StagedMf
    afqmc_frozen: int | NDArray
    trial_frozen: int | NDArray

    def __init__(self, mf_or_cc: Any, frozen: int | ArrayLike | None):
        from pyscf.cc.ccsd import CCSD
        from pyscf.cc.gccsd import GCCSD
        from pyscf.cc.uccsd import UCCSD
        from pyscf.scf.ghf import GHF
        from pyscf.scf.hf import RHF
        from pyscf.scf.rohf import ROHF
        from pyscf.scf.uhf import UHF

        if isinstance(mf_or_cc, (CCSD, UCCSD, GCCSD)):
            mf_or_cc = StagedCc(mf_or_cc, frozen)
            mf = mf_or_cc.mf
            source = "cc"
        elif isinstance(mf_or_cc, (RHF, ROHF, UHF, GHF)):
            mf_or_cc = StagedMf(mf_or_cc, frozen)
            mf = mf_or_cc
            source = "mf"
        else:
            raise TypeError(f"Unreachable: '{type(mf_or_cc)}'")

        object.__setattr__(self, "mf_or_cc", mf_or_cc)
        object.__setattr__(self, "mf", mf)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "kind", mf_or_cc.kind)
        object.__setattr__(self, "afqmc_frozen", mf_or_cc.afqmc_frozen)
        object.__setattr__(self, "trial_frozen", mf_or_cc.trial_frozen)

    @property
    def norb(self) -> int:
        return self.mf.mo_coeff.shape[-1]

    def __getattr__(self, name: str):
        if name in StagedMfOrCc._delegate_cc:
            return getattr(self.mf_or_cc, name)
        elif name in StagedMfOrCc._delegate_mf:
            return getattr(self.mf, name)
        elif self.source == "cc" and hasattr(self.mf_or_cc.cc, name):
            raise AttributeError(
                f"Attribute '{name}' exists in the CC object but not in this wrapper."
            )
        elif hasattr(self.mf.mf, name):
            raise AttributeError(
                f"Attribute '{name}' exists in the SCF object but not in this wrapper."
            )
        else:
            raise AttributeError(
                f"Attribute '{name}' does not exist in the SCF and CC objects or in this wrapper."
            )


# public API
def stage(
    obj: Any,
    *,
    norb_frozen_core: int | None = None,
    norb_frozen: int | None = None,
    frozen_orbitals: ArrayLike | None = None,
    chol_cut: float = 1e-5,
    cache: Union[str, Path] | None = None,
    overwrite: bool = False,
    verbose: bool = False,
    log: Any | None = None,
    ham: HamInput | None = None,
    trial: TrialInput | None = None,
) -> StagedInputs:
    """
    Stage inputs from a pyscf mf or cc object.

    Args:
        obj:
            pyscf mf object (RHF/ROHF/UHF) or cc object (CCSD/UCCSD).
        norb_frozen_core:
            Preferred name for the number of lowest occupied core orbitals removed from the
            AFQMC Hamiltonian.
        norb_frozen:
            Backward-compatible alias for ``norb_frozen_core``.
            For CC objects with integer ``cc.frozen``, this is inferred from ``cc.frozen``.
            For restricted CCSD objects with list-valued ``cc.frozen``, trial-space frozen
            occupied/virtual blocks are inferred from ``cc.frozen`` while
            ``norb_frozen_core``/``norb_frozen`` control the occupied core orbitals removed
            from the AFQMC Hamiltonian.
        frozen_orbitals:
            Explicit orbital list for LNO-style staging. Generic AFQMC/FNO staging should use
            ``norb_frozen_core`` instead.
        chol_cut:
            Cholesky decomposition cutoff.
        cache:
            Optional path to write on disk. If it exists and overwrite=False,
            loads it. Otherwise computes and writes it.
        overwrite:
            If True and cache is provided, recompute and overwrite cache.
        verbose:
            Print timing/info.
        ham:
            Optionally provide HamInput. If None, will be staged from obj.
        trial:
            Optionally provide TrialInput. If None, will be staged from obj.

    Returns:
        StagedInputs containing HamInput, TrialInput, and metadata.
    """
    cache_path = Path(cache).expanduser().resolve() if cache is not None else None
    if cache_path is not None and cache_path.exists() and not overwrite:
        return load(cache_path, log=log)

    t0 = time.time()

    resolved_frozen = _resolve_stage_frozen_arg(norb_frozen_core, norb_frozen, frozen_orbitals)
    obj = StagedMfOrCc(obj, resolved_frozen)
    mol = obj.mol

    if ham is None:
        t_ham = _stage_begin("building Hamiltonian", log=log)
        ham = _stage_ham_input(
            obj,
            chol_cut=chol_cut,
            verbose=verbose,
            log=log,
        )
        _stage_end(
            t_ham,
            "Hamiltonian ready",
            details=f"norb={ham.norb} nchol={ham.chol.shape[0]}",
            log=log,
        )

    if trial is None:
        t_trial = _stage_begin("building trial input", log=log)
        trial = _stage_trial_input(obj)
        _stage_end(t_trial, "trial input ready", details=f"kind={trial.kind}", log=log)

    meta: Dict[str, Any] = {
        "format_version": STAGE_FORMAT_VERSION,
        "timestamp_unix": time.time(),
        "source_kind": obj.source,
        "frozen": _freeze_meta_value(obj.afqmc_frozen),
        "chol_cut": ham.chol_cut if ham is not None else chol_cut,
        "mol": {
            "nao": int(mol.nao),
            "nelectron": int(mol.nelectron),
            "spin": int(mol.spin),
            "charge": int(mol.charge),
            "basis": getattr(mol, "basis", None),
        },
    }

    staged = StagedInputs(ham=ham, trial=trial, meta=meta)

    if cache_path is not None:
        dump(staged, cache_path, log=log)

    if verbose:
        dt = time.time() - t0
        text = f"[stage] done in {dt:.2f}s | norb={ham.norb} nchol={ham.chol.shape[0]}"
        if log is None:
            print(text)
        else:
            log.debug("%s", text)

    return staged


def dump(staged: StagedInputs, path: Union[str, Path], *, log: Any | None = None) -> None:
    """
    Save staged inputs to a single h5 file.

    Args:
        staged: StagedInputs to serialize
        path: output file path
    """
    p = Path(path).expanduser().resolve()
    t_dump = _stage_begin(f"writing staged inputs to {p}", log=log)
    _dump_h5(staged, p)
    _stage_end(t_dump, "staged inputs written", log=log)


def load(path: Union[str, Path], *, log: Any | None = None) -> StagedInputs:
    """
    Load staged inputs from a single file written by dump().

    Args:
        path: input file path

    Returns:
        StagedInputs
    """
    p = Path(path).expanduser().resolve()
    t_load = _stage_begin(f"loading staged inputs from {p}", log=log)
    staged = _load_h5(p)
    _stage_end(
        t_load,
        "staged inputs loaded",
        details=f"norb={staged.ham.norb} nchol={staged.ham.chol.shape[0]} trial={staged.trial.kind}",
        log=log,
    )
    return staged


def _is_cc_like(obj: Any) -> bool:
    return hasattr(obj, "t1") and hasattr(obj, "t2")


def _stage_ham_input(
    obj: StagedMfOrCc,
    *,
    chol_cut: float,
    verbose: bool,
    log: Any | None = None,
) -> HamInput:
    """
    Produce h0/h1/chol in a single orthonormal basis.
    For UHF, we use the alpha MO basis for integrals.
    """
    from pyscf import mcscf

    mol = obj.mol
    scf_obj = obj.mf

    if scf_obj.kind in ("rhf", "rohf", "ghf"):
        basis_coeff = np.asarray(scf_obj.mo_coeff)
    elif scf_obj.kind == "uhf":
        basis_coeff = np.asarray(scf_obj.mo_coeff[0])
    else:
        raise ValueError(f"Unreachable: '{scf_obj.kind}'.")

    if scf_obj.kind in ("rhf", "rohf", "uhf"):
        ham_basis = "restricted"
    elif scf_obj.kind == "ghf":
        ham_basis = "generalized"
    else:
        raise ValueError(f"Unreachable: '{scf_obj.kind}'.")

    # nuclear energy (without frozen core correction)
    h0 = float(scf_obj.energy_nuc())

    # one body
    hcore = scf_obj.get_hcore()
    h1 = basis_coeff.T.conj() @ hcore @ basis_coeff
    h1 = np.asarray(h1)

    # ao cholesky
    t0 = time.time()
    chol_vec = chunked_cholesky(mol, max_error=chol_cut, verbose=verbose)
    if verbose:
        text = f"[stage] AO cholesky: nchol={chol_vec.shape[0]} in {time.time() - t0:.2f}s"
        if log is None:
            print(text)
        else:
            log.debug("%s", text)

    # full space electron count
    nelec: Tuple[int, int] = (int(mol.nelec[0]), int(mol.nelec[1]))
    norb_frozen = scf_obj.afqmc_frozen

    assert isinstance(norb_frozen, int)

    # mo Cholesky
    C = np.asarray(basis_coeff)
    if scf_obj.kind != "ghf":
        norb = int(basis_coeff.shape[1])
        chol = _rotate_chol_to_mo(chol_vec, C)
    else:
        norb = basis_coeff.shape[1] // 2
        chol = _rotate_chol_to_ghf_mo(chol_vec, C)

    # freeze core
    if norb_frozen > 0 and scf_obj.kind != "ghf":

        if isinstance(norb_frozen, int):
            if norb_frozen > min(nelec):
                raise ValueError(f"norb_frozen={norb_frozen} exceeds min(nelec)={min(nelec)}")

            nelec_frozen = 2 * norb_frozen
            ncas = basis_coeff.shape[1] - norb_frozen
            nelecas = mol.nelectron - nelec_frozen

        if nelecas <= 0 or ncas <= 0:
            raise ValueError("Frozen core left no active electrons/orbitals.")

        mc = mcscf.CASSCF(scf_obj.mf, ncas, nelecas)
        mc.mo_coeff = basis_coeff  # type: ignore
        h1_eff, ecore = mc.get_h1eff()  # type: ignore
        i0 = int(mc.ncore)  # type: ignore
        i1 = i0 + int(mc.ncas)  # type: ignore

        h0 = float(ecore)
        h1 = np.asarray(h1_eff)
        chol = np.array(chol[:, i0:i1, i0:i1], copy=True)
        norb = int(ncas)
        nelec = tuple(int(x) for x in mc.nelecas)  # type: ignore
    elif norb_frozen > 0 and scf_obj.kind == "ghf":
        raise NotImplementedError(
            "Frozen core approximation not available for generalised integrals."
        )

    return HamInput(
        h0=h0,
        h1=np.asarray(h1),
        chol=np.asarray(chol),
        nelec=nelec,
        norb=norb,
        chol_cut=float(chol_cut),
        frozen=norb_frozen,
        source_kind=obj.source,
        basis=ham_basis,
    )


def _stage_trial_input(obj: StagedMfOrCc) -> TrialInput:
    """
    Produce TrialInput consistent with the Hamiltonian basis and frozen core choice
    """

    if obj.kind in ("rhf", "rohf", "uhf", "ghf"):
        stage_tr_fun = _stage_mf_input
    elif obj.kind == "ccsd":
        stage_tr_fun = _stage_cisd_input
    elif obj.kind == "uccsd":
        stage_tr_fun = _stage_ucisd_input
    elif obj.kind == "gccsd":
        stage_tr_fun = _stage_gcisd_input
    elif obj.kind == "pt2ccsd":
        stage_tr_fun = _stage_pt2ccsd_input
    else:
        raise ValueError(f"Unreachable: '{obj.kind}'.")

    return stage_tr_fun(obj)


def _stage_mf_input(obj: StagedMfOrCc) -> TrialInput:

    mol = obj.mol
    S = obj.get_ovlp(mol)
    frozen = obj.afqmc_frozen

    if obj.mf.kind in ("rhf", "rohf", "ghf"):
        Ca = np.asarray(obj.mo_coeff)
        mo = _mf_coeff_helper(Ca, Ca, S, frozen)
        data = {"mo": np.asarray(mo)}
    elif obj.mf.kind == "uhf":
        Ca = np.asarray(obj.mo_coeff[0])
        Cb = np.asarray(obj.mo_coeff[1])

        # basis is alpha MOs, represent alpha and beta orbitals in this basis
        moa = _mf_coeff_helper(Ca, Ca, S, frozen)
        mob = _mf_coeff_helper(Ca, Cb, S, frozen)
        data = {"mo_a": np.asarray(moa), "mo_b": np.asarray(mob)}
    else:
        raise ValueError(f"Unreachable: '{obj.kind}'.")

    return TrialInput(
        kind=obj.kind,
        data=data,
        frozen=frozen,
        source_kind=obj.source,
    )


def _mf_coeff_helper(
    Ca: NDArray,
    Cb: NDArray,
    S: NDArray,
    frozen: int | NDArray,
) -> NDArray:
    q, r = np.linalg.qr(Ca.T @ S @ Cb)
    sgn = np.sign(np.diag(r))
    q = q * sgn[None, :]
    if isinstance(frozen, int):
        q = q[frozen:, frozen:]
    elif isinstance(frozen, np.ndarray):
        idx = np.delete(np.arange(len(q)), frozen)
        q = q[np.ix_(idx, idx)]
    else:
        raise TypeError(
            f"frozen must be an integer or a np.ndarray, but received '{type(frozen)}'."
        )

    return q


def _stage_cisd_input(obj: StagedMfOrCc) -> TrialInput:
    if obj.kind != "ccsd":
        raise ValueError(f"Unreachable: '{obj.kind}'.")

    t1_arr = np.asarray(obj.t1)
    t2_arr = np.asarray(obj.t2)
    nocc_t_core = 0
    nvir_t_outer = 0

    if isinstance(obj.trial_frozen, (np.ndarray)):
        if obj.mol.nelec[0] != obj.mol.nelec[1]:
            raise ValueError(
                "List-valued cc.frozen is currently supported only for closed-shell restricted CCSD."
            )

        assert isinstance(obj.afqmc_frozen, int)

        nocc_t_core, nvir_t_outer = _infer_restricted_trial_freeze_from_cc(
            cc_frozen=obj.trial_frozen,
            nmo_full=int(obj.mf.mo_coeff.shape[-1]),
            nocc_full=int(obj.mol.nelectron // 2),
            norb_frozen=int(obj.afqmc_frozen),
            t1_shape=(int(t1_arr.shape[0]), int(t1_arr.shape[1])),
        )

    ci2 = t2_arr + np.einsum("ia,jb->ijab", t1_arr, t1_arr)
    ci2 = ci2.transpose(0, 2, 1, 3)  # (i,a,j,b) -> (i,j,a,b)
    ci1 = t1_arr

    data = {
        "ci1": ci1,
        "ci2": ci2,
        "nocc_t_core": np.array(nocc_t_core, dtype=np.int64),
        "nvir_t_outer": np.array(nvir_t_outer, dtype=np.int64),
    }
    return TrialInput(
        kind="cisd",
        data=data,
        frozen=obj.trial_frozen,
        source_kind=obj.source,
    )


def _stage_ucisd_input(obj: StagedMfOrCc) -> TrialInput:
    if obj.kind != "uccsd":
        raise ValueError(f"Unreachable: '{obj.kind}'.")

    t1a, t1b = obj.t1
    t2aa, t2ab, t2bb = obj.t2

    ci2aa = np.asarray(t2aa) + 2.0 * np.einsum("ia,jb->ijab", np.asarray(t1a), np.asarray(t1a))
    ci2aa = 0.5 * (ci2aa - ci2aa.transpose(0, 1, 3, 2))
    ci2aa = ci2aa.transpose(0, 2, 1, 3)

    ci2bb = np.asarray(t2bb) + 2.0 * np.einsum("ia,jb->ijab", np.asarray(t1b), np.asarray(t1b))
    ci2bb = 0.5 * (ci2bb - ci2bb.transpose(0, 1, 3, 2))
    ci2bb = ci2bb.transpose(0, 2, 1, 3)

    ci2ab = np.asarray(t2ab) + np.einsum("ia,jb->ijab", np.asarray(t1a), np.asarray(t1b))
    ci2ab = ci2ab.transpose(0, 2, 1, 3)

    _uhf_input = _stage_mf_input(obj)
    moa = _uhf_input.data["mo_a"]
    mob = _uhf_input.data["mo_b"]

    data = {
        "mo_coeff_a": np.asarray(moa),
        "mo_coeff_b": np.asarray(mob),
        "ci1a": np.asarray(t1a),
        "ci1b": np.asarray(t1b),
        "ci2aa": np.asarray(ci2aa),
        "ci2ab": np.asarray(ci2ab),
        "ci2bb": np.asarray(ci2bb),
    }

    return TrialInput(
        kind="ucisd",
        data=data,
        frozen=obj.trial_frozen,
        source_kind=obj.source,
    )


def _stage_gcisd_input(obj: StagedMfOrCc) -> TrialInput:
    if obj.kind != "gccsd":
        raise ValueError(f"Unreachable: '{obj.kind}'.")

    t1 = obj.t1
    t2 = obj.t2

    ci2 = (
        np.einsum("ijab->iajb", t2)
        + np.einsum("ia,jb->iajb", t1, t1)
        - np.einsum("ib,ja->iajb", t1, t1)
    )
    ci1 = np.asarray(t1)

    _ghf_input = _stage_mf_input(obj)
    mo = _ghf_input.data["mo"]

    data = {"mo_coeff": mo, "ci1": ci1, "ci2": ci2}

    return TrialInput(
        kind="gcisd",
        data=data,
        frozen=obj.trial_frozen,
        source_kind=obj.source,
    )


def _stage_pt2ccsd_input(obj):
    # TODO obj.kind is frozen... figure out how to assign more flexible trial
    # if obj.kind != "pt2ccsd":
    #     raise ValueError(f"Unreachable: '{obj.kind}'.")

    t1 = obj.t1
    t2 = obj.t2
    nocc, nvir = t1.shape
    norb = nocc + nvir

    t1 = np.asarray(t1)
    t2 = np.asarray(t2)
    t2 = t2.transpose(0, 2, 1, 3)  # (i,j,a,b) -> (i,a,j,b)

    def _thouless(init_slater, t1):
        # Thouless transformation: |psi'> = exp(t1_ia a+ i)|psi>
        # init slater: mo_coeff of psi (in mo basis)
        # return mo_coeff of psi' (in mo basis)
        norb, nvir = t1.shape
        norb = nocc + nvir
        exp_t1 = np.eye(norb, dtype=np.float64)
        exp_t1[:nocc, nocc:] = t1
        # exp_t1 = jsp.linalg.expm(t1_full)
        return exp_t1.T @ init_slater

    mo_coeff = np.eye(norb, dtype=np.float64)[:, :nocc]
    mo_t = _thouless(mo_coeff, t1)

    data = {"mo_t": mo_t, "t2": t2}
    return TrialInput(
        kind="pt2ccsd",
        data=data,
        frozen=obj.trial_frozen,
        source_kind=obj.source,
    )


def _dump_h5(staged: StagedInputs, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["meta_json"] = json.dumps(staged.meta)

        gham = f.create_group("ham")
        gham.create_dataset("h0", data=np.array(staged.ham.h0))
        gham.create_dataset("h1", data=staged.ham.h1)
        gham.create_dataset("chol", data=staged.ham.chol)
        gham.create_dataset("nelec", data=np.array(staged.ham.nelec, dtype=np.int64))
        gham.attrs["norb"] = staged.ham.norb
        gham.attrs["chol_cut"] = staged.ham.chol_cut
        _dump_frozen(gham, staged.ham.frozen)
        gham.attrs["source_kind"] = staged.ham.source_kind
        gham.attrs["basis"] = staged.ham.basis

        gtr = f.create_group("trial")
        gtr.attrs["kind"] = staged.trial.kind
        _dump_frozen(gtr, staged.trial.frozen)
        gtr.attrs["source_kind"] = staged.trial.source_kind
        gdata = gtr.create_group("data")
        for k, v in staged.trial.data.items():
            gdata.create_dataset(k, data=np.asarray(v))


def _to_json_str(x: Any) -> str:
    # np scalar -> python scalar
    if isinstance(x, np.ndarray):
        x = x.item()
    # bytes like -> decode
    if isinstance(x, (bytes, bytearray, np.bytes_)):
        return bytes(x).decode("utf-8")
    return str(x)


def _load_h5(path: Path) -> StagedInputs:
    with h5py.File(path, "r") as f:
        meta = json.loads(_to_json_str(f.attrs["meta_json"]))
        if "frozen" in meta:
            meta["frozen"] = _freeze_from_meta_value(meta["frozen"])

        t_ham = _stage_begin("reading Hamiltonian from cache")
        gham: Any = f["ham"]
        ham = HamInput(
            h0=float(np.array(gham["h0"]).item()),
            h1=np.array(gham["h1"]),
            chol=np.array(gham["chol"]),
            nelec=(int(np.array(gham["nelec"])[0]), int(np.array(gham["nelec"])[1])),
            norb=int(gham.attrs["norb"]),
            chol_cut=float(gham.attrs["chol_cut"]),
            frozen=_load_frozen(gham),
            source_kind=str(gham.attrs["source_kind"]),
            basis=cast(HamBasis, str(gham.attrs["basis"])),
        )
        _stage_end(
            t_ham, "Hamiltonian loaded", details=f"norb={ham.norb} nchol={ham.chol.shape[0]}"
        )

        t_trial = _stage_begin("reading trial input from cache")
        gtr: Any = f["trial"]
        gdata = gtr["data"]
        trial_data = {k: np.array(gdata[k]) for k in gdata.keys()}
        trial = TrialInput(
            kind=str(gtr.attrs["kind"]),
            data=trial_data,
            frozen=_load_frozen(gtr),
            source_kind=str(gtr.attrs["source_kind"]),
        )
        _stage_end(t_trial, "trial input loaded", details=f"kind={trial.kind}")

        return StagedInputs(ham=ham, trial=trial, meta=meta)


def build_ham_lno(
    obj: Any,
    *,
    frozen_orbitals: ArrayLike,
    chol_cut: float,
) -> HamInput:
    from pyscf import ao2mo, mcscf

    obj = StagedMfOrCc(obj, 0)
    mf = obj.mf.mf
    mol = mf.mol

    norb = obj.norb
    frozen = _normalize_frozen_list(frozen_orbitals, nmo=norb)
    basis_coeff = mf.mo_coeff

    nelec_frozen = 2 * np.sum(frozen < mol.nelec[0])
    nact = basis_coeff.shape[1] - frozen.size
    nelec_act = mol.nelectron - nelec_frozen
    mc = mcscf.CASSCF(mf, nact, nelec_act)
    mc.frozen = frozen  # type: ignore
    nelec = mc.nelecas  # type: ignore
    h1, h0 = mc.get_h1eff()  # type: ignore
    act = [i for i in range(norb) if i not in frozen]
    e = np.asarray(ao2mo.kernel(mf.mol, mf.mo_coeff[:, act]))  # , compact=False)
    chol = modified_cholesky(e, max_error=chol_cut)
    chol = chol.reshape((-1, nact, nact))

    ham = HamInput(
        h0=float(h0),
        h1=np.asarray(h1),
        chol=np.asarray(chol),
        nelec=nelec,
        norb=nact,
        chol_cut=float(chol_cut),
        frozen=frozen,
        source_kind=obj.source,
        basis="restricted",
    )

    return ham
