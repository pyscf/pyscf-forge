"""Collinear Kohn-Sham, CAS, and target-state matrix densities."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

import numpy
import scipy.linalg
import torch
import warnings
from pyscf.dft import numint
from pyscf.fci.addons import _unpack_nelec
from pyscf.scf.hf import get_hcore

from .active_space import ActiveSpace, ActiveSpaceError
from .spin import SpinType


@dataclass(frozen=True)
class AOGridBatch:
    weights: Optional[torch.Tensor]
    ao_value: torch.Tensor
    grad_ao_value: Optional[torch.Tensor] = None
    lapl_ao_value: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class OneBodyOperatorTable:
    spin_rows: torch.Tensor
    spin_cols: torch.Tensor
    orbital_rows: torch.Tensor
    orbital_cols: torch.Tensor
    det_rows: torch.Tensor
    det_cols: torch.Tensor
    values: torch.Tensor
    shape: tuple


@dataclass(frozen=True)
class CASGuessComponents:
    orbital_coefficients: torch.Tensor
    orbital_rotation_params: torch.nn.Parameter
    active_space: ActiveSpace
    s2_matrix: torch.Tensor
    det_to_csf: torch.Tensor
    nstate: int
    ndet: int


def _spin_orbital_index(spin, orbital, norb):
    if spin == 0:
        return norb + orbital
    if spin == 1:
        return orbital
    raise ValueError("spin index must be 0 or 1")


def _apply_annihilation(det, spin_orbital):
    if ((det >> spin_orbital) & 1) == 0:
        return None
    phase = -1.0 if bin(det & ((1 << spin_orbital) - 1)).count("1") % 2 else 1.0
    return det ^ (1 << spin_orbital), phase


def _apply_creation(det, spin_orbital):
    if ((det >> spin_orbital) & 1) == 1:
        return None
    phase = -1.0 if bin(det & ((1 << spin_orbital) - 1)).count("1") % 2 else 1.0
    return det | (1 << spin_orbital), phase


def _apply_one_body_operator(det, create_spin, annihilate_spin, p, q, norb):
    annihilated = _apply_annihilation(
        det, _spin_orbital_index(annihilate_spin, q, norb))
    if annihilated is None:
        return None
    created = _apply_creation(
        annihilated[0], _spin_orbital_index(create_spin, p, norb))
    if created is None:
        return None
    return created[0], annihilated[1] * created[1]


def antisymmetric_matrix(elements, n):
    expected = (n * (n - 1)) // 2
    if elements.numel() != expected:
        raise ValueError(f"expected {expected} independent elements, got {elements.numel()}")
    result = torch.zeros((n, n), dtype=elements.dtype, device=elements.device)
    rows, columns = torch.triu_indices(n, n, offset=1, device=elements.device)
    result[rows, columns] = elements.reshape(-1)
    result[columns, rows] = -elements.reshape(-1)
    return result


def stiefel_tangent_block_matrix(block, n, k):
    if block.size() != torch.Size([n - k, k]):
        raise ValueError(f"Stiefel block must have shape ({n-k}, {k})")
    result = torch.zeros((n, n), dtype=block.dtype, device=block.device)
    result[:k, k:] = -block.T
    result[k:, :k] = block
    return result


def orbital_guess(mol, guess="random", seed=None):
    nao = mol.nao_nr()
    if isinstance(guess, numpy.ndarray):
        if guess.shape != (nao, nao):
            raise ValueError(f"orbital guess must have shape {(nao, nao)}")
        return guess
    if guess == "rohf":
        from pyscf import scf
        mean_field = scf.ROHF(mol)
        mean_field.verbose = 0
        mean_field.kernel()
        return mean_field.mo_coeff
    overlap = mol.intor_symmetric("int1e_ovlp")
    if guess == "hcore":
        matrix = get_hcore(mol)
    elif guess == "random":
        matrix = numpy.random.default_rng(seed).random((nao, nao))
        matrix = 0.5 * (matrix + matrix.T)
    else:
        raise NotImplementedError(f"orbital guess {guess!r} is not implemented")
    return scipy.linalg.eigh(matrix, overlap)[1]


def reorder_active_orbitals(mol, mo_coeff, active_orbitals, nelecas):
    active_orbitals = list(active_orbitals)
    nelec = mol.tot_electrons()
    nclosed = (nelec - nelecas) // 2
    homo = (nelec + 1) // 2 - 1
    lumo = homo + 1
    for index, value in enumerate(active_orbitals):
        if isinstance(value, str):
            active_orbitals[index] = eval(
                value, {"__builtins__": {}}, {"H": homo, "HOMO": homo, "L": lumo, "LUMO": lumo}
            )
    if len(set(active_orbitals)) != len(active_orbitals):
        raise ValueError("active orbital indices must be unique")
    result = numpy.copy(mo_coeff)
    for frontier, active in zip(range(nclosed, nclosed + len(active_orbitals)), active_orbitals):
        result[:, frontier], result[:, active] = mo_coeff[:, active], mo_coeff[:, frontier]
    return result


def _validate_spin_type(spin_type):
    if not isinstance(spin_type, SpinType):
        raise ValueError(f"spin_type must be a SpinType, got {spin_type!r}")


def _validate_active_space_for_molecule(mol, norb, nelec):
    neleca, nelecb = _unpack_nelec(nelec)
    nactive = neleca + nelecb
    ntotal = mol.tot_electrons()
    if neleca > norb or nelecb > norb:
        raise ActiveSpaceError(
            "Active-space spin electron counts cannot exceed the number of active orbitals"
        )
    if nactive > ntotal:
        raise ActiveSpaceError(
            f"More active electrons ({nactive}) than total number of electrons ({ntotal})"
        )
    if ntotal % 2 != nactive % 2:
        raise ActiveSpaceError(
            "For odd (even) total number of electrons, number of active electrons must "
            f"be odd (even) as well, got {ntotal} electrons in total, but {nactive} active electrons."
        )
    ndouble = (ntotal - nactive) // 2
    if norb > mol.nao_nr() - ndouble:
        raise ActiveSpaceError(
            f"More active orbitals ({norb}) than molecular orbitals (nmo={mol.nao_nr()}) "
            f"minus doubly occupied orbitals (ndouble={ndouble})."
        )
    return ndouble


def _cas_components(mol, norb, nelec, spin_symmetry, spin_type, max_level, guess, seed):
    _validate_spin_type(spin_type)
    _validate_active_space_for_molecule(mol, norb, nelec)
    neleca, nelecb = _unpack_nelec(nelec)
    active_space = ActiveSpace(norb, nelec, max_level=max_level, spin_range=[neleca - nelecb])
    s2 = active_space.total_spin_matrix()
    if not len(s2):
        raise ActiveSpaceError(f"Active space (norb={norb}, nelec={nelec}) is empty")
    eigenvalues, eigenvectors = numpy.linalg.eigh(s2)
    if spin_symmetry:
        target = 0.5 * mol.spin
        selected = numpy.flatnonzero(numpy.isclose(eigenvalues, target * (target + 1), atol=1.0e-2))
    else:
        selected = numpy.arange(len(eigenvalues))
    if not len(selected):
        raise ActiveSpaceError("There are no states with total spin matching the molecule")
    coefficients = torch.from_numpy(orbital_guess(mol, guess, seed)).double()
    nmo = coefficients.size(1)
    orbital_params = torch.nn.Parameter(torch.zeros((nmo * (nmo - 1)) // 2, dtype=torch.double))
    return (
        coefficients,
        orbital_params,
        active_space,
        torch.from_numpy(s2).double(),
        torch.from_numpy(eigenvectors[:, selected]).double(),
    )


def cas_guess_components(mol, norb, nelec, spin_symmetry=True,
                         spin_type=SpinType.UNPOLARIZED, max_level=numpy.inf,
                         guess="hcore", seed=None):
    values = _cas_components(
        mol, norb, nelec, spin_symmetry, spin_type, max_level, guess, seed)
    orbital_coefficients, orbital_params, active_space, s2_matrix, det_to_csf = values
    return CASGuessComponents(
        orbital_coefficients=orbital_coefficients,
        orbital_rotation_params=orbital_params,
        active_space=active_space,
        s2_matrix=s2_matrix,
        det_to_csf=det_to_csf,
        nstate=det_to_csf.size(1),
        ndet=det_to_csf.size(0),
    )


class MultistateMatrixDensity(torch.nn.Module, ABC):
    def __init__(self, mol):
        super().__init__()
        self.mol = mol

    @property
    @abstractmethod
    def number_of_states(self):
        pass

    @property
    def number_of_electrons(self):
        return self.mol.tot_electrons()

    @property
    def device(self):
        return next(self.parameters(), torch.empty(0)).device

    @abstractmethod
    def density_matrices_ao(self):
        pass

    @abstractmethod
    def spin_multiplicity(self):
        pass

    def state_weights(self):
        return self.spin_multiplicity()

    @staticmethod
    def _check_input(tensor, name, expected_size):
        if tensor.size() != expected_size:
            raise ValueError(
                f"Input '{name}' has to be of size {expected_size}, but got {tensor.size()}")
        return tensor

    def evaluate(self, coords, dm_ao=None, need_gradient=True, need_laplacian=True):
        if dm_ao is None:
            dm_ao = self.density_matrices_ao()
        need_gradient = need_gradient or need_laplacian
        deriv = 2 if need_laplacian else 1 if need_gradient else 0
        ao = torch.from_numpy(numint.eval_ao(self.mol, coords, deriv=deriv)).to(
            dtype=dm_ao.dtype, device=dm_ao.device
        )
        if deriv == 0:
            batch = AOGridBatch(None, ao)
        else:
            laplacian = ao[4] + ao[7] + ao[9] if need_laplacian else None
            batch = AOGridBatch(None, ao[0], ao[1:4], laplacian)
        return self.evaluate_from_ao_batch(batch, dm_ao, need_gradient, need_laplacian)

    def evaluate_from_ao_batch(self, batch, dm_ao=None, need_gradient=True, need_laplacian=True):
        if dm_ao is None:
            dm_ao = self.density_matrices_ao()
        need_gradient = need_gradient or need_laplacian
        ao = batch.ao_value
        density = torch.einsum("stabij,ra,rb->strij", dm_ao, ao, ao)
        gradient = None
        if need_gradient:
            if batch.grad_ao_value is None:
                raise ValueError("AO gradients are missing from the grid batch")
            gradient = (
                torch.einsum("stabij,xra,rb->strxij", dm_ao, batch.grad_ao_value, ao)
                + torch.einsum("stabij,ra,xrb->strxij", dm_ao, ao, batch.grad_ao_value)
            )
        laplacian = None
        if need_laplacian:
            if batch.lapl_ao_value is None:
                raise ValueError("AO Laplacians are missing from the grid batch")
            laplacian = (
                torch.einsum("stabij,ra,rb->strij", dm_ao, batch.lapl_ao_value, ao)
                + 2 * torch.einsum("stabij,xra,xrb->strij", dm_ao, batch.grad_ao_value, batch.grad_ao_value)
                + torch.einsum("stabij,ra,rb->strij", dm_ao, ao, batch.lapl_ao_value)
            )
        return density, gradient, laplacian


class MultistateMatrixDensityKohnSham(MultistateMatrixDensity):
    @classmethod
    def from_guess(cls, mol, guess="hcore", seed=None):
        coefficients = torch.from_numpy(orbital_guess(mol, guess, seed)).double()
        nmo = coefficients.size(1)
        params = torch.nn.Parameter(torch.zeros((nmo * (nmo - 1)) // 2, dtype=torch.double))
        return cls(mol, coefficients, params)

    def __init__(self, mol, orbital_coefficients, orbital_rotation_params):
        super().__init__(mol)
        self.register_buffer("mo_coeff_guess", orbital_coefficients.detach())
        self.nao, self.nmo = orbital_coefficients.size()
        self.orbital_rotation_params = self._check_input(
            orbital_rotation_params, "orbital_rotation_params",
            torch.Size([(self.nmo * (self.nmo - 1)) // 2]))

    @property
    def number_of_states(self):
        return 1

    def orbital_coefficients(self):
        return self.mo_coeff_guess @ torch.matrix_exp(antisymmetric_matrix(self.orbital_rotation_params, self.nmo))

    def density_matrices_ao(self):
        coefficients = self.orbital_coefficients()
        occupations = torch.zeros((2, 2, self.nmo), dtype=coefficients.dtype, device=coefficients.device)
        occupations[0, 0, :self.mol.nelec[0]] = 1
        occupations[1, 1, :self.mol.nelec[1]] = 1
        density = torch.einsum("ap,stp,bp->stab", coefficients, occupations, coefficients)
        return density[..., None, None]

    def spin_multiplicity(self):
        return torch.tensor([self.mol.spin + 1], device=self.orbital_rotation_params.device)

    def basis_transformation(self, transformation):
        if transformation.size() != torch.Size([1, 1]):
            raise ValueError("a Kohn-Sham density contains one state")


class MultistateMatrixDensityCAS(MultistateMatrixDensity):
    @classmethod
    def from_guess(cls, mol, norb, nelec, spin_symmetry=True, spin_type=SpinType.UNPOLARIZED,
                   max_level=numpy.inf, guess="hcore", seed=None):
        coefficients, orbital_params, active_space, s2, det_to_csf = _cas_components(
            mol, norb, nelec, spin_symmetry, spin_type, max_level, guess, seed
        )
        nstate = det_to_csf.size(1)
        state_params = torch.nn.Parameter(torch.zeros((nstate * (nstate - 1)) // 2, dtype=torch.double))
        return cls(mol, norb, nelec, coefficients, orbital_params, torch.eye(nstate, dtype=torch.double),
                   state_params, spin_symmetry, spin_type, max_level)

    def __init__(self, mol, norb, nelec, orbital_coefficients, orbital_rotation_params,
                 state_coefficients, state_rotation_params, spin_symmetry=True,
                 spin_type=SpinType.UNPOLARIZED, max_level=numpy.inf):
        super().__init__(mol)
        _validate_spin_type(spin_type)
        coefficients, _, active_space, s2, det_to_csf = _cas_components(
            mol, norb, nelec, spin_symmetry, spin_type, max_level,
            orbital_coefficients.detach().cpu().numpy(), None
        )
        self.active_space = active_space
        self.spin_type = spin_type
        self.spin_symmetry = spin_symmetry
        self.register_buffer("mo_coeff_guess", coefficients)
        self.register_buffer("s2_matrix", s2)
        self.register_buffer("det_to_csf", det_to_csf)
        self.register_buffer("ci_coeff_guess", state_coefficients.detach())
        self.nao, self.nmo = coefficients.size()
        self.orbital_rotation_params = self._check_input(
            orbital_rotation_params, "orbital_rotation_params",
            torch.Size([(self.nmo * (self.nmo - 1)) // 2]))
        self.nstate = det_to_csf.size(1)
        self.ndet = det_to_csf.size(0)
        self._check_input(state_coefficients, "state_coefficients",
                          torch.Size([self.nstate, self.nstate]))
        self.state_rotation_params = self._check_input(
            state_rotation_params, "state_rotation_params",
            torch.Size([(self.nstate * (self.nstate - 1)) // 2]))
        neleca, nelecb = _unpack_nelec(nelec)
        ndouble = _validate_active_space_for_molecule(mol, norb, (neleca, nelecb))
        active_density = active_space.matrix_density_mo()
        density = numpy.zeros((2, 2, self.nmo, self.nmo, self.ndet, self.ndet))
        for orbital in range(ndouble):
            density[0, 0, orbital, orbital] = numpy.eye(self.ndet)
            density[1, 1, orbital, orbital] = numpy.eye(self.ndet)
        density[:, :, ndouble:ndouble+norb, ndouble:ndouble+norb] = active_density
        density = numpy.einsum("stpqIJ,IS,JT->stpqST", density, det_to_csf.numpy(), det_to_csf.numpy())
        self.register_buffer("dm_mo_spin", torch.from_numpy(density).double())

    @property
    def number_of_states(self):
        return self.nstate

    @property
    def number_of_determinants(self):
        return self.ndet

    def orbital_coefficients(self):
        return self.mo_coeff_guess @ torch.matrix_exp(antisymmetric_matrix(self.orbital_rotation_params, self.nmo))

    def state_coefficients(self):
        return self.ci_coeff_guess @ torch.matrix_exp(antisymmetric_matrix(self.state_rotation_params, self.nstate))

    def determinant_coefficients(self):
        return self.det_to_csf @ self.state_coefficients()

    def density_matrices_ao(self):
        ci = self.state_coefficients()
        density_mo = torch.einsum("...st,si,tj->...ij", self.dm_mo_spin, ci, ci)
        coefficients = self.orbital_coefficients()
        return torch.einsum("stpqij,ap,bq->stabij", density_mo, coefficients, coefficients)

    def spin_s2_expectation(self):
        coefficients = self.determinant_coefficients()
        return torch.einsum("ab,ai,bi->i", self.s2_matrix, coefficients, coefficients)

    def spin_multiplicity(self):
        return torch.round(torch.sqrt(1.0 + 4.0 * self.spin_s2_expectation())).int()

    def occupation_labels(self):
        return self.active_space.occupation_labels()

    def basis_transformation(self, transformation):
        ci = self.state_coefficients() @ transformation.T
        self.register_buffer("ci_coeff_guess", ci.detach())
        self.state_rotation_params = torch.nn.Parameter(torch.zeros_like(self.state_rotation_params))


class TargetStateMultistateMatrixDensityCAS(MultistateMatrixDensity):
    RESTART_SCHEMA_VERSION = 2
    TARGET_PARAMETERIZATION_FULL_ROTATION = "full_rotation"
    TARGET_PARAMETERIZATION_STIEFEL_K = "stiefel_k"
    TARGET_PARAMETERIZATIONS = {
        TARGET_PARAMETERIZATION_FULL_ROTATION,
        TARGET_PARAMETERIZATION_STIEFEL_K,
    }
    FULL_ROTATION = TARGET_PARAMETERIZATION_FULL_ROTATION
    STIEFEL_K = TARGET_PARAMETERIZATION_STIEFEL_K

    @classmethod
    def from_guess(cls, mol, norb, nelec, target_states, spin_symmetry=True,
                   spin_type=SpinType.UNPOLARIZED, max_level=numpy.inf, guess="hcore",
                   seed=None, target_parameterization=FULL_ROTATION):
        if spin_type is not SpinType.UNPOLARIZED:
            raise NotImplementedError("target-state LDMA supports SpinType.UNPOLARIZED only")
        coefficients, orbital_params, active_space, s2, det_to_csf = _cas_components(
            mol, norb, nelec, spin_symmetry, spin_type, max_level, guess, seed
        )
        if not 1 <= target_states <= det_to_csf.size(1):
            raise ActiveSpaceError("target_states exceeds the selected CSF dimension")
        return cls(mol, norb, nelec, target_states, coefficients, orbital_params,
                   det_to_csf, s2, active_space, spin_symmetry, spin_type,
                   target_parameterization=target_parameterization)

    def __init__(self, mol, norb, nelec, target_states, orbital_coefficients,
                 orbital_rotation_params, det_to_csf, s2_matrix, active_space,
                 spin_symmetry=True, spin_type=SpinType.UNPOLARIZED,
                 target_rotation_params=None, target_parameterization=FULL_ROTATION,
                 orthonormality_threshold=1.0e-8, spin_mixing_threshold=1.0e-8):
        super().__init__(mol)
        if spin_type is not SpinType.UNPOLARIZED:
            raise NotImplementedError("target-state LDMA supports SpinType.UNPOLARIZED only")
        if target_parameterization not in (self.FULL_ROTATION, self.STIEFEL_K):
            raise ValueError("target_parameterization must be 'full_rotation' or 'stiefel_k'")
        self.norb, self.nelec, self.ntarget = norb, nelec, target_states
        self.nao, self.nmo = orbital_coefficients.size()
        self.nspin = 2
        self.ncsf = det_to_csf.size(1)
        self.ndet = det_to_csf.size(0)
        self.active_space = active_space
        self.spin_symmetry = spin_symmetry
        self.spin_type = spin_type
        self.target_parameterization = target_parameterization
        self.orthonormality_threshold = orthonormality_threshold
        self.spin_mixing_threshold = spin_mixing_threshold
        self.register_buffer("mo_coeff_guess", orbital_coefficients.detach())
        self.register_buffer("det_to_csf", det_to_csf.detach())
        self.register_buffer("s2_matrix", s2_matrix.detach())
        self.orbital_rotation_params = self._check_input(
            orbital_rotation_params, "orbital_rotation_params",
            torch.Size([(self.nmo * (self.nmo - 1)) // 2]))
        neleca, nelecb = _unpack_nelec(nelec)
        self.ndouble = _validate_active_space_for_molecule(mol, norb, (neleca, nelecb))
        if target_parameterization == self.FULL_ROTATION:
            shape = ((self.ncsf * (self.ncsf - 1)) // 2,)
        else:
            shape = (self.ncsf - self.ntarget, self.ntarget)
        if target_rotation_params is None:
            target_rotation_params = torch.nn.Parameter(torch.zeros(shape, dtype=orbital_coefficients.dtype))
        if target_rotation_params.size() != torch.Size(shape):
            raise ValueError(f"target_rotation_params must have shape {shape}")
        self.target_rotation_params = target_rotation_params
        table = self._build_one_body_operator_table(orbital_coefficients.dtype)
        self.register_buffer("one_body_spin_rows", table.spin_rows)
        self.register_buffer("one_body_spin_cols", table.spin_cols)
        self.register_buffer("one_body_orbital_rows", table.orbital_rows)
        self.register_buffer("one_body_orbital_cols", table.orbital_cols)
        self.register_buffer("one_body_det_rows", table.det_rows)
        self.register_buffer("one_body_det_cols", table.det_cols)
        self.register_buffer("one_body_values", table.values)
        self.one_body_shape = table.shape
        self._last_orbital_grid_density = None
        self.enforce_invariants()

    @property
    def number_of_states(self):
        return self.ntarget

    @property
    def number_of_csfs(self):
        return self.ncsf

    @property
    def number_of_determinants(self):
        return self.ndet

    def _build_one_body_operator_table(self, dtype):
        entries = []
        determinants = self.active_space.slater_determinants()
        determinant_indices = {det: index for index, det in enumerate(determinants)}
        for ket, determinant in enumerate(determinants):
            for create_spin in range(self.nspin):
                for annihilate_spin in range(self.nspin):
                    for p in range(self.norb):
                        for q in range(self.norb):
                            result = _apply_one_body_operator(
                                determinant, create_spin, annihilate_spin,
                                p, q, self.norb)
                            if result is None:
                                continue
                            bra = determinant_indices.get(result[0])
                            if bra is not None:
                                entries.append((
                                    create_spin, annihilate_spin,
                                    self.ndouble + q, self.ndouble + p,
                                    bra, ket, result[1]))
        for orbital in range(self.ndouble):
            for determinant in range(self.ndet):
                for spin in range(self.nspin):
                    entries.append((spin, spin, orbital, orbital,
                                    determinant, determinant, 1.0))
        if entries:
            data = numpy.asarray(entries, dtype=numpy.float64)
            integer_columns = [torch.from_numpy(data[:, index].astype(numpy.int64))
                               for index in range(6)]
            values = torch.from_numpy(data[:, 6]).to(dtype=dtype)
        else:
            integer_columns = [torch.empty(0, dtype=torch.long) for _ in range(6)]
            values = torch.empty(0, dtype=dtype)
        return OneBodyOperatorTable(
            integer_columns[0], integer_columns[1], integer_columns[2],
            integer_columns[3], integer_columns[4], integer_columns[5],
            values,
            (self.nspin, self.nspin, self.nmo, self.nmo, self.ndet, self.ndet))

    def orbital_coefficients(self):
        return self.mo_coeff_guess @ torch.matrix_exp(antisymmetric_matrix(self.orbital_rotation_params, self.nmo))

    def target_coefficients(self):
        if self.target_parameterization == self.FULL_ROTATION:
            generator = antisymmetric_matrix(self.target_rotation_params, self.ncsf)
        else:
            generator = stiefel_tangent_block_matrix(self.target_rotation_params, self.ncsf, self.ntarget)
        return torch.matrix_exp(generator)[:, :self.ntarget]

    def determinant_coefficients(self):
        return self.det_to_csf @ self.target_coefficients()

    def transition_1rdm_mo(self):
        coefficients = self.determinant_coefficients()
        gamma = torch.zeros((2, 2, self.nmo, self.nmo, self.ntarget, self.ntarget),
                            dtype=coefficients.dtype, device=coefficients.device)
        values = self.one_body_values.to(coefficients)
        entry_values = (
            values[:, None, None]
            * coefficients[self.one_body_det_rows, :, None]
            * coefficients[self.one_body_det_cols, None, :]
        )
        gamma.index_put_(
            (
                self.one_body_spin_rows,
                self.one_body_spin_cols,
                self.one_body_orbital_rows,
                self.one_body_orbital_cols,
            ),
            entry_values,
            accumulate=True,
        )
        return gamma

    def density_matrices_ao(self):
        return torch.einsum("stpqij,ap,bq->stabij", self.transition_1rdm_mo(),
                            self.orbital_coefficients(), self.orbital_coefficients())

    def evaluate_from_mo_grid_batch(self, batch, gamma_mo=None):
        gamma_mo = self.transition_1rdm_mo() if gamma_mo is None else gamma_mo
        mo_values = batch.ao_value @ self.orbital_coefficients()
        return torch.einsum("stpqij,gp,gq->stgij", gamma_mo, mo_values, mo_values), None, None

    def spin_s2_expectation(self):
        coefficients = self.determinant_coefficients()
        return torch.einsum("ab,ai,bi->i", self.s2_matrix, coefficients, coefficients)

    def spin_multiplicity(self):
        return torch.round(torch.sqrt(1.0 + 4.0 * self.spin_s2_expectation())).int()

    def target_orthonormality_error(self):
        coefficients = self.target_coefficients()
        identity = torch.eye(
            self.ntarget, dtype=coefficients.dtype, device=coefficients.device)
        return torch.linalg.norm(coefficients.T @ coefficients - identity)

    def orbital_orthonormality_error(self):
        coefficients = self.orbital_coefficients()
        overlap = torch.from_numpy(self.mol.intor_symmetric("int1e_ovlp")).to(coefficients)
        identity = torch.eye(
            self.nmo, dtype=coefficients.dtype, device=coefficients.device)
        return torch.linalg.norm(coefficients.T @ overlap @ coefficients - identity)

    def invariant_diagnostics(self):
        spin_s2 = self.spin_s2_expectation().detach()
        return {
            "target_orthonormality_error": float(self.target_orthonormality_error().detach()),
            "orbital_orthonormality_error": float(self.orbital_orthonormality_error().detach()),
            "spin_s2_min": float(torch.min(spin_s2)),
            "spin_s2_max": float(torch.max(spin_s2)),
        }

    def check_invariants(self, fail=True):
        diagnostics = self.invariant_diagnostics()
        errors = []
        if diagnostics["target_orthonormality_error"] > self.orthonormality_threshold:
            errors.append("target coefficients are not orthonormal")
        if diagnostics["orbital_orthonormality_error"] > self.orthonormality_threshold:
            errors.append("orbital coefficients are not orthonormal")
        if errors:
            message = "; ".join(errors)
            if fail:
                raise RuntimeError(message)
            warnings.warn(message, RuntimeWarning)
        return diagnostics

    def state_diagnostics(self, reference_target_coefficients=None, dominant_count=5):
        coefficients = self.target_coefficients().detach()
        diagnostics = {
            "spin_s2": self.spin_s2_expectation().detach().cpu().tolist(),
            "spin_multiplicity": self.spin_multiplicity().detach().cpu().tolist(),
            "dominant_csfs": [],
        }
        nitems = min(dominant_count, self.ncsf)
        for state in range(self.ntarget):
            values, indices = torch.topk(coefficients[:, state].square(), k=nitems)
            diagnostics["dominant_csfs"].append([
                {"csf_index": int(index), "weight": float(value),
                 "label": f"CSF {int(index)}"}
                for value, index in zip(values.cpu(), indices.cpu())
            ])
        if reference_target_coefficients is not None:
            reference = reference_target_coefficients.detach().to(coefficients)
            overlap = reference.T @ coefficients
            diagnostics["reference_overlap"] = overlap.cpu().tolist()
            diagnostics["reference_overlap_abs"] = torch.abs(overlap).cpu().tolist()
        return diagnostics

    def warn_if_spin_mixed(self):
        if not self.spin_symmetry or self.ntarget <= 1:
            return None
        coefficients = self.determinant_coefficients()
        s2_matrix = self.s2_matrix.to(coefficients)
        target_s2 = coefficients.T @ s2_matrix @ coefficients
        off_diagonal = target_s2 - torch.diag(torch.diagonal(target_s2))
        defect = torch.max(torch.abs(off_diagonal)).detach()
        if defect > self.spin_mixing_threshold:
            warnings.warn(
                "Target-state coefficients mix spin sectors: off-diagonal S^2 defect "
                f"{float(defect)} > {self.spin_mixing_threshold}", RuntimeWarning)
        return defect

    def enforce_spin_symmetry(self):
        defect = self.warn_if_spin_mixed()
        if defect is not None and defect > self.spin_mixing_threshold:
            raise RuntimeError(
                "Target-state coefficients mix spin sectors: off-diagonal S^2 defect "
                f"{float(defect)} > {self.spin_mixing_threshold}")
        return defect

    def enforce_invariants(self):
        diagnostics = self.check_invariants(fail=True)
        self.enforce_spin_symmetry()
        return diagnostics

    def basis_transformation(self, transformation):
        target = self.target_coefficients() @ transformation.T
        target, _ = torch.linalg.qr(target.detach())
        completion = torch.eye(self.ncsf, dtype=target.dtype, device=target.device)
        completion[:, :self.ntarget] = target[:, :self.ntarget]
        full_basis, _ = torch.linalg.qr(completion)
        self.register_buffer("det_to_csf", self.det_to_csf @ full_basis)
        shape = self.target_rotation_params.size()
        self.target_rotation_params = torch.nn.Parameter(
            torch.zeros(shape, dtype=target.dtype, device=target.device)
        )
        self.enforce_invariants()

    def restart_state(self):
        return {
            "schema_version": self.RESTART_SCHEMA_VERSION,
            "class": self.__class__.__name__,
            "norb": self.norb,
            "nelec": self.nelec,
            "target_states": self.ntarget,
            "ncsf": self.ncsf,
            "nmo": self.nmo,
            "spin_symmetry": self.spin_symmetry,
            "spin_type": self.spin_type.name,
            "target_parameterization": self.target_parameterization,
            "orthonormality_threshold": self.orthonormality_threshold,
            "spin_mixing_threshold": self.spin_mixing_threshold,
            "orbital_rotation_params": self.orbital_rotation_params.detach().cpu().clone(),
            "target_rotation_params": self.target_rotation_params.detach().cpu().clone(),
            "diagnostics": self.invariant_diagnostics(),
        }

    def _validate_restart_state(self, state):
        schema_version = state.get("schema_version")
        if schema_version not in {1, self.RESTART_SCHEMA_VERSION}:
            raise ValueError(
                "Unsupported restart schema version: "
                f"{schema_version} not in {{1, {self.RESTART_SCHEMA_VERSION}}}"
            )
        restart_parameterization = state.get(
            "target_parameterization", self.TARGET_PARAMETERIZATION_FULL_ROTATION)
        if restart_parameterization != self.target_parameterization:
            raise ValueError(
                "Restart target_parameterization does not match this object: "
                f"{restart_parameterization} != {self.target_parameterization}."
            )
        expected = {
            "target_states": self.ntarget,
            "ncsf": self.ncsf,
            "nmo": self.nmo,
            "norb": self.norb,
            "nelec": self.nelec,
            "spin_symmetry": self.spin_symmetry,
            "spin_type": self.spin_type.name,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(
                    f"Restart {key} does not match this object: {state.get(key)} != {value}."
                )

    def load_restart_state(self, state):
        if "schema_version" not in state:
            if self.target_parameterization != self.TARGET_PARAMETERIZATION_FULL_ROTATION:
                raise ValueError(
                    "Legacy target-state restart files can only be loaded into "
                    "target_parameterization='full_rotation'."
                )
            if state["target_states"] != self.ntarget:
                raise ValueError("Restart target-state count does not match this object.")
        else:
            self._validate_restart_state(state)
        with torch.no_grad():
            self.orbital_rotation_params.copy_(torch.as_tensor(
                state["orbital_rotation_params"]).to(self.orbital_rotation_params))
            self.target_rotation_params.copy_(torch.as_tensor(
                state["target_rotation_params"]).to(self.target_rotation_params))
        self.enforce_invariants()

    def save_restart_file(self, path):
        state = self.restart_state()
        state["orbital_rotation_params"] = state["orbital_rotation_params"].tolist()
        state["target_rotation_params"] = state["target_rotation_params"].tolist()
        path = Path(path)
        path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def load_restart_file(self, path):
        self.load_restart_state(json.loads(Path(path).read_text(encoding="utf-8")))


__all__ = [
    "AOGridBatch", "CASGuessComponents", "OneBodyOperatorTable",
    "MultistateMatrixDensity", "MultistateMatrixDensityKohnSham",
    "MultistateMatrixDensityCAS", "TargetStateMultistateMatrixDensityCAS",
    "antisymmetric_matrix", "cas_guess_components", "stiefel_tangent_block_matrix", "orbital_guess",
    "reorder_active_orbitals",
]
