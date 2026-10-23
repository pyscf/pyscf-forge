"""Collinear LDMA Hamiltonians and state-averaged optimization."""

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy
import torch
from pyscf import dft
from pyscf.dft import numint

from ..nn.functional import trace_average
from ..optim.torch_optimizer import TorchLBFGSOptimizer, WrappedOptimizer
from .density import AOGridBatch, MultistateMatrixDensity, TargetStateMultistateMatrixDensityCAS
from .hartree import HartreeFunctionalAO
from .integrals import OneElectronIntegralCache
from .kinetic import KineticFunctionalAO
from .nuclear import NuclearFunctionalAO
from .spin import SpinType
from .xc import lda_c_chachiyo, lda_x_dirac


class Hamiltonian(ABC):
    @abstractmethod
    def matrix_elements(self, msmd: MultistateMatrixDensity):
        pass


class HamiltonianSemilocal(Hamiltonian):
    """Semilocal Hamiltonian for unpolarized or collinearly polarized densities."""

    def __init__(self, mol, kinetic_functional=None, exchange_functional=lda_x_dirac,
                 correlation_functional=lda_c_chachiyo, exact_exchange_functional=None,
                 exchange_correlation_functional=None,
                 spin_type=SpinType.UNPOLARIZED, grid_level=3,
                 grid_chunks=1, target_memory_fraction=None,
                 max_grid_points_per_chunk=None, min_grid_points_per_chunk=1):
        if not isinstance(spin_type, SpinType):
            raise ValueError("spin_type must be a SpinType")
        if isinstance(grid_chunks, int):
            if grid_chunks < 1:
                raise ValueError("grid_chunks must be at least 1")
        elif grid_chunks not in (None, "auto"):
            raise ValueError("grid_chunks must be an integer, 'auto', or None")
        if min_grid_points_per_chunk < 1:
            raise ValueError("min_grid_points_per_chunk must be at least 1")
        if max_grid_points_per_chunk is not None and max_grid_points_per_chunk < min_grid_points_per_chunk:
            raise ValueError("max_grid_points_per_chunk is smaller than the minimum")
        self.mol = mol
        self.spin_type = spin_type
        self.kinetic = kinetic_functional
        self.exchange = exchange_functional
        self.correlation = correlation_functional
        self.exact_exchange = exact_exchange_functional
        self.exchange_correlation = exchange_correlation_functional
        self.grid_level = grid_level
        self.grid_chunks = grid_chunks
        self.target_memory_fraction = target_memory_fraction
        self.max_grid_points_per_chunk = max_grid_points_per_chunk
        self.min_grid_points_per_chunk = min_grid_points_per_chunk
        self.grids = dft.gen_grid.Grids(mol)
        self.grids.level = grid_level
        self.grids.build()
        self.nuclear_ao = NuclearFunctionalAO(mol)
        self.hartree_ao = HartreeFunctionalAO(mol)
        self.kinetic_ao = KineticFunctionalAO(mol)
        functionals = (self.kinetic, self.exchange, self.correlation, self.exchange_correlation)
        self.need_laplacian = any(getattr(functional, "need_laplacian", False) for functional in functionals)
        self.need_gradient = self.need_laplacian or any(
            getattr(functional, "need_gradient", False) for functional in functionals
        )
        self._ao_grid_cache = None
        self._ao_grid_cache_key = None
        self._resolved_grid_chunks = None

    def __call__(self, msmd):
        return self.matrix_elements(msmd)

    @property
    def resolved_grid_chunks(self):
        return self._resolved_grid_chunks

    def _estimate_bytes_per_grid_point(self, dtype, nstate):
        element_size = torch.empty((), dtype=dtype).element_size()
        derivative_factor = 5 if self.need_laplacian else 4 if self.need_gradient else 1
        channels = 4 if self.spin_type is SpinType.POLARIZED else 3
        return int(element_size * 4 * (self.mol.nao_nr() * derivative_factor + channels * nstate**2 + 3 * nstate))

    def _available_memory_bytes(self, device):
        if device.type == "cuda" and torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info(device)
            return int(free * (self.target_memory_fraction or 0.35))
        try:
            import psutil
            return int(psutil.virtual_memory().available * (self.target_memory_fraction or 0.5))
        except ImportError:
            return 1024**3

    def _resolve_grid_chunks(self, dtype, device, msmd=None):
        if isinstance(self.grid_chunks, int):
            return self.grid_chunks
        if msmd is None:
            return 1
        ngrids = len(self.grids.coords)
        budget = self._available_memory_bytes(device)
        points = max(1, budget // max(1, self._estimate_bytes_per_grid_point(dtype, msmd.number_of_states)))
        points = max(points, self.min_grid_points_per_chunk)
        if self.max_grid_points_per_chunk is not None:
            points = min(points, self.max_grid_points_per_chunk)
        return max(1, int(numpy.ceil(ngrids / points)))

    def _get_ao_grid_cache(self, dtype, device, msmd=None):
        chunks = self._resolve_grid_chunks(dtype, device, msmd)
        chunks = min(chunks, max(1, len(self.grids.coords)))
        self._resolved_grid_chunks = chunks
        key = (dtype, device, chunks, self.need_gradient, self.need_laplacian)
        if key == self._ao_grid_cache_key and self._ao_grid_cache is not None:
            return self._ao_grid_cache
        deriv = 2 if self.need_laplacian else 1 if self.need_gradient else 0
        batches = []
        for coords, weights in zip(numpy.array_split(self.grids.coords, chunks),
                                   numpy.array_split(self.grids.weights, chunks)):
            ao = torch.from_numpy(numint.eval_ao(self.mol, coords, deriv=deriv)).to(dtype=dtype, device=device)
            weights = torch.from_numpy(weights).to(dtype=dtype, device=device)
            if deriv == 0:
                batches.append(AOGridBatch(weights, ao))
            else:
                laplacian = ao[4] + ao[7] + ao[9] if self.need_laplacian else None
                batches.append(AOGridBatch(weights, ao[0], ao[1:4], laplacian))
        self._ao_grid_cache_key = key
        self._ao_grid_cache = batches
        return batches

    @staticmethod
    def _scale(value, factor):
        return None if value is None else factor * value

    def _unpolarized_xc(self, density, gradient, laplacian):
        if self.exchange_correlation is not None:
            return self.exchange_correlation(density, gradient, laplacian)
        result = 0.0
        if self.exchange is not None:
            result = result + 2.0 * self.exchange(
                density / 2.0, self._scale(gradient, 0.5), self._scale(laplacian, 0.5)
            )
        if self.correlation is not None:
            result = result + self.correlation(density, gradient, laplacian)
        return result

    def _polarized_xc(self, spin_density, spin_gradient, spin_laplacian, density, gradient, laplacian):
        if self.exchange_correlation is not None:
            raise NotImplementedError("fused XC is available only for SpinType.UNPOLARIZED")
        result = 0.0
        if self.exchange is not None:
            result = result + self.exchange(
                spin_density[0, 0], None if spin_gradient is None else spin_gradient[0, 0],
                None if spin_laplacian is None else spin_laplacian[0, 0]
            ) + self.exchange(
                spin_density[1, 1], None if spin_gradient is None else spin_gradient[1, 1],
                None if spin_laplacian is None else spin_laplacian[1, 1]
            )
        if self.correlation is not None:
            result = result + self.correlation(density, gradient, laplacian)
        return result

    def matrix_elements(self, msmd):
        if self.mol is not msmd.mol:
            raise ValueError("Hamiltonian and matrix density must use the same molecule")
        if hasattr(msmd, "spin_type") and msmd.spin_type is not self.spin_type:
            raise ValueError("Hamiltonian and matrix density must use the same spin_type")
        spin_dm = msmd.density_matrices_ao()
        dm = torch.einsum("ss...->...", spin_dm)
        hamiltonian = self.nuclear_ao(dm) + self.hartree_ao(dm)
        hamiltonian = hamiltonian + self.mol.get_enuc() * torch.eye(
            msmd.number_of_states, dtype=dm.dtype, device=dm.device
        )
        if self.kinetic is None:
            hamiltonian = hamiltonian + self.kinetic_ao(dm)
        if self.exact_exchange is not None:
            if self.spin_type is SpinType.UNPOLARIZED:
                hamiltonian = hamiltonian + 2.0 * self.exact_exchange(dm / 2.0)
            else:
                hamiltonian = hamiltonian + self.exact_exchange(spin_dm[0, 0]) + self.exact_exchange(spin_dm[1, 1])
        for batch in self._get_ao_grid_cache(spin_dm.dtype, spin_dm.device, msmd):
            spin_density, spin_gradient, spin_laplacian = msmd.evaluate_from_ao_batch(
                batch, spin_dm, self.need_gradient, self.need_laplacian
            )
            density = torch.einsum("ss...->...", spin_density)
            gradient = None if spin_gradient is None else torch.einsum("ss...->...", spin_gradient)
            laplacian = None if spin_laplacian is None else torch.einsum("ss...->...", spin_laplacian)
            if self.spin_type is SpinType.UNPOLARIZED:
                xc = self._unpolarized_xc(density, gradient, laplacian)
            else:
                xc = self._polarized_xc(spin_density, spin_gradient, spin_laplacian,
                                        density, gradient, laplacian)
            weights = batch.weights[:, None, None]
            hamiltonian = hamiltonian + torch.sum(xc * weights, dim=0)
            if self.kinetic is not None:
                hamiltonian = hamiltonian + torch.sum(
                    self.kinetic(density, gradient, laplacian) * weights, dim=0
                )
        if not torch.isfinite(hamiltonian).all():
            raise RuntimeError("Hamiltonian contains non-finite values")
        return hamiltonian


class HamiltonianTargetStateLDMA(Hamiltonian):
    """Unpolarized target-state Hamiltonian with MO integral contraction."""

    def __init__(self, mol, exchange_functional=lda_x_dirac,
                 correlation_functional=lda_c_chachiyo,
                 exchange_correlation_functional=None,
                 spin_type=SpinType.UNPOLARIZED, grid_level=3, grid_chunks=1,
                 hartree_backend="ao"):
        if spin_type is not SpinType.UNPOLARIZED:
            raise NotImplementedError("target-state LDMA supports SpinType.UNPOLARIZED only")
        if hartree_backend != "ao":
            raise ValueError("the core port provides only hartree_backend='ao'")
        if not isinstance(grid_chunks, int) or grid_chunks < 1:
            raise ValueError("grid_chunks must be an integer of at least 1")
        self.mol = mol
        self.exchange = exchange_functional
        self.correlation = correlation_functional
        self.exchange_correlation = exchange_correlation_functional
        self.spin_type = spin_type
        self.grid_chunks = grid_chunks
        self.grids = dft.gen_grid.Grids(mol)
        self.grids.level = grid_level
        self.grids.build()
        self.one_electron_cache = OneElectronIntegralCache(mol)
        self.hartree = HartreeFunctionalAO(mol)

    def __call__(self, msmd):
        return self.matrix_elements(msmd)

    def one_electron_matrix(self, msmd):
        return self.one_electron_cache.matrix_elements_from_gamma(
            msmd.transition_1rdm_mo(), msmd.orbital_coefficients()
        )

    def hartree_matrix(self, msmd):
        return self.hartree(torch.einsum("ss...->...", msmd.density_matrices_ao()))

    def exchange_correlation_matrix(self, msmd):
        gamma = msmd.transition_1rdm_mo()
        result = torch.zeros((msmd.number_of_states, msmd.number_of_states),
                             dtype=gamma.dtype, device=gamma.device)
        chunks = min(self.grid_chunks, max(1, len(self.grids.coords)))
        for coords, weights in zip(numpy.array_split(self.grids.coords, chunks),
                                   numpy.array_split(self.grids.weights, chunks)):
            ao = torch.from_numpy(numint.eval_ao(self.mol, coords, deriv=0)).to(gamma)
            weights = torch.from_numpy(weights).to(gamma)
            spin_density, _, _ = msmd.evaluate_from_mo_grid_batch(AOGridBatch(weights, ao), gamma)
            density = torch.einsum("ss...->...", spin_density)
            if self.exchange_correlation is not None:
                xc = self.exchange_correlation(density, None, None)
            else:
                xc = 0.0
                if self.exchange is not None:
                    xc = xc + 2.0 * self.exchange(density / 2.0, None, None)
                if self.correlation is not None:
                    xc = xc + self.correlation(density, None, None)
            result = result + torch.sum(xc * weights[:, None, None], dim=0)
        return result

    def matrix_elements(self, msmd: TargetStateMultistateMatrixDensityCAS):
        if self.mol is not msmd.mol:
            raise ValueError("Hamiltonian and matrix density must use the same molecule")
        identity = torch.eye(msmd.number_of_states, dtype=msmd.orbital_rotation_params.dtype,
                             device=msmd.orbital_rotation_params.device)
        return (self.mol.get_enuc() * identity + self.one_electron_matrix(msmd)
                + self.hartree_matrix(msmd) + self.exchange_correlation_matrix(msmd))


def minimize_subspace_energy(hamiltonian, msmd, state_average=None, optimizer="wrapped",
                             enforce_invariants=True, convergence=None, return_info=False,
                             **optimizer_kwds):
    if optimizer in ("wrapped", "numpy", "bfgs"):
        optimizer_instance = WrappedOptimizer(msmd.parameters(), **optimizer_kwds)
    elif optimizer == "torch_lbfgs":
        optimizer_instance = TorchLBFGSOptimizer(msmd.parameters(), **optimizer_kwds)
    else:
        raise ValueError("optimizer must be 'wrapped' or 'torch_lbfgs'")
    for parameter in msmd.parameters():
        parameter.requires_grad_(True)
    if enforce_invariants and hasattr(msmd, "enforce_invariants"):
        msmd.enforce_invariants()
    convergence = convergence or {}
    history = []

    def closure():
        optimizer_instance.zero_grad()
        with torch.no_grad():
            weights = msmd.state_weights()
        value = trace_average(
            hamiltonian(msmd), weights=weights, subspace_dim=state_average
        )
        if convergence or return_info:
            value.backward(retain_graph=True)
            gradients = [parameter.grad.detach().norm() for parameter in msmd.parameters()
                         if parameter.grad is not None]
            history.append({"loss": float(value.detach()),
                            "grad_norm": float(torch.linalg.norm(torch.stack(gradients))) if gradients else 0.0})
            optimizer_instance.zero_grad()
        return value

    optimizer_instance.step(closure)
    msmd.optimizer_telemetry = optimizer_instance.telemetry.as_dict()
    final = history[-1] if history else {"loss": None, "grad_norm": None}
    loss_delta = abs(history[-1]["loss"] - history[-2]["loss"]) if len(history) > 1 else None
    gtol = convergence.get("gtol")
    ftol = convergence.get("ftol")
    converged_grad = None if gtol is None or final["grad_norm"] is None else final["grad_norm"] <= gtol
    converged_loss = None if ftol is None or loss_delta is None else loss_delta <= ftol
    checks = [value for value in (converged_grad, converged_loss) if value is not None]
    msmd.optimizer_convergence = {
        "closure_count": len(history), "final_loss": final["loss"],
        "final_grad_norm": final["grad_norm"], "loss_delta": loss_delta,
        "gtol": gtol, "ftol": ftol, "converged_grad": converged_grad,
        "converged_loss": converged_loss,
        "converged": all(checks) if checks else None,
        "history_tail": history[-10:],
    }
    if enforce_invariants and hasattr(msmd, "enforce_invariants"):
        msmd.enforce_invariants()
    matrix = hamiltonian(msmd)
    energies, eigenvectors = torch.linalg.eigh(matrix)
    msmd.basis_transformation(eigenvectors.T)
    if enforce_invariants and hasattr(msmd, "enforce_invariants"):
        msmd.enforce_invariants()
    if return_info:
        return energies, msmd, {"optimizer_telemetry": msmd.optimizer_telemetry,
                                "optimizer_convergence": msmd.optimizer_convergence}
    return energies, msmd


__all__ = ["Hamiltonian", "HamiltonianSemilocal", "HamiltonianTargetStateLDMA",
           "minimize_subspace_energy"]
