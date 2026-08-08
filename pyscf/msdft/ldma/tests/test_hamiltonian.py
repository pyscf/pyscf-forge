from unittest import mock

import pytest
import torch
from pyscf import dft, gto

from pyscf.msdft.ldma import (HamiltonianSemilocal,
                              MultistateMatrixDensityCAS,
                              MultistateMatrixDensityKohnSham, SpinType)


def h2_density():
    mol = gto.M(atom="H 0 0 -0.35; H 0 0 0.35", basis="sto-3g",
                spin=0, verbose=0)
    density = MultistateMatrixDensityCAS.from_guess(
        mol, 2, 2, spin_symmetry=True, spin_type=SpinType.POLARIZED,
        guess="hcore"
    )
    return mol, density


def test_grid_chunking_and_cache_preserve_hamiltonian():
    mol, density = h2_density()
    single = HamiltonianSemilocal(mol, spin_type=SpinType.POLARIZED,
                                  grid_level=0, grid_chunks=1)
    chunked = HamiltonianSemilocal(mol, spin_type=SpinType.POLARIZED,
                                   grid_level=0, grid_chunks=3)
    torch.testing.assert_close(single(density), chunked(density), atol=1.0e-10,
                               rtol=1.0e-10)
    dm = density.density_matrices_ao()
    with mock.patch("pyscf.msdft.ldma.dft.hamiltonian.numint.eval_ao",
                    wraps=__import__("pyscf").dft.numint.eval_ao) as eval_ao:
        first = chunked._get_ao_grid_cache(dm.dtype, dm.device, density)
        second = chunked._get_ao_grid_cache(dm.dtype, dm.device, density)
    assert first is second
    assert eval_ao.call_count == 0  # Existing matrix evaluation populated the cache.
    assert all(batch.grad_ao_value is None for batch in first)


def test_auto_chunking_respects_point_limit():
    mol, density = h2_density()
    hamiltonian = HamiltonianSemilocal(
        mol, spin_type=SpinType.POLARIZED, grid_level=0, grid_chunks="auto",
        max_grid_points_per_chunk=8,
    )
    assert hamiltonian._resolve_grid_chunks(torch.double, torch.device("cpu"), density) > 1


def test_hamiltonian_rejects_mismatched_spin_type():
    mol, density = h2_density()
    hamiltonian = HamiltonianSemilocal(
        mol, spin_type=SpinType.UNPOLARIZED, grid_level=0)
    with pytest.raises(ValueError, match="same spin_type"):
        hamiltonian(density)


def test_default_unpolarized_hamiltonian_matches_pyscf_rks():
    mol = gto.M(atom="H 0 0 -0.35; H 0 0 0.35", basis="sto-3g",
                spin=0, verbose=0)
    mean_field = dft.RKS(mol)
    mean_field.xc = "LDA_X,LDA_C_CHACHIYO"
    mean_field.grids.level = 0
    mean_field.verbose = 0
    mean_field.kernel()
    density = MultistateMatrixDensityKohnSham.from_guess(
        mol, guess=mean_field.mo_coeff)
    hamiltonian = HamiltonianSemilocal(mol, grid_level=0)
    assert hamiltonian.spin_type is SpinType.UNPOLARIZED
    torch.testing.assert_close(
        hamiltonian(density)[0, 0],
        torch.tensor(mean_field.e_tot, dtype=torch.double),
        rtol=1.0e-7, atol=1.0e-7)
