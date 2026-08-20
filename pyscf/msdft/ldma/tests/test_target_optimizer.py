import torch
import copy
import pytest
import numpy
from unittest import mock
from pyscf.dft import numint
from pyscf import gto
from pyscf.scf.hf import get_hcore

from pyscf.msdft.ldma import (HamiltonianTargetStateLDMA, SpinType,
                              TargetStateMultistateMatrixDensityCAS,
                              minimize_subspace_energy)
from pyscf.msdft.ldma.dft.integrals import OneElectronIntegralCache
from pyscf.msdft.ldma.dft.active_space import ActiveSpace
from pyscf.msdft.ldma.dft.density import AOGridBatch
from pyscf.msdft.ldma.dft.density import cas_guess_components


def target_density(parameterization="full_rotation"):
    mol = gto.M(atom="H 0 0 -0.35; H 0 0 0.35", basis="sto-3g",
                spin=0, verbose=0)
    density = TargetStateMultistateMatrixDensityCAS.from_guess(
        mol, 2, 2, target_states=2, spin_symmetry=True,
        spin_type=SpinType.UNPOLARIZED, guess="hcore",
        target_parameterization=parameterization,
    )
    return mol, density


def test_target_parameterizations_agree_at_origin():
    mol, full = target_density("full_rotation")
    _, stiefel = target_density("stiefel_k")
    torch.testing.assert_close(full.transition_1rdm_mo(), stiefel.transition_1rdm_mo())
    assert full.target_orthonormality_error() < 1.0e-12
    components = cas_guess_components(mol, 2, 2)
    assert components.nstate == components.det_to_csf.size(1)
    assert components.ndet == components.det_to_csf.size(0)


def test_target_full_space_matches_dense_without_dense_table_construction():
    mol, _ = target_density()
    from pyscf.msdft.ldma import MultistateMatrixDensityCAS
    dense = MultistateMatrixDensityCAS.from_guess(mol, 2, 2, guess="hcore")
    full = TargetStateMultistateMatrixDensityCAS.from_guess(
        mol, 2, 2, target_states=dense.number_of_states, guess="hcore")
    torch.testing.assert_close(full.density_matrices_ao(), dense.density_matrices_ao())
    with mock.patch.object(ActiveSpace, "matrix_density_mo",
                           side_effect=AssertionError("dense path used")):
        sparse = TargetStateMultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, target_states=2, guess="hcore")
    assert sparse.one_body_values.numel() > 0
    assert sparse.one_body_shape[-2:] == (sparse.number_of_determinants,) * 2
    assert sparse._last_orbital_grid_density is None


def test_target_density_autograd():
    mol, original = target_density()
    coords = torch.from_numpy(numint.eval_ao(
        mol, numpy.asarray([[0.1, -0.2, 0.3]]), deriv=0)).double()
    batch = AOGridBatch(None, coords)

    def evaluate(orbital_params, target_params):
        density = TargetStateMultistateMatrixDensityCAS(
            mol, 2, 2, 2, original.mo_coeff_guess, orbital_params,
            original.det_to_csf, original.s2_matrix, original.active_space,
            target_rotation_params=target_params)
        return density.evaluate_from_mo_grid_batch(batch)[0]

    orbital_params = torch.zeros_like(original.orbital_rotation_params,
                                      requires_grad=True)
    target_params = torch.zeros_like(original.target_rotation_params,
                                     requires_grad=True)
    assert torch.autograd.gradcheck(evaluate, (orbital_params, target_params),
                                    eps=1.0e-6, atol=1.0e-5)


def test_cached_mo_integrals_match_ao_contraction():
    mol, density = target_density()
    cache = OneElectronIntegralCache(mol)
    from_mo = cache.matrix_elements_from_gamma(
        density.transition_1rdm_mo(), density.orbital_coefficients()
    )
    dm = torch.einsum("ss...->...", density.density_matrices_ao())
    from_ao = torch.einsum("ab,abij->ij", cache.integrals_ao, dm)
    torch.testing.assert_close(from_mo, from_ao, atol=1.0e-10, rtol=1.0e-10)


def test_one_electron_cache_includes_scalar_ecp():
    mol = gto.M(
        atom="Na 0 0 0", basis="lanl2dz", ecp="lanl2dz",
        spin=1, verbose=0)
    cache = OneElectronIntegralCache(mol)
    torch.testing.assert_close(
        cache.integrals_ao,
        torch.from_numpy(get_hcore(mol)).double(),
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_torch_optimizer_reports_telemetry_and_preserves_targets():
    mol, density = target_density()
    hamiltonian = HamiltonianTargetStateLDMA(mol, grid_level=0, grid_chunks=2)
    energies, optimized, info = minimize_subspace_energy(
        hamiltonian, density, optimizer="torch_lbfgs", maxiter=1,
        return_info=True, convergence={"gtol": 1.0e6},
    )
    assert torch.isfinite(energies).all()
    assert optimized.target_orthonormality_error() < 1.0e-10
    assert info["optimizer_telemetry"]["numpy_parameter_transfers"] == 0
    assert info["optimizer_telemetry"]["closure_calls"] > 0
    assert "converged_grad" in info["optimizer_convergence"]


def test_restart_schema_2_round_trip_and_validation(tmp_path):
    _, density = target_density()
    with torch.no_grad():
        density.orbital_rotation_params.add_(0.01)
        density.target_rotation_params.add_(0.02)
    state = density.restart_state()
    assert state["schema_version"] == 2
    _, restored = target_density()
    restored.load_restart_state(state)
    torch.testing.assert_close(restored.orbital_rotation_params,
                               density.orbital_rotation_params)
    torch.testing.assert_close(restored.target_rotation_params,
                               density.target_rotation_params)
    path = density.save_restart_file(tmp_path / "restart.json")
    restored.load_restart_file(path)
    invalid = copy.deepcopy(state)
    invalid["ncsf"] += 1
    with pytest.raises(ValueError, match="ncsf"):
        restored.load_restart_state(invalid)
    invalid = copy.deepcopy(state)
    invalid["schema_version"] = 99
    with pytest.raises(ValueError, match="Unsupported restart schema"):
        restored.load_restart_state(invalid)


def test_legacy_restart_only_supports_full_rotation():
    _, full = target_density("full_rotation")
    legacy = {
        "target_states": full.number_of_states,
        "orbital_rotation_params": full.orbital_rotation_params.detach().clone(),
        "target_rotation_params": full.target_rotation_params.detach().clone(),
    }
    full.load_restart_state(legacy)
    _, stiefel = target_density("stiefel_k")
    with pytest.raises(ValueError, match="Legacy"):
        stiefel.load_restart_state(legacy)


def test_schema_1_restart_defaults_to_full_rotation():
    _, density = target_density("full_rotation")
    state = density.restart_state()
    state["schema_version"] = 1
    state.pop("target_parameterization")
    _, restored = target_density("full_rotation")
    restored.load_restart_state(state)
    torch.testing.assert_close(restored.target_rotation_params,
                               density.target_rotation_params)
