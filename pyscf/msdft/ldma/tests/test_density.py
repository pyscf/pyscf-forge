import numpy
import pytest
import torch
from pyscf import gto

from pyscf.msdft.ldma import (MultistateMatrixDensityCAS,
                              MultistateMatrixDensityKohnSham, SpinType)


@pytest.fixture
def h2():
    return gto.M(atom="H 0 0 -0.35; H 0 0 0.35", basis="sto-3g", spin=0,
                 verbose=0)


def test_spin_type_is_strictly_collinear():
    assert list(SpinType) == [SpinType.UNPOLARIZED, SpinType.POLARIZED]


@pytest.mark.parametrize("spin_type", [SpinType.UNPOLARIZED, SpinType.POLARIZED])
def test_cas_density_is_normalized_and_collinear(h2, spin_type):
    density = MultistateMatrixDensityCAS.from_guess(
        h2, 2, 2, spin_symmetry=True, spin_type=spin_type, guess="hcore"
    )
    dm = density.density_matrices_ao()
    overlap = torch.from_numpy(h2.intor_symmetric("int1e_ovlp")).to(dm)
    integrated = torch.einsum("ssabij,ab->ij", dm, overlap)
    torch.testing.assert_close(
        integrated,
        h2.tot_electrons() * torch.eye(density.number_of_states, dtype=dm.dtype),
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    assert torch.count_nonzero(dm[0, 1]) == 0
    assert torch.count_nonzero(dm[1, 0]) == 0


def test_ao_batch_matches_direct_density_evaluation(h2):
    density = MultistateMatrixDensityKohnSham.from_guess(h2, guess="hcore")
    coords = numpy.array([[0.0, 0.0, 0.0], [0.1, -0.2, 0.3]])
    value, gradient, laplacian = density.evaluate(
        coords, need_gradient=False, need_laplacian=False
    )
    assert value.shape[2] == len(coords)
    assert gradient is None
    assert laplacian is None


def test_density_gradient_matches_finite_difference(h2):
    density = MultistateMatrixDensityCAS.from_guess(h2, 2, 2, guess="hcore")
    coords = numpy.array([[0.21, -0.13, 0.34]])
    value, gradient, _ = density.evaluate(coords, need_gradient=True, need_laplacian=False)
    step = 1.0e-5
    for axis in range(3):
        direction = numpy.zeros(3)
        direction[axis] = step
        plus = density.evaluate(coords + direction, need_gradient=False,
                                need_laplacian=False)[0]
        minus = density.evaluate(coords - direction, need_gradient=False,
                                 need_laplacian=False)[0]
        torch.testing.assert_close(
            gradient[:, :, :, axis], (plus - minus) / (2.0 * step),
            rtol=1.0e-5, atol=1.0e-6)


def test_density_laplacian_matches_finite_difference(h2):
    density = MultistateMatrixDensityCAS.from_guess(h2, 2, 2, guess="hcore")
    coords = numpy.array([[0.21, -0.13, 0.34]])
    value, _, laplacian = density.evaluate(
        coords, need_gradient=False, need_laplacian=True)
    step = 1.0e-3
    numerical = torch.zeros_like(value)
    for axis in range(3):
        direction = numpy.zeros(3)
        direction[axis] = step
        plus = density.evaluate(coords + direction, need_gradient=False,
                                need_laplacian=False)[0]
        minus = density.evaluate(coords - direction, need_gradient=False,
                                 need_laplacian=False)[0]
        numerical += (plus - 2.0 * value + minus) / step**2
    torch.testing.assert_close(laplacian, numerical, rtol=2.0e-4, atol=2.0e-5)


def test_density_parameter_autograd(h2):
    original = MultistateMatrixDensityCAS.from_guess(h2, 2, 2, guess="hcore")
    orbital = original.orbital_coefficients().detach()
    states = original.state_coefficients().detach()
    coords = numpy.array([[0.2, 0.1, -0.3]])

    def evaluate(orbital_params, state_params):
        density = MultistateMatrixDensityCAS(
            h2, 2, 2, orbital, orbital_params, states, state_params)
        return density.evaluate(coords, need_gradient=False, need_laplacian=False)[0]

    orbital_params = torch.zeros_like(original.orbital_rotation_params, requires_grad=True)
    state_params = torch.zeros_like(original.state_rotation_params, requires_grad=True)
    assert torch.autograd.gradcheck(evaluate, (orbital_params, state_params),
                                    eps=1.0e-6, atol=1.0e-5)


def test_cas_basis_transformation(h2):
    density = MultistateMatrixDensityCAS.from_guess(h2, 2, 2, guess="hcore")
    coords = numpy.array([[0.1, 0.2, 0.3]])
    before = density.evaluate(coords, need_gradient=False, need_laplacian=False)[0]
    generator = torch.randn(density.number_of_states, density.number_of_states,
                            dtype=torch.double)
    rotation = torch.matrix_exp(generator - generator.T)
    expected = torch.einsum("ia,strab,bj->strij", rotation, before, rotation.T)
    density.basis_transformation(rotation)
    after = density.evaluate(coords, need_gradient=False, need_laplacian=False)[0]
    torch.testing.assert_close(after, expected)


def test_invalid_active_spaces_restore_source_checks():
    lithium = gto.M(atom="Li 0 0 0", basis="sto-3g", spin=1, verbose=0)
    with pytest.raises(ValueError, match="odd.*even|Odd.*even"):
        MultistateMatrixDensityCAS.from_guess(lithium, 2, 2)
    hydrogen = gto.M(atom="H 0 0 0", basis="sto-3g", spin=1, verbose=0)
    with pytest.raises(ValueError, match="More active electrons"):
        MultistateMatrixDensityCAS.from_guess(hydrogen, 2, 3)
    molecule = gto.M(atom="H 0 0 -0.35; H 0 0 0.35", basis="sto-3g",
                     spin=0, verbose=0)
    with pytest.raises(ValueError, match="spin electron counts"):
        MultistateMatrixDensityCAS.from_guess(molecule, 1, (2, 0))
    with pytest.raises(ValueError, match="More active orbitals"):
        MultistateMatrixDensityCAS.from_guess(hydrogen, 4, 1)
