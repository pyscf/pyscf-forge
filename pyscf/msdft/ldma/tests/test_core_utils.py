import numpy
import pytest
import torch

from pyscf.msdft.ldma.dft.configurations import (
    configurations_by_excitation_levels, excitation_levels, matrix_density_mo,
    occupation_labels, total_spin_matrix)
from pyscf.msdft.ldma.dft.pure import LDA, PureXCFunctional
from pyscf.msdft.ldma.dft.spin import (index_within_multiplicity,
                                      merge_multiplet_energies)
from pyscf.msdft.ldma.nn.functional import robust_symmetric_eigh, trace_average
from pyscf.msdft.ldma.optim import ClosureProfiler


def test_robust_symmetric_eigh_validates_input():
    matrix = torch.tensor([[2.0, 0.3], [0.3, 1.0]], dtype=torch.double)
    values, vectors = robust_symmetric_eigh(matrix)
    torch.testing.assert_close(vectors @ torch.diag(values) @ vectors.T, matrix)
    with pytest.raises(ValueError, match="symmetric"):
        robust_symmetric_eigh(torch.tensor([[1.0, 1.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="finite"):
        robust_symmetric_eigh(torch.tensor([[1.0, 0.0], [0.0, float("nan")]]))


def test_weighted_trace_average_uses_a_symmetric_similarity_transform():
    matrix = torch.tensor([[2.0, 1.0], [1.0, 4.0]], dtype=torch.double)
    weights = torch.tensor([1.0, 3.0], dtype=torch.double)
    normalized = 2.0 * weights / weights.sum()
    sqrt_weights = torch.sqrt(normalized)
    expected = torch.linalg.eigvalsh(
        sqrt_weights[:, None] * matrix * sqrt_weights[None, :]
    )[0]
    torch.testing.assert_close(
        trace_average(matrix, weights=weights, subspace_dim=1).squeeze(), expected)
    with pytest.raises(ValueError, match="non-negative"):
        trace_average(matrix, weights=torch.tensor([1.0, -1.0]))


def test_configuration_reference_values():
    assert excitation_levels(4, 2) == [0, 1, 1, 1, 1, 2]
    assert configurations_by_excitation_levels(2, (1, 1), 1) == [(0, 0), (0, 1), (1, 0)]
    assert occupation_labels(2, 2) == ["2.", "ab", "ba", ".2"]
    numpy.testing.assert_allclose(numpy.linalg.eigvalsh(total_spin_matrix(2, 2)),
                                  [0.0, 0.0, 0.0, 2.0])
    density = matrix_density_mo(2, (1, 1))
    numpy.testing.assert_allclose(numpy.einsum("sppij->sij", density),
                                  numpy.stack([numpy.eye(4), numpy.eye(4)]))


def test_spin_multiplet_helpers():
    energies, multiplicities = merge_multiplet_energies(
        numpy.array([1.0, 2.0, 2.0, 2.0]), numpy.array([1, 3, 3, 3]))
    numpy.testing.assert_array_equal(energies, [1.0, 2.0])
    numpy.testing.assert_array_equal(multiplicities, [1, 3])
    numpy.testing.assert_array_equal(index_within_multiplicity([1, 3, 1, 1, 3]),
                                     [1, 1, 2, 3, 2])


def test_pure_functional_base_and_closure_profiler():
    density = torch.eye(2, dtype=torch.double)
    base = PureXCFunctional()
    torch.testing.assert_close(base.exchange_correlation(density, None),
                               torch.zeros_like(density))
    assert torch.isfinite(LDA().exchange_correlation(density, None)).all()
    profiler = ClosureProfiler()
    wrapped = profiler.wrap(lambda value: value + 1)
    assert wrapped(2) == 3
    assert profiler.as_dict()["closure_calls"] == 1
