"""Collinear spin choices for LDMA."""

from enum import Enum
import numpy


class SpinType(str, Enum):
    """Supported collinear treatments of the matrix density."""

    UNPOLARIZED = "spin_unpolarized"
    POLARIZED = "spin_polarized"


def merge_multiplet_energies(energies, spin_multiplicities):
    energies_merged = []
    multiplicities_merged = []
    index = 0
    while index < len(energies):
        multiplicity = spin_multiplicities[index]
        numpy.testing.assert_allclose(
            energies[index:index + multiplicity] - energies[index],
            numpy.zeros(multiplicity), atol=1.0e-5)
        energies_merged.append(energies[index])
        multiplicities_merged.append(multiplicity)
        index += multiplicity
    return numpy.asarray(energies_merged), numpy.asarray(multiplicities_merged)


def index_within_multiplicity(spin_multiplicities):
    indices = numpy.zeros_like(spin_multiplicities, dtype=int)
    for multiplicity in numpy.unique(spin_multiplicities):
        counter = 1
        for index, value in enumerate(spin_multiplicities):
            if value == multiplicity:
                indices[index] = counter
                counter += 1
    return indices


__all__ = ["SpinType", "merge_multiplet_energies", "index_within_multiplicity"]
