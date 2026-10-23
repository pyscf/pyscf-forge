"""Collinear determinant configuration utilities."""

import numpy
import warnings
from pyscf.fci.addons import _unpack_nelec
from pyscf.fci.cistring import gen_occslst, make_strings, num_strings
from pyscf.fci.direct_spin1 import trans_rdm1s
from pyscf.fci.spin_op import contract_ss


def excitation_levels(norb, nelec):
    if nelec < 0:
        raise ValueError("nelec must be non-negative")
    reference = (1 << nelec) - 1
    return [bin(int(det) & ~reference).count("1") for det in make_strings(range(norb), nelec)]


def configurations_by_excitation_levels(norb=2, nelec=2, max_level=numpy.inf):
    neleca, nelecb = _unpack_nelec(nelec)
    if neleca != nelecb and max_level < numpy.inf:
        warnings.warn(
            "For neleca != nelecb, an excitation-truncated space may contain states "
            "with incorrect spin.", RuntimeWarning)
    levels_a = excitation_levels(norb, neleca)
    levels_b = excitation_levels(norb, nelecb)
    return [
        (ia, ib)
        for ia in range(num_strings(norb, neleca))
        for ib in range(num_strings(norb, nelecb))
        if levels_a[ia] + levels_b[ib] <= max_level
    ]


def occupation_labels(norb=2, nelec=2, max_level=numpy.inf):
    neleca, nelecb = _unpack_nelec(nelec)
    occ_a = gen_occslst(range(norb), neleca)
    occ_b = gen_occslst(range(norb), nelecb)
    labels = []
    for ia, ib in configurations_by_excitation_levels(norb, nelec, max_level):
        label = ""
        for orbital in range(norb):
            alpha = orbital in occ_a[ia]
            beta = orbital in occ_b[ib]
            label += "2" if alpha and beta else "a" if alpha else "b" if beta else "."
        labels.append(label)
    return labels


def total_spin_matrix(norb=2, nelec=2, max_level=numpy.inf):
    neleca, nelecb = _unpack_nelec(nelec)
    na = num_strings(norb, neleca)
    nb = num_strings(norb, nelecb)
    selected = configurations_by_excitation_levels(norb, nelec, max_level)
    matrix = numpy.zeros((len(selected), len(selected)))
    for ket, (ja, jb) in enumerate(selected):
        vector = numpy.zeros((na, nb))
        vector[ja, jb] = 1.0
        contracted = contract_ss(vector.ravel(), norb, (neleca, nelecb))
        for bra, (ia, ib) in enumerate(selected):
            matrix[bra, ket] = contracted[ia, ib]
    return matrix


def matrix_density_mo(norb=2, nelec=2, max_level=numpy.inf):
    neleca, nelecb = _unpack_nelec(nelec)
    na = num_strings(norb, neleca)
    nb = num_strings(norb, nelecb)
    selected = configurations_by_excitation_levels(norb, nelec, max_level)
    density = numpy.zeros((2, norb, norb, len(selected), len(selected)))
    for ket, (ja, jb) in enumerate(selected):
        ket_vector = numpy.zeros((na, nb))
        ket_vector[ja, jb] = 1.0
        for bra, (ia, ib) in enumerate(selected):
            bra_vector = numpy.zeros((na, nb))
            bra_vector[ia, ib] = 1.0
            density[0, :, :, bra, ket], density[1, :, :, bra, ket] = trans_rdm1s(
                bra_vector.ravel(), ket_vector.ravel(), norb, nelec)
    return density


__all__ = ["excitation_levels", "configurations_by_excitation_levels",
           "occupation_labels", "total_spin_matrix", "matrix_density_mo"]
