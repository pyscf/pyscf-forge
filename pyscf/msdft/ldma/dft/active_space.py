"""Collinear determinant active spaces used by LDMA matrix densities."""

import numpy
from pyscf.fci.addons import _unpack_nelec
from pyscf.fci.cistring import make_strings


class ActiveSpaceError(ValueError):
    pass


def int2binvec(value, width):
    """Return a least-significant-bit-first binary vector of fixed width."""
    return numpy.asarray([(value >> index) & 1 for index in range(width)])


def _bits(value, width):
    return int2binvec(value, width)


def _annihilate(det, orbital):
    if not ((det >> orbital) & 1):
        return None
    phase = -1.0 if bin(det & ((1 << orbital) - 1)).count("1") % 2 else 1.0
    return det ^ (1 << orbital), phase


def _create(det, orbital):
    if (det >> orbital) & 1:
        return None
    phase = -1.0 if bin(det & ((1 << orbital) - 1)).count("1") % 2 else 1.0
    return det | (1 << orbital), phase


class ActiveSpace:
    """Determinants with one fixed spin projection and bounded excitation rank."""

    def __init__(self, norb, nelec, max_level=numpy.inf, spin_range=None):
        self.norb = norb
        self.max_level = max_level
        neleca, nelecb = _unpack_nelec(nelec)
        self.nelec = neleca + nelecb
        if spin_range is None:
            spin_range = [neleca - nelecb]
        self.spin_range = list(spin_range)
        for spin in self.spin_range:
            if spin % 2 != self.nelec % 2:
                raise ActiveSpaceError("2*Sz and the electron count must have the same parity")

    def slater_determinants(self):
        determinants = []
        orbitals = range(self.norb)
        for neleca in range(self.nelec + 1):
            nelecb = self.nelec - neleca
            if neleca > self.norb or nelecb > self.norb:
                continue
            if neleca - nelecb not in self.spin_range:
                continue
            reference = (((1 << neleca) - 1) << self.norb) | ((1 << nelecb) - 1)
            for alpha in make_strings(orbitals, neleca):
                for beta in make_strings(orbitals, nelecb):
                    det = (int(alpha) << self.norb) | int(beta)
                    if bin(det & ~reference).count("1") <= self.max_level:
                        determinants.append(det)
        return determinants

    def occupation_labels(self):
        labels = []
        for det in self.slater_determinants():
            occupations = _bits(det, 2 * self.norb)
            label = ""
            for orbital in range(self.norb):
                beta = occupations[orbital]
                alpha = occupations[self.norb + orbital]
                label += "2" if alpha and beta else "a" if alpha else "b" if beta else "."
            labels.append(label)
        return labels

    def spin_projection_sz(self):
        result = []
        for det in self.slater_determinants():
            occupations = _bits(det, 2 * self.norb)
            result.append(0.5 * (sum(occupations[self.norb:]) - sum(occupations[:self.norb])))
        return numpy.asarray(result)

    def total_spin_matrix(self):
        determinants = self.slater_determinants()
        size = len(determinants)
        matrix = numpy.zeros((size, size))
        # S^2 = Sz^2 + (S+S- + S-S+)/2, evaluated by explicit spin flips.
        index = {det: position for position, det in enumerate(determinants)}
        for ket_index, det in enumerate(determinants):
            sz = self.spin_projection_sz()[ket_index]
            matrix[ket_index, ket_index] += sz * sz
            for first_from_beta in (True, False):
                intermediate_terms = [(det, 1.0)]
                for from_beta in (first_from_beta, not first_from_beta):
                    next_terms = []
                    for current, coefficient in intermediate_terms:
                        for orbital in range(self.norb):
                            source = orbital if from_beta else self.norb + orbital
                            target = self.norb + orbital if from_beta else orbital
                            removed = _annihilate(current, source)
                            if removed is None:
                                continue
                            created = _create(removed[0], target)
                            if created is not None:
                                next_terms.append((created[0], coefficient * removed[1] * created[1]))
                    intermediate_terms = next_terms
                for bra_det, coefficient in intermediate_terms:
                    bra_index = index.get(bra_det)
                    if bra_index is not None:
                        matrix[bra_index, ket_index] += 0.5 * coefficient
        return 0.5 * (matrix + matrix.T)

    def matrix_density_mo(self):
        determinants = self.slater_determinants()
        index = {det: position for position, det in enumerate(determinants)}
        ndet = len(determinants)
        density = numpy.zeros((2, 2, self.norb, self.norb, ndet, ndet))
        for ket_index, det in enumerate(determinants):
            for spin in range(2):
                for p in range(self.norb):
                    for q in range(self.norb):
                        source = q if spin == 1 else self.norb + q
                        target = p if spin == 1 else self.norb + p
                        removed = _annihilate(det, source)
                        if removed is None:
                            continue
                        created = _create(removed[0], target)
                        if created is None:
                            continue
                        bra_index = index.get(created[0])
                        if bra_index is not None:
                            density[spin, spin, q, p, bra_index, ket_index] = removed[1] * created[1]
        return density

    def __repr__(self):
        return (
            f"ActiveSpace(active orbitals: {self.norb}, active electrons: {self.nelec}, "
            f"maximum excitation level: {self.max_level})"
        )


__all__ = ["ActiveSpace", "ActiveSpaceError", "int2binvec"]
