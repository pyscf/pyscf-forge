#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: Bhavnesh Jangid

import unittest
import numpy as np
from numpy.testing import assert_allclose
from scipy import linalg

from pyscf.fci import cistring
from pyscf.siso import biortho


def _occupied_strings(norb, nelec):
    strings = cistring.make_strings(range(norb), nelec)
    return [[i for i in range(norb) if string & (1 << i)]
            for string in strings]


def _determinant_overlap(ci_left, ci_right, orbital_overlap, ncore,
                         ncas, nelec):
    """Direct determinant-minor reference for two restricted CAS states."""
    alpha = _occupied_strings(ncas, nelec[0])
    beta = _occupied_strings(ncas, nelec[1])
    core = list(range(ncore))
    value = 0.0

    for ia, occa_left in enumerate(alpha):
        occa_left = core + [ncore + i for i in occa_left]
        for ib, occb_left in enumerate(beta):
            occb_left = core + [ncore + i for i in occb_left]
            coefficient_left = ci_left[ia, ib].conjugate()
            for ja, occa_right in enumerate(alpha):
                occa_right = core + [ncore + i for i in occa_right]
                overlap_alpha = np.linalg.det(
                    orbital_overlap[np.ix_(occa_left, occa_right)])
                for jb, occb_right in enumerate(beta):
                    occb_right = core + [ncore + i for i in occb_right]
                    overlap_beta = np.linalg.det(
                        orbital_overlap[np.ix_(occb_left, occb_right)])
                    value += (coefficient_left * ci_right[ja, jb]
                              * overlap_alpha * overlap_beta)
    return value


class KnownValues(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(19)

    def _orthonormal_columns(self, nao, norb):
        matrix = self.rng.standard_normal((nao, norb))
        return np.linalg.qr(matrix)[0]

    def test_lu_and_orbital_transformations(self):
        """All transformation stages produce mutually biorthonormal MOs."""
        ncore, ncas = 2, 3
        nocc = ncore + ncas
        left = self._orthonormal_columns(8, nocc)
        right = self._orthonormal_columns(8, nocc)
        overlap = left.T @ right

        tra_left, tra_right = biortho.compute_trans_mat(
            overlap, (ncore, ncas), lu_threshold=1e-12)
        coeff_left = biortho.orbital_transformation(tra_left)
        coeff_right = biortho.orbital_transformation(tra_right)

        assert_allclose(
            coeff_left.T @ overlap @ coeff_right,
            np.eye(nocc),
            atol=1e-12,
        )

        # Exercise the LU routine directly, including a one-orbital block.
        lu_left, lu_right = biortho.lu_pp_decomposition(
            np.eye(3), np.eye(3), (1, 2))
        assert_allclose(lu_left, np.eye(3))
        assert_allclose(lu_right, np.eye(3))

    def test_identity_ci_transformation(self):
        ncore, ncas, nelec = 1, 4, (2, 1)
        ci = self.rng.standard_normal((6, 4))
        transformed = biortho.transform_ci(
            ci, np.eye(ncore + ncas), ncore, ncas, nelec)
        assert_allclose(transformed, ci, atol=1e-14)

    def test_same_active_space_and_ci_on_both_sides(self):
        ncore, ncas, nelec = 1, 3, (2, 1)
        nocc = ncore + ncas
        mo = self._orthonormal_columns(7, nocc)
        ci = self.rng.standard_normal((3, 3))
        ci /= np.linalg.norm(ci)

        values = biortho.biorthogonalize(
            mo, mo, ci, ci, np.eye(7), ncore, ncas, nelec, nelec)
        _, _, mo_left, mo_right, ci_left, ci_right = values

        assert_allclose(mo_left.T @ mo_right, np.eye(nocc), atol=1e-13)
        assert_allclose(ci_left, ci, atol=1e-13)
        assert_allclose(ci_right, ci, atol=1e-13)
        assert_allclose(np.vdot(ci_left, ci_right), 1.0, atol=1e-13)

    def test_different_active_orbital_spaces(self):
        """The counter-transformed CI overlap equals the determinant result."""
        ncore, ncas, nelec = 1, 3, (2, 1)
        nocc = ncore + ncas
        mo_left = self._orthonormal_columns(7, nocc)
        mo_right = self._orthonormal_columns(7, nocc)
        ci_left = self.rng.standard_normal((3, 3))
        ci_right = self.rng.standard_normal((3, 3))
        ci_left /= np.linalg.norm(ci_left)
        ci_right /= np.linalg.norm(ci_right)

        orbital_overlap = mo_left.T @ mo_right
        reference = _determinant_overlap(
            ci_left, ci_right, orbital_overlap, ncore, ncas, nelec)
        values = biortho.biorthogonalize(
            mo_left, mo_right, ci_left, ci_right, np.eye(7),
            ncore, ncas, nelec, nelec, lu_threshold=1e-12)
        _, _, mo_left_bi, mo_right_bi, ci_left_bi, ci_right_bi = values

        assert_allclose(
            mo_left_bi.T @ mo_right_bi, np.eye(nocc), atol=1e-12)
        assert_allclose(
            np.vdot(ci_left_bi, ci_right_bi), reference, atol=1e-12)

    def test_single_determinant(self):
        """Include inactive orbitals and a single occupied active determinant."""
        ncore, ncas, nelec = 1, 2, (1, 1)
        nocc = ncore + ncas
        mo_left = self._orthonormal_columns(6, nocc)
        mo_right = self._orthonormal_columns(6, nocc)
        ci = np.zeros((2, 2))
        ci[0, 0] = 1.0

        reference = _determinant_overlap(
            ci, ci, mo_left.T @ mo_right, ncore, ncas, nelec)
        values = biortho.biorthogonalize(
            mo_left, mo_right, ci, ci, np.eye(6),
            ncore, ncas, nelec, nelec, lu_threshold=1e-12)
        ci_left_bi, ci_right_bi = values[-2:]
        assert_allclose(
            np.vdot(ci_left_bi, ci_right_bi), reference, atol=1e-12)

    def test_no_inactive_orbitals(self):
        ncore, ncas, nelec = 0, 3, (1, 1)
        mo = self._orthonormal_columns(5, ncas)
        ci = self.rng.standard_normal((3, 3))
        ci /= np.linalg.norm(ci)
        values = biortho.biorthogonalize(
            mo, mo, ci, ci, np.eye(5), ncore, ncas, nelec, nelec)
        assert_allclose(np.vdot(values[-2], values[-1]), 1.0, atol=1e-13)

    def test_singular_orbital_overlap(self):
        mo_left = np.eye(4)[:, :3]
        mo_right = np.eye(4)[:, [0, 1, 3]]
        ci = np.eye(2)
        with self.assertRaises(linalg.LinAlgError):
            biortho.biorthogonalize(
                mo_left, mo_right, ci, ci, np.eye(4),
                1, 2, (1, 1), (1, 1))


if __name__ == "__main__":
    unittest.main()
