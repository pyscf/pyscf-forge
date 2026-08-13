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

'''
Test Description: Testing various functions in the SISO Hamiltonian module.

Test-1: Check the Clebsch-Gordan coefficients used for spin coupling.
Test-2: Check assembly of model-space intermediates.
Test-3: Check density-matrix dispatch and output shapes.
Test-4: Check construction of Hamiltonian intermediates.
Test-5: Compare a singlet-triplet Hamiltonian with its reference matrix.
Test-6: Check that the kernel diagonalizes the SISO Hamiltonian.
'''

import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose

from pyscf.siso import siso


class KnownValues(unittest.TestCase):

    def test_clebsch_gordan_coefficients(self):
        cg_same = siso.compute_cg_coefficients(2, Ms=0)
        cg_plus = siso.compute_cg_coefficients(0, Ms=1)

        self.assertEqual(cg_same.shape, (3, 3, 3))
        self.assertEqual(cg_plus.shape, (3, 1, 3))
        self.assertAlmostEqual(np.linalg.norm(cg_same), 1.0, 14)
        assert_allclose(cg_plus[0, 0, 0], 1.0 / np.sqrt(3.0), atol=1e-14)
        assert_allclose(cg_plus[1, 0, 1], 1.0 / np.sqrt(3.0), atol=1e-14)
        assert_allclose(cg_plus[2, 0, 2], 1.0 / np.sqrt(3.0), atol=1e-14)
        with self.assertRaises(AssertionError):
            siso.compute_cg_coefficients(2, Ms=2)

    def test_assemble_model_space_intermediates(self):
        ci = [
            np.arange(9).reshape(3, 3),
            np.arange(9, 18).reshape(1, 9),
            np.arange(3).reshape(3, 1),
        ]
        mc = SimpleNamespace(
            ci=ci,
            e_states=np.asarray([-2.0, -1.8, -1.5]),
            ncas=3,
            nelecas=(1, 1),
        )
        my_siso = SimpleNamespace(
            mc=mc,
            twoslst=np.asarray([0, 2]),
            statelis=[2, 0, 1],
            imds=SimpleNamespace(),
        )

        cimat = siso.assemble_civecs(my_siso)
        energy = siso.assemble_energy(my_siso)

        self.assertEqual([x.shape for x in cimat], [(2, 3, 3), (1, 3, 1)])
        assert_allclose(cimat[0][1].ravel(), ci[1].ravel())
        assert_allclose(energy[0], [-2.0, -1.8])
        assert_allclose(energy[1], [-1.5])

    def test_build_imds(self):
        my_siso = SimpleNamespace(
            somf=True,
            amf=True,
            mmf=False,
            soc1e=True,
            soc2e=True,
            ham='DKH',
            imds=SimpleNamespace(),
        )
        intermediates = {
            'z': np.zeros((3, 2, 2)),
            'a': ['link'],
            'c': [np.zeros((1, 2, 2))],
            'e': [np.asarray([-1.0])],
            'd': [np.zeros((1, 1, 1, 1))],
        }
        with mock.patch.object(
                siso, 'calculate_zmat', return_value=intermediates['z']):
            with mock.patch.object(
                    siso, 'assemble_amat', return_value=intermediates['a']):
                with mock.patch.object(
                        siso, 'assemble_civecs', return_value=intermediates['c']):
                    with mock.patch.object(
                            siso, 'assemble_energy',
                            return_value=intermediates['e']):
                        with mock.patch.object(
                                siso, 'compute_dmat',
                                return_value=intermediates['d']):
                            result = siso.build_imds(my_siso)

        self.assertIs(result, my_siso)
        for key, value in intermediates.items():
            self.assertIs(getattr(my_siso.imds, key), value)

    # Build a four-state model from one singlet and the three components of one
    # triplet.  The test checks that singlet-triplet coupling and the internal
    # triplet SOC block are scaled and assembled before spin-free energies are
    # added to form the total state-interaction Hamiltonian.
    def test_singlet_triplet_hamiltonian(self):
        coupling = np.asarray([[0.2 + 0.1j, -0.3j, 0.05]])
        triplet_soc = np.asarray([
            [0.0, 0.02j, -0.03],
            [-0.02j, 0.01, 0.04j],
            [-0.03, -0.04j, -0.01],
        ])
        scale = np.sqrt(3.0 / 2.0)
        triplet_scale = np.sqrt(6.0) / 2.0
        imds = SimpleNamespace(
            d=[
                np.zeros((1, 1)),
                coupling / scale,
                -coupling.conj().T / scale,
                triplet_soc / triplet_scale,
            ],
            e=[np.asarray([-1.0]), np.asarray([-0.8])],
        )
        my_siso = SimpleNamespace(
            statelis=[1, 0, 1],
            twoslst=np.asarray([0, 2]),
            stuples=[(0, 0), (0, 2), (2, 0), (2, 2)],
            imds=imds,
        )

        soc_hamiltonian = siso.compute_soc_hamiltonian(my_siso)
        soc_reference = np.block([
            [np.zeros((1, 1)), coupling],
            [coupling.conj().T, triplet_soc],
        ])
        self.assertEqual(soc_hamiltonian.shape, (4, 4))
        assert_allclose(soc_hamiltonian, soc_reference, atol=1e-14)

        hamiltonian = siso.compute_hamiltonian(my_siso)
        reference = np.block([
            [np.asarray([[-1.0]]), coupling],
            [coupling.conj().T, -0.8 * np.eye(3) + triplet_soc],
        ])
        assert_allclose(hamiltonian, reference, atol=1e-14)
        assert_allclose(hamiltonian, hamiltonian.conj().T, atol=1e-14)

    def test_kernel_diagonalizes_hamiltonian(self):
        class FakeSISO:
            def __init__(self):
                self.finalized = False
                self.verbose = 0

            def build_imds(self):
                return self

            def compute_hamiltonian(self):
                return np.asarray([[1.0, 0.2j], [-0.2j, 2.0]])

            def _finalize(self):
                self.finalized = True

        my_siso = FakeSISO()
        energies, si_vecs = siso.kernel(my_siso)
        assert_allclose(energies, [0.9614835192865496, 2.0385164807134504])
        assert_allclose(si_vecs.conj().T.dot(si_vecs), np.eye(2), atol=1e-14)
        self.assertTrue(my_siso.finalized)


if __name__ == '__main__':
    unittest.main()
