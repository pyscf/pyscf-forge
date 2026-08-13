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
Test Description: SISO calculations based on CASSCF model spaces.
Test-1: Check the AVAS state-averaged CASSCF model space for a B atom.
Test-2: Compare BP and DKH SISO energies obtained with AMFI integrals.
Test-3: Compare SISO energies obtained with MMFI spin-orbit integrals.
'''

import unittest

import numpy as np
from numpy.testing import assert_allclose

from pyscf import gto, mcscf, scf, siso
from pyscf.mcscf import avas


class KnownValues(unittest.TestCase):
    # These values are generated with this code only.
    amfi_energies = {
        'BP': np.array([
            -24.189280508070294, -24.189280508070283,
            -24.189214701394956, -24.189214701394950,
            -24.189214701394945, -24.189214701394942,
        ]),
        'DKH': np.array([
            -24.189280525971345, -24.189280525971330,
            -24.189214692444434, -24.189214692444420,
            -24.189214692444416, -24.189214692444410,
        ]),
    }
    mmfi_energies = {
        'BP': np.array([
            -24.189289226636127, -24.189289226636124,
            -24.189210342112066, -24.189210342112027,
            -24.189210342112027, -24.189210342112005,
        ]),
    }

    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(
            atom='B 0 0 0', basis='sto-3g', spin=1, verbose=0)
        cls.mf = scf.ROHF(cls.mol).run()
        ncas, nelecas, mo = avas.avas(cls.mf, ['B 2s', 'B 2p'])
        cls.modelspace = [(3, 2)]
        cls.mc = mcscf.CASSCF(cls.mf, ncas, nelecas)
        cls.mc = siso.sacasscf_solver(cls.mc, cls.modelspace).run(mo)

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf, cls.mc, cls.modelspace

    def test_sa_casscf_model_space(self):
        self.assertTrue(self.mc.converged)
        self.assertEqual(len(self.mc.e_states), 3)
        spin_square = self.mc.fcisolver.states_spin_square(
            self.mc.ci, self.mc.ncas, self.mc.nelecas)[0]
        assert_allclose(
            spin_square, [0.75, 0.75, 0.75], atol=1e-8, rtol=0)

    def test_amfi_siso(self):
        for ham, reference in self.amfi_energies.items():
            with self.subTest(ham=ham):
                my_siso = siso.SISO(
                    self.mc, self.modelspace, ham=ham, amf=True)
                energies, si_vecs = my_siso.kernel()

                self.assertEqual(energies.shape, (6,))
                self.assertEqual(si_vecs.shape, (6, 6))
                self.assertEqual(my_siso.imds.z.shape, (3, 4, 4))
                assert_allclose(energies, reference, atol=1e-8, rtol=0)
                hamiltonian = my_siso.compute_hamiltonian()
                assert_allclose(
                    hamiltonian, hamiltonian.conj().T,
                    atol=1e-8, rtol=0)

    def test_mmfi_siso(self):
        for ham, reference in self.mmfi_energies.items():
            with self.subTest(ham=ham):
                mmfi_siso = siso.SISO(
                    self.mc, self.modelspace, ham=ham,
                    amf=False, mmf=True)
                energies = mmfi_siso.kernel()[0]

                self.assertEqual(energies.shape, (6,))
                assert_allclose(energies, reference, atol=1e-8, rtol=0)


if __name__ == '__main__':
    unittest.main()
