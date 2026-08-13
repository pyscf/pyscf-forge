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

'''
Test Description: SISO calculations using L-PDFT states.
Test-1: Compare odd-multiplicity L-PDFT and SISO energies with references.
Test-2: Compare even-multiplicity L-PDFT and SISO energies with references.
'''

import unittest

import numpy as np
from numpy.testing import assert_allclose

from pyscf import gto, mcpdft, scf, siso


GRID_LEVEL = 1

# Absolute references generated with the small systems and fixed numerical
# grid specified in the tests below.
# These values are generated with Pyscf (v2.14, 14fb93158ec9c97ae1d67462ea47b5b27ce63aca) 
# PySCF-Forge (commit:cbc3808b92cbc4a5fff2c7343349770de845a93c)

LPDFT_REFERENCES = {
    'odd': np.array([
        -1.0272721697832456,
        -0.8672386060830728,
    ]),
    'even': np.array([
        -24.223960276355974,
        -24.223960276355974,
        -24.221603998434166,
        -24.094507287650040,
    ]),
}
SISO_REFERENCES = {
    'odd': np.array([
        -1.0272721697832456,
        -0.8672386060830728,
        -0.8672386060830728,
        -0.8672386060830728,
    ]),
    'even': np.array([
        -24.223982609418087,
        -24.223982609418066,
        -24.223938347621807,
        -24.223938347621782,
        -24.221603594106277,
        -24.221603594106266,
        -24.094507287650053,
        -24.094507287650046,
        -24.094507287650032,
        -24.094507287650025,
    ]),
}


class KnownValues(unittest.TestCase):
    def run_lpdft(self, mf, ncas, nelecas, modelspace):
        mf.run()
        mc = mcpdft.CASSCF(
            mf, 'tPBE', ncas, nelecas, grids_level=GRID_LEVEL)
        mc = siso.sacasscf_solver(mc, modelspace, ms='lin').run()
        my_siso = siso.SISO(mc, modelspace, ham='BP')
        energies, si_vecs = my_siso.kernel()
        return mc, my_siso, energies, si_vecs

    def check_siso_result(
            self, parity, mc, my_siso, energies, si_vecs):
        self.assertTrue(mc.converged)
        self.assertIsInstance(mc, mcpdft.MultiStateMCPDFTSolver)
        self.assertEqual(mc.grids.level, GRID_LEVEL)
        self.assertEqual(energies.shape, SISO_REFERENCES[parity].shape)
        self.assertEqual(si_vecs.shape, (energies.size, energies.size))
        # Degenerate state-average roots can be returned in different orders.
        assert_allclose(
            np.sort(mc.e_states), LPDFT_REFERENCES[parity],
            atol=1e-8, rtol=0.0)
        assert_allclose(
            energies, SISO_REFERENCES[parity], atol=1e-8, rtol=0.0)
        assert_allclose(
            si_vecs.conj().T @ si_vecs, np.eye(energies.size),
            atol=1e-12, rtol=0.0)
        hamiltonian = my_siso.compute_hamiltonian()
        assert_allclose(
            hamiltonian, hamiltonian.conj().T, atol=1e-12, rtol=0.0)

    def test_odd_multiplicities(self):
        mol = gto.M(
            atom='H 0 0 -0.7; H 0 0 0.7', basis='sto-3g', verbose=0)
        modelspace = [(1, 1), (1, 3)]
        result = self.run_lpdft(
            scf.RHF(mol), ncas=2, nelecas=2, modelspace=modelspace)
        self.check_siso_result('odd', *result)

    def test_even_multiplicities(self):
        mol = gto.M(
            atom='B 0 0 0', basis='sto-3g', spin=1, verbose=0)
        modelspace = [(3, 2), (1, 4)]
        result = self.run_lpdft(
            scf.ROHF(mol), ncas=4, nelecas=3, modelspace=modelspace)
        self.check_siso_result('even', *result)


if __name__ == '__main__':
    unittest.main()
