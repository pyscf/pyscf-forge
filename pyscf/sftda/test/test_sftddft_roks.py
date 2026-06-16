# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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

import unittest
import numpy as np
from pyscf import gto
from pyscf import sftda

def diagonalize_tddft(mf, extype=1, collinear_samples=50, nstates=5):
    a, b = sftda.uhf_sf.get_ab_sf(mf, collinear_samples=collinear_samples)
    A_baba, A_abab = a
    B_baab, B_abba = b
    n_occ_a, n_virt_b = A_abab.shape[0], A_abab.shape[1]
    n_occ_b, n_virt_a = B_abba.shape[2], B_abba.shape[3]
    A_abab_2d = A_abab.reshape((n_occ_a*n_virt_b, n_occ_a*n_virt_b), order='C')
    B_abba_2d = B_abba.reshape((n_occ_a*n_virt_b, n_occ_b*n_virt_a), order='C')
    B_baab_2d = B_baab.reshape((n_occ_b*n_virt_a, n_occ_a*n_virt_b), order='C')
    A_baba_2d = A_baba.reshape((n_occ_b*n_virt_a, n_occ_b*n_virt_a), order='C')
    Casida_matrix = np.block([[ A_abab_2d, B_abba_2d],
                              [-B_baab_2d, -A_baba_2d]])
    eigenvals, eigenvecs = np.linalg.eig(Casida_matrix)
    idx = eigenvals.real.argsort()
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    norms = np.linalg.norm(eigenvecs[:n_occ_a*n_virt_b], axis=0)**2
    norms -= np.linalg.norm(eigenvecs[n_occ_a*n_virt_b:], axis=0)**2
    if extype == 1:
        mask = norms > 1e-3
        valid_e = eigenvals[mask].real
    else: 
        mask = norms < -1e-3
        valid_e = eigenvals[mask].real
        valid_e = -valid_e
    lowest_e = np.sort(valid_e)[:nstates]
    return lowest_e

class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.verbose = 0
        mol.output = '/dev/null'
        mol.atom = '''
        O     0.   0.       0.
        H     0.   -0.757   0.587
        H     0.   0.757    0.587'''
        mol.spin = 2
        mol.basis = '631g'
        cls.mol = mol.build()

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()

    def test_hf_tddft(self):
        mf = self.mol.ROKS(xc='HF').run()
        ref = np.array([0.4629615319, 0.5364066061])
        td = sftda.TDDFT_SF(mf).set(extype=0, collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tddft(mf, extype=0, collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

        ref = np.array([-0.2217270249, -0.005551913 ])
        td = sftda.TDDFT_SF(mf).set(extype=1, collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tddft(mf, extype=1, collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

    def test_mcol_lda_tddft(self):
        mf = self.mol.ROKS(xc='SVWN').run()
        ref = np.array([0.4502145189, 0.5768298978])
        td = sftda.TDDFT_SF(mf).set(extype=0, collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tddft(mf, extype=0, collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

        ref = np.array([-0.3273393356, -0.0007545328])
        td = sftda.TDDFT_SF(mf).set(extype=1, collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tddft(mf, extype=1, collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

    def test_mcol_b3lyp_tddft(self):
        mf = self.mol.ROKS(xc='B3LYP').run()
        ref = np.array([0.4587747607, 0.5730181452])
        td = sftda.TDDFT_SF(mf).set(extype=0, collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tddft(mf, extype=0, collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

        ref = np.array([-0.2979532666, -0.0013290418])
        td = sftda.TDDFT_SF(mf).set(extype=1, collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tddft(mf, extype=1, collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

    def test_col_b3lyp_tddft(self):
        mf = self.mol.ROKS(xc='B3LYP').run()
        ref = np.array([0.4745987469, 0.6060827854])
        td = sftda.TDDFT_SF(mf).set(extype=0, collinear_samples=-50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tddft(mf, extype=0, collinear_samples=-50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

        ref = np.array([-0.2865625142,  0.041539444 ])
        td = sftda.TDDFT_SF(mf).set(extype=1, collinear_samples=-50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tddft(mf, extype=1, collinear_samples=-50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

    def test_mcol_tpss_tddft(self):
        mf = self.mol.ROKS(xc='TPSS').run()
        ref = np.array([0.4486464854, 0.5651121494])
        td = sftda.TDDFT_SF(mf).set(extype=0, collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tddft(mf, extype=0, collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

        ref = np.array([-0.2887465343, -0.0012673515])
        td = sftda.TDDFT_SF(mf).set(extype=1, collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tddft(mf, extype=1, collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

    def test_mcol_cam_tddft(self):
        mf = self.mol.ROKS(xc='CAM-B3LYP').run()
        ref = np.array([0.4609174566, 0.5715410876])
        td = sftda.TDDFT_SF(mf).set(extype=0, collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tddft(mf, extype=0, collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

        ref = np.array([-0.2992951841, -0.001365193 ])
        td = sftda.TDDFT_SF(mf).set(extype=1, collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tddft(mf, extype=1, collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

    def test_col_cam_tddft(self):
        mf = self.mol.ROKS(xc='CAM-B3LYP').run()
        ref = np.array([0.4762639961, 0.6039715045])
        td = sftda.TDDFT_SF(mf).set(extype=0, collinear_samples=-50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tddft(mf, extype=0, collinear_samples=-50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

        ref = np.array([-0.288822456 ,  0.0288018552])
        td = sftda.TDDFT_SF(mf).set(extype=1, collinear_samples=-50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tddft(mf, extype=1, collinear_samples=-50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

if __name__ == "__main__":
    print("Full Tests for spin-flip-TDDFT with multicollinear functionals and collinear functionals based on ROKS reference")
    unittest.main()