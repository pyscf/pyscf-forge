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

def diagonalize_tda(mf, extype=1, collinear_samples=50, nstates=5):
    a, b = sftda.uhf_sf.get_ab_sf(mf, collinear_samples=collinear_samples)
    A_baba, A_abab = a
    B_baab, B_abba = b
    n_occ_a, n_virt_b = A_abab.shape[0], A_abab.shape[1]
    n_occ_b, n_virt_a = B_abba.shape[2], B_abba.shape[3]
    A_abab_2d = A_abab.reshape((n_occ_a*n_virt_b, n_occ_a*n_virt_b), order='C')
    B_abba_2d = B_abba.reshape((n_occ_a*n_virt_b, n_occ_b*n_virt_a), order='C')
    B_baab_2d = B_baab.reshape((n_occ_b*n_virt_a, n_occ_a*n_virt_b), order='C')
    A_baba_2d = A_baba.reshape((n_occ_b*n_virt_a, n_occ_b*n_virt_a), order='C')
    Casida_matrix = np.block([[ A_abab_2d, np.zeros_like(B_abba_2d)],
                              [np.zeros_like(-B_baab_2d), -A_baba_2d]])
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

    def test_hf_tda(self):
        mf = self.mol.ROKS(xc='HF').run()
        ref = np.array([0.4728164461, 0.5570168495])
        td = sftda.TDA_SF(mf).set(extype=0, collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tda(mf, extype=0, collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

        ref = np.array([-0.2204522712, -0.0023966488])
        td = sftda.TDA_SF(mf).set(extype=1, collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tda(mf, extype=1, collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

    def test_mcol_lda_tda(self):
        mf = self.mol.ROKS(xc='SVWN').run()
        ref = np.array([0.4508285718, 0.5792438533])
        td = sftda.TDA_SF(mf).set(extype=0, collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tda(mf, extype=0, collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

        ref = np.array([-0.3271863485, -0.000335489 ])
        td = sftda.TDA_SF(mf).set(extype=1, collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tda(mf, extype=1, collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

    def test_mcol_b3lyp_tda(self):
        mf = self.mol.ROKS(xc='B3LYP').run()
        ref = np.array([0.4606522787, 0.5780790823])
        td = sftda.TDA_SF(mf).set(extype=0, collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tda(mf, extype=0, collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

        ref = np.array([-0.2976172461, -0.000605713 ])
        td = sftda.TDA_SF(mf).set(extype=1, collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tda(mf, extype=1, collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

    def test_col_b3lyp_tda(self):
        mf = self.mol.ROKS(xc='B3LYP').run()
        ref = np.array([0.4749475052, 0.6067006254])
        td = sftda.TDA_SF(mf).set(extype=0, collinear_samples=-50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tda(mf, extype=0, collinear_samples=-50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

        ref = np.array([-0.2865096495,  0.0416989745])
        td = sftda.TDA_SF(mf).set(extype=1, collinear_samples=-50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tda(mf, extype=1, collinear_samples=-50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

    def test_mcol_tpss_tda(self):
        mf = self.mol.ROKS(xc='TPSS').run()
        ref = np.array([0.4506804512, 0.5704060695])
        td = sftda.TDA_SF(mf).set(extype=0, collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tda(mf, extype=0, collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

        ref = np.array([-0.2883457218, -0.0005819225])
        td = sftda.TDA_SF(mf).set(extype=1, collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tda(mf, extype=1, collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

    def test_mcol_cam_tda(self):
        mf = self.mol.ROKS(xc='CAM-B3LYP').run()
        ref = np.array([0.463106907 , 0.5771522483])
        td = sftda.TDA_SF(mf).set(extype=0, collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tda(mf, extype=0, collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

        ref = np.array([-0.2989172623, -0.0006272555])
        td = sftda.TDA_SF(mf).set(extype=1, collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tda(mf, extype=1, collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

    def test_col_cam_tda(self):
        mf = self.mol.ROKS(xc='CAM-B3LYP').run()
        ref = np.array([0.4768390356, 0.6049192733])
        td = sftda.TDA_SF(mf).set(extype=0, collinear_samples=-50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tda(mf, extype=0, collinear_samples=-50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

        ref = np.array([-0.2887349087,  0.0291502911])
        td = sftda.TDA_SF(mf).set(extype=1, collinear_samples=-50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tda(mf, extype=1, collinear_samples=-50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

if __name__ == "__main__":
    print("Full Tests for spin-flip-TDA with multicollinear functionals and collinear functionals based on ROKS reference")
    unittest.main()