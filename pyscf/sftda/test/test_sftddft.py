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
        mf = self.mol.UKS(xc='HF').run()
        ref = np.array([0.4562711038, 0.5371281911])
        td = mf.TDDFT_SF().set(extype=0, collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tddft(mf, extype=0, collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

        ref = np.array([-0.2170068763,  0.0000000730])
        td = mf.TDDFT_SF().set(extype=1, collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tddft(mf, extype=1, collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

    def test_mcol_lda_tddft(self):
        mf = self.mol.UKS(xc='SVWN').run()
        ref = np.array([0.4496080754, 0.5767662743])
        td = mf.TDDFT_SF().set(extype=0, collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tddft(mf, extype=0, collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

        ref = np.array([-0.3265808777, -0.0000058921])
        td = mf.TDDFT_SF().set(extype=1, collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tddft(mf, extype=1, collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

    def test_mcol_b3lyp_tddft(self):
        mf = self.mol.UKS(xc='B3LYP').run()
        ref = np.array([0.4575220751, 0.5729490946])
        td = mf.TDDFT_SF().set(extype=0, collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tddft(mf, extype=0, collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

        ref = np.array([-0.2966235764, -0.0000019816])
        td = mf.TDDFT_SF().set(extype=1, collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tddft(mf, extype=1, collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

    def test_col_b3lyp_tddft(self):
        mf = self.mol.UKS(xc='B3LYP').run()
        ref = np.array([0.4733607067, 0.6059909951])
        td = mf.TDDFT_SF().set(extype=0, collinear_samples=-50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tddft(mf, extype=0, collinear_samples=-50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

        ref = np.array([-0.2852484489,  0.0427271935])
        td = mf.TDDFT_SF().set(extype=1, collinear_samples=-50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tddft(mf, extype=1, collinear_samples=-50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

    def test_mcol_tpss_tddft(self):
        mf = self.mol.UKS(xc='TPSS').run()
        ref = np.array([0.4478240804, 0.5654751068])
        td = mf.TDDFT_SF().set(extype=0, collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tddft(mf, extype=0, collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

        ref = np.array([-0.2873951146,  0.0000023945])
        td = mf.TDDFT_SF().set(extype=1, collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tddft(mf, extype=1, collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

    def test_mcol_cam_tddft(self):
        mf = self.mol.UKS(xc='CAM-B3LYP').run()
        ref = np.array([0.4595200285, 0.5715324610])
        td = mf.TDDFT_SF().set(extype=0, collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tddft(mf, extype=0, collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

        ref = np.array([-0.2979398223, -0.0000007521])
        td = mf.TDDFT_SF().set(extype=1, collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tddft(mf, extype=1, collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

    def test_col_cam_tddft(self):
        mf = self.mol.UKS(xc='CAM-B3LYP').run()
        ref = np.array([0.4749112597, 0.6040293216])
        td = mf.TDDFT_SF().set(extype=0, collinear_samples=-50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tddft(mf, extype=0, collinear_samples=-50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

        ref = np.array([-0.2874776809,  0.0300218092])
        td = mf.TDDFT_SF().set(extype=1, collinear_samples=-50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tddft(mf, extype=1, collinear_samples=-50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

if __name__ == "__main__":
    print("Full Tests for spin-flip-TDDFT with multicollinear functionals and collinear functionals")
    unittest.main()