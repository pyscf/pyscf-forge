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
from pyscf.grad import tduks_sf


def cal_exact_sf_tda_gradient(mf, extype=1, collinear_samples=50, state=1):
    a, b = sftda.uhf_sf.get_ab_sf(mf, collinear_samples=collinear_samples)
    A_baba, A_abab = a
    B_baab, B_abba = b

    n_occ_a, n_virt_b = A_abab.shape[0], A_abab.shape[1]
    n_occ_b, n_virt_a = B_abba.shape[2], B_abba.shape[3]

    A_abab_2d = A_abab.reshape((n_occ_a * n_virt_b, n_occ_a * n_virt_b), order='C')
    B_abba_2d = B_abba.reshape((n_occ_a * n_virt_b, n_occ_b * n_virt_a), order='C')
    B_baab_2d = B_baab.reshape((n_occ_b * n_virt_a, n_occ_a * n_virt_b), order='C')
    A_baba_2d = A_baba.reshape((n_occ_b * n_virt_a, n_occ_b * n_virt_a), order='C')

    Casida_matrix = np.block([[A_abab_2d, np.zeros_like(B_abba_2d)], [-np.zeros_like(B_baab_2d), -A_baba_2d]])

    eigenvals, eigenvecs = np.linalg.eig(Casida_matrix)
    idx = eigenvals.real.argsort()
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]

    norms = np.linalg.norm(eigenvecs[: n_occ_a * n_virt_b], axis=0) ** 2
    norms -= np.linalg.norm(eigenvecs[n_occ_a * n_virt_b :], axis=0) ** 2

    if extype == 1:
        mask = norms > 1e-3
        valid_e = eigenvals[mask].real
        eigenvecs = eigenvecs[:, mask].transpose(1, 0)

        def norm_xy(z):
            x = z[: n_occ_a * n_virt_b].reshape(n_occ_a, n_virt_b)
            y = z[n_occ_a * n_virt_b :].reshape(n_occ_b, n_virt_a)
            norm = np.linalg.norm(x) ** 2 - np.linalg.norm(y) ** 2
            norm = np.sqrt(1.0 / norm)
            return (x * norm, 0)
    else:
        mask = norms < -1e-3
        valid_e = -eigenvals[mask].real
        idx = np.argsort(valid_e)
        valid_e = valid_e[::-1]
        eigenvecs = eigenvecs[:, mask][:, ::-1].transpose(1, 0)

        def norm_xy(z):
            x = z[n_occ_a * n_virt_b :].reshape(n_occ_b, n_virt_a)
            y = z[: n_occ_a * n_virt_b].reshape(n_occ_a, n_virt_b)
            norm = np.linalg.norm(x) ** 2 - np.linalg.norm(y) ** 2
            norm = np.sqrt(1.0 / norm)
            return (x * norm, 0)

    fake_td = mf.TDA_SF()
    fake_td.extype = extype
    fake_td.collinear_samples = collinear_samples
    fake_td.e = valid_e
    fake_td.xy = [norm_xy(z) for z in eigenvecs]

    tdg = fake_td.Gradients()
    return tdg.kernel(state=state)


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.verbose = 0
        mol.output = '/dev/null'
        mol.atom = """
        O     0.   0.       0.
        H     0.   -0.757   0.587
        H     0.   0.757    0.587"""
        mol.spin = 2
        mol.basis = '631g'
        cls.mol = mol.build()

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()

    def test_hf_tda(self):
        mf = self.mol.UKS(xc='HF').run()
        td = mf.TDA_SF(extype=0, collinear_samples=50).run()
        grad_iter = td.Gradients().kernel(state=1)
        grad_exact = cal_exact_sf_tda_gradient(mf, extype=0, collinear_samples=50, state=1)
        ref = np.array(
            [
                [-3.9062848726e-17, -3.8312759625e-14, 3.3682079790e-01],
                [-9.7053273214e-17, 3.1640759776e-01, -1.6841039895e-01],
                [1.3611612194e-16, -3.1640759776e-01, -1.6841039895e-01],
            ]
        )
        self.assertAlmostEqual(abs(grad_exact - ref).max(), 0, 5)
        self.assertAlmostEqual(abs(grad_iter - ref).max(), 0, 5)

        td = mf.TDA_SF(extype=1, collinear_samples=50).run()
        grad_iter = td.Gradients().kernel(state=1)
        grad_exact = cal_exact_sf_tda_gradient(mf, extype=1, collinear_samples=50, state=1)
        ref = np.array(
            [
                [2.9628379088e-18, 2.9939944619e-15, -1.6746699427e-02],
                [-3.6072117518e-16, 1.4739029263e-02, 8.3733497137e-03],
                [3.5775833727e-16, -1.4739029263e-02, 8.3733497137e-03],
            ]
        )
        self.assertAlmostEqual(abs(grad_exact - ref).max(), 0, 5)
        self.assertAlmostEqual(abs(grad_iter - ref).max(), 0, 5)

    def test_mcol_lda_tda(self):
        mf = self.mol.UKS(xc='SVWN').run()
        td = mf.TDA_SF(extype=0, collinear_samples=50).run()
        grad_iter = td.Gradients().kernel(state=1)
        grad_exact = cal_exact_sf_tda_gradient(mf, extype=0, collinear_samples=50, state=1)
        ref = np.array(
            [
                [-1.1347069729e-16, -1.1084528691e-14, 2.8558017754e-01],
                [-1.1092137160e-16, 2.7449714063e-01, -1.4279559143e-01],
                [5.9671903444e-17, -2.7449714063e-01, -1.4279559143e-01],
            ]
        )
        self.assertAlmostEqual(abs(grad_exact - ref).max(), 0, 5)
        self.assertAlmostEqual(abs(grad_iter - ref).max(), 0, 5)

        td = mf.TDA_SF(extype=1, collinear_samples=50).run()
        grad_iter = td.Gradients().kernel(state=1)
        grad_exact = cal_exact_sf_tda_gradient(mf, extype=1, collinear_samples=50, state=1)
        ref = np.array(
            [
                [-3.9699882577e-16, 3.2848830357e-14, 3.6816914140e-03],
                [8.4541133573e-17, 1.4522652407e-02, -1.8472200133e-03],
                [9.7761986797e-17, -1.4522652407e-02, -1.8472200133e-03],
            ]
        )
        self.assertAlmostEqual(abs(grad_exact - ref).max(), 0, 5)
        self.assertAlmostEqual(abs(grad_iter - ref).max(), 0, 5)

    def test_mcol_b3lyp_tda(self):
        mf = self.mol.UKS(xc='B3LYP').run()
        td = mf.TDA_SF(extype=0, collinear_samples=50).run()
        grad_iter = td.Gradients().kernel(state=1)
        grad_exact = cal_exact_sf_tda_gradient(mf, extype=0, collinear_samples=50, state=1)
        ref = np.array(
            [
                [-1.8081449997e-16, -8.2435367236e-15, 3.0303551390e-01],
                [-4.0213738471e-17, 2.8716424898e-01, -1.5151853722e-01],
                [2.0517979988e-16, -2.8716424898e-01, -1.5151853722e-01],
            ]
        )
        self.assertAlmostEqual(abs(grad_exact - ref).max(), 0, 5)
        self.assertAlmostEqual(abs(grad_iter - ref).max(), 0, 5)

        td = mf.TDA_SF(extype=1, collinear_samples=50).run()
        grad_iter = td.Gradients().kernel(state=1)
        grad_exact = cal_exact_sf_tda_gradient(mf, extype=1, collinear_samples=50, state=1)
        ref = np.array(
            [
                [-2.1754590583e-16, 5.1510183861e-15, -2.8870940206e-03],
                [2.6851737303e-16, 1.1221659593e-02, 1.4375770430e-03],
                [-1.1021313913e-16, -1.1221659593e-02, 1.4375770429e-03],
            ]
        )
        self.assertAlmostEqual(abs(grad_exact - ref).max(), 0, 5)
        self.assertAlmostEqual(abs(grad_iter - ref).max(), 0, 5)

    def test_col_b3lyp_tda(self):
        mf = self.mol.UKS(xc='B3LYP').run()
        td = mf.TDA_SF(extype=0, collinear_samples=-50).run()
        grad_iter = td.Gradients().kernel(state=1)
        grad_exact = cal_exact_sf_tda_gradient(mf, extype=0, collinear_samples=-50, state=1)
        ref = np.array(
            [
                [-2.5468184163e-16, -1.1180009418e-14, 2.9019466200e-01],
                [-9.0775873947e-17, 2.8359926220e-01, -1.4509830953e-01],
                [1.1512910412e-16, -2.8359926220e-01, -1.4509830953e-01],
            ]
        )
        self.assertAlmostEqual(abs(grad_exact - ref).max(), 0, 5)
        self.assertAlmostEqual(abs(grad_iter - ref).max(), 0, 5)

        td = mf.TDA_SF(extype=1, collinear_samples=-50).run()
        grad_iter = td.Gradients().kernel(state=1)
        grad_exact = cal_exact_sf_tda_gradient(mf, extype=1, collinear_samples=-50, state=1)
        ref = np.array(
            [
                [-1.3918149905e-16, 2.8187893403e-15, -7.5626763433e-03],
                [6.2566519798e-17, 8.4214463746e-03, 3.7759834021e-03],
                [-1.1867186914e-16, -8.4214463746e-03, 3.7759834021e-03],
            ]
        )
        self.assertAlmostEqual(abs(grad_exact - ref).max(), 0, 5)
        self.assertAlmostEqual(abs(grad_iter - ref).max(), 0, 5)

    def test_mcol_tpss_tda(self):
        mf = self.mol.UKS(xc='TPSS').run()
        td = mf.TDA_SF(extype=0, collinear_samples=50).run()
        grad_iter = td.Gradients().kernel(state=1)
        grad_exact = cal_exact_sf_tda_gradient(mf, extype=0, collinear_samples=50, state=1)
        ref = np.array(
            [
                [7.8326284520e-17, -1.3071806385e-14, 3.0276587070e-01],
                [7.2426310078e-17, 2.8575699030e-01, -1.5139697086e-01],
                [-1.9546658108e-16, -2.8575699030e-01, -1.5139697086e-01],
            ]
        )
        self.assertAlmostEqual(abs(grad_exact - ref).max(), 0, 5)
        self.assertAlmostEqual(abs(grad_iter - ref).max(), 0, 5)

        td = mf.TDA_SF(extype=1, collinear_samples=50).run()
        grad_iter = td.Gradients().kernel(state=1)
        grad_exact = cal_exact_sf_tda_gradient(mf, extype=1, collinear_samples=50, state=1)
        ref = np.array(
            [
                [1.4878309854e-16, 1.8283875050e-14, 8.7167801429e-03],
                [4.6186501645e-17, 1.4280801923e-02, -4.3644887121e-03],
                [-1.1302550886e-16, -1.4280801923e-02, -4.3644887121e-03],
            ]
        )
        self.assertAlmostEqual(abs(grad_exact - ref).max(), 0, 5)
        self.assertAlmostEqual(abs(grad_iter - ref).max(), 0, 5)

    def test_mcol_cam_tda(self):
        mf = self.mol.UKS(xc='CAM-B3LYP').run()
        td = mf.TDA_SF(extype=0, collinear_samples=50).run()
        grad_iter = td.Gradients().kernel(state=1)
        grad_exact = cal_exact_sf_tda_gradient(mf, extype=0, collinear_samples=50, state=1)
        ref = np.array(
            [
                [-8.0651321211e-17, -2.3767777669e-15, 3.0670229779e-01],
                [4.1732199809e-18, 2.9083187407e-01, -1.5335290807e-01],
                [7.9000081855e-18, -2.9083187407e-01, -1.5335290807e-01],
            ]
        )
        self.assertAlmostEqual(abs(grad_exact - ref).max(), 0, 5)
        self.assertAlmostEqual(abs(grad_iter - ref).max(), 0, 5)

        td = mf.TDA_SF(extype=1, collinear_samples=50).run()
        grad_iter = td.Gradients().kernel(state=1)
        grad_exact = cal_exact_sf_tda_gradient(mf, extype=1, collinear_samples=50, state=1)
        ref = np.array(
            [
                [1.5152147863e-16, 1.9028098735e-15, -1.0403780070e-02],
                [-1.0684663896e-16, 1.1793709730e-02, 5.1952611230e-03],
                [-1.1109928193e-16, -1.1793709730e-02, 5.1952611230e-03],
            ]
        )
        self.assertAlmostEqual(abs(grad_exact - ref).max(), 0, 5)
        self.assertAlmostEqual(abs(grad_iter - ref).max(), 0, 5)

    def test_col_cam_tda(self):
        mf = self.mol.UKS(xc='CAM-B3LYP').run()
        td = mf.TDA_SF(extype=0, collinear_samples=-50).run()
        grad_iter = td.Gradients().kernel(state=1)
        grad_exact = cal_exact_sf_tda_gradient(mf, extype=0, collinear_samples=-50, state=1)
        ref = np.array(
            [
                [-2.3403358733e-16, -8.8750321950e-15, 2.9440091136e-01],
                [-1.8329779505e-17, 2.8746882738e-01, -1.4720233631e-01],
                [2.8537278462e-17, -2.8746882738e-01, -1.4720233631e-01],
            ]
        )
        self.assertAlmostEqual(abs(grad_exact - ref).max(), 0, 5)
        self.assertAlmostEqual(abs(grad_iter - ref).max(), 0, 5)

        td = mf.TDA_SF(extype=1, collinear_samples=-50).run()
        grad_iter = td.Gradients().kernel(state=1)
        grad_exact = cal_exact_sf_tda_gradient(mf, extype=1, collinear_samples=-50, state=1)
        ref = np.array(
            [
                [-7.4172801642e-17, 2.8382044273e-15, -1.5815249811e-02],
                [-2.2604851938e-16, 8.5651006595e-03, 7.9014387760e-03],
                [1.0633060174e-16, -8.5651006595e-03, 7.9014387760e-03],
            ]
        )
        self.assertAlmostEqual(abs(grad_exact - ref).max(), 0, 5)
        self.assertAlmostEqual(abs(grad_iter - ref).max(), 0, 5)


if __name__ == '__main__':
    print('Full Tests for spin-flip-TDA analytic gradient with multicollinear functionals and collinear functionals')
    unittest.main()
