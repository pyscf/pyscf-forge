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


def cal_exact_sf_tddft_gradient(mf, extype=1, collinear_samples=50, state=1):
    a, b = sftda.uhf_sf.get_ab_sf(mf, collinear_samples=collinear_samples)
    A_baba, A_abab = a
    B_baab, B_abba = b

    n_occ_a, n_virt_b = A_abab.shape[0], A_abab.shape[1]
    n_occ_b, n_virt_a = B_abba.shape[2], B_abba.shape[3]

    A_abab_2d = A_abab.reshape((n_occ_a * n_virt_b, n_occ_a * n_virt_b), order='C')
    B_abba_2d = B_abba.reshape((n_occ_a * n_virt_b, n_occ_b * n_virt_a), order='C')
    B_baab_2d = B_baab.reshape((n_occ_b * n_virt_a, n_occ_a * n_virt_b), order='C')
    A_baba_2d = A_baba.reshape((n_occ_b * n_virt_a, n_occ_b * n_virt_a), order='C')

    Casida_matrix = np.block([[A_abab_2d, B_abba_2d], [-B_baab_2d, -A_baba_2d]])

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
            return (x * norm, y * norm)
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
            return (x * norm, y * norm)

    fake_td = mf.TDDFT_SF()
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

    def test_hf_tddft(self):
        mf = self.mol.UKS(xc='HF').run()
        td = mf.TDDFT_SF(extype=0, collinear_samples=50).run()
        grad_iter = td.Gradients().kernel(state=1)
        grad_exact = cal_exact_sf_tddft_gradient(mf, extype=0, collinear_samples=50, state=1)
        ref = np.array(
            [
                [-2.2719396368e-17, -4.0913995956e-15, 3.5288985123e-01],
                [-9.8623500302e-17, 3.2334703193e-01, -1.7644492561e-01],
                [1.2134289667e-16, -3.2334703193e-01, -1.7644492561e-01],
            ]
        )
        self.assertAlmostEqual(abs(grad_exact - ref).max(), 0, 5)
        self.assertAlmostEqual(abs(grad_iter - ref).max(), 0, 5)

        td = mf.TDDFT_SF(extype=1, collinear_samples=50).run()
        grad_iter = td.Gradients().kernel(state=1)
        grad_exact = cal_exact_sf_tddft_gradient(mf, extype=1, collinear_samples=50, state=1)
        ref = np.array(
            [
                [-3.1451899376e-17, 1.2325674955e-15, -1.7602719006e-02],
                [-1.0377532915e-16, 1.4559257186e-02, 8.8013595029e-03],
                [1.3522722853e-16, -1.4559257186e-02, 8.8013595029e-03],
            ]
        )
        self.assertAlmostEqual(abs(grad_exact - ref).max(), 0, 5)
        self.assertAlmostEqual(abs(grad_iter - ref).max(), 0, 5)

    def test_mcol_lda_tddft(self):
        mf = self.mol.UKS(xc='SVWN').run()
        td = mf.TDDFT_SF(extype=0, collinear_samples=50).run()
        grad_iter = td.Gradients().kernel(state=1)
        grad_exact = cal_exact_sf_tddft_gradient(mf, extype=0, collinear_samples=50, state=1)
        ref = np.array(
            [
                [-8.4809773264e-16, 8.3401070355e-15, 2.8683261827e-01],
                [-1.2920607139e-17, 2.7494364759e-01, -1.4342181199e-01],
                [-1.9612973720e-17, -2.7494364759e-01, -1.4342181199e-01],
            ]
        )
        self.assertAlmostEqual(abs(grad_exact - ref).max(), 0, 5)
        self.assertAlmostEqual(abs(grad_iter - ref).max(), 0, 5)

        td = mf.TDDFT_SF(extype=1, collinear_samples=50).run()
        grad_iter = td.Gradients().kernel(state=1)
        grad_exact = cal_exact_sf_tddft_gradient(mf, extype=1, collinear_samples=50, state=1)
        ref = np.array(
            [
                [-6.7904482786e-16, 1.2476156342e-14, 3.6571945011e-03],
                [-9.3757276431e-17, 1.4533596842e-02, -1.8349716472e-03],
                [4.8848910182e-17, -1.4533596842e-02, -1.8349716472e-03],
            ]
        )
        self.assertAlmostEqual(abs(grad_exact - ref).max(), 0, 5)
        self.assertAlmostEqual(abs(grad_iter - ref).max(), 0, 5)

    def test_mcol_b3lyp_tddft(self):
        mf = self.mol.UKS(xc='B3LYP').run()
        td = mf.TDDFT_SF(extype=0, collinear_samples=50).run()
        grad_iter = td.Gradients().kernel(state=1)
        grad_exact = cal_exact_sf_tddft_gradient(mf, extype=0, collinear_samples=50, state=1)
        ref = np.array(
            [
                [6.3207724582e-17, -3.3299964897e-15, 3.0669181757e-01],
                [-7.4706674154e-17, 2.8848086768e-01, -1.5334666728e-01],
                [3.3531102689e-19, -2.8848086768e-01, -1.5334666728e-01],
            ]
        )
        self.assertAlmostEqual(abs(grad_exact - ref).max(), 0, 5)
        self.assertAlmostEqual(abs(grad_iter - ref).max(), 0, 5)

        td = mf.TDDFT_SF(extype=1, collinear_samples=50).run()
        grad_iter = td.Gradients().kernel(state=1)
        grad_exact = cal_exact_sf_tddft_gradient(mf, extype=1, collinear_samples=50, state=1)
        ref = np.array(
            [
                [8.5327168247e-17, 1.2525864103e-15, -2.9494220961e-03],
                [2.7815342784e-17, 1.1241681452e-02, 1.4687030974e-03],
                [-1.7036279713e-17, -1.1241681452e-02, 1.4687030974e-03],
            ]
        )
        self.assertAlmostEqual(abs(grad_exact - ref).max(), 0, 5)
        self.assertAlmostEqual(abs(grad_iter - ref).max(), 0, 5)

    def test_col_b3lyp_tddft(self):
        mf = self.mol.UKS(xc='B3LYP').run()
        td = mf.TDDFT_SF(extype=0, collinear_samples=-50).run()
        grad_iter = td.Gradients().kernel(state=1)
        grad_exact = cal_exact_sf_tddft_gradient(mf, extype=0, collinear_samples=-50, state=1)
        ref = np.array(
            [
                [-1.5105624594e-16, 1.2602467888e-15, 2.9075066796e-01],
                [6.5823041051e-18, 2.8383333522e-01, -1.4537631436e-01],
                [-7.5347051462e-17, -2.8383333522e-01, -1.4537631436e-01],
            ]
        )
        self.assertAlmostEqual(abs(grad_exact - ref).max(), 0, 5)
        self.assertAlmostEqual(abs(grad_iter - ref).max(), 0, 5)

        td = mf.TDDFT_SF(extype=1, collinear_samples=-50).run()
        grad_iter = td.Gradients().kernel(state=1)
        grad_exact = cal_exact_sf_tddft_gradient(mf, extype=1, collinear_samples=-50, state=1)
        ref = np.array(
            [
                [-7.5510879830e-17, 1.0627607632e-15, -7.5959284737e-03],
                [-9.4837014163e-17, 8.4101877276e-03, 3.7926098698e-03],
                [-1.0223646526e-17, -8.4101877276e-03, 3.7926098698e-03],
            ]
        )
        self.assertAlmostEqual(abs(grad_exact - ref).max(), 0, 5)
        self.assertAlmostEqual(abs(grad_iter - ref).max(), 0, 5)

    def test_mcol_tpss_tddft(self):
        mf = self.mol.UKS(xc='TPSS').run()
        td = mf.TDDFT_SF(extype=0, collinear_samples=50).run()
        grad_iter = td.Gradients().kernel(state=1)
        grad_exact = cal_exact_sf_tddft_gradient(mf, extype=0, collinear_samples=50, state=1)
        ref = np.array(
            [
                [-2.8308341496e-16, 1.0704054160e-15, 3.0645165527e-01],
                [-1.1971039167e-16, 2.8704791604e-01, -1.5324080858e-01],
                [3.3992205735e-17, -2.8704791604e-01, -1.5324080858e-01],
            ]
        )
        self.assertAlmostEqual(abs(grad_exact - ref).max(), 0, 5)
        self.assertAlmostEqual(abs(grad_iter - ref).max(), 0, 5)

        td = mf.TDDFT_SF(extype=1, collinear_samples=50).run()
        grad_iter = td.Gradients().kernel(state=1)
        grad_exact = cal_exact_sf_tddft_gradient(mf, extype=1, collinear_samples=50, state=1)
        ref = np.array(
            [
                [-1.6711533117e-16, 1.0947265850e-14, 8.8032106411e-03],
                [-1.0451471451e-16, 1.4360403474e-02, -4.4080354878e-03],
                [-4.7204899979e-20, -1.4360403474e-02, -4.4080354878e-03],
            ]
        )
        self.assertAlmostEqual(abs(grad_exact - ref).max(), 0, 5)
        self.assertAlmostEqual(abs(grad_iter - ref).max(), 0, 5)

    def test_mcol_cam_tddft(self):
        mf = self.mol.UKS(xc='CAM-B3LYP').run()
        td = mf.TDDFT_SF(extype=0, collinear_samples=50).run()
        grad_iter = td.Gradients().kernel(state=1)
        grad_exact = cal_exact_sf_tddft_gradient(mf, extype=0, collinear_samples=50, state=1)
        ref = np.array(
            [
                [-3.7538914920e-16, -6.9527321033e-15, 3.1103703156e-01],
                [-1.4713567467e-16, 2.9247173978e-01, -1.5552026728e-01],
                [-2.4033397519e-17, -2.9247173978e-01, -1.5552026728e-01],
            ]
        )
        self.assertAlmostEqual(abs(grad_exact - ref).max(), 0, 5)
        self.assertAlmostEqual(abs(grad_iter - ref).max(), 0, 5)

        td = mf.TDDFT_SF(extype=1, collinear_samples=50).run()
        grad_iter = td.Gradients().kernel(state=1)
        grad_exact = cal_exact_sf_tddft_gradient(mf, extype=1, collinear_samples=50, state=1)
        ref = np.array(
            [
                [-4.5098470283e-16, 2.7736173739e-15, -1.0510769880e-02],
                [-1.9483417972e-16, 1.1800860322e-02, 5.2487299209e-03],
                [1.1710777201e-16, -1.1800860322e-02, 5.2487299209e-03],
            ]
        )
        self.assertAlmostEqual(abs(grad_exact - ref).max(), 0, 5)
        self.assertAlmostEqual(abs(grad_iter - ref).max(), 0, 5)

    def test_col_cam_tddft(self):
        mf = self.mol.UKS(xc='CAM-B3LYP').run()
        td = mf.TDDFT_SF(extype=0, collinear_samples=-50).run()
        grad_iter = td.Gradients().kernel(state=1)
        grad_exact = cal_exact_sf_tddft_gradient(mf, extype=0, collinear_samples=-50, state=1)
        ref = np.array(
            [
                [1.1028977623e-17, -8.7353942702e-15, 2.9529372572e-01],
                [-1.3201775114e-16, 2.8788637730e-01, -1.4764874713e-01],
                [3.1443202277e-17, -2.8788637730e-01, -1.4764874713e-01],
            ]
        )
        self.assertAlmostEqual(abs(grad_exact - ref).max(), 0, 5)
        self.assertAlmostEqual(abs(grad_iter - ref).max(), 0, 5)

        td = mf.TDDFT_SF(extype=1, collinear_samples=-50).run()
        grad_iter = td.Gradients().kernel(state=1)
        grad_exact = cal_exact_sf_tddft_gradient(mf, extype=1, collinear_samples=-50, state=1)
        ref = np.array(
            [
                [-2.5925322984e-17, 3.1312811099e-15, -1.5876133062e-02],
                [4.8403599468e-17, 8.5399089682e-03, 7.9318808125e-03],
                [-7.8228805841e-17, -8.5399089682e-03, 7.9318808125e-03],
            ]
        )
        self.assertAlmostEqual(abs(grad_exact - ref).max(), 0, 5)
        self.assertAlmostEqual(abs(grad_iter - ref).max(), 0, 5)


if __name__ == '__main__':
    print('Full Tests for spin-flip-TDDFT analytic gradient with multicollinear functionals and collinear functionals')
    unittest.main()
