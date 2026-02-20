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
        td = mf.TDDFT_SF().set(extype=0, collinear_samples=50, nstates=2).run()
        tdg = td.Gradients().run()
        ref = np.array(
            [[ 9.8635123327e-17, -2.6531978332e-15,  3.5288902041e-01],
            [-2.0752411168e-16,  3.2334648658e-01, -1.7644451021e-01],
            [ 1.0888898835e-16, -3.2334648658e-01, -1.7644451021e-01]]
        )
        self.assertAlmostEqual(abs(tdg.de - ref).max(), 0, 4)

        td = mf.TDDFT_SF().set(extype=1, collinear_samples=50, nstates=2).run()
        tdg = td.Gradients().run()
        ref = np.array(
            [[ 1.2217345207e-16, -1.4073039905e-15, -1.7603015843e-02],
            [-3.1381655282e-16,  1.4559153763e-02,  8.8015079214e-03],
            [ 1.9164310076e-16, -1.4559153763e-02,  8.8015079214e-03]]
        )
        self.assertAlmostEqual(abs(tdg.de - ref).max(), 0, 4)

    def test_mcol_lda_tddft(self):
        mf = self.mol.UKS(xc='SVWN').run()
        td = mf.TDDFT_SF().set(extype=0, collinear_samples=50, nstates=2).run()
        tdg = td.Gradients().run()
        ref = np.array(
            [[-1.2757624379e-16,  1.9190538241e-15,  2.8683262201e-01],
            [ 3.2461388658e-17,  2.7494365566e-01, -1.4342181386e-01],
            [-7.3881676371e-17, -2.7494365566e-01, -1.4342181386e-01]]
        )
        self.assertAlmostEqual(abs(tdg.de - ref).max(), 0, 4)

        td = mf.TDDFT_SF().set(extype=1, collinear_samples=50, nstates=2).run()
        tdg = td.Gradients().run()
        ref = np.array(
            [[-1.4814561008e-16,  1.5733130704e-14,  3.6572276035e-03],
            [-1.0110540434e-16,  1.4533618985e-02, -1.8349881981e-03],
            [ 1.7090139659e-17, -1.4533618985e-02, -1.8349881981e-03]]
        )
        self.assertAlmostEqual(abs(tdg.de - ref).max(), 0, 4)

    def test_mcol_b3lyp_tddft(self):
        mf = self.mol.UKS(xc='B3LYP').run()
        td = mf.TDDFT_SF().set(extype=0, collinear_samples=50, nstates=2).run()
        tdg = td.Gradients().run()
        ref = np.array(
            [[ 3.2603455049e-16,  1.7646595764e-16,  3.0669181702e-01],
            [-9.7156306981e-17,  2.8848085531e-01, -1.5334666701e-01],
            [ 1.5134230155e-17, -2.8848085531e-01, -1.5334666701e-01]]
        )
        self.assertAlmostEqual(abs(tdg.de - ref).max(), 0, 4)

        td = mf.TDDFT_SF().set(extype=1, collinear_samples=50, nstates=2).run()
        tdg = td.Gradients().run()
        ref = np.array(
            [[ 3.9440438558e-16,  5.8563320419e-15, -2.9490371434e-03],
            [-1.2116370071e-17,  1.1241921015e-02,  1.4685106244e-03],
            [-8.3644726181e-17, -1.1241921015e-02,  1.4685106244e-03]]
        )
        self.assertAlmostEqual(abs(tdg.de - ref).max(), 0, 4)

    def test_col_b3lyp_tddft(self):
        mf = self.mol.UKS(xc='B3LYP').run()
        td = mf.TDDFT_SF().set(extype=0, collinear_samples=-50, nstates=2).run()
        tdg = td.Gradients().run()
        ref = np.array(
            [[ 2.5161393249e-16, -2.5218247762e-15,  2.9075065556e-01],
            [-7.3124160510e-17,  2.8383332799e-01, -1.4537630816e-01],
            [ 7.0137639062e-17, -2.8383332799e-01, -1.4537630816e-01]]
        )
        self.assertAlmostEqual(abs(tdg.de - ref).max(), 0, 4)

        td = mf.TDDFT_SF().set(extype=1, collinear_samples=-50, nstates=2).run()
        tdg = td.Gradients().run()
        ref = np.array(
            [[ 3.6858848913e-16,  4.0065663116e-15, -7.5967156655e-03],
            [-2.0596267663e-17,  8.4096408373e-03,  3.7930034810e-03],
            [-9.5790330150e-18, -8.4096408373e-03,  3.7930034810e-03]]
        )
        self.assertAlmostEqual(abs(tdg.de - ref).max(), 0, 4)

    def test_mcol_tpss_tddft(self):
        mf = self.mol.UKS(xc='TPSS').run()
        td = mf.TDDFT_SF().set(extype=0, collinear_samples=50, nstates=2).run()
        tdg = td.Gradients().run()
        ref = np.array(
        [[-3.0513915325e-16, -2.6005974810e-15,  3.0645164672e-01],
        [ 6.9190185801e-18,  2.8704790411e-01, -1.5324080433e-01],
        [-1.0512951278e-16, -2.8704790411e-01, -1.5324080433e-01]]
        )
        self.assertAlmostEqual(abs(tdg.de - ref).max(), 0, 4)

        td = mf.TDDFT_SF().set(extype=1, collinear_samples=50, nstates=2).run()
        tdg = td.Gradients().run()
        ref = np.array(
            [[-3.4214134238e-16, -3.5767789007e-15,  8.8035914294e-03],
            [-7.8492425982e-18,  1.4360641011e-02, -4.4082258897e-03],
            [-5.7805037162e-17, -1.4360641011e-02, -4.4082258897e-03]]
        )
        self.assertAlmostEqual(abs(tdg.de - ref).max(), 0, 4)

    def test_mcol_cam_tddft(self):
        mf = self.mol.UKS(xc='CAM-B3LYP').run()
        td = mf.TDDFT_SF().set(extype=0, collinear_samples=50, nstates=2).run()
        tdg = td.Gradients().run()
        ref = np.array(
            [[-2.1149902016e-16, -1.2801272120e-15,  3.1103703527e-01],
            [ 8.8772726516e-18,  2.9247173996e-01, -1.5552026913e-01],
            [ 8.1548873113e-18, -2.9247173996e-01, -1.5552026913e-01]]
        )
        self.assertAlmostEqual(abs(tdg.de - ref).max(), 0, 4)

        td = mf.TDDFT_SF().set(extype=1, collinear_samples=50, nstates=2).run()
        tdg = td.Gradients().run()
        ref = np.array(
            [[-1.2850649394e-16,  9.1261102573e-16, -1.0510318371e-02],
            [-4.1480095616e-17,  1.1801184295e-02,  5.2485041660e-03],
            [ 8.8065545693e-19, -1.1801184295e-02,  5.2485041660e-03]]
        )
        self.assertAlmostEqual(abs(tdg.de - ref).max(), 0, 4)

    def test_col_cam_tddft(self):
        mf = self.mol.UKS(xc='CAM-B3LYP').run()
        td = mf.TDDFT_SF().set(extype=0, collinear_samples=-50, nstates=2).run()
        tdg = td.Gradients().run()
        ref = np.array(
            [[-2.9070712607e-16,  7.8946412672e-16,  2.9529370943e-01],
            [-1.0423166195e-16,  2.8788636795e-01, -1.4764873898e-01],
            [-4.6938480146e-17, -2.8788636795e-01, -1.4764873898e-01]]
        )
        self.assertAlmostEqual(abs(tdg.de - ref).max(), 0, 4)

        td = mf.TDDFT_SF().set(extype=1, collinear_samples=-50, nstates=2).run()
        tdg = td.Gradients().run()
        ref = np.array(
            [[-3.4311860805e-16,  8.3116047642e-16, -1.5876227600e-02],
            [ 5.5628047850e-17,  8.5399145588e-03,  7.9319280810e-03],
            [-1.1167593801e-16, -8.5399145588e-03,  7.9319280810e-03]]
        )
        self.assertAlmostEqual(abs(tdg.de - ref).max(), 0, 4)


if __name__ == "__main__":
    print("Full Tests for spin-flip-TDDFT analytic gradient with multicollinear functionals and collinear functionals")
    unittest.main()