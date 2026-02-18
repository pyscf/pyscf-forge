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

    def test_hf_tda(self):
        mf = self.mol.UKS(xc='HF').run()
        td = mf.TDA_SF().set(extype=0, collinear_samples=50, nstates=2).run()
        tdg = td.Gradients().run()
        ref = np.array(
            [[-3.0157070982e-18,  3.7144943268e-15,  3.3681971737e-01],
            [ 8.9129157073e-17,  3.1640695246e-01, -1.6840985868e-01],
            [-8.6113449975e-17, -3.1640695246e-01, -1.6840985868e-01]]
        )
        self.assertAlmostEqual(abs(tdg.de - ref).max(), 0, 4)

        td = mf.TDA_SF().set(extype=1, collinear_samples=50, nstates=2).run()
        tdg = td.Gradients().run()
        ref = np.array(
            [[ 9.3014403269e-17,  1.5956923862e-16, -1.6746993353e-02],
            [ 5.8351890612e-17,  1.4738926078e-02,  8.3734966765e-03],
            [-1.5136629388e-16, -1.4738926078e-02,  8.3734966765e-03]]
        )
        self.assertAlmostEqual(abs(tdg.de - ref).max(), 0, 4)

    def test_mcol_lda_tda(self):
        mf = self.mol.UKS(xc='SVWN').run()
        td = mf.TDA_SF().set(extype=0, collinear_samples=50, nstates=2).run()
        tdg = td.Gradients().run()
        ref = np.array(
            [[-9.2546359854e-17, -6.0541677143e-15,  2.8558017756e-01],
            [-1.2529623683e-16,  2.7449714062e-01, -1.4279559145e-01],
            [ 1.9188009639e-16, -2.7449714062e-01, -1.4279559145e-01]]
        )
        self.assertAlmostEqual(abs(tdg.de - ref).max(), 0, 4)

        td = mf.TDA_SF().set(extype=1, collinear_samples=50, nstates=2).run()
        tdg = td.Gradients().run()
        ref = np.array(
            [[ 1.2454225094e-17, -1.7849576353e-14,  3.6816914140e-03],
            [ 1.2927014603e-16,  1.4522652407e-02, -1.8472200133e-03],
            [-1.3166296104e-16, -1.4522652407e-02, -1.8472200133e-03]]
        )
        self.assertAlmostEqual(abs(tdg.de - ref).max(), 0, 4)

    def test_mcol_b3lyp_tda(self):
        mf = self.mol.UKS(xc='B3LYP').run()
        td = mf.TDA_SF().set(extype=0, collinear_samples=50, nstates=2).run()
        tdg = td.Gradients().run()
        ref = np.array(
            [[ 3.3755805526e-16,  5.1347236507e-15,  3.0303548147e-01],
            [-5.7420768990e-17,  2.8716422104e-01, -1.5151852101e-01],
            [ 5.9375519925e-17, -2.8716422104e-01, -1.5151852101e-01]]
        )
        self.assertAlmostEqual(abs(tdg.de - ref).max(), 0, 4)

        td = mf.TDA_SF().set(extype=1, collinear_samples=50, nstates=2).run()
        tdg = td.Gradients().run()
        ref = np.array(
            [[ 3.8904964283e-16, -1.6116646615e-15, -2.8870940206e-03],
            [ 3.0341520419e-17,  1.1221659593e-02,  1.4375770430e-03],
            [-1.1785487464e-16, -1.1221659593e-02,  1.4375770430e-03]]
        )
        self.assertAlmostEqual(abs(tdg.de - ref).max(), 0, 4)

    def test_col_b3lyp_tda(self):
        mf = self.mol.UKS(xc='B3LYP').run()
        td = mf.TDA_SF().set(extype=0, collinear_samples=-50, nstates=2).run()
        tdg = td.Gradients().run()
        ref = np.array(
            [[-3.2259149630e-17,  7.6179452385e-16,  2.9019466200e-01],
            [-2.5052609688e-16,  2.8359926220e-01, -1.4509830953e-01],
            [ 1.1342578584e-16, -2.8359926220e-01, -1.4509830953e-01]]
        )
        self.assertAlmostEqual(abs(tdg.de - ref).max(), 0, 4)

        td = mf.TDA_SF().set(extype=1, collinear_samples=-50, nstates=2).run()
        tdg = td.Gradients().run()
        ref = np.array(
            [[-4.7652535574e-17,  2.7295834367e-17, -7.5626763433e-03],
            [ 4.6782068927e-17,  8.4214463746e-03,  3.7759834021e-03],
            [-2.5715216150e-18, -8.4214463746e-03,  3.7759834021e-03]]
        )
        self.assertAlmostEqual(abs(tdg.de - ref).max(), 0, 4)

    def test_mcol_tpss_tda(self):
        mf = self.mol.UKS(xc='TPSS').run()
        td = mf.TDA_SF().set(extype=0, collinear_samples=50, nstates=2).run()
        tdg = td.Gradients().run()
        ref = np.array(
            [[-1.0618116650e-16,  8.5065843025e-16,  3.0276587070e-01],
            [-1.7544086463e-16,  2.8575699030e-01, -1.5139697086e-01],
            [-2.1559126598e-17, -2.8575699030e-01, -1.5139697086e-01]]
        )
        self.assertAlmostEqual(abs(tdg.de - ref).max(), 0, 4)

        td = mf.TDA_SF().set(extype=1, collinear_samples=50, nstates=2).run()
        tdg = td.Gradients().run()
        ref = np.array(
            [[-8.0765155817e-17, -5.4813611036e-15,  8.7167801429e-03],
            [ 4.8264725316e-17,  1.4280801923e-02, -4.3644887121e-03],
            [-1.2028615208e-16, -1.4280801923e-02, -4.3644887121e-03]]
        )
        self.assertAlmostEqual(abs(tdg.de - ref).max(), 0, 4)

    def test_mcol_cam_tda(self):
        mf = self.mol.UKS(xc='CAM-B3LYP').run()
        td = mf.TDA_SF().set(extype=0, collinear_samples=50, nstates=2).run()
        tdg = td.Gradients().run()
        ref = np.array(
            [[-2.4425492658e-16,  7.1480887150e-16,  3.0670229779e-01],
            [-1.0665896206e-17,  2.9083187407e-01, -1.5335290807e-01],
            [ 1.4274133369e-17, -2.9083187407e-01, -1.5335290807e-01]]
        )
        self.assertAlmostEqual(abs(tdg.de - ref).max(), 0, 4)

        td = mf.TDA_SF().set(extype=1, collinear_samples=50, nstates=2).run()
        tdg = td.Gradients().run()
        ref = np.array(
            [[-3.9676112181e-16, -2.0321658431e-15, -1.0403780070e-02],
            [ 1.3911348534e-16,  1.1793709730e-02,  5.1952611230e-03],
            [ 3.4897930373e-18, -1.1793709730e-02,  5.1952611230e-03]]
        )
        self.assertAlmostEqual(abs(tdg.de - ref).max(), 0, 4)

    def test_col_cam_tda(self):
        mf = self.mol.UKS(xc='CAM-B3LYP').run()
        td = mf.TDA_SF().set(extype=0, collinear_samples=-50, nstates=2).run()
        tdg = td.Gradients().run()
        ref = np.array(
            [[-3.7095391429e-16,  2.3070693682e-15,  2.9440091136e-01],
            [-1.1663709778e-16,  2.8746882738e-01, -1.4720233631e-01],
            [-2.3132142127e-18, -2.8746882738e-01, -1.4720233631e-01]]
        )
        self.assertAlmostEqual(abs(tdg.de - ref).max(), 0, 4)

        td = mf.TDA_SF().set(extype=1, collinear_samples=-50, nstates=2).run()
        tdg = td.Gradients().run()
        ref = np.array(
            [[-4.7123223583e-16,  9.9126039837e-16, -1.5815377851e-02],
            [ 4.1675117676e-17,  8.5650850136e-03,  7.9015027954e-03],
            [-7.4950221683e-17, -8.5650850136e-03,  7.9015027954e-03]]
        )
        self.assertAlmostEqual(abs(tdg.de - ref).max(), 0, 4)


if __name__ == "__main__":
    print("Full Tests for spin-flip-TDA analytic gradient with multicollinear functionals and collinear functionals")
    unittest.main()