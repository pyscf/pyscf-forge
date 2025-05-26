#/usr/bin/env python
# Copyright 2014-2024 The PySCF Developers. All Rights Reserved.
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
import numpy
from pyscf import lib, gto, scf, dft
from pyscf import sftda
try:
    import mcfun
except ImportError:
    mcfun = None

def setUpModule():
    global mol, mf_lda, mf_bp86, mf_b3lyp, mf_tpss
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = '/dev/null'
    mol.atom = '''
    O     0.   0.       0.
    H     0.   -0.757   0.587
    H     0.   0.757    0.587'''
    mol.spin = 2
    mol.basis = '631g'
    mol.build()

    mf_lda = dft.UKS(mol).set(xc='lda', conv_tol=1e-12)
    mf_lda.grids.prune = None
    mf_lda = mf_lda.newton().run()
    mf_bp86 = dft.UKS(mol).set(xc='b88,p86', conv_tol=1e-12)
    mf_bp86.grids.prune = None
    mf_bp86 = mf_bp86.newton().run()
    mf_b3lyp = dft.UKS(mol).set(xc='b3lyp', conv_tol=1e-12)
    mf_b3lyp.grids.prune = None
    mf_b3lyp = mf_b3lyp.newton().run()
    mf_tpss = dft.UKS(mol).set(xc='tpss', conv_tol=1e-12)
    mf_tpss.grids.prune = None
    mf_tpss = mf_tpss.newton().run()

def tearDownModule():
    global mol, mf_lda, mf_bp86 , mf_b3lyp, mf_tpss
    mol.stdout.close()
    del mol, mf_lda, mf_bp86 , mf_b3lyp, mf_tpss


class KnownValues(unittest.TestCase):
    @unittest.skipIf(mcfun is None, "mcfun library not found.")
    def test_tddft_lda(self):
        td = sftda.TDDFT_SF(mf_lda).set(conv_tol=1e-12)
        td.extype = 0
        td.collinear_samples = 200
        es = td.kernel(nstates=6)[0]
        self.assertAlmostEqual(lib.fp(es[1:4]* 27.2114), 7.948103693468726, 4)
        ref = [2.90934687e-01, 4.17504148e-01, 5.39294897e-01, 1.00809769e+00]
        self.assertAlmostEqual(abs(es[1:5] - ref).max(), 0, 4)

        td = sftda.TDDFT_SF(mf_lda).set(conv_tol=1e-12)
        td.extype = 1
        td.collinear_samples = 200
        es = td.kernel(nstates=6)[0]
        self.assertAlmostEqual(lib.fp(es[1:4]* 27.2114), 0.9727213492347719, 4)
        ref = [2.48203085e-02, 9.20937945e-02, 9.33131348e-02, 2.42567238e-01]
        self.assertAlmostEqual(abs(es[1:5] - ref).max(), 0, 4)

    @unittest.skipIf(mcfun is None, "mcfun library not found.")
    def test_tddft_bp86(self):
        td = sftda.uks_sf.TDDFT_SF(mf_bp86).set(conv_tol=1e-12)
        td.extype = 0
        td.collinear_samples = 200
        es = td.kernel(nstates=6)[0]
        self.assertAlmostEqual(lib.fp(es[1:4]* 27.2114), 8.368198437727397, 4)
        ref = [3.03220029e-01, 4.48165285e-01, 5.71527356e-01, 1.04310453e+00]
        self.assertAlmostEqual(abs(es[1:5] - ref).max(), 0, 4)

        td = sftda.uks_sf.TDDFT_SF(mf_bp86).set(conv_tol=1e-12)
        td.extype = 1
        td.collinear_samples = 200
        es = td.kernel(nstates=6)[0]
        self.assertAlmostEqual(lib.fp(es[1:4]* 27.2114), 0.6454471439667748, 4)
        ref = [1.82489014e-02, 8.23546536e-02, 9.37783834e-02, 2.38795063e-01]
        self.assertAlmostEqual(abs(es[1:5] - ref).max(), 0, 4)

    @unittest.skipIf(mcfun is None, "mcfun library not found.")
    def test_tddft_b3lyp(self):
        td = sftda.TDDFT_SF(mf_b3lyp).set(conv_tol=1e-12)
        td.extype = 0
        td.collinear_samples = 200
        es = td.kernel(nstates=6)[0]
        self.assertAlmostEqual(lib.fp(es[1:4]* 27.2114), 8.310118087916951, 4)
        ref = [2.96621765e-01, 4.57522349e-01, 5.72949431e-01, 1.06284181e+00]
        self.assertAlmostEqual(abs(es[1:5] - ref).max(), 0, 4)

        td = sftda.TDDFT_SF(mf_b3lyp).set(conv_tol=1e-12)
        td.extype = 1
        td.collinear_samples = 200
        es = td.kernel(nstates=6)[0]
        self.assertAlmostEqual(lib.fp(es[1:4]* 27.2114), 0.7265669784640774, 4)
        ref = [1.83133171e-02, 8.50138300e-02, 9.02221325e-02, 2.35296638e-01]
        self.assertAlmostEqual(abs(es[1:5] - ref).max(), 0, 4)

    @unittest.skipIf(mcfun is None, "mcfun library not found.")
    def test_tddft_tpss(self):
        td = mf_tpss.TDDFT_SF().set(conv_tol=1e-12)
        td.extype = 0
        td.collinear_samples = 200
        es = td.kernel(nstates=6)[0]
        self.assertAlmostEqual(lib.fp(es[1:4]* 27.2114), 8.001090007698803, 4)
        ref = [2.87394817e-01, 4.47824155e-01, 5.65475295e-01, 1.05244100e+00]
        self.assertAlmostEqual(abs(es[1:5] - ref).max(), 0, 4)

        td = mf_tpss.TDDFT_SF().set(conv_tol=1e-12)
        td.extype = 1
        td.collinear_samples = 200
        es = td.kernel(nstates=6)[0]
        self.assertAlmostEqual(lib.fp(es[1:4]* 27.2114), 0.6465536096511839, 4)
        ref = [2.18608622e-02, 8.76824835e-02, 1.09277571e-01, 2.53604876e-01]
        self.assertAlmostEqual(abs(es[1:5] - ref).max(), 0, 4)

    def test_init(self):
        ks = scf.UKS(mol)
        self.assertTrue(isinstance(sftda.TDDFT_SF(ks), sftda.uks_sf.CasidaTDDFT))

if __name__ == "__main__":
    print("Full Tests for SF-TDDFT")
    unittest.main()
