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
from pyscf import gto, dft
from pyscf import sftda
try:
    import mcfun
except ImportError:
    mcfun = None

# ToDo: Add the SF-TDDFT Grad tests.

def setUpModule():
    global mol, pmol, mf_lda, mf_bp86, mf_b3lyp, mf_tpss, nstates
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = '/dev/null'
    mol.atom = [
        ['O' , (0. , 0. , 2.070)],
        ['O' , (0. , 0. , 0.)], ]
    mol.unit = 'B'
    mol.spin = 2
    mol.basis = '6-31g'
    mol.build()
    pmol = mol.copy()

    mf_lda  = dft.UKS(mol).set(xc='lda,', conv_tol=1e-12)
    mf_lda.kernel()
    mf_bp86 = dft.UKS(mol).set(xc='bp86', conv_tol=1e-12)
    mf_bp86.kernel()
    mf_b3lyp= dft.UKS(mol).set(xc='b3lyp',conv_tol=1e-12)
    mf_b3lyp.kernel()
    mf_tpss = dft.UKS(mol).set(xc='tpss', conv_tol=1e-12)
    mf_tpss.kernel()

def tearDownModule():
    global mol, pmol, mf_lda, mf_bp86, mf_b3lyp, mf_tpss
    mol.stdout.close()
    del mol, pmol, mf_lda, mf_bp86, mf_b3lyp, mf_tpss

class KnownValues(unittest.TestCase):
    @unittest.skipIf(mcfun is None, "mcfun library not found.")
    def test_tdasfd_anag_lda(self):
        td = sftda.TDA_SF(mf_lda).set(conv_tol=1e-12)
        td.extype = 1
        td.collinear_samples = 200
        td.kernel(nstates=5)[0]

        tdg = td.nuc_grad_method()
        g1 = tdg.kernel(td.xy[2])
        self.assertAlmostEqual(g1[0,2], -0.37025743446132253, 6)

        td_solver = td.as_scanner()
        e1 = td_solver(pmol.set_geom_('O 0 0 2.071; O 0 0 0', unit='B'))
        e2 = td_solver(pmol.set_geom_('O 0 0 2.069; O 0 0 0', unit='B'))
        self.assertAlmostEqual((e1[2]-e2[2])/.002, g1[0,2], 4)

    @unittest.skipIf(mcfun is None, "mcfun library not found.")
    def test_tdasfu_anag_bp86(self):
        td = sftda.TDA_SF(mf_bp86).set(conv_tol=1e-12)
        td.extype = 0
        td.collinear_samples = 200
        td.kernel(nstates=5)[0]

        tdg = td.nuc_grad_method()
        g1 = tdg.kernel(td.xy[2])
        self.assertAlmostEqual(g1[0,2], -1.4387734863741706, 6)

        td_solver = td.as_scanner()
        e1 = td_solver(pmol.set_geom_('O 0 0 2.071; O 0 0 0', unit='B'))
        e2 = td_solver(pmol.set_geom_('O 0 0 2.069; O 0 0 0', unit='B'))
        self.assertAlmostEqual((e1[2]-e2[2])/.002, g1[0,2], 4)

    @unittest.skipIf(mcfun is None, "mcfun library not found.")
    def test_tdasfu_anag_b3lyp(self):
        td = sftda.TDA_SF(mf_b3lyp).set(conv_tol=1e-12)
        td.extype = 1
        td.collinear_samples = 200
        td.kernel(nstates=5)[0]

        tdg = td.Gradients()
        g1 = tdg.kernel(td.xy[3])
        self.assertAlmostEqual(g1[0,2], -0.3428495646475067, 6)

        td_solver = td.as_scanner()
        e1 = td_solver(pmol.set_geom_('O 0 0 2.071; O 0 0 0', unit='B'))
        e2 = td_solver(pmol.set_geom_('O 0 0 2.069; O 0 0 0', unit='B'))
        self.assertAlmostEqual((e1[3]-e2[3])/.002, g1[0,2], 4)

    @unittest.skipIf(mcfun is None, "mcfun library not found.")
    def test_tdasfu_anag_tpss(self):
        td = sftda.TDA_SF(mf_tpss).set(conv_tol=1e-12)
        td.extype = 0
        td.collinear_samples = 200
        td.kernel(nstates=5)[0]

        tdg = td.Gradients()
        g1 = tdg.kernel(td.xy[3])
        self.assertAlmostEqual(g1[0,2], -1.027043808560098, 6)

        td_solver = td.as_scanner()
        e1 = td_solver(pmol.set_geom_('O 0 0 2.071; O 0 0 0', unit='B'))
        e2 = td_solver(pmol.set_geom_('O 0 0 2.069; O 0 0 0', unit='B'))
        self.assertAlmostEqual((e1[3]-e2[3])/.002, g1[0,2], 4)

if __name__ == "__main__":
    print("Full Tests for SF-TD-UKS gradients")
    unittest.main()
