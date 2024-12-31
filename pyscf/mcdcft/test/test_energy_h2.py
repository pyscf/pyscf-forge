#!/usr/bin/env python
# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
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
#
from pyscf import gto, scf, lib, mcscf
from pyscf.mcdcft import mcdcft, dcfnal
#from pyscf.fci import csf_solver
import unittest
import tempfile
import os

def run(r, xc, chkfile):
    r /= 2
    mol = gto.M(atom=f'H  0 0 {r}; H 0 0 -{r}', basis='cc-pvtz',
          symmetry=False, verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    mc = mcdcft.CASSCF(mf, xc, 2, 2, grids_level=6)
    mc.fix_spin_(ss=0)
    mc.chkfile = chkfile
    mc.kernel()
    return mc.e_tot

def run2(r, xc, chkfile):
    r /= 2
    mol = gto.M(atom=f'H  0 0 {r}; H 0 0 -{r}', basis='cc-pvtz',
          symmetry=False, verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    xc0 = 'DC24'
    mc = mcdcft.CASSCF(mf, xc0, 2, 2, grids_level=6)
    mc.fix_spin_(ss=0)
    mc.kernel()
    mc.recalculate_with_xc(xc, load_chk=chkfile)
    return mc.e_tot

def restart(xc, chkfile):
    mol = lib.chkfile.load_mol(chkfile)
    mol.verbose = 0
    mf = scf.RHF(mol)
    mc = mcdcft.CASSCF(mf, xc, 2, 2, grids_level=6)
    mc.load_mcdcft_chk(chkfile)
    mc.recalculate_with_xc(xc, dump_chk=chkfile)
    return mc.e_tot

cPBE_preset = dict(args=dict(f=dcfnal.f_v1, negative_rho=True), xc_code='PBE')
cBLYP_preset = dict(args=dict(f=dcfnal.f_v1, negative_rho=True), xc_code='BLYP')

class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dcfnal.register_dcfnal_('cPBE', cPBE_preset)
        dcfnal.register_dcfnal_('cBLYP', cBLYP_preset)

    @classmethod
    def tearDownClass(cls):
        dcfnal.unregister_dcfnal_('cPBE')
        dcfnal.unregister_dcfnal_('cBLYP')

    def test_cPBE(self):
        with tempfile.NamedTemporaryFile() as chkfile1:
            chkname1 = chkfile1.name
            with tempfile.NamedTemporaryFile() as chkfile2:
                chkname2 = chkfile2.name
                self.assertAlmostEqual(run(8.00, 'cPBE', chkname1) -
                                       run(0.78, 'cPBE', chkname2), 0.14898997201251052, 5)
                self.assertAlmostEqual(run2(8.00, 'cPBE', chkname1) -
                                       run2(0.78, 'cPBE', chkname2), 0.14898997201251052, 5)
                self.assertAlmostEqual(restart('cBLYP', chkname1) -
                                       restart('cBLYP', chkname2), 0.15624825293702616, 5)

    def test_cBLYP(self):
        with tempfile.NamedTemporaryFile() as chkfile1:
            chkname1 = chkfile1.name
            with tempfile.NamedTemporaryFile() as chkfile2:
                chkname2 = chkfile2.name
                self.assertAlmostEqual(run(8.00, 'cBLYP', chkname1) -
                                       run(0.78, 'cBLYP', chkname2), 0.15624825293702616, 5)
                self.assertAlmostEqual(run2(8.00, 'cBLYP', chkname1) -
                                       run2(0.78, 'cBLYP', chkname2), 0.15624825293702616, 5)
                self.assertAlmostEqual(restart('cPBE', chkname1) -
                                       restart('cPBE', chkname2), 0.14898997201251052, 5)

    def test_DC24(self):
        self.assertAlmostEqual(run(0.78, 'DC24', None), -1.25248849, 5)

if __name__ == "__main__":
    print("Full Tests for MC-DCFT energies of H2 molecule")
    unittest.main()

