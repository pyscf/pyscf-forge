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
from pyscf.mcdcft import mcdcft
#from pyscf.fci import csf_solver
import unittest
import tempfile
import os

def run(r, xc, ot_name, chkfile):
    r /= 2
    mol = gto.M(atom=f'H  0 0 {r}; H 0 0 -{r}', basis='cc-pvtz', 
          symmetry=False, verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    mc = mcdcft.CASSCF(mf, xc, 2, 2, ot_name=ot_name, 
                       grids_level=6)
    #mc.fcisolver = csf_solver(mol, smult=1)
    mc.fix_spin_(ss=0)
    mc.chkfile = chkfile
    mc.kernel()
    mc.dump_mcdcft_chk(chkfile)
    return mc.e_tot

def restart(xc, ot_name, chkfile):
    mol = lib.chkfile.load_mol(chkfile)
    mol.verbose = 0
    mf = scf.RHF(mol)
    mc = mcdcft.CASSCF(mf, None, 2, 2, grids_level=6)
    mc.load_mcdcft_chk(chkfile)
    mc.recalculate_with_xc(xc, ot_name=ot_name, dump_chk=chkfile)
    return mc.e_tot

class KnownValues(unittest.TestCase):

    def test_cPBE(self):
        with tempfile.NamedTemporaryFile() as chkfile1:
            chkname1 = chkfile1.name
            with tempfile.NamedTemporaryFile() as chkfile2:
                chkname2 = chkfile2.name
                self.assertAlmostEqual(run(8.00, 'PBE', 'cPBE', chkname1) -
                                       run(0.78, 'PBE', 'cPBE', chkname2), 0.14898997201251052, 5)
                self.assertAlmostEqual(restart('BLYP', 'cBLYP', chkname1) -
                                       restart('BLYP', 'cBLYP', chkname2), 0.15624825293702616, 5)

    def test_cBLYP(self):
        with tempfile.NamedTemporaryFile() as chkfile1:
            chkname1 = chkfile1.name
            with tempfile.NamedTemporaryFile() as chkfile2:
                chkname2 = chkfile2.name
                self.assertAlmostEqual(run(8.00, 'BLYP', 'cBLYP', chkname1) -
                                       run(0.78, 'BLYP', 'cBLYP', chkname2), 0.15624825293702616, 5)
                self.assertAlmostEqual(restart('PBE', 'cPBE', chkname1) -
                                       restart('PBE', 'cPBE', chkname2), 0.14898997201251052, 5)

if __name__ == "__main__":
    print("Full Tests for MC-DCFT energies of H2 molecule")
    unittest.main()

