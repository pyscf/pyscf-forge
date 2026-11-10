#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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

import copy
import unittest
import tempfile
import numpy as np
from pyscf.tools import molden
from pyscf import gto, scf, lib, mcscf
from pyscf.mcscf.las.lasscf_sync_o0 import LASSCF

def setUpModule():
    global mf, las, las2
    xyz='''Li 0 0 0,
           H 2 0 0,
           Li 10 0 0,
           H 12 0 0'''
    mol = gto.M (atom=xyz, basis='6-31g', symmetry='C2v', verbose=0, output='/dev/null')
    mf = scf.RHF (mol).run ()
    from pyscf.tools import molden
    mc = mcscf.CASSCF (mf, 4, 4).run ()
    mc.analyze ()
    
    las = LASSCF (mf, (2,2), (2,2))
    mo = las.localize_init_guess (([0,1],[2,3]), mc.mo_coeff, freeze_cas_spaces=True)
    las.kernel (mo)
    assert (las.converged)
    las = las.state_average_(weights=[1,0,0,0,0],
                             charges=[[0,0],[1,-1],[1,-1],[-1,1],[-1,1]],
                             spins=[[0,0],[1,-1],[-1,1],[1,-1],[-1,1]],
                             smults=[[1,1],[2,2],[2,2],[2,2],[2,2]],
                             wfnsyms=[[0,0],[0,0],[0,0],[0,0],[0,0]])
    las.lasci ()
    with tempfile.NamedTemporaryFile() as chkfile:
        las.dump_chk (chkfile=chkfile.name)
        las2 = LASSCF (mf, (2,2), (2,2))
        las2.load_chk_(chkfile=chkfile.name)

def tearDownModule():
    global mf, las, las2
    mf.mol.stdout.close ()
    del mf, las, las2

class KnownValues(unittest.TestCase):
    def test_config (self):
        self.assertEqual (las.ncore, las2.ncore)
        self.assertEqual (las.nfrags, las2.nfrags)
        self.assertEqual (las.nroots, las2.nroots)
        self.assertListEqual (list(las.ncas_sub), list(las2.ncas_sub))
        self.assertListEqual (las.nelecas_sub.tolist(), las2.nelecas_sub.tolist ())
        self.assertListEqual (list(las.weights), list(las2.weights))

    def test_results (self):
        self.assertEqual (las.e_tot, las2.e_tot)
        self.assertListEqual (list(las.e_states), list(las2.e_states))
        self.assertListEqual (list(las.states_converged), list(las2.states_converged))
        self.assertAlmostEqual (lib.fp (las.mo_coeff), lib.fp (las2.mo_coeff), 9)
        for i in range (2):
            for j in range (las.nroots):
                self.assertAlmostEqual (lib.fp (las.ci[i][j]), lib.fp (las2.ci[i][j]), 9)

if __name__ == "__main__":
    print("Full Tests for LASSCF chkfile")
    unittest.main()

