# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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

import numpy as np
import unittest
from pyscf import gto, scf, mcscf, lib
from pyscf.csf_fci import csf_solver
from pyscf.mcscf.las.addons import state_average_n_mix

def setUpModule():
    global mol, mf, mc, anion, rad1, rad2
    mol = gto.M (atom = 'O 0 0 0; H 1.145 0 0', basis='6-31g', symmetry=True, charge=-1, spin=0, verbose=0, output='/dev/null')
    mf = scf.RHF (mol).set (conv_tol=1e-10).run ()
    mc = mcscf.CASSCF (mf, 8, 8).set (conv_tol=1e-10).run ()
    mc.ci = None

    anion = csf_solver (mol, smult=1)
    anion.wfnsym = 'A1'

    rad1 = csf_solver (mol, smult=2)
    rad1.spin = 1
    rad1.charge = 1
    rad1.wfnsym = 'E1x'

    rad2 = csf_solver (mol, smult=2)
    rad2.spin = 1
    rad2.charge = 1
    rad2.wfnsym = 'E1y'

    mc = state_average_n_mix (mc, [anion, rad1, rad2], [1.0/3.0,]*3)
    mc.kernel ()

def tearDownModule():
    global mol, mf, mc, anion, rad1, rad2
    mol.stdout.close ()
    del mol, mf, mc, anion, rad1, rad2


class KnownValues(unittest.TestCase):
    def test_energies (self):
        self.assertAlmostEqual (mc.e_tot, -75.43571816597, 9)
        self.assertAlmostEqual (mc.e_states[0], -75.4219814122, 7)
        self.assertAlmostEqual (mc.e_states[1], -75.4425865429, 7)
        self.assertAlmostEqual (mc.e_states[2], -75.4425865429, 7)

    def test_occ (self):
        dm1 = mc.fcisolver.make_rdm1 (mc.ci, mc.ncas, mc.nelecas)
        dm1_states = mc.fcisolver.states_make_rdm1 (mc.ci, mc.ncas, mc.nelecas)
        self.assertAlmostEqual (np.trace (dm1), 7.3333333333, 9)
        self.assertAlmostEqual (np.trace (dm1_states[0]), 8.0, 9)
        self.assertAlmostEqual (np.trace (dm1_states[1]), 7.0, 9)
        self.assertAlmostEqual (np.trace (dm1_states[2]), 7.0, 9)

if __name__ == "__main__":
    print("Full Tests for state-average mix with different charges")
    unittest.main()




