#!/usr/bin/env python
# Copyright 2014-2023 The PySCF Developers. All Rights Reserved.
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
# Author: Matthew Hennefarth <mhennefarth@uchicago.edu>

import numpy as np
import unittest

from pyscf import scf, gto, mcscf, lib
from pyscf import mcpdft
from mrh.my_pyscf.fci import csf_solver

def setUpModule():
    global h2_sto3g, lih
    h2_sto3g = scf.RHF(gto.M(atom=f'H 0 0 0; H 1.2 0 0', basis='sto-3g',
                       output='/dev/null', verbose=0)).run()
    lih = scf.RHF(gto.M(atom=f'Li 0 0 0; H 1.2 0 0', basis='sto-3g',
                        output='/dev/null', verbose=0)).run()


def tearDownModule():
    global h2_sto3g, lih
    h2_sto3g.mol.stdout.close()
    lih.mol.stdout.close()
    del h2_sto3g, lih


class KnownValues(unittest.TestCase):

    def assertListAlmostEqual(self, first_list, second_list, expected):
        self.assertTrue(len(first_list) == len(second_list))
        for first, second in zip(first_list, second_list):
            self.assertAlmostEqual(first, second, expected)

    def test_h2_sto3g(self):
        for nstates in range(2,4):
            for fnal in ('tLDA,VWN3', 'ftLDA,VWN3', 'tPBE', 'ftPBE'):
                with self.subTest(nstates=nstates, fnal=fnal):
                    mc = mcpdft.CASSCF(h2_sto3g, fnal, 2, 2, grids_level=1)
                    mc.fcisolver = csf_solver(mc.mol, smult=1)
                    weights = [1.0 / nstates, ] * nstates
                    lpdft = mc.multi_state(weights, method='lin').run()
                    lpdft_grad = lpdft.nuc_grad_method()

                    de = np.zeros(nstates)
                    for state in range(nstates):
                        de[state] = lpdft_grad.kernel(state=state)[1, 0]

                    lscanner = lpdft.as_scanner()
                    mol = lpdft.mol
                    lscanner(mol.set_geom_('H 0 0 0; H 1.20001 0 0'))
                    e1 = lscanner.e_states
                    lscanner(mol.set_geom_('H 0 0 0; H 1.19999 0 0'))
                    e2 = lscanner.e_states
                    lscanner(mol.set_geom_('H 0 0 0; H 1.2 0 0')) # reset
                    de_ref = (e1 - e2) / 0.00002 * lib.param.BOHR
                    self.assertListAlmostEqual(de, de_ref, 8)


if __name__ == "__main__":
    print("Full Tests for L-PDFT gradients API")
    unittest.main()
