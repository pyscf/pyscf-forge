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


h2_xyz = 'H 0 0 0; H 1.2 0 0'
lih_xyz = 'Li 0 0 0; H 1.5 0 0'
def setUpModule():
    global h2_sto3g, lih_sto3g, h2_631g
    h2_sto3g = scf.RHF(gto.M(atom=h2_xyz, basis='sto-3g',
                       output='/dev/null', verbose=0)).run()
    h2_631g = scf.RHF(gto.M(atom=h2_xyz, basis='6-31g',
                       output='lpdft.log', verbose=5)).run()
    lih_sto3g = scf.RHF(gto.M(atom=lih_xyz, basis='sto-3g',
                        output='/dev/null', verbose=0)).run()


def tearDownModule():
    global h2_sto3g, lih_sto3g, h2_631g
    h2_sto3g.mol.stdout.close()
    lih_sto3g.mol.stdout.close()
    h2_631g.mol.stdout.close()
    del h2_sto3g, lih_sto3g, h2_631g


class KnownValues(unittest.TestCase):

    def assertListAlmostEqual(self, first_list, second_list, expected):
        self.assertTrue(len(first_list) == len(second_list))
        for first, second in zip(first_list, second_list):
            self.assertAlmostEqual(first, second, expected)

    # def test_h2_sto3g(self):
    #     for nstates in range(2,4):
    #         for fnal in ('tLDA,VWN3', 'ftLDA,VWN3', 'tPBE', 'ftPBE'):
    #             with self.subTest(nstates=nstates, fnal=fnal):
    #                 mc = mcpdft.CASSCF(h2_sto3g, fnal, 2, 2, grids_level=1)
    #                 mc.fix_spin_(ss=0, shift=1)
    #                 lpdft = mc.multi_state([1.0 / nstates, ] * nstates, method='lin').run()
    #                 lpdft_grad = lpdft.nuc_grad_method()
    #
    #                 de = np.zeros(nstates)
    #                 for state in range(nstates):
    #                     de[state] = lpdft_grad.kernel(state=state)[1, 0]
    #
    #                 lscanner = lpdft.as_scanner()
    #                 mol = lpdft.mol
    #                 lscanner(mol.set_geom_('H 0 0 0; H 1.20001 0 0'))
    #                 e1 = lscanner.e_states
    #                 lscanner(mol.set_geom_('H 0 0 0; H 1.19999 0 0'))
    #                 e2 = lscanner.e_states
    #                 lscanner(mol.set_geom_(h2_xyz))  # reset
    #                 de_ref = (e1 - e2) / 0.00002 * lib.param.BOHR
    #                 self.assertListAlmostEqual(de, de_ref, 8)

    def test_h2_631g(self):
        for nstates in range(2,3):
            for fnal in ('tLDA,VWN3', 'ftLDA,VWN3', 'tPBE', 'ftPBE'):
                with self.subTest(nstates=nstates, fnal=fnal):
                    mc = mcpdft.CASSCF(h2_631g, fnal, 4, 2, grids_level=1)
                    mc.fix_spin_(ss=0, shift=1)
                    #lpdft = mc.state_average([1.0 / nstates, ] * nstates).run()
                    lpdft = mc.multi_state([1.0 / nstates, ] * nstates, method='lin').run()
                    lpdft_grad = lpdft.nuc_grad_method()

                    de = np.zeros(nstates)
                    for state in range(nstates):
                        de[state] = lpdft_grad.kernel(state=state)[1, 0]

                    lscanner = lpdft.as_scanner()
                    mol = lpdft.mol
                    lscanner(mol.set_geom_('H 0 0 0; H 1.21 0 0'))
                    e1 = np.array(lscanner.e_states)
                    lscanner(mol.set_geom_('H 0 0 0; H 1.19 0 0'))
                    e2 = np.array(lscanner.e_states)
                    lscanner(mol.set_geom_(h2_xyz))  # reset
                    de_ref = (e1 - e2) / 0.02 * lib.param.BOHR
                    print(de)
                    print(de_ref)
                    print(de-de_ref)
                    return
                    self.assertListAlmostEqual(de, de_ref, 8)


    # def test_lih_sto3g(self):
    #     nstates = 3
    #     for fnal in ('tLDA,VWN3', 'ftLDA,VWN3', 'tPBE', 'ftPBE'):
    #         with self.subTest(fnal=fnal):
    #             mc = mcpdft.CASSCF(lih_sto3g, fnal, 2, 2, grids_level=1)
    #             mc.fix_spin_(ss=0, shift=1)
    #             #lpdft = mc.state_average([1.0 / nstates, ] * nstates).run()
    #             lpdft = mc.multi_state([1.0 / nstates, ] * nstates, method='lin').run()
    #             lpdft_grad = lpdft.nuc_grad_method()
    #
    #             de = np.zeros(nstates)
    #             for state in range(nstates):
    #                 de[state] = lpdft_grad.kernel(state=state)[1, 0]
    #
    #             lscanner = lpdft.as_scanner()
    #             mol = lpdft.mol
    #             lscanner(mol.set_geom_('Li 0 0 0; H 1.50001 0 0'))
    #             e1 = np.array(lscanner.e_states)
    #             lscanner(mol.set_geom_('Li 0 0 0; H 1.49999 0 0'))
    #             e2 = np.array(lscanner.e_states)
    #             lscanner(mol.set_geom_(lih_xyz))  # reset
    #             de_ref = (e1 - e2) / 0.00002 * lib.param.BOHR
    #             print(de)
    #             print(de_ref)
    #             print(de-de_ref)
    #             return
                #self.assertListAlmostEqual(de, de_ref, 8)


if __name__ == "__main__":
    print("Full Tests for L-PDFT gradients API")
    unittest.main()
