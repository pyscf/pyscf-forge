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
import numpy as np
from pyscf import gto, scf, fci
from pyscf import mcpdft
import unittest


def get_lih (r, n_states=2, functional='ftLDA,VWN3', basis='sto3g'):
    mol = gto.M (atom='Li 0 0 0\nH {} 0 0'.format (r), basis=basis,
                 output='/dev/null', verbose=0)
    mf = scf.RHF (mol).run ()
    if n_states == 2:
        mc = mcpdft.CASSCF (mf, functional, 2, 2, grids_level=1)

    else:
        mc = mcpdft.CASSCF(mf, functional, 5, 2, grids_level=1)

    mc.fix_spin_(ss=0)
    weights = [1.0/float(n_states), ] * n_states
    
    mc = mc.multi_state(weights, "lin")
    mc = mc.run()
    return mc

def get_water(functional='tpbe', basis='6-31g'):
    mol = gto.M(atom='''
 O     0.    0.000    0.1174
 H     0.    0.757   -0.4696
 H     0.   -0.757   -0.4696
    ''',symmetry=True, basis=basis, output='/dev/null', verbose=0)

    mf = scf.RHF(mol).run()

    weights = [0.5, 0.5]
    solver1 = fci.direct_spin1_symm.FCI(mol)
    solver1.wfnsym = 'A1'
    solver1.spin = 0
    solver2 = fci.direct_spin1_symm.FCI(mol)
    solver2.wfnsym = 'A2'
    solver2.spin = 0

    mc = mcpdft.CASSCF(mf, functional, 4, 4, grids_level=1)
    mc = mc.multi_state_mix([solver1, solver2], weights, "lin")
    mc.run()
    return mc

def get_cc(r, functional='tPBE', basis='cc-pvdz'):
    mol = gto.Mole(atom=[
        ['C', (0., 0., -r / 2)],
        ['C', (0., 0., r / 2)], ], basis=basis, unit='B', symmetry=True, output='tmp.log', verbose=5)

    mf = scf.RHF(mol)
    mf.irrep_nelec = {'A1g': 4, 'E1gx': 0, 'E1gy': 0, 'A1u': 4,
                      'E1uy': 2, 'E1ux': 2, 'E2gx': 0, 'E2gy': 0, 'E2uy': 0, 'E2ux': 0}

    mf.kernel()
    weights = np.ones(3) / 3
    solver1 = fci.direct_spin1_symm.FCI(mol)
    solver1.spin = 2
    solver1 = fci.addons.fix_spin(solver1, shift=.2, ss=2)
    solver1.nroots = 1
    solver2 = fci.direct_spin0_symm.FCI(mol)
    solver2.spin = 0
    solver2.nroots = 2

    mc = mcpdft.CASSCF(mf, functional, 8, 8, grids_level=1)
    mc = mc.multi_state_mix([solver1, solver2], weights, "lin")
    mc.run()
    return mc


def setUpModule():
    global lih, lih_4, lih_tpbe, lih_tpbe0, water, cc
    lih = get_lih(1.5)
    lih_4 = get_lih(1.5, n_states=4, basis="6-31G")
    lih_tpbe = get_lih(1.5, functional="tPBE")
    lih_tpbe0 = get_lih(1.5, functional="tPBE0")
    water = get_water()
    cc = get_cc(1.8)

def tearDownModule():
    global lih, lih_4, lih_tpbe0, lih_tpbe, water, cc
    lih.mol.stdout.close()
    lih_4.mol.stdout.close()
    lih_tpbe0.mol.stdout.close()
    lih_tpbe.mol.stdout.close()
    water.mol.stdout.close()
    cc.mol.stdout.close()
    del lih, lih_4, lih_tpbe0, lih_tpbe, water, cc

class KnownValues(unittest.TestCase):

    def assertListAlmostEqual(self, first_list, second_list, expected):
        self.assertTrue(len(first_list) == len(second_list))
        for first, second in zip(first_list, second_list):
            self.assertAlmostEqual(first, second, expected)

    def test_lih_2_states_adiabat(self):
        e_mcscf_avg = np.dot (lih.e_mcscf, lih.weights)
        hcoup = abs(lih.lpdft_ham[1,0])
        hdiag = lih.get_lpdft_diag()

        e_states = lih.e_states

        # Reference values from OpenMolcas v22.02, tag 177-gc48a1862b
        E_MCSCF_AVG_EXPECTED = -7.78902185
        
        # Below reference values from 
        #   - PySCF commit 71fc2a41e697fec76f7f9a5d4d10fd2f2476302c
        #   - mrh   commit c5fc02f1972c1c8793061f20ed6989e73638fc5e
        HCOUP_EXPECTED = 0.01663680
        HDIAG_EXPECTED = [-7.87848993, -7.72984482]

        E_STATES_EXPECTED = [-7.88032921, -7.72800554]

        self.assertAlmostEqual(e_mcscf_avg, E_MCSCF_AVG_EXPECTED, 7)
        self.assertAlmostEqual(abs(hcoup), HCOUP_EXPECTED, 7)
        self.assertListAlmostEqual(hdiag, HDIAG_EXPECTED, 7)
        self.assertListAlmostEqual(e_states, E_STATES_EXPECTED, 7)

    def test_lih_4_states_adiabat(self):
        e_mcscf_avg = np.dot(lih_4.e_mcscf, lih_4.weights)
        hdiag = lih_4.get_lpdft_diag()
        hcoup = lih_4.lpdft_ham[np.triu_indices(4, k=1)]
        e_states = lih_4.e_states

        # References values from
        #     - PySCF       commit 71fc2a41e697fec76f7f9a5d4d10fd2f2476302c
        #     - PySCF-forge commit 00183c314ebbf541f8461e7b7e5ee9e346fd6ff5
        E_MCSCF_AVG_EXPECTED = -7.88112386
        HDIAG_EXPECTED = [-7.99784259, -7.84720560, -7.80476518, -7.80476521]
        HCOUP_EXPECTED = [0.01479405,0,0,0,0,0]
        E_STATES_EXPECTED = [-7.99928176, -7.84576642, -7.80476519, -7.80476519]

        self.assertAlmostEqual(e_mcscf_avg, E_MCSCF_AVG_EXPECTED, 7)
        self.assertListAlmostEqual(hdiag, HDIAG_EXPECTED, 7)
        self.assertListAlmostEqual(list(map(abs, hcoup)), HCOUP_EXPECTED, 7)
        self.assertListAlmostEqual(e_states, E_STATES_EXPECTED, 7)


    def test_lih_hybrid_tPBE_adiabat(self):
        e_mcscf_tpbe_avg = np.dot(lih_tpbe.e_mcscf, lih_tpbe.weights)
        e_mcscf_tpbe0_avg = np.dot(lih_tpbe0.e_mcscf, lih_tpbe0.weights)

        hlpdft_ham = 0.75 * lih_tpbe.lpdft_ham
        idx = np.diag_indices_from(hlpdft_ham)
        hlpdft_ham[idx] += 0.25 * lih_tpbe.e_mcscf
        e_hlpdft, si_hlpdft = lih_tpbe._eig_si(hlpdft_ham)

        # References values from
        #     - PySCF       commit 8ae2bb2eefcd342c52639097517b1eda7ca5d1cd
        #     - PySCF-forge commit a7b8b3bb291e528088f9cefab007438d9e0f4701
        E_MCSCF_AVG_EXPECTED = -7.78902182
        E_TPBE_STATES_EXPECTED = [-7.93389909, -7.78171959]


        self.assertAlmostEqual(e_mcscf_tpbe_avg, E_MCSCF_AVG_EXPECTED, 7)
        self.assertAlmostEqual(e_mcscf_tpbe_avg, e_mcscf_tpbe0_avg, 9)
        self.assertListAlmostEqual(lih_tpbe.e_states, E_TPBE_STATES_EXPECTED, 7)
        self.assertListAlmostEqual(lih_tpbe0.e_states, e_hlpdft, 9)
        self.assertListAlmostEqual(hlpdft_ham.flatten(), lih_tpbe0.lpdft_ham.flatten(), 9)

    def test_water_spatial_samix(self):
        e_mcscf_avg = np.dot(water.e_mcscf, water.weights)
        hdiag = water.get_lpdft_diag()
        e_states = water.e_states

        # References values from
        #     - PySCF       commit 8ae2bb2eefcd342c52639097517b1eda7ca5d1cd
        #     - PySCF-forge commit 5338d3060033d60b47e0c89cfcfe9427c34ff24a
        E_MCSCF_AVG_EXPECTED = -75.81489195169507
        HDIAG_EXPECTED = [-76.29913074162732, -75.93502437481517]

        self.assertAlmostEqual(e_mcscf_avg, E_MCSCF_AVG_EXPECTED, 7)
        self.assertListAlmostEqual(hdiag, HDIAG_EXPECTED, 7)
        # The off-diagonal should be identical to zero because of symmetry
        self.assertListAlmostEqual(e_states, hdiag, 10)

    def test_C2_spin_samix(self):
        print(cc.e_states)

if __name__ == "__main__":
    print("Full Tests for Linearized-PDFT")
    unittest.main()
