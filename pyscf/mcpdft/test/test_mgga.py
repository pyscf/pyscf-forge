#!/usr/bin/env python
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
#
# Author: Bhavnesh Jangid <jangidbhavnesh@uchicago.com>

import tempfile
import numpy as np
from pyscf import gto, scf, dft, fci
from pyscf import mcpdft
import unittest

'''
In this unit-test, test the MCPDFT energies calculated for the LiH
molecule at the state-specific and state-average (2-states) using
1. Meta-GGA functional (M06L)
2. Hybrid-meta-GGA functional M06L0
3. MC23 Functional
4. Customized Functional

Test the MCPDFT energies calculated for the triplet water molecule at the
5. Meta-GGA functional (M06L)
6. MC23 Functional

Note: The reference values from OpenMolcas v22.02, tag 177-gc48a1862b
'''

def get_lih (r, stateaverage=False, functional='tM06L', basis='sto3g'):
    mol = gto.M (atom='Li 0 0 0\nH {} 0 0'.format (r), basis=basis,
                 output='/dev/null', verbose=0)
    mf = scf.RHF (mol).run ()
    if stateaverage:
        mc = mcpdft.CASSCF (mf, functional, 2, 2, grids_level=6)
        mc = mc.state_average_([0.5, 0.5])
    else:
        mc = mcpdft.CASSCF(mf, functional, 5, 2, grids_level=6)

    mc.fix_spin_(ss=0)
    mc = mc.run()
    return mc

def get_water_triplet(functional='tM06L', basis='6-31G'):
    mol = gto.M(atom='''
 O     0.    0.000    0.1174
 H     0.    0.757   -0.4696
 H     0.   -0.757   -0.4696
    ''',basis=basis, output='/dev/null', verbose=0)

    mf = scf.RHF(mol).run()

    mc = mcpdft.CASSCF(mf, functional, 4, 4, grids_level=6)
    solver1 = fci.addons.fix_spin(solver1, ss=2)
    solver1.spin = 2
    mc.fcisolver = solver1
    mc.run()
    return mc

def setUpModule():
    global original_grids
    global get_lih, lih_tm06l, lih_tmc23, lih_tm06l_sa2, lih_tmc23_sa2
    global get_lih_tm06l0, lih_tm06l0
    global get_lih_custom, lih_custom
    global get_water_triplet, water_tm06l, water_tmc23

    original_grids = dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS
    dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False
    lih_tm06l = get_lih(1.5, functional='tM06L')
    lih_tmc23 = get_lih(1.5, functional='tMC23')
    lih_tm06l_sa2 = get_lih(1.5, stateaverage=True, functional='tM06L')
    lih_tmc23_sa2 = get_lih(1.5, stateaverage=True, functional='tMC23')
    lih_tm06l0 = get_lih_tm06l0(1.5)
    lih_custom = get_lih_custom(1.5)
    water_tm06l = get_water_triplet()
    water_tmc23 = get_water_triplet(functional='tMC23')

def tearDownModule():
    global original_grids, lih_tm06l, lih_tmc23, lih_tm06l_sa2, lih_tmc23_sa2
    global lih_tm06l0, get_lih_custom, lih_custom, water_tm06l, water_tmc23
    dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = original_grids
    lih_tm06l.mol.stdout.close()#Done
    lih_tmc23.mol.stdout.close()#Done
    lih_tm06l_sa2.mol.stdout.close()#Done
    lih_tmc23_sa2.mol.stdout.close()#Done
    lih_tm06l0.mol.stdout.close()#Done
    lih_custom.mol.stdout.close()#Done
    water_tm06l.mol.stdout.close()#Done
    water_tmc23.mol.stdout.close()#Done
    del original_grids, lih_tm06l, lih_tmc23, lih_tm06l_sa2, lih_tmc23_sa2
    del lih_tm06l0, get_lih_custom, lih_custom, water_tm06l, water_tmc23
    
class KnownValues(unittest.TestCase):

    def assertListAlmostEqual(self, first_list, second_list, expected):
        self.assertTrue(len(first_list) == len(second_list))
        for first, second in zip(first_list, second_list):
            self.assertAlmostEqual(first, second, expected)

    def test_tmgga(self):
        e_mcscf = lih_tm06l.e_mcscf
        epdft = lih_tm06l.e_pdft
        
        sa_e_mcscf = lih_tm06l_sa2.e_mcscf
        sa_epdft = lih_tm06l_sa2.e_states

        E_CASSCF_EXPECTED = -7.88112386
        E_MCPDFT_EXPECTED =   -7.72800554
        SA_E_CASSCF_EXPECTED = [-7.88112386]
        SA_E_MCPDFT_EXPECTED = [-7.88112386]

        self.assertAlmostEqual(e_mcscf, E_CASSCF_EXPECTED, 7)
        self.assertAlmostEqual(epdft, E_MCPDFT_EXPECTED, 7)
        self.assertListAlmostEqual(sa_e_mcscf, SA_E_CASSCF_EXPECTED, 7)
        self.assertListAlmostEqual(sa_epdft, SA_E_MCPDFT_EXPECTED, 7)
    
    def test_t_hyb_mgga(self):
        e_mcscf = lih_tm06l0.e_mcscf
        epdft = lih_tm06l0.e_pdft
    
        E_CASSCF_EXPECTED = -7.88112386
        E_MCPDFT_EXPECTED =   -7.72800554
      
        self.assertAlmostEqual(e_mcscf, E_CASSCF_EXPECTED, 7)
        self.assertAlmostEqual(epdft, E_MCPDFT_EXPECTED, 7)
      
    def test_tmc23(self):
        e_mcscf = lih_tmc23.e_mcscf
        epdft = lih_tmc23.e_pdft
        
        sa_e_mcscf = lih_tmc23_sa2.e_mcscf
        sa_epdft = lih_tmc23_sa2.e_states

        E_CASSCF_EXPECTED = -7.88112386
        E_MCPDFT_EXPECTED =   -7.72800554
        SA_E_CASSCF_EXPECTED = [-7.88112386]
        SA_E_MCPDFT_EXPECTED = [-7.88112386]

        self.assertAlmostEqual(e_mcscf, E_CASSCF_EXPECTED, 7)
        self.assertAlmostEqual(epdft, E_MCPDFT_EXPECTED, 7)
        self.assertListAlmostEqual(sa_e_mcscf, SA_E_CASSCF_EXPECTED, 7)
        self.assertListAlmostEqual(sa_epdft, SA_E_MCPDFT_EXPECTED, 7)


    def test_water_triplet_tm06l(self):
        e_mcscf = water_tm06l.e_mcscf
        epdft = water_tm06l.e_pdft
    
        E_CASSCF_EXPECTED = -7.88112386
        E_MCPDFT_EXPECTED =   -7.72800554
      
        self.assertAlmostEqual(e_mcscf, E_CASSCF_EXPECTED, 7)
        self.assertAlmostEqual(epdft, E_MCPDFT_EXPECTED, 7)
    
    def test_water_triplet_tmc23(self):
        e_mcscf = water_tmc23.e_mcscf
        epdft = water_tmc23.e_pdft
    
        E_CASSCF_EXPECTED = -7.88112386
        E_MCPDFT_EXPECTED =   -7.72800554
      
        self.assertAlmostEqual(e_mcscf, E_CASSCF_EXPECTED, 7)
        self.assertAlmostEqual(epdft, E_MCPDFT_EXPECTED, 7)
    
    def test_custom_ot_functional(self):
        e_mcscf = lih_custom.e_mcscf
        epdft = lih_custom.e_pdft
    
        E_CASSCF_EXPECTED = -7.88112386
        E_MCPDFT_EXPECTED =   -7.72800554
      
        self.assertAlmostEqual(e_mcscf, E_CASSCF_EXPECTED, 7)
        self.assertAlmostEqual(epdft, E_MCPDFT_EXPECTED, 7)



if __name__ == "__main__":
    print("Full Tests for MGGAs and MC23")
    unittest.main()
