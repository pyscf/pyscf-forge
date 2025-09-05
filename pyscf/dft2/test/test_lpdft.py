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
# Author: Matthew Hennefarth <mhennefarth@uchicago.com>

import tempfile, h5py
import numpy as np
from pyscf import gto, scf, dft, fci, lib
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
    solver2.spin = 2

    mc = mcpdft.CASSCF(mf, functional, 4, 4, grids_level=1)
    mc.chkfile = tempfile.NamedTemporaryFile().name 
    # mc.chk_ci = True
    mc = mc.multi_state_mix([solver1, solver2], weights, "lin")
    mc.run()
    return mc

def get_water_triplet(functional='tPBE', basis="6-31G"):
    mol = gto.M(atom='''
    O     0.    0.000    0.1174
    H     0.    0.757   -0.4696
    H     0.   -0.757   -0.4696
       ''', symmetry=True, basis=basis, output='/dev/null', verbose=0)

    mf = scf.RHF(mol).run()

    weights = np.ones(3) / 3
    solver1 = fci.direct_spin1_symm.FCI(mol)
    solver1.spin = 2
    solver1 = fci.addons.fix_spin(solver1, shift=.2, ss=2)
    solver1.nroots = 1
    solver2 = fci.direct_spin0_symm.FCI(mol)
    solver2.spin = 0
    solver2.nroots = 2

    mc = mcpdft.CASSCF(mf, functional, 4, 4, grids_level=1)
    mc.chkfile = tempfile.NamedTemporaryFile().name 
    # mc.chk_ci = True
    mc = mc.multi_state_mix([solver1, solver2], weights, "lin")
    mc.run()
    return mc


def setUpModule():
    global lih, lih_4, lih_tpbe, lih_tpbe0, lih_mc23, water, t_water, original_grids

    from importlib import reload
    from pyscf import dft2
    dft.libxc = dft2.libxc
    reload (mcpdft)
    reload (mcpdft.otfnal)

    original_grids = dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS
    dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False
    lih = get_lih(1.5)
    lih_4 = get_lih(1.5, n_states=4, basis="6-31G")
    lih_tpbe = get_lih(1.5, functional="tPBE")
    lih_tpbe0 = get_lih(1.5, functional="tPBE0")
    lih_mc23 = get_lih(1.5, functional="MC23")
    water = get_water()
    t_water = get_water_triplet()

def tearDownModule():
    global lih, lih_4, lih_tpbe0, lih_tpbe, t_water, water, original_grids, lih_mc23
    dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = original_grids
    lih.mol.stdout.close()
    lih_4.mol.stdout.close()
    lih_tpbe0.mol.stdout.close()
    lih_tpbe.mol.stdout.close()
    water.mol.stdout.close()
    t_water.mol.stdout.close()
    del lih, lih_4, lih_tpbe0, lih_tpbe, t_water, water, original_grids, lih_mc23

class KnownValues(unittest.TestCase):

    def assertListAlmostEqual(self, first_list, second_list, expected):
        self.assertTrue(len(first_list) == len(second_list))
        for first, second in zip(first_list, second_list):
            self.assertAlmostEqual(first, second, expected)

    def test_lih_mc23_adiabat(self):
        e_mcscf_mc23_avg = np.dot(lih_mc23.e_mcscf, lih_mc23.weights)
        hcoup = abs(lih_mc23.lpdft_ham[1,0])
        hdiag = lih_mc23.get_lpdft_diag()

        # Reference values from 
        #     - PySCF       commit 9a0bb6ddded7049bdacdaf4cfe422f7ce826c2c7
        #     - PySCF-forge commit eb0ad96f632994d2d1846009ecce047193682526
        E_MCSCF_AVG_EXPECTED = -7.78902182
        E_MC23_EXPECTED = [-7.94539408, -7.80094952]
        HCOUP_EXPECTED = 0.01285147
        HDIAG_EXPECTED = [-7.94424147, -7.80210214]

        self.assertAlmostEqual(e_mcscf_mc23_avg, E_MCSCF_AVG_EXPECTED, 7)
        self.assertAlmostEqual(hcoup, HCOUP_EXPECTED, 7)
        self.assertAlmostEqual(lib.fp(hdiag), lib.fp(HDIAG_EXPECTED), 7)
        self.assertAlmostEqual(lib.fp(lih_mc23.e_states), lib.fp(E_MC23_EXPECTED), 7)

if __name__ == "__main__":
    print("Full Tests for Linearized-PDFT")
    unittest.main()
