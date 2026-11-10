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
import numpy as np
from pyscf import lib, gto, scf, dft, fci, mcscf, df
from c2h4n4_struct import structure as struct
from pyscf.mcscf.las.lasscf_sync_o0 import LASSCF

def setUpModule():
    global mol, mf, mf_df, mol_hs, mf_hs, mf_hs_df, frags
    dr_nn = 3.0
    mol = struct (dr_nn, dr_nn, '6-31g', symmetry=False)
    mol.verbose = lib.logger.DEBUG
    mol.output = '/dev/null'
    mol.build ()
    mf = scf.RHF (mol).run ()
    mf_df = mf.density_fit (auxbasis = df.aug_etb (mol)).run ()

    mol_hs = mol.copy ()
    mol_hs.spin = 8
    mol_hs.build ()
    mf_hs = scf.RHF (mol_hs).run ()
    mf_hs_df = mf_hs.density_fit (auxbasis = df.aug_etb (mol_hs)).run ()

    frags = (list (range (3)), list (range (7,10)))

def tearDownModule():
    global mol, mf, mf_df, mol_hs, mf_hs, mf_hs_df, frags
    mol.stdout.close ()
    mol_hs.stdout.close ()
    del mol, mf, mf_df, mol_hs, mf_hs, mf_hs_df, frags


class KnownValues(unittest.TestCase):
    def test_dia (self):
        las = LASSCF (mf, (4,4), (4,4), spin_sub=(1,1))
        mo_coeff = las.localize_init_guess (frags)
        las.kernel (mo_coeff)
        self.assertTrue (las.converged)
        self.assertAlmostEqual (las.e_tot, -295.44779578419946, 7)

    def test_dia_df (self):
        las = LASSCF (mf_df, (4,4), (4,4), spin_sub=(1,1))
        mo_coeff = las.localize_init_guess (frags)
        las.kernel (mo_coeff)
        self.assertTrue (las.converged)
        self.assertAlmostEqual (las.e_tot, -295.44716017803967, 7)

    def test_ferro (self):
        las = LASSCF (mf_hs, (4,4), ((4,0),(4,0)), spin_sub=(5,5))
        mo_coeff = las.localize_init_guess (frags)
        las.kernel (mo_coeff)
        self.assertTrue (las.converged)
        self.assertAlmostEqual (las.e_tot, mf_hs.e_tot, 7)

    def test_ferro_df (self):
        las = LASSCF (mf_hs_df, (4,4), ((4,0),(4,0)), spin_sub=(5,5))
        mo_coeff = las.localize_init_guess (frags)
        las.kernel (mo_coeff)
        self.assertTrue (las.converged)
        self.assertAlmostEqual (las.e_tot, mf_hs_df.e_tot, 7)

    def test_af (self):
        las = LASSCF (mf_hs, (4,4), ((4,0),(0,4)), spin_sub=(5,5))
        mo_coeff = las.localize_init_guess (frags)
        las.kernel (mo_coeff)
        self.assertTrue (las.converged)
        self.assertAlmostEqual (las.e_tot, -295.44724798042466, 7)

    def test_af_df (self):
        las = LASSCF (mf_hs_df, (4,4), ((4,0),(0,4)), spin_sub=(5,5))
        mo_coeff = las.localize_init_guess (frags)
        las.kernel (mo_coeff)
        self.assertTrue (las.converged)
        self.assertAlmostEqual (las.e_tot, -295.4466638852035, 7)


if __name__ == "__main__":
    print("Full Tests for LASSCF c2h4n4")
    unittest.main()

