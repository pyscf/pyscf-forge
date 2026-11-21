#!/usr/bin/env python
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

# End-to-end tests of LASCI (mf, ncas_sub, nelecas_sub).kernel () and of the branches of the
# LASCI factory function. Two H2 molecules 50 Angstrom apart don't interact, so LASCI equals
# CASCI in the same orbitals. When they are close, LASCI is bounded below by that CASCI.

import unittest
from pyscf import gto, scf, mcscf, lib
from pyscf.mcscf.las import _DFLASCI
from pyscf.mcscf.las.lasci import LASCI, LASCINoSymm, LASCISymm

# Planar, so that each fragment is invariant under the Cs mirror plane
xyz_far = 'H 0 0 0; H 0.74 0 0; H 0.3 50 0; H 1.2 50.3 0'
xyz_near = 'H 0 0 0; H 0.74 0 0; H 0.3 2.0 0; H 1.2 2.3 0'
frags = ((0,1),(2,3))

def setUpModule():
    global mol, mf, mf_symm, mf_df, mf_near
    mol = gto.M (atom=xyz_far, basis='6-31g', symmetry=False, verbose=0, output='/dev/null')
    mf = scf.RHF (mol).run ()
    mol_symm = gto.M (atom=xyz_far, basis='6-31g', symmetry=True, verbose=0, output='/dev/null')
    mf_symm = scf.RHF (mol_symm).run ()
    mf_df = scf.RHF (mol).density_fit ().run ()
    mol_near = gto.M (atom=xyz_near, basis='6-31g', symmetry=False, verbose=0, output='/dev/null')
    mf_near = scf.RHF (mol_near).run ()

def tearDownModule():
    global mol, mf, mf_symm, mf_df, mf_near
    for m in (mf, mf_symm, mf_df, mf_near): m.mol.stdout.close ()
    del mol, mf, mf_symm, mf_df, mf_near

def _run (my_mf):
    las = LASCI (my_mf, (2,2), (2,2))
    mo = las.localize_init_guess (frags, my_mf.mo_coeff)
    las.kernel (mo)
    mc = mcscf.CASCI (my_mf, 4, 4)
    mc.kernel (mo)
    return las, mc

class KnownValues (unittest.TestCase):

    def test_nosymm (self):
        las, mc = _run (mf)
        self.assertIsInstance (las, LASCINoSymm)
        self.assertNotIsInstance (las, LASCISymm)
        self.assertTrue (las.converged)
        self.assertAlmostEqual (las.e_tot, mc.e_tot, 8)

    def test_symm (self):
        las, mc = _run (mf_symm)
        self.assertIsInstance (las, LASCISymm)
        self.assertTrue (las.converged)
        self.assertAlmostEqual (las.e_tot, mc.e_tot, 8)

    def test_df (self):
        las, mc = _run (mf_df)
        self.assertIsInstance (las, _DFLASCI)
        self.assertTrue (las.converged)
        self.assertAlmostEqual (las.e_tot, mc.e_tot, 8)

    def test_mol (self):
        las = LASCI (mol, (2,2), (2,2))
        self.assertIsInstance (las._scf, scf.hf.RHF)
        self.assertIsInstance (las, LASCINoSymm)
        # LASCI (mol, ...) does not run the SCF, so the guess orbitals must be passed explicitly
        las._scf.run ()
        las.kernel (las.localize_init_guess (frags, las._scf.mo_coeff))
        self.assertTrue (las.converged)
        self.assertAlmostEqual (las.e_tot, _run (mf)[0].e_tot, 8)

    def test_variational_bound (self):
        las, mc = _run (mf_near)
        self.assertTrue (las.converged)
        # The fragments interact, so LASCI is strictly above CASCI in the same orbitals
        self.assertGreater (las.e_tot, mc.e_tot + 1e-6)

if __name__ == "__main__":
    print("Full Tests for LASCI kernel")
    unittest.main()
