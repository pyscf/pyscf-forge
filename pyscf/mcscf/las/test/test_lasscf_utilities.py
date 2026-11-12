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

# Tests of LAS utility methods that the rest of this test suite doesn't otherwise check:
# get_single_state_las (from mrh's tests/lasscf/test_utilities.py), h1e_for_las, get_sym_fr, and
# LASSCF.lasci/lasci_ against LASCI.kernel.

import unittest
import numpy as np
from pyscf import gto, scf, symm
from pyscf.mcscf.las.lasci import LASCI
from pyscf.mcscf.las.lasscf_sync_o0 import LASSCF

# Planar, so that each fragment is invariant under the Cs mirror plane
xyz_near = 'H 0 0 0; H 0.74 0 0; H 0.3 2.0 0; H 1.2 2.3 0'
frags = ((0,1),(2,3))
# Three rootspaces, with fragments of different sizes, so that a swapped index would show up as a
# wrong shape
ncas_sub, nelecas_sub = (2,3), (2,2)
sa_kwargs = {'weights': [1.0/3,]*3,
             'spins':   [[0,0],[2,0],[0,2]],
             'smults':  [[1,1],[3,1],[1,3]]}

def setUpModule():
    global mf, mf_symm, mf_n2
    mol = gto.M (atom=xyz_near, basis='6-31g', symmetry=False, verbose=0, output='/dev/null')
    mf = scf.RHF (mol).run ()
    mol = gto.M (atom=xyz_near, basis='6-31g', symmetry=True, verbose=0, output='/dev/null')
    mf_symm = scf.RHF (mol).run ()
    mol = gto.M (atom='N 0 0 0; N 3 0 0', basis='cc-pvdz', symmetry=False, verbose=0,
                 output='/dev/null')
    mf_n2 = scf.RHF (mol).run ()

def tearDownModule():
    global mf, mf_symm, mf_n2
    for m in (mf, mf_symm, mf_n2): m.mol.stdout.close ()
    del mf, mf_symm, mf_n2

def _sa_las (cls, my_mf):
    las = cls (my_mf, ncas_sub, nelecas_sub)
    las.state_average_(**sa_kwargs)
    mo = las.localize_init_guess (frags, my_mf.mo_coeff)
    return las, mo

class KnownValues (unittest.TestCase):

    def test_get_single_state_las (self):
        # N2 at 3 Angstrom as two quartet N atoms. The rootspaces are those that mrh's
        # lassi.spaces.spin_shuffle generates from the reference ((3,0),(0,3)): every way to
        # distribute local Sz with total Sz = 0.
        las = LASSCF (mf_n2, (6,6), ((3,0),(0,3)))
        mo = las.localize_init_guess (([0],[1]), mf_n2.mo_coeff)
        las.state_average_(weights=[0.25,]*4, charges=[[0,0],]*4,
                           spins=[[3,-3],[-3,3],[-1,1],[1,-1]], smults=[[4,4],]*4)
        las.kernel (mo)
        self.assertTrue (las.converged)
        for i in range (len (las.e_states)):
            with self.subTest (state=i):
                las1 = las.get_single_state_las (state=i)
                self.assertEqual (las1.nroots, 1)
                las1.lasci ()
                self.assertAlmostEqual (las1.e_tot, las.e_states[i], 6)
                las1.kernel ()
                self.assertTrue (las1.converged)
                self.assertLessEqual (las1.e_tot, las.e_states[i])

    def test_h1e_for_las (self):
        las, mo = _sa_las (LASCI, mf)
        las.kernel (mo)
        h1e_fr = las.h1e_for_las ()
        self.assertEqual (len (h1e_fr), len (ncas_sub))
        for ifrag, (h1e_r, no) in enumerate (zip (h1e_fr, ncas_sub)):
            with self.subTest (frag=ifrag):
                self.assertEqual (h1e_r.shape, (las.nroots, 2, no, no))
                self.assertAlmostEqual (np.amax (np.abs (h1e_r - h1e_r.transpose (0,1,3,2))),
                                        0, 12)

    def test_get_sym_fr (self):
        las = _sa_las (LASCI, mf)[0]
        sym_fr = las.get_sym_fr ()
        self.assertEqual (sym_fr.shape, (len (ncas_sub), las.nroots))
        self.assertTrue (np.issubdtype (sym_fr.dtype, np.integer))
        self.assertTrue (np.all (sym_fr == 0))
        # With symmetry, irrep names are converted to IDs
        las = _sa_las (LASCI, mf_symm)[0]
        las.fciboxes[0].fcisolvers[1].wfnsym = 'A"'
        las.fciboxes[1].fcisolvers[2].wfnsym = "A'"
        sym_fr = las.get_sym_fr ()
        self.assertEqual (sym_fr.shape, (len (ncas_sub), las.nroots))
        self.assertTrue (np.issubdtype (sym_fr.dtype, np.integer))
        ref = np.zeros ((len (ncas_sub), las.nroots), dtype=int)
        ref[0,1] = symm.irrep_name2id ('Cs', 'A"')
        ref[1,2] = symm.irrep_name2id ('Cs', "A'")
        self.assertEqual (sym_fr.tolist (), ref.tolist ())
        self.assertEqual (ref[0,1], 1)

    def test_lasscf_lasci (self):
        # LASSCF.lasci and LASSCF.lasci_ do the same thing as LASCI.kernel
        for label, my_mf in (('nosymm', mf), ('symm', mf_symm)):
            las_ref, mo = _sa_las (LASCI, my_mf)
            las_ref.kernel (mo.copy ())
            self.assertTrue (las_ref.converged)
            for method in ('lasci', 'lasci_'):
                with self.subTest (label, method=method):
                    las = _sa_las (LASSCF, my_mf)[0]
                    getattr (las, method) (mo.copy ())
                    self.assertTrue (las.converged)
                    self.assertAlmostEqual (las.e_tot, las_ref.e_tot, 9)
                    for e, e_ref in zip (las.e_states, las_ref.e_states):
                        self.assertAlmostEqual (e, e_ref, 9)

if __name__ == "__main__":
    print("Full Tests for LAS utility methods")
    unittest.main()
