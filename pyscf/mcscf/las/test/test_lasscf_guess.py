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

# Options of localize_init_guess not covered elsewhere: the three ways of specifying fragments
# (atom indices, AO indices with frags_by_AOs=True, AO-label strings), and lo_coeff. 6-31G,
# because in a minimal basis Lowdin and meta-Lowdin orbitals coincide.

import unittest
import numpy as np
from pyscf import gto, scf, lo
from pyscf.mcscf.las.lasscf_sync_o0 import LASSCF
from pyscf.mcscf.las.lasscf_guess import interpret_frags_atoms

frags_atoms = ((0,1),(2,3))
frags_labels = (['0 H', '1 H'], ['2 H', '3 H'])

def setUpModule():
    global mol, mf, las, frags_aos, mo_default
    mol = gto.M (atom='H 0 0 0; H 0.8 0 0; H 0.2 2.6 0.1; H 1.0 2.8 -0.1', basis='6-31g',
                 verbose=0, output='/dev/null')
    mf = scf.RHF (mol).run ()
    las = LASSCF (mf, (2,2), (2,2), spin_sub=(1,1))
    off = mol.offset_ao_by_atom ()
    frags_aos = [list (range (off[0,2], off[1,3])), list (range (off[2,2], off[3,3]))]
    mo_default = las.localize_init_guess (frags_atoms, mf.mo_coeff)

def tearDownModule():
    global mol, mf, las, frags_aos, mo_default
    mol.stdout.close ()
    del mol, mf, las, frags_aos, mo_default

def active_projector (mo):
    c = mo[:,las.ncore:las.ncore+las.ncas]
    return c @ c.conj ().T

class KnownValues (unittest.TestCase):

    def test_interpret_frags_atoms (self):
        self.assertEqual (interpret_frags_atoms (mol, frags_atoms), frags_aos)
        self.assertEqual (interpret_frags_atoms (mol, frags_aos, frags_by_AOs=True), frags_aos)
        self.assertEqual ([list (map (int, f)) for f in interpret_frags_atoms (mol, frags_labels)],
                          frags_aos)
        with self.assertRaises (RuntimeError):
            interpret_frags_atoms (mol, ((0, '1 H'), (2,3)))

    def test_frags_by_AOs (self):
        mo = las.localize_init_guess (frags_aos, mf.mo_coeff, frags_by_AOs=True)
        self.assertAlmostEqual (np.abs (mo - mo_default).max (), 0, 10)

    def test_frags_by_labels (self):
        mo = las.localize_init_guess (frags_labels, mf.mo_coeff)
        self.assertAlmostEqual (np.abs (mo - mo_default).max (), 0, 10)

    def test_lo_coeff_default (self):
        mo = las.localize_init_guess (frags_atoms, mf.mo_coeff,
                                      lo_coeff=lo.orth_ao (mol, 'meta_lowdin'))
        self.assertAlmostEqual (np.abs (mo - mo_default).max (), 0, 10)

    def test_lo_coeff_lowdin (self):
        mo = las.localize_init_guess (frags_atoms, mf.mo_coeff, lo_coeff=lo.orth_ao (mol, 'lowdin'))
        s = mf.get_ovlp ()
        with self.subTest ('orthonormal'):
            self.assertAlmostEqual (np.abs (mo.T @ s @ mo - np.eye (mo.shape[1])).max (), 0, 10)
        with self.subTest ('lo_coeff changes the active space'):
            dP = active_projector (mo) - active_projector (mo_default)
            self.assertGreater (np.abs (dP).max (), 1e-3)
        with self.subTest ('active orbitals localized on their fragments'):
            w, v = np.linalg.eigh (s)
            c = (v * np.sqrt (w)) @ v.T @ mo[:,las.ncore:las.ncore+las.ncas] # Lowdin basis
            for ifrag, aos in enumerate (frags_aos):
                pop = (c[aos,2*ifrag:2*ifrag+2]**2).sum (0)
                self.assertTrue (np.all (pop > 0.99), pop)

if __name__ == "__main__":
    print("Full Tests for LASSCF localize_init_guess options")
    unittest.main()
