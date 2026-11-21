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
# because in a minimal basis Lowdin and meta-Lowdin orbitals coincide. Also the inactive orbitals
# of the freeze_cas_spaces=False guess, on C2H4N4 with N 2pz fragments: before they were chosen by
# aufbau on fock, the C=C pi orbital (occupied in RHF) ended up virtual and LASCI was 1 Eh too high.

import os
import unittest
import numpy as np
from pyscf import gto, scf, lo
from pyscf.mcscf.las.lasci import LASCI
from pyscf.mcscf.las.lasscf import LASSCF
from pyscf.mcscf.las.localize import interpret_frags_atoms

frags_atoms = ((0,1),(2,3))
frags_labels = (['0 H', '1 H'], ['2 H', '3 H'])
frags_2pz = (['1 N 2pz', '2 N 2pz'], ['7 N 2pz', '8 N 2pz'])
topdir = os.path.abspath (os.path.join (__file__, '..'))

def setUpModule():
    global mol, mf, las, frags_aos, mo_default, mf_2pz
    mol = gto.M (atom='H 0 0 0; H 0.8 0 0; H 0.2 2.6 0.1; H 1.0 2.8 -0.1', basis='6-31g',
                 verbose=0, output='/dev/null')
    mf = scf.RHF (mol).run ()
    las = LASSCF (mf, (2,2), (2,2), spin_sub=(1,1))
    off = mol.offset_ao_by_atom ()
    frags_aos = [list (range (off[0,2], off[1,3])), list (range (off[2,2], off[3,3]))]
    mo_default = las.localize_init_guess (frags_atoms, mf.mo_coeff)
    with open (os.path.join (topdir, 'c2h4n4.xyz'), 'r') as f:
        mol_2pz = gto.M (atom=f.read (), basis='6-31g', verbose=0, output='/dev/null')
    mf_2pz = scf.RHF (mol_2pz).run ()

def tearDownModule():
    global mol, mf, las, frags_aos, mo_default, mf_2pz
    mol.stdout.close ()
    mf_2pz.mol.stdout.close ()
    del mol, mf, las, frags_aos, mo_default, mf_2pz

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

    def test_unfrozen_core_aufbau (self):
        lasci = LASCI (mf_2pz, (2,2), (2,2))
        mo = lasci.localize_init_guess (frags_2pz, mf_2pz.mo_coeff)
        s = mf_2pz.get_ovlp ()
        occ = mf_2pz.mo_coeff[:,mf_2pz.mo_occ>0]
        virt = mo[:,lasci.ncore+lasci.ncas:]
        with self.subTest ('RHF occupied orbitals not in the virtual space'):
            # 1.01 with the old guess, 0.053 with the aufbau guess
            sv = virt.T @ s @ occ
            self.assertLess ((sv**2).sum (), 0.1)
        with self.subTest ('LASCI energy'):
            e = lasci.kernel (mo)[0]
            self.assertTrue (lasci.converged)
            self.assertAlmostEqual (e, -295.5386252538, 6)

if __name__ == "__main__":
    print("Full Tests for LASSCF localize_init_guess options")
    unittest.main()
