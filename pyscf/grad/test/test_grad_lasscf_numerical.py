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

# LASSCF nuclear gradients with two fragments against central finite differences of the energy.
# (test_grad_lasscf.py has only one fragment, where LASSCF is CASSCF.)
# Single-state LASSCF of this H4 converges reliably in STO-3G, not in 6-31G.

import unittest
import numpy as np
from pyscf import gto, scf, lib
from pyscf.mcscf.las.lasscf_sync_o0 import LASSCF, LASSCFNoSymm

atoms = ['H', 'H', 'H', 'H']
coords0 = np.array ([[0.0, 0.0,  0.0],
                     [0.8, 0.0,  0.0],
                     [0.2, 2.6,  0.1],
                     [1.0, 2.8, -0.1]]) # Angstrom
step = 1e-4 # Bohr
# (atom, xyz) components to check
components = [(0,0), (1,1), (2,1), (3,2)]

def run_las (coords, df=False, sa=False):
    mol = gto.M (atom=list (zip (atoms, coords.tolist ())), basis='sto-3g', unit='angstrom',
                 verbose=0, output='/dev/null')
    mf = scf.RHF (mol)
    if df: mf = mf.density_fit ()
    mf.run (conv_tol=1e-12)
    las = LASSCF (mf, (2,2), (2,2), spin_sub=(1,1))
    las.conv_tol_grad = 1e-6
    mo = las.localize_init_guess (((0,1),(2,3)), mf.mo_coeff)
    if sa:
        las.state_average_(weights=[.5,.5], spins=[[0,0],[2,-2]], smults=[[1,1],[3,3]])
    las.kernel (mo)
    mol.stdout.close ()
    return las

def finite_difference (ia, ix, **kwargs):
    h = step * lib.param.BOHR
    cp = coords0.copy ()
    cp[ia,ix] += h
    cm = coords0.copy ()
    cm[ia,ix] -= h
    las_p, las_m = run_las (cp, **kwargs), run_las (cm, **kwargs)
    assert (las_p.converged and las_m.converged)
    return (las_p.e_tot - las_m.e_tot) / (2*step)

class KnownValues (unittest.TestCase):

    def _check (self, **kwargs):
        las = run_las (coords0, **kwargs)
        self.assertTrue (las.converged)
        de = las.nuc_grad_method ().kernel ()
        for ia, ix in components:
            with self.subTest (atom=ia, xyz=ix):
                self.assertAlmostEqual (de[ia,ix], finite_difference (ia, ix, **kwargs), 6)

    def test_single_state (self):
        self._check ()

    def test_state_average (self):
        self._check (sa=True)

    def test_df_raises (self):
        # Density fitting isn't implemented; the gradient used to come out ~1e-5 off finite
        # differences without warning
        las = run_las (coords0, df=True)
        with self.subTest ('DF LAS object'):
            with self.assertRaises (NotImplementedError):
                las.nuc_grad_method ().kernel ()
        with self.subTest ('conventional LAS object on a DF mean field'):
            las1 = LASSCFNoSymm (las._scf, (2,2), (2,2), spin_sub=(1,1))
            las1.mo_coeff, las1.ci = las.mo_coeff, las.ci
            with self.assertRaises (NotImplementedError):
                las1.nuc_grad_method ().kernel ()

if __name__ == "__main__":
    print("Full Tests for LASSCF gradients against finite differences")
    unittest.main()
