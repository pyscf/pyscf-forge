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
from scipy import linalg
from copy import deepcopy
from itertools import product
from pyscf import lib, gto, scf, dft, fci, mcscf, df
from pyscf.tools import molden
from c2h4n4_struct import structure as struct
from pyscf.mcscf.las.lasscf_sync_o0 import LASSCF

def setUpModule():
    global mol, mol_symm, las, las_symm, las_ref, states, states_symm, weights, mo, mo_symm, lroots, _check_
    xyz = '''6        2.215130000      3.670330000      0.000000000
    1        3.206320000      3.233120000      0.000000000
    1        2.161870000      4.749620000      0.000000000
    6        1.117440000      2.907720000      0.000000000
    1        0.141960000      3.387820000      0.000000000
    1       -0.964240000      1.208850000      0.000000000
    6        1.117440000      1.475850000      0.000000000
    1        2.087280000      0.983190000      0.000000000
    6        0.003700000      0.711910000      0.000000000
    6       -0.003700000     -0.711910000      0.000000000
    6       -1.117440000     -1.475850000      0.000000000
    1        0.964240000     -1.208850000      0.000000000
    1       -2.087280000     -0.983190000      0.000000000
    6       -1.117440000     -2.907720000      0.000000000
    6       -2.215130000     -3.670330000      0.000000000
    1       -0.141960000     -3.387820000      0.000000000
    1       -2.161870000     -4.749620000      0.000000000
    1       -3.206320000     -3.233120000      0.000000000'''
    mol = gto.M (atom = xyz, basis='STO-3G', symmetry=False, verbose=0, output='/dev/null')
    mol_symm = gto.M (atom = xyz, basis='STO-3G', symmetry='Cs', verbose=0, output='/dev/null')
    mf = scf.RHF (mol).run ()
    mf_symm = scf.RHF (mol_symm).run ()
    las = LASSCF (mf, (2,2,2,2),((1,1),(1,1),(1,1),(1,1)))
    las_symm = LASSCF (mf_symm, (2,2,2,2),((1,1),(1,1),(1,1),(1,1)))
    a = list (range (18))
    frags = [a[:5], a[5:9], a[9:13], a[13:18]]
    mo = las.localize_init_guess (frags, mf.mo_coeff)
    mo_symm = las_symm.localize_init_guess (frags, mf_symm.mo_coeff)
    del a, frags
    #las.kernel (mo)
    
    # State list contains a couple of different 4-frag interactions
    states  = {'charges': [[0,0,0,0],[1,1,-1,-1],[2,-1,-1,0],[1,0,0,-1],[1,1,-1,-1],[2,-1,-1,0],[1,0,0,-1]],
               'spins':   [[0,0,0,0],[1,1,-1,-1],[0,1,-1,0], [1,0,0,-1],[-1,-1,1,1],[0,-1,1,0], [-1,0,0,1]],
               'smults':  [[1,1,1,1],[2,2,2,2],  [1,2,2,1],  [2,1,1,2], [2,2,2,2],  [1,2,2,1],  [2,1,1,2]],
               'wfnsyms': [[0,0,0,0],]*7}
    states_symm = copy.deepcopy (states)
    states_symm['wfnsyms'] = [[0,0,0,0],[1,1,1,1],[0,1,1,0],[1,0,0,1],[1,1,1,1],[0,1,1,0],[1,0,0,1]]
    weights = [1.0,] + [0.0,]*6
    las_ref = [None, None]

    # Local excitation list (avoid exciting zero electrons)
    lroots = np.array ([[3,2,1,2,2,1,2],
                        [3,2,2,2,2,2,2],
                        [3,2,2,2,2,2,2],
                        [3,2,2,2,2,2,2]])

def _check_():
    if not las.converged: las.kernel (mo)
    assert (las.converged)
    if not las_symm.converged: las_symm.kernel (mo_symm)
    assert (las_symm.converged)
    if las_ref[0] is None:
        las_ref[0] = las.state_average (weights=weights, **states)
        las_ref[0].frozen = range (mo.shape[1])
        las_ref[0].kernel ()
        assert (las_ref[0].converged)
    if las_ref[1] is None:
        las_ref[1] = las_symm.state_average (weights=weights, **states_symm)
        las_ref[1].frozen = range (mo_symm.shape[1])
        las_ref[1].kernel ()
        assert (las_ref[1].converged)
        
def _lrootsout (las):
    lout = np.zeros_like (lroots)
    for i in range (4):
        for j in range (7):
            nroots = las.ci[i][j].shape[0] if las.ci[i][j].ndim>2 else 1
            lout[i,j] = nroots
    return lout

def tearDownModule():
    global mol, mol_symm, las, las_symm, las_ref, states, states_symm, weights, mo, mo_symm, lroots, _check_, _lrootsout
    mol.stdout.close (), mol_symm.stdout.close ()
    del mol, mol_symm, las, las_symm, las_ref, states, states_symm, weights, mo, mo_symm, lroots, _check_, _lrootsout

class KnownValues(unittest.TestCase):
    def test_sanity (self):
        _check_()
        las_test = las_ref[0].state_average (weights=weights, **states)
        las_test.lasci (lroots=lroots)
        self.assertAlmostEqual (lib.fp (las_test.e_states), lib.fp (las_ref[0].e_states), 5)
        self.assertTrue (las_test.converged)
        self.assertTrue (np.all (_lrootsout(las_test)==lroots))
        e_lexc = np.concatenate ([item for sublist in las_test.e_lexc for item in sublist])
        self.assertTrue (np.all (e_lexc>-1e-8))

    def test_convergence_slow (self):
        _check_()
        las_test = las.state_average (weights=weights, **states)
        las_test.lasci (lroots=lroots)
        self.assertAlmostEqual (lib.fp (las_test.e_states), lib.fp (las_ref[0].e_states), 5)
        self.assertTrue (las_test.converged)
        self.assertTrue (np.all (_lrootsout(las_test)==lroots))
        e_lexc = np.concatenate ([item for sublist in las_test.e_lexc for item in sublist])
        self.assertTrue (np.all (e_lexc>-1e-8))

    def test_sanity_symm (self):
        _check_()
        las_test = las_ref[1].state_average (weights=weights, **states_symm)
        las_test.lasci (lroots=lroots)
        self.assertAlmostEqual (lib.fp (las_test.e_states), lib.fp (las_ref[1].e_states), 5)
        self.assertTrue (las_test.converged)
        self.assertTrue (np.all (_lrootsout(las_test)==lroots))
        e_lexc = np.concatenate ([item for sublist in las_test.e_lexc for item in sublist])
        self.assertTrue (np.all (e_lexc>-1e-8))

    def test_convergence_symm_slow (self):
        _check_()
        las_test = las_symm.state_average (weights=weights, **states_symm)
        las_test.lasci (lroots=lroots)
        self.assertAlmostEqual (lib.fp (las_test.e_states), lib.fp (las_ref[1].e_states), 5)
        self.assertTrue (las_test.converged)
        self.assertTrue (np.all (_lrootsout(las_test)==lroots))
        e_lexc = np.concatenate ([item for sublist in las_test.e_lexc for item in sublist])
        self.assertTrue (np.all (e_lexc>-1e-8))


if __name__ == "__main__":
    print("Full Tests for LASCI calculation")
    unittest.main()

