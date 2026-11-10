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

# Partially frozen orbitals. With one fragment, LASSCF is CASSCF, so LASSCF with las.frozen must
# match CASSCF with mc.frozen, and (with canonicalization off) leave the frozen columns of
# mo_coeff untouched, as CASSCF does.
#
# TODO: both fail, also in mrh before the migration. Every macrocycle of lasscf_sync_o0.kernel
# rediagonalizes the inactive and virtual blocks (_eig_inactive_virtual) regardless of
# las.frozen, mixing frozen orbitals into unfrozen ones. Frozen 1s cores: energy ~1e-7 below
# CASSCF; frozen virtuals: ~1e-3 below. Remove expectedFailure when fixed.

import unittest
import numpy as np
from pyscf import scf, mcscf
from me2n2_struct import structure as struct
from pyscf.mcscf.las.lasscf_sync_o0 import LASSCF

def setUpModule():
    global mol, mf
    mol = struct (3.0, '6-31g')
    mol.output = '/dev/null'
    mol.verbose = 0
    mol.build ()
    mf = scf.RHF (mol).run ()

def tearDownModule():
    global mol, mf
    mol.stdout.close ()
    del mol, mf

class KnownValues (unittest.TestCase):

    def _check (self, frozen):
        mc = mcscf.CASSCF (mf, 4, 4).set (frozen=frozen, conv_tol=1e-11, conv_tol_grad=1e-7)
        mc.canonicalization = False
        mc.kernel ()
        las = LASSCF (mf, (4,), (4,), spin_sub=(1,)).set (conv_tol_grad=1e-7, frozen=frozen)
        las.canonicalization = False
        las.kernel ()
        self.assertTrue (las.converged)
        self.assertAlmostEqual (las.e_tot, mc.e_tot, 8)
        self.assertAlmostEqual (np.abs (las.mo_coeff[:,frozen] - mf.mo_coeff[:,frozen]).max (),
                                0, 8)

    @unittest.expectedFailure
    def test_frozen_core (self):
        self._check ([0,1,2,3])

    @unittest.expectedFailure
    def test_frozen_virtual (self):
        self._check ([20,21])

if __name__ == "__main__":
    print("Full Tests for LASSCF with partially frozen orbitals")
    unittest.main()
