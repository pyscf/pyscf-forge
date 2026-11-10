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
import tempfile
import numpy as np
from pyscf import lib, gto, scf
from pyscf.mcscf.las.lasscf_sync_o0 import LASSCF as LASSCFRef
from pyscf.mcscf.las.lasscf_rdm import LASSCF as LASSCFTest

def setUpModule():
    global mol, mf, las_test, las_ref, mo_loc
    xyz = '''H 0.0 0.0 0.0
             H 1.0 0.0 0.0
             H 0.2 3.9 0.1
             H 1.159166 4.1 -0.1'''
    mol = gto.M (atom = xyz, basis = '6-31g', output='/dev/null',
        verbose=lib.logger.INFO)
    mf = scf.RHF (mol).run ()
    las_ref = LASSCFRef (mf, (2,2), (2,2), spin_sub=(1,1))
    las_ref.chkfile = tempfile.NamedTemporaryFile ().name
    las_test = LASSCFTest (mf, (2,2), (2,2), spin_sub=(1,1))
    las_test.chkfile = tempfile.NamedTemporaryFile ().name
    mo_loc = las_ref.localize_init_guess (((0,1),(2,3)), mf.mo_coeff)

def tearDownModule():
    global mol, mf, las_test, las_ref, mo_loc
    mol.stdout.close ()
    del mol, mf, las_test, las_ref, mo_loc

class KnownValues(unittest.TestCase):
    def test_etot (self):
        las_ref.ah_level_shift = 1e-8
        las_ref.max_cycle_macro = 50
        las_ref.kernel ()
        self.assertTrue (las_ref.converged)
        las_test.kernel ()
        self.assertTrue (las_test.converged)
        self.assertAlmostEqual (las_test.e_tot, las_ref.e_tot, 6)

    def test_derivs (self):
        las_ref.ah_level_shift = 1e-8
        las_ref.max_cycle_macro = 10
        las_ref.kernel (mo_loc, None)
        self.assertTrue (las_ref.converged)
        ugg_ref = las_ref.get_ugg ()
        hop_ref = las_ref.get_hop (ugg=ugg_ref)
        las_test.casdm1frs = las_ref.states_make_casdm1s_sub ()
        las_test.casdm2fr = las_ref.states_make_casdm2_sub ()
        las_test.mo_coeff = las_ref.mo_coeff
        ugg_test = las_test.get_ugg ()
        hop_test = las_test.get_hop (ugg=ugg_test)
        with self.subTest (fn = 'grad'):
            g_test = hop_test.get_grad ()
            g_ref = hop_ref.get_grad ()[:g_test.size]
            self.assertAlmostEqual (lib.fp (g_test), lib.fp (g_ref), 6)
        with self.subTest (fn = 'prec'):
            np.random.seed (0)
            x_test = np.random.rand (ugg_test.nvar_tot)
            x_ref = np.zeros (ugg_ref.nvar_tot)
            x_ref[:ugg_ref.nvar_orb] = x_test[:]
            prec_test = hop_test.get_prec ()(x_test)
            prec_ref = hop_ref.get_prec ()(x_ref)[:prec_test.size]
            self.assertAlmostEqual (lib.fp (prec_test), lib.fp (prec_ref), 6)
        with self.subTest (fn = 'hx'):
            hx_test = hop_test._matvec (x_test)
            hx_ref = hop_ref._matvec (x_ref)[:hx_test.size]
            self.assertAlmostEqual (lib.fp (hx_test), lib.fp (hx_ref), 6)


if __name__ == "__main__":
    print("Full Tests for LASSCF RDM module functions")
    unittest.main()

