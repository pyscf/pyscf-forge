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
from pyscf import lib, gto, scf, dft, fci, mcscf, df
from pyscf.mcscf import newton_casscf
from me2n2_struct import structure as struct
from pyscf.mcscf.las.lasscf_sync_o0 import LASSCF

def setUpModule():
    global mol, mf, mf_df, mc, mc_df, mol_hs, mf_hs, mf_hs_df
    r_nn = 3.0
    mol = struct (3.0, '6-31g')
    mol.output = '/dev/null'
    mol.verbose = 0
    mol.build ()
    mf = scf.RHF (mol).run ()
    mc = mcscf.CASSCF (mf, 4, 4).run ()
    mf_df = mf.density_fit (auxbasis = df.aug_etb (mol)).run ()
    mc_df = mcscf.CASSCF (mf_df, 4, 4).run ()
    
    mol_hs = mol.copy ()
    mol_hs.spin = 4
    mol_hs.build ()
    mf_hs = scf.RHF (mol_hs).run ()
    mf_hs_df = mf_hs.density_fit (auxbasis = df.aug_etb (mol_hs)).run ()

def tearDownModule():
    global mol, mf, mf_df, mc, mc_df, mol_hs, mf_hs, mf_hs_df
    mol.stdout.close ()
    mol_hs.stdout.close ()
    del mol, mf, mf_df, mc, mc_df, mol_hs, mf_hs, mf_hs_df


class KnownValues(unittest.TestCase):
    def test_energy_slow (self):
        las = LASSCF (mf, (4,), (4,), spin_sub=(1,)).set (conv_tol_grad=1e-5).run ()
        self.assertTrue (las.converged)
        self.assertAlmostEqual (las.e_tot, mc.e_tot, 6)
        with self.subTest ('chkfile'):
            las2 = LASSCF (mf, (4,), (4,), spin_sub=(1,)).set (max_cycle_macro=1)
            with tempfile.NamedTemporaryFile() as chkfile:
                las.dump_chk (chkfile=chkfile.name)
                las2.load_chk_(chkfile=chkfile.name)
            las2.kernel ()
            self.assertTrue (las2.converged)
            self.assertAlmostEqual (las.e_tot, las2.e_tot, 8)

    def test_energy_df_slow (self):
        las = LASSCF (mf_df, (4,), (4,), spin_sub=(1,)).set (conv_tol_grad=1e-5).run ()
        self.assertTrue (las.converged)
        self.assertAlmostEqual (las.e_tot, mc_df.e_tot, 6)

    def test_energy_hs (self):
        las = LASSCF (mf_hs, (4,), ((4,0),), spin_sub=(5,)).set (conv_tol_grad=1e-5).run ()
        self.assertTrue (las.converged)
        self.assertAlmostEqual (las.e_tot, mf_hs.e_tot, 8)

    def test_energy_hs_df (self):
        las = LASSCF (mf_hs_df, (4,), ((4,0),), spin_sub=(5,)).set (conv_tol_grad=1e-5).run ()
        self.assertTrue (las.converged)
        self.assertAlmostEqual (las.e_tot, mf_hs_df.e_tot, 8)

    def test_grad_hess_vs_newton_casscf (self):
        np.random.seed(1)
        las = LASSCF (mf, (4,), (4,), spin_sub=(1,)).set (max_cycle_macro=1, ah_level_shift=0).run ()
        ugg = las.get_ugg ()
        ci0_csf = np.random.rand (ugg.ncsf_sub[0][0])
        ci0_csf /= np.linalg.norm (ci0_csf)
        ci0 = ugg.ci_transformers[0][0].vec_csf2det (ci0_csf)
        las_gorb, las_gci = las.get_grad (mo_coeff=mf.mo_coeff, ci=[[ci0]])[:2]
        las_grad = np.append (las_gorb, las_gci)
        las_hess = las.get_hop (ugg=ugg, mo_coeff=mf.mo_coeff, ci=[[ci0]])
        self.assertAlmostEqual (lib.fp (las_grad), lib.fp (las_hess.get_grad ()), 8)
        cas_grad, _, cas_hess, _ = newton_casscf.gen_g_hop (mc, mf.mo_coeff, ci0, mc.ao2mo (mf.mo_coeff))
        _pack_ci, _unpack_ci = newton_casscf._pack_ci_get_H (mc, mf.mo_coeff, ci0)[-2:]
        def pack_cas (kappa, ci1):
            return np.append (mc.pack_uniq_var (kappa), _pack_ci (ci1))
        def unpack_cas (x):
            return mc.unpack_uniq_var (x[:ugg.nvar_orb]), _unpack_ci (x[ugg.nvar_orb:])
        def cas2las (y):
            yorb, yci = unpack_cas (y)
            yc = yci[0].ravel ().dot (ci0.ravel ())
            yci[0] -= yc * ci0
            yorb *= 0.5
            return ugg.pack (yorb, [yci])
        def las2cas (y):
            yorb, yci = ugg.unpack (y)
            yc = yci[0][0].ravel ().dot (ci0.ravel ())
            yci[0][0] -= yc * ci0
            yorb *= 0.5
            return pack_cas (yorb, yci[0])
        cas_grad = cas2las (cas_grad)
        self.assertAlmostEqual (lib.fp (las_grad), lib.fp (cas_grad), 8)
        x = np.random.rand (ugg.nvar_tot)
        # orb on input
        x_las = x.copy ()
        x_las[ugg.nvar_orb:] = 0.0
        x_cas = las2cas (x_las)
        hx_las = las_hess._matvec (x_las)
        hx_cas = cas2las (cas_hess (x_cas))
        self.assertAlmostEqual (lib.fp (hx_las), lib.fp (hx_cas), 8)
        # CI on input
        x_las = x.copy ()
        x_las[:ugg.nvar_orb] = 0.0
        x_cas = las2cas (x_las)
        hx_las = las_hess._matvec (x_las)
        hx_cas = cas2las (cas_hess (x_cas))
        self.assertAlmostEqual (lib.fp (hx_las), lib.fp (hx_cas), 8)
        # I have to do these separately because there is no straightforward way
        # for H_co, H_oc, and H_cc to all be simultaneously correct given the
        # convention difference 

    def test_grad_hess_vs_newton_casscf_df (self):
        np.random.seed(1)
        las = LASSCF (mf_df, (4,), (4,), spin_sub=(1,)).set (max_cycle_macro=1, ah_level_shift=0).run ()
        ugg = las.get_ugg ()
        ci0_csf = np.random.rand (ugg.ncsf_sub[0][0])
        ci0_csf /= np.linalg.norm (ci0_csf)
        ci0 = ugg.ci_transformers[0][0].vec_csf2det (ci0_csf)
        las_gorb, las_gci = las.get_grad (mo_coeff=mf_df.mo_coeff, ci=[[ci0]])[:2]
        las_grad = np.append (las_gorb, las_gci)
        las_hess = las.get_hop (ugg=ugg, mo_coeff=mf_df.mo_coeff, ci=[[ci0]])
        self.assertAlmostEqual (lib.fp (las_grad), lib.fp (las_hess.get_grad ()), 8)
        cas_grad, _, cas_hess, _ = newton_casscf.gen_g_hop (mc_df, mf_df.mo_coeff, ci0, mc_df.ao2mo (mf_df.mo_coeff))
        _pack_ci, _unpack_ci = newton_casscf._pack_ci_get_H (mc_df, mf_df.mo_coeff, ci0)[-2:]
        def pack_cas (kappa, ci1):
            return np.append (mc_df.pack_uniq_var (kappa), _pack_ci (ci1))
        def unpack_cas (x):
            return mc_df.unpack_uniq_var (x[:ugg.nvar_orb]), _unpack_ci (x[ugg.nvar_orb:])
        def cas2las (y):
            yorb, yci = unpack_cas (y)
            yc = yci[0].ravel ().dot (ci0.ravel ())
            yci[0] -= yc * ci0
            yorb *= 0.5
            return ugg.pack (yorb, [yci])
        def las2cas (y):
            yorb, yci = ugg.unpack (y)
            yc = yci[0][0].ravel ().dot (ci0.ravel ())
            yci[0][0] -= yc * ci0
            yorb *= 0.5
            return pack_cas (yorb, yci[0])
        cas_grad = cas2las (cas_grad)
        self.assertAlmostEqual (lib.fp (las_grad), lib.fp (cas_grad), 8)
        x = np.random.rand (ugg.nvar_tot)
        # orb on input
        x_las = x.copy ()
        x_las[ugg.nvar_orb:] = 0.0
        x_cas = las2cas (x_las)
        hx_las = las_hess._matvec (x_las)
        hx_cas = cas2las (cas_hess (x_cas))
        self.assertAlmostEqual (lib.fp (hx_las[:ugg.nvar_orb]), lib.fp (hx_cas[:ugg.nvar_orb]), 8)
        self.assertAlmostEqual (lib.fp (hx_las[ugg.nvar_orb:]), lib.fp (hx_cas[ugg.nvar_orb:]), 8)
        # CI on input
        x_las = x.copy ()
        x_las[:ugg.nvar_orb] = 0.0
        x_cas = las2cas (x_las)
        hx_las = las_hess._matvec (x_las)
        hx_cas = cas2las (cas_hess (x_cas))
        self.assertAlmostEqual (lib.fp (hx_las[:ugg.nvar_orb]), lib.fp (hx_cas[:ugg.nvar_orb]), 8)
        self.assertAlmostEqual (lib.fp (hx_las[ugg.nvar_orb:]), lib.fp (hx_cas[ugg.nvar_orb:]), 8)
        # I have to do these separately because there is no straightforward way
        # for H_co, H_oc, and H_cc to all be simultaneously correct given the
        # convention difference 

if __name__ == "__main__":
    print("Full Tests for LASSCF me2n2")
    unittest.main()

