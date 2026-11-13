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
from pyscf.mcscf import newton_casscf
from pyscf.mcscf.addons import state_average_mix
from me2n2_struct import structure as struct
from pyscf.mcscf.las.lasscf import LASSCF
from pyscf.csf_fci import csf_solver

def setUpModule():
    global mol, mf, mf_df, mc, mc_df
    r_nn = 3.0
    mol = struct (3.0, '6-31g')
    mol.output = '/dev/null'
    mol.verbose = lib.logger.DEBUG
    mol.build ()
    mf = scf.RHF (mol).run ()
    mc = state_average_mix (mcscf.CASSCF (mf, 4, 4),
        [csf_solver (mol, smult=m2+1).set (spin=m2) for m2 in (0,2)],
        [0.5, 0.5]).run ()
    mf_df = mf.density_fit (auxbasis = df.aug_etb (mol)).run ()
    mc_df = state_average_mix (mcscf.CASSCF (mf_df, 4, 4),
        [csf_solver (mol, smult=m2+1).set (spin=m2) for m2 in (0,2)],
        [0.5, 0.5]).run ()

def tearDownModule():
    global mol, mf, mf_df, mc, mc_df 
    mol.stdout.close ()
    del mol, mf, mf_df, mc, mc_df


class KnownValues(unittest.TestCase):
    def test_energy_slow (self):
        las = LASSCF (mf, (4,), (4,), spin_sub=(1,)).set (conv_tol_grad=1e-5)
        las.state_average_(weights=[0.5,0.5], spins=[0,2]).run ()
        self.assertTrue (las.converged)
        self.assertAlmostEqual (las.e_tot, mc.e_tot, 8)
        self.assertAlmostEqual (las.e_states[0], mc.e_states[0], 7)
        self.assertAlmostEqual (las.e_states[1], mc.e_states[1], 7)

    def test_energy_df_slow (self):
        las = LASSCF (mf_df, (4,), (4,), spin_sub=(1,)).set (conv_tol_grad=1e-5)
        las.state_average_(weights=[0.5,0.5], charges=[0,0], spins=[0,2], smults=[1,3]).run ()
        self.assertTrue (las.converged)
        self.assertAlmostEqual (las.e_tot, mc_df.e_tot, 8)
        self.assertAlmostEqual (las.e_states[0], mc_df.e_states[0], 7)
        self.assertAlmostEqual (las.e_states[1], mc_df.e_states[1], 7)

    def test_grad_hess_vs_newton_casscf (self):
        np.random.seed(1)
        las = LASSCF (mf, (4,), (4,), spin_sub=(1,)).set (max_cycle_macro=1, ah_level_shift=0)
        las.state_average_(weights=[0.5,0.5], charges=[0,0], spins=[0,2], smults=[1,3]).run ()
        ugg = las.get_ugg ()
        ci0_csf = [np.random.rand (ncsf) for ncsf in ugg.ncsf_sub[0]]
        ci0_csf = [c / np.linalg.norm (c) for c in ci0_csf]
        ci0 = [t.vec_csf2det (c) for t, c in zip (ugg.ci_transformers[0], ci0_csf)]
        las_gorb, las_gci = las.get_grad (mo_coeff=mf.mo_coeff, ci=[ci0])[:2]
        las_grad = np.append (las_gorb, las_gci)
        las_hess = las.get_hop (ugg=ugg, mo_coeff=mf.mo_coeff, ci=[ci0])
        self.assertAlmostEqual (lib.fp (las_grad), lib.fp (las_hess.get_grad ()), 8)
        cas_grad, _, cas_hess, _ = newton_casscf.gen_g_hop (mc, mf.mo_coeff, ci0, mc.ao2mo (mf.mo_coeff))
        _pack_ci, _unpack_ci = newton_casscf._pack_ci_get_H (mc, mf.mo_coeff, ci0)[-2:]
        def pack_cas (kappa, ci1):
            return np.append (mc.pack_uniq_var (kappa), _pack_ci (ci1))
        def unpack_cas (x):
            return mc.unpack_uniq_var (x[:ugg.nvar_orb]), _unpack_ci (x[ugg.nvar_orb:])
        def cas2las (y):
            yorb, yci = unpack_cas (y)
            yci = [2 * (yc - c * c.ravel ().dot (yc.ravel ())) for c, yc in zip (ci0, yci)]
            yorb *= 0.5
            return ugg.pack (yorb, [yci])
        def las2cas (y):
            yorb, yci = ugg.unpack (y)
            yci = [yc - c * c.ravel ().dot (yc.ravel ()) for c, yc in zip (ci0, yci[0])]
            yorb *= 0.5
            return pack_cas (yorb, yci)
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
        las = LASSCF (mf_df, (4,), (4,), spin_sub=(1,)).set (max_cycle_macro=1, ah_level_shift=0)
        las.state_average_(weights=[0.5,0.5], charges=[0,0], spins=[0,2], smults=[1,3]).run ()
        ugg = las.get_ugg ()
        ci0_csf = [np.random.rand (ncsf) for ncsf in ugg.ncsf_sub[0]]
        ci0_csf = [c / np.linalg.norm (c) for c in ci0_csf]
        ci0 = [t.vec_csf2det (c) for t, c in zip (ugg.ci_transformers[0], ci0_csf)]
        las_gorb, las_gci = las.get_grad (mo_coeff=mf_df.mo_coeff, ci=[ci0])[:2]
        las_grad = np.append (las_gorb, las_gci)
        las_hess = las.get_hop (ugg=ugg, mo_coeff=mf_df.mo_coeff, ci=[ci0])
        self.assertAlmostEqual (lib.fp (las_grad), lib.fp (las_hess.get_grad ()), 8)
        cas_grad, _, cas_hess, _ = newton_casscf.gen_g_hop (mc_df, mf_df.mo_coeff, ci0, mc_df.ao2mo (mf_df.mo_coeff))
        _pack_ci, _unpack_ci = newton_casscf._pack_ci_get_H (mc_df, mf_df.mo_coeff, ci0)[-2:]
        def pack_cas (kappa, ci1):
            return np.append (mc_df.pack_uniq_var (kappa), _pack_ci (ci1))
        def unpack_cas (x):
            return mc_df.unpack_uniq_var (x[:ugg.nvar_orb]), _unpack_ci (x[ugg.nvar_orb:])
        def cas2las (y):
            yorb, yci = unpack_cas (y)
            yci = [2 * (yc - c * c.ravel ().dot (yc.ravel ())) for c, yc in zip (ci0, yci)]
            yorb *= 0.5
            return ugg.pack (yorb, [yci])
        def las2cas (y):
            yorb, yci = ugg.unpack (y)
            yci = [yc - c * c.ravel ().dot (yc.ravel ()) for c, yc in zip (ci0, yci[0])]
            yorb *= 0.5
            return pack_cas (yorb, yci)
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

if __name__ == "__main__":
    print("Full Tests for SA-LASSCF me2n2")
    unittest.main()

