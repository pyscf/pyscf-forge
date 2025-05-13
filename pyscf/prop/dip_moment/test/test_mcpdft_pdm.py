#!/usr/bin/env python
# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
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
#
import os
import numpy as np
from scipy import linalg
from pyscf import gto, scf, df
from pyscf import mcpdft
#from pyscf.fci import csf_solver
#from pyscf.df.grad import dfmcpdft as mcpdft_grad
import unittest
topdir = os.path.abspath (os.path.join (__file__, '..'))

h2co_casscf66_631g_xyz = \
'''
C  0.534004  0.000000  0.000000
O -0.676110  0.000000  0.000000
H  1.102430  0.000000  0.920125
H  1.102430  0.000000 -0.920125
'''
mol_nosymm = gto.M (atom = h2co_casscf66_631g_xyz, basis = '6-31g', symmetry = False, output='/dev/null', verbose = 0)
mol_symm = gto.M (atom = h2co_casscf66_631g_xyz, basis = '6-31g', symmetry = True, output='/dev/null', verbose = 0)
def get_mc_ref (mol, ri=False, sa2=False):
    mf = scf.RHF (mol)
#    if ri: mf = mf.density_fit (auxbasis = df.aug_etb (mol))
    mc = mcpdft.CASSCF (mf.run (), 'tPBE', 6, 6, grids_level=9)
#    if sa2:
#        fcisolvers = [csf_solver (mol, smult=((2*i)+1)) for i in (0,1)]
#        if mol.symmetry:
#            fcisolvers[0].wfnsym = 'A1'
#            fcisolvers[1].wfnsym = 'A2'
#        mc = mc.state_average_mix_(fcisolvers, [0.5,0.5])
#        ref = np.load (os.path.join (topdir, 'h2co_sa2_tpbe66_631g_grad_num.npy'))
#        ref = ref.reshape (2,2,4,3)[int(ri)]
#    else:
#        ref = np.load (os.path.join (topdir, 'h2co_tpbe66_631g_grad_num.npy'))[int(ri)]
 #this reference is obtained by mcpdft numerical differentiation in pyscf 
 #using central differences with a second order accuracy and step size of 1e-4 
    ref = np.load (os.path.join (topdir, 'h2co_tpbe66_631g_edipole_num.npy'))[int(ri)]
    return mc.run (), ref

def tearDownModule():
    global mol_nosymm, mol_symm
    mol_nosymm.stdout.close ()
    mol_symm.stdout.close ()
    del mol_nosymm, mol_symm

class KnownValues(unittest.TestCase):

    def test_ss (self):
        for mol in (mol_nosymm, mol_symm):
            mc_conv, ref_conv = get_mc_ref (mol, ri=False, sa2=False)
            mc_conv_dip = mc_conv.dip_moment (unit='Debye')
#            mc_df, ref_df = get_mc_ref (mol, ri=True, sa2=False)
#            mc_df_grad = mcpdft_grad.Gradients (mc_df)
            lbl, mc_dip, ref = 'conv', mc_conv_dip, ref_conv
            with self.subTest (symm=mol.symmetry, eri=lbl):
                test = mc_dip
                self.assertLessEqual (linalg.norm (test-ref), 1e-3)
#            for lbl, mc_grad, ref in (('conv', mc_conv_grad, ref_conv), ('DF', mc_df_grad, ref_df)):
#                with self.subTest (symm=mol.symmetry, eri=lbl):
#                    test = mc_dip.kernel ()
#                    self.assertLessEqual (linalg.norm (test-ref), 1e-3)

#    def test_sa (self):
#        for mol in (mol_nosymm, mol_symm):
#            mc_conv, ref_conv = get_mc_ref (mol, ri=False, sa2=True)
#            mc_conv_grad = mc_conv.nuc_grad_method ()
#            mc_df, ref_df = get_mc_ref (mol, ri=True, sa2=True)
#            mc_df_grad = mcpdft_grad.Gradients (mc_df)
#            for lbl, mc_grad, ref in (('conv', mc_conv_grad, ref_conv), ('DF', mc_df_grad, ref_df)):
#                with self.subTest (symm=mol.symmetry, eri=lbl):
#                    test = mc_grad.kernel (state=0)
#                    self.assertLessEqual (linalg.norm (test-ref[0]), 1e-4)
#                    test = mc_grad.kernel (state=1)
#                    self.assertLessEqual (linalg.norm (test-ref[1]), 1e-4)


if __name__ == "__main__":
    print("Full Tests for MC-PDFT electric dipole moment of H2CO molecule")
    unittest.main()






