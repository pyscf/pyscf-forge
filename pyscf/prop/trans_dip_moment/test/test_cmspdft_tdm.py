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
import numpy as np
from pyscf import gto, scf, mcscf
from pyscf import mcpdft
import unittest
#from pyscf.fci import csf_solver
from pyscf.fci.addons import fix_spin_

geom_furan= '''
C        0.000000000     -0.965551055     -2.020010585
C        0.000000000     -1.993824223     -1.018526668
C        0.000000000     -1.352073201      0.181141565
O        0.000000000      0.000000000      0.000000000
C        0.000000000      0.216762264     -1.346821565
H        0.000000000     -1.094564216     -3.092622941
H        0.000000000     -3.062658055     -1.175803180
H        0.000000000     -1.688293885      1.206105691
H        0.000000000      1.250242874     -1.655874372
'''
# Three doublets of A1 symmetry
def get_furan(iroots=3):
    weights = [1/iroots]*iroots
    mol = gto.M(atom = geom_furan, basis = 'sto-3g',
             symmetry='C2v', output='/dev/null', verbose=0)
    mf = scf.RHF(mol).run()
    mc = mcpdft.CASSCF(mf,'tPBE', 5, 6, grids_level=1)
    #mc.fcisolver = csf_solver(mol, smult=1, symm='A1')
    fix_spin_(mc.fcisolver, ss=0, shift=.2)
    mc.fcisolver.wfnsym = 'A1'
    mc = mc.multi_state(weights, 'cms')
    mc.max_cycle_macro = 200
    mo = mcscf.sort_mo(mc, mf.mo_coeff, [12,17,18,19,20])
    mc.conv_tol = 1e-11
    mc.kernel(mo)
    return mc

# Two doublets of A2 symmetry
def get_furan_cation(iroots=2):
    weights = [1/iroots]*iroots
    mol = gto.M(atom = geom_furan, basis = 'sto-3g', charge=1, spin=1,
             symmetry='C2v', output='/dev/null', verbose=0)
    mf = scf.RHF(mol).run()
    mc = mcpdft.CASSCF(mf,'tBLYP', 5, 5, grids_level=1)
    #mc.fcisolver = csf_solver(mol, smult=2, symm='A2')
    fix_spin_(mc.fcisolver, ss=0.75)
    mc.fcisolver.wfnsym = 'A2'
    mc = mc.multi_state(weights, 'cms')
    mc.max_cycle_macro = 200
    mo = mcscf.sort_mo(mc, mf.mo_coeff, [12,17,18,19,20])
    mc.conv_tol = 1e-11
    mc.kernel(mo)
    return mc

class KnownValues(unittest.TestCase):
    '''
    The reference values were obtained by numerical differentiation of the model-space Hamiltonian
    in the intermediate-state basis w.r.t the electric field extrapolated to the zero step size. 
    The fields were applied along each direction in the XYZ frame, and derivatives 
    were evaluated using 2-point central difference formula.   
    '''
    def test_furan_cms3_tpbe_sto3g(self):
        tdm_ref = np.array(\
        [[0.0000, -0.2049, -0.2104],
        [ 0.0000, -0.6832, -0.7015],
        [ 0.0000, -0.7035, -0.7223]])
        delta = 0.001
        message = "Transition dipoles are not equal within {} D".format(delta)

        iroots=3
        k=0
        mc = get_furan(iroots)
        for i in range(iroots):
            for j in range(i):
                with self.subTest (k=k):
                    tdm_test = mc.trans_moment(\
                        unit='Debye', origin="mass_center",state=[i,j])
                    for tdmt,tdmr in zip(tdm_test,tdm_ref[k]):
                        try:
                            self.assertAlmostEqual(tdmt, tdmr, None, message, delta)
                        except:
                            self.assertAlmostEqual(-tdmt, tdmr, None, message, delta )
                k += 1

    def test_furan_cation_cms2_tblyp_sto3g(self):
        tdm_ref = np.array(\
        [[0.0000,  -0.0742, -0.0762]])
        delta = 0.001
        message = "Transition dipoles are not equal within {} D".format(delta)

        iroots=2
        k=0
        mc = get_furan_cation(iroots)
        for i in range(iroots):
            for j in range(i):
                with self.subTest (k=k):
                    tdm_test = mc.trans_moment(\
                        unit='AU', origin="Charge_center",state=[i,j])
                    for tdmt,tdmr in zip(tdm_test,tdm_ref[k]):
                        try:
                            self.assertAlmostEqual(tdmt, tdmr, None, message, delta)
                        except:
                            self.assertAlmostEqual(-tdmt, tdmr, None, message, delta )
                k += 1  

if __name__ == "__main__":
    print("Test for CMS-PDFT transition dipole moments")
    unittest.main()
