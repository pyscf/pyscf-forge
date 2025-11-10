#!/usr/bin/env python
# Copyright 2014-2025 The PySCF Developers. All Rights Reserved.
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
from pyscf.csf_fci import csf_solver
from pyscf.mcpdft.lpdft import _LPDFT
from pyscf.prop.dip_moment.lpdft import _LPDFTDipole

geom_h2o='''
O  0.00000000   0.08111156   0.00000000
H  0.78620605   0.66349738   0.00000000
H -0.78620605   0.66349738   0.00000000
'''

mol_h2o = gto.M(atom = geom_h2o, basis = 'aug-cc-pVDZ', symmetry='c2v', output='/dev/null', verbose=0)

# Three singlets all of A1 symmetry
def get_h2o_ftpbe(mol,iroots=3): 
    weights = [1/iroots]*iroots
    mf = scf.RHF(mol).run()
    mc = mcpdft.CASSCF(mf,'ftPBE', 4, 4, grids_level=9)
    mc.fcisolver = csf_solver(mol, smult=1, symm='A1')
    mc = mc.multi_state(weights, "lin")
    mo = mcscf.sort_mo(mc, mf.mo_coeff, [4,5,8,9])
    mc.conv_tol = 1e-12
    mc.conv_tol_grad = 1e-6
    mc.kernel(mo)
    return mc

def get_h2o_ftlda(mol,iroots=3):
    weights = [1/iroots]*iroots
    mf = scf.RHF(mol).run()
    mc = mcpdft.CASSCF(mf,'ftLDA', 4, 4, grids_level=9)
    mc.fcisolver = csf_solver(mol, smult=1, symm='A1')
    mc = mc.multi_state(weights, "lin")
    mo = mcscf.sort_mo(mc, mf.mo_coeff, [4,5,8,9])
    mc.conv_tol = 1e-12
    mc.conv_tol_grad = 1e-6
    mc.kernel(mo)
    return mc


class KnownValues(unittest.TestCase):
    '''
    The reference values were obtained by numeric differentiation of the energy 
    with respect to the electric field strength extrapolated to zero step size. 
    The fields were applied along each direction in the XYZ frame, and derivatives 
    were evaluated using 2-point central difference formula.   
    '''
    def test_h2o_lpdft_ftpbe_augccpvdz(self): 
        dm_ref = np.array(\
            [[0.0000, 1.9902, 0.0000],  # State 0: x, y, z
            [ 0.0000,-1.4528, 0.0000],  # State 1: x, y, z
            [ 0.0000, 3.3628, 0.0000]]) # State 2: x, y, z
        delta = 0.001
        message = "Dipoles are not equal within {} D".format(delta)
        iroots=3
        mc = get_h2o_ftpbe(mol_h2o, iroots)
        for i in range(3):
            with self.subTest (i=i):
                dm_test = mc.dip_moment(unit='Debye', origin="Coord_center",state=i)
                for dmt,dmr in zip(dm_test,dm_ref[i]):
                    self.assertAlmostEqual(dmt, dmr, None, message, delta)

    def test_h2o_lpdft_ftlda_augccpvdz(self): 
        dm_ref = np.array(\
            [[0.0000, 1.8875, 0.0000],  # State 0: x, y, z
            [ 0.0000,-1.4480, 0.0000],  # State 1: x, y, z
            [ 0.0000, 3.3715, 0.0000]]) # State 2: x, y, z
        delta = 0.001
        message = "Dipoles are not equal within {} D".format(delta)
        iroots=3
        mc = get_h2o_ftlda(mol_h2o, iroots)
        for i in range(3):
            with self.subTest (i=i):
                dm_test = mc.dip_moment(unit='Debye', origin="Coord_center",state=i)
                for dmt,dmr in zip(dm_test,dm_ref[i]):
                    self.assertAlmostEqual(dmt, dmr, None, message, delta)


if __name__ == "__main__":
    print("Test for L-PDFT permanent dipole moments")
    unittest.main()
