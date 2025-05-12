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
from pyscf import gto, dft, scf, mcscf
from pyscf import mcpdft
import unittest
from pyscf.fci.addons import fix_spin_
#from pyscf.fci import csf_solver

geom_h2o='''
O  0.00000000   0.08111156   0.00000000
H  0.78620605   0.66349738   0.00000000
H -0.78620605   0.66349738   0.00000000
'''
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
mol_h2o = gto.M(atom = geom_h2o, basis = 'aug-cc-pVDZ', symmetry='c2v', output='/dev/null', verbose=0)
mol_furan_cation = gto.M(atom = geom_furan, basis = 'sto-3g', charge=1, spin=1, symmetry=False, output='/dev/null', verbose=0)
def get_h2o(mol,iroots=3):
    weights = [1/iroots]*iroots
    mf = scf.RHF(mol).run()
    mc = mcpdft.CASSCF(mf,'tPBE', 4, 4, grids_level=1)
    #mc.fcisolver = csf_solver(mol, smult=1, symm='A1')
    fix_spin_(mc.fcisolver, ss=0)
    mc.fcisolver.wfnsym = 'A1'
    mc = mc.state_average_(weights)
    mo = mcscf.sort_mo(mc, mf.mo_coeff, [4,5,8,9])
    mc.conv_tol = 1e-11
    mc.kernel(mo)
    return mc

def setUpModule():
    global mol_h2o, mol_furan_cation, original_grids
    original_grids = dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS
    dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False

def tearDownModule():
    global mol_h2o, mol_furan_cation, original_grids
    dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = original_grids
    mol_h2o.stdout.close ()
    mol_furan_cation.stdout.close ()
    del mol_h2o, mol_furan_cation, original_grids

class KnownValues(unittest.TestCase):
    '''
    The reference values were obtained by numeric differentiation of the energy 
    with respect to the electric field strength extrapolated to zero step size. 
    The fields were applied along each direction in the XYZ frame, and derivatives 
    were evaluated using 2-point central difference formula.   
    '''
    def test_h2o_sa3_tpbe_631g(self):
        dm_ref = np.array(\
            [[0.0000,  1.3147, 0.0000],
            [ 0.0000, -0.8687, 0.0000],
            [ 0.0000,  2.6450, 0.0000]])
        delta = 0.001
        message = "Dipoles are not equal within {} D".format(delta)

        iroots=3
        mc = get_h2o(mol_h2o, iroots)
        for i in range(iroots):
            with self.subTest (i=i):
                dm_test = mc.dip_moment(unit='Debye', origin="Coord_center",state=i)
                for dmt,dmr in zip(dm_test,dm_ref[i]):
                    self.assertAlmostEqual(dmt, dmr, None, message, delta)

if __name__ == "__main__":
    print("Test for SA-PDFT permanent dipole moments")
    unittest.main()
