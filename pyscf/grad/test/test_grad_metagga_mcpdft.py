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
# Author: Matthew Hennefarth <mhennefarth@uchicago.edu>

import unittest

from pyscf import scf, gto, df, dft, lib
from pyscf import mcpdft


from mrh.my_pyscf.fci import csf_solver

def diatomic(
    atom1,
    atom2,
    r,
    fnal,
    basis,
    ncas,
    nelecas,
    nstates,
    charge=None,
    spin=None,
    symmetry=False,
    cas_irrep=None,
    density_fit=False,
    grids_level=9,
):
    """Used for checking diatomic systems to see if the Lagrange Multipliers are working properly."""
    global mols
    xyz = "{:s} 0.0 0.0 0.0; {:s} {:.3f} 0.0 0.0".format(atom1, atom2, r)
    mol = gto.M(
        atom=xyz,
        basis=basis,
        charge=charge,
        spin=spin,
        symmetry=symmetry,
        verbose=0,
        output="/dev/null",
    )
    mols.append(mol)
    mf = scf.RHF(mol)
    if density_fit:
        mf = mf.density_fit(auxbasis=df.aug_etb(mol))

    mc = mcpdft.CASSCF(mf.run(), fnal, ncas, nelecas, grids_level=grids_level)
    if spin is None:
        spin = mol.nelectron % 2

    ss = spin * (spin + 2) * 0.25
    if nstates > 1:
        mc = mc.start_average(
            [
                1.0 / float(nstates),
            ]
            * nstates,
        )

    # mc.fix_spin_(ss=ss, shift=2)
    mc.fcisolver = csf_solver(mol, 1)

    mc.conv_tol = 1e-12
    mc.conv_grad_tol = 1e-6
    mo = None
    if symmetry and (cas_irrep is not None):
        mo = mc.sort_mo_by_irrep(cas_irrep)

    return mc.run(mo)
    mc_grad = mc.run(mo).nuc_grad_method()
    mc_grad.conv_rtol = 1e-12
    return mc_grad


def setUpModule():
    global mols, original_grids
    mols = []
    original_grids = dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS
    dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False

def tearDownModule():
    global mols, diatomic, original_grids
    [m.stdout.close() for m in mols]
    dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = original_grids
    del mols, diatomic, original_grids

class KnownValues(unittest.TestCase):
    def test_grad_lih_sstm06l22_sto3g(self):
        n_states = 1
        r = 0.8
        mc = diatomic("Li", "H", r, "tM06L", "STO-3G", 2, 2, n_states, grids_level=1)
       
        mc_grad = mc.nuc_grad_method()
        de_ana = mc_grad.kernel()[1,0]
        print("Ana: ", de_ana)
    
        scanner = mc.as_scanner()

        import numpy as np
        for i in np.arange(2,10,0.1):
            delta = np.exp(-i)
            scanner(f"Li 0 0 0; H {r+delta} 0 0")
            e_states_f = scanner.e_tot
            
            scanner(f"Li 0 0 0; H {r-delta} 0 0")
            e_states_b = scanner.e_tot

            de_num = (e_states_f - e_states_b)/(2*delta)
            # unit conversion
            de_num /= 1.8897259886

            print("Delta: ", delta, "Num: ", de_num, "Diff: ", de_ana-de_num)




if __name__ == "__main__":
    print("Full Tests for MC-PDFT gradients with meta-GGA functionals")
    unittest.main()
