#!/usr/bin/env python
#
# Author: Shreya Verma <shreyav@uchicago.edu>

# The following tests are broken down into a couple of different categories.
#   1. Check accuracy of LASSCF analytical gradients with single fragment to CASSCF gradients for a diatomic molecule.
#   2. Check the implemmentation as scanner object.

import unittest

from pyscf import scf, gto, df, lib
from pyscf import mcscf
from pyscf.mcscf.las.lasscf_sync_o0 import LASSCF
import numpy as np


def diatomic(atom1, atom2, r, basis, ncas, nelecas, charge=None, spin=None, cas_irrep=None, density_fit=False):
    """Used for checking diatomic systems to see if the Lagrange Multipliers are working properly."""
    global mols
    xyz = '{:s} 0.0 0.0 0.0; {:s} {:.3f} 0.0 0.0'.format(atom1, atom2, r)
    mol = gto.M(atom=xyz, basis=basis, charge=charge, spin=spin, verbose=0, output='/dev/null')
    mols.append(mol)
    mf = scf.RHF(mol)

    mc = mcscf.CASSCF(mf.run(), ncas, nelecas)
    if spin is None:
        spin = mol.nelectron % 2

    ss = spin * (spin + 2) * 0.25
    mo = mc.mo_coeff
    #mc = mc.multi_state([1.0 / float(nstates), ] * nstates, 'lin')
    mc.fix_spin_(ss=ss)
    #mc.conv_tol = 1e-12
    #mc.conv_grad_tol = 1e-6
    #mo = None

    mc_grad = mc.run(mo).nuc_grad_method()
    #mc_grad.conv_rtol = 1e-12

    las = LASSCF (mf,(2,),(2,), spin_sub=(1))
    frag_atom_list = [list (range (2))]
    mo_coeff = las.localize_init_guess (frag_atom_list, mf.mo_coeff)
    las.kernel (mo_coeff)
    #las_grad = las.Gradients()
    las_grad = las.nuc_grad_method()

    return mc_grad,las_grad


def setUpModule():
    global mols
    mols = []


def tearDownModule():
    global mols, diatomic
    [m.stdout.close() for m in mols]
    del mols, diatomic


class KnownValues(unittest.TestCase):

    def test_grad_h2_sto3g(self):
        mc_grad,las_grad = diatomic('H', 'H', 1.0, 'sto-3g', 2, 2)

        de_las = las_grad.kernel()
        de_cas = mc_grad.kernel()

        self.assertAlmostEqual (lib.fp (de_cas), lib.fp (de_las), 6)

    def test_grad_scanner(self):
        # Tests and Scanner capabilities
        mc_grad1, las_grad1 = diatomic('H', 'H', 1.0, 'sto-3g', 2, 2)
        mol1 = mc_grad1.base.mol
        mc_grad2, las_grad2 = diatomic('H', 'H', 1.0, 'sto-3g', 2, 2)
        las_grad2 = las_grad2.as_scanner()
        de1 = las_grad1.kernel()
        e1 = las_grad1.base.e_states[0]
        e2, de2 = las_grad2(mol1)
        self.assertAlmostEqual(e1, e2, 6)
        self.assertAlmostEqual(lib.fp(de1), lib.fp(de2), 6)


if __name__ == "__main__":
    print("Full Tests for LASSCF gradients")
    unittest.main()
