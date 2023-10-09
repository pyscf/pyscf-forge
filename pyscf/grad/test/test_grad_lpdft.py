#!/usr/bin/env python
# Copyright 2014-2023 The PySCF Developers. All Rights Reserved.
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

# The following tests are broken down into a couple of different categories.
#   1. Check accuracy of analytical gradients to numerical gradients for different Lagrange multiplier situations. Some
#      tests are redundant since as long as the tests with both MO and CI Lagrange multipliers pass, then everything
#      should be good. All other tests are marked as slow and used for thorough debugging.
#   2. Check API as scanner object.
#   3. L-PDFT gradients for multi_state_mix type objects.

import unittest

from pyscf import scf, gto, mcscf, df, lib, fci
from pyscf.data.nist import BOHR
from pyscf import mcpdft


def diatomic(atom1, atom2, r, fnal, basis, ncas, nelecas, nstates,
             charge=None, spin=None, symmetry=False, cas_irrep=None,
             density_fit=False, grids_level=9):
    """Used for checking diatomic systems to see if the Lagrange Multipliers are working properly."""
    global mols
    xyz = '{:s} 0.0 0.0 0.0; {:s} {:.3f} 0.0 0.0'.format(atom1, atom2, r)
    mol = gto.M(atom=xyz, basis=basis, charge=charge, spin=spin, symmetry=symmetry, verbose=0, output='/dev/null')
    mols.append(mol)
    mf = scf.RHF(mol)
    if density_fit:
        mf = mf.density_fit(auxbasis=df.aug_etb(mol))

    mc = mcpdft.CASSCF(mf.run(), fnal, ncas, nelecas, grids_level=grids_level)
    if spin is None:
        spin = mol.nelectron % 2

    ss = spin * (spin + 2) * 0.25
    mc = mc.multi_state([1.0 / float(nstates), ] * nstates, 'lin')
    mc.fix_spin_(ss=ss, shift=2)
    mc.conv_tol = 1e-12
    mc.conv_grad_tol = 1e-12
    mo = None
    if symmetry and (cas_irrep is not None):
        mo = mc.sort_mo_by_irrep(cas_irrep)

    mc.kernel(mo)
    return mc.nuc_grad_method()


def auto_setup(xyz='Li 0 0 0\nH 1.5 0 0', symmetry=False, mix=False):
    global mols
    mol = gto.M(atom=xyz, basis='sto3g',
                output='/dev/null', verbose=0, symmetry=symmetry)
    mols.append(mol)
    mf = scf.RHF(mol).run()
    mc = mcpdft.CASSCF(mf, 'ftLDA,VWN3', 5, 2, grids_level=1).run()

    if not mix:
        return mc.multi_state([1.0 / 5, ] * 5, method="lin").run()

    if symmetry:
        solver_A1 = fci.solver(mol).set(wfnsym='A1', nroots=3)
        solver_E1x = fci.solver(mol).set(wfnsym='E1x', nroots=1, spin=2)
        solver_E1y = fci.solver(mol).set(wfnsym='E1y', nroots=1, spin=2)

        solvers = [solver_A1, solver_E1x, solver_E1y]
    else:
        solver_S = fci.solver(mol, singlet=True).set(spin=0, nroots=2)
        solver_T = fci.solver(mol, singlet=False).set(spin=2, nroots=3)
        solvers = [solver_S, solver_T]

    return mc.multi_state_mix(solvers, [1.0 / 5, ] * 5, method="lin").run()


def setUpModule():
    global mols
    mols = []


def tearDownModule():
    global mols, diatomic
    [m.stdout.close() for m in mols]
    del mols, diatomic


class KnownValues(unittest.TestCase):

    def test_grad_h2_lin3ftlda22_sto3g_slow(self):
        """ System has the following Lagrange multiplier sectors:
            orb:    no
            ci:     no
        """
        mc_grad = diatomic('H', 'H', 1.3, 'ftLDA,VWN3', 'STO-3G', 2, 2, 3)

        # Numerical from this software
        # PySCF commit:         824218f997
        # PySCF-forge commit:   426bfa33e1
        NUM_REF = [0.2483709972, -0.235934352, -0.7202397013]
        for i in range(3):
            with self.subTest(state=i):
                de = mc_grad.kernel(state=i)[1, 0] / BOHR
                self.assertAlmostEqual(de, NUM_REF[i], 7)

    def test_grad_h2_lin2ftlda22_sto3g_slow(self):
        """ System has the following Lagrange multiplier sectors:
            orb:    no
            ci:     yes
        """
        mc_grad = diatomic('H', 'H', 1.3, 'ftLDA,VWN3', 'STO-3G', 2, 2, 2)

        # Numerical from this software
        # PySCF commit:         824218f997
        # PySCF-forge commit:   426bfa33e1
        NUM_REF = [0.3045653798, -0.2328036389]
        for i in range(2):
            with self.subTest(state=i):
                de = mc_grad.kernel(state=i)[1, 0] / BOHR
                self.assertAlmostEqual(de, NUM_REF[i], 7)

    def test_grad_h2_lin3ftlda22_631g_slow(self):
        """ System has the following Lagrange multiplier sectors:
            orb:    yes
            ci:     no
        """
        mc_grad = diatomic('H', 'H', 1.3, 'ftLDA,VWN3', '6-31G', 2, 2, 3)

        # Numerical from this software
        # PySCF commit:         824218f997
        # PySCF-forge commit:   426bfa33e1
        NUM_REF = [0.1892819283, -0.1459936283, -0.4812691849]
        for i in range(3):
            with self.subTest(state=i):
                de = mc_grad.kernel(state=i)[1, 0] / BOHR
                self.assertAlmostEqual(de, NUM_REF[i], 7)

    def test_grad_h2_lin2ftlda22_631g(self):
        """ System has the following Lagrange multiplier sectors:
            orb:    yes
            ci:     yes
        """
        mc_grad = diatomic('H', 'H', 1.3, 'ftLDA,VWN3', '6-31G', 2, 2, 2)

        # Numerical from this software
        # PySCF commit:         824218f997
        # PySCF-forge commit:   426bfa33e1
        NUM_REF = [0.2181136615, -0.1549280412]
        for i in range(2):
            with self.subTest(state=i):
                de = mc_grad.kernel(state=i)[1, 0] / BOHR
                self.assertAlmostEqual(de, NUM_REF[i], delta=1e-5)

    def test_grad_lih_lin2ftlda44_sto3g_slow(self):
        """ System has the following Lagrange multiplier sectors:
            orb:    no
            ci:     yes
        """
        mc_grad = diatomic('Li', 'H', 1.8, 'ftLDA,VWN3', 'STO-3G', 4, 4, 2, symmetry=True, cas_irrep={'A1': 4})

        # Numerical from this software
        # PySCF commit:         824218f997
        # PySCF-forge commit:   426bfa33e1
        NUM_REF = [0.0706137599, -0.0065958760]
        for i in range(2):
            with self.subTest(state=i):
                de = mc_grad.kernel(state=i)[1, 0] / BOHR
                self.assertAlmostEqual(de, NUM_REF[i], delta=1e-6)

    def test_grad_lih_lin3ftlda22_sto3g_slow(self):
        """ System has the following Lagrange multiplier sectors:
            orb:    yes
            ci:     no
        """
        mc_grad = diatomic('Li', 'H', 2.4, 'ftLDA,VWN3', 'STO-3G', 2, 2, 3)

        # Numerical from this software
        # PySCF commit:         824218f997
        # PySCF-forge commit:   426bfa33e1
        NUM_REF = [0.1258966287, 0.0072754131, -0.1113474634]
        for i in range(3):
            with self.subTest(state=i):
                de = mc_grad.kernel(state=i)[1, 0] / BOHR
                self.assertAlmostEqual(de, NUM_REF[i], delta=1e-4)

    def test_grad_lih_lin2ftlda22_sto3g_slow(self):
        """ System has the following Lagrange multiplier sectors:
            orb:    yes
            ci:     yes
        """
        mc_grad = diatomic('Li', 'H', 2.4, 'ftLDA,VWN3', 'STO-3G', 2, 2, 2)

        # Numerical from this software
        # PySCF commit:         824218f997
        # PySCF-forge commit:   426bfa33e1
        NUM_REF = [0.1072597373, 0.0235666209]
        for i in range(2):
            with self.subTest(state=i):
                de = mc_grad.kernel(state=i)[1, 0] / BOHR
                print(de - NUM_REF[i])
                self.assertAlmostEqual(de, NUM_REF[i], delta=1e-4)

    def test_grad_lih_lin2ftpbe22_sto3g(self):
        """ System has the following Lagrange multiplier sectors:
            orb:    yes
            ci:     yes
        """
        mc_grad = diatomic('Li', 'H', 2.4, 'ftPBE', 'STO-3G', 2, 2, 2)

        # Numerical from this software
        # PySCF commit:         824218f997
        # PySCF-forge commit:   426bfa33e1
        NUM_REF = [0.1024946074, 0.0190815023]
        for i in range(2):
            with self.subTest(state=i):
                de = mc_grad.kernel(state=i)[1, 0] / BOHR
                self.assertAlmostEqual(de, NUM_REF[i], delta=1e-4)

    def test_grad_scanner(self):
        # Tests API and Scanner capabilities
        mc_grad1 = diatomic("Li", "H", 1.5, "ftLDA,VWN3", "STO-3G", 2, 2, 2, grids_level=1)
        mol1 = mc_grad1.base.mol
        mc_grad2 = diatomic("Li", "H", 1.6, "ftLDA,VWN3", "STO-3G", 2, 2, 2, grids_level=1).as_scanner()

        for state in range(2):
            with self.subTest(state=state):
                de1 = mc_grad1.kernel(state=state)
                e1 = mc_grad1.base.e_states[state]
                e2, de2 = mc_grad2(mol1, state=state)
                self.assertTrue(mc_grad1.converged)
                self.assertTrue(mc_grad2.converged)
                self.assertAlmostEqual(e1, e2, 6)
                self.assertAlmostEqual(lib.fp(de1), lib.fp(de2), 6)

    # Need to debug. Something not right with CI Lagrange multiplier terms...Maybe projection issue??
    # def test_grad_multi_state_mix(self):
    #     mc_nosym = auto_setup(mix=True, symmetry=False).nuc_grad_method()
    #     mc_sym = auto_setup(mix=True, symmetry=True).nuc_grad_method()
    #
    #     for state in range(5):
    #         state_nosym = np.argsort(mc_nosym.e_states)[state]
    #         state_sym = np.argsort(mc_sym.e_states)[state]
    #         de1 = mc_nosym.kernel(state=state_nosym)[0,0]
    #         de2 = mc_sym.kernel(state=state_sym)[0,0]
    #         print(f"de1: {de1} de2: {de2} diff:\t{de1-de2}")


if __name__ == "__main__":
    print("Full Tests for L-PDFT gradients API")
    unittest.main()
