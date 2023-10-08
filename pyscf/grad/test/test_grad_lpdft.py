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

import numpy as np
import unittest

from pyscf import scf, gto, mcscf, df
from pyscf.data.nist import BOHR
from pyscf import mcpdft

def diatomic(atom1, atom2, r, fnal, basis, ncas, nelecas, nstates,
             charge=None, spin=None, symmetry=False, cas_irrep=None,
             density_fit=False):
    global mols
    xyz = '{:s} 0.0 0.0 0.0; {:s} {:.3f} 0.0 0.0'.format(atom1, atom2, r)
    mol = gto.M(atom=xyz, basis=basis, charge=charge, spin=spin, symmetry=symmetry, verbose=0, output='/dev/null')
    mols.append(mol)
    mf = scf.RHF(mol)
    if density_fit:
        mf = mf.density_fit(auxbasis=df.aug_etb(mol))

    mc = mcpdft.CASSCF(mf.run(), fnal, ncas, nelecas, grids_level=9)
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
    return mc, mc.nuc_grad_method()


def setUpModule():
    global mols
    mols = []


def tearDownModule():
    global mols, diatomic
    [m.stdout.close() for m in mols]
    del mols, diatomic

np.set_printoptions(precision=10)
def gen(scanner, start=1.3):
    for i in range(0, 30):
        delta = 0.1 / (1.1 ** i) / 2
        scanner(f'Li 0 0 0; H {start + delta} 0 0')
        e1 = np.array(scanner.e_states)
        scanner(f'Li 0 0 0; H {start - delta} 0 0')
        e2 = np.array(scanner.e_states)
        de_ref = (e1 - e2) / (2 * delta)
        if len(de_ref) == 2:
            print(f"{(2 * delta) ** 2}, {de_ref[0]:.10f}, {de_ref[1]:.10f}")

        else:
            print(f"{(2 * delta) ** 2}, {de_ref[0]:.10f}, {de_ref[1]:.10f}, {de_ref[2]:.10f}")

    scanner(f'Li 0 0 0; H {start} 0 0')  # reset


class KnownValues(unittest.TestCase):

    def test_grad_h2_lin3ftlda22_sto3g_slow(self):
        # orb:    no
        # ci:     no
        mc, mc_grad = diatomic('H', 'H', 1.3, 'ftLDA,VWN3', 'STO-3G', 2, 2, 3)

        # Numerical from this software
        NUM_REF = [0.2483709972, -0.235934352, -0.7202397013]
        for i in range(3):
            with self.subTest(state=i):
                de = mc_grad.kernel(state=i)[1, 0] / BOHR
                self.assertAlmostEqual(de, NUM_REF[i], 7)

    def test_grad_h2_lin2ftlda22_sto3g_slow(self):
        # orb:    no
        # ci:     yes
        mc, mc_grad = diatomic('H', 'H', 1.3, 'ftLDA,VWN3', 'STO-3G', 2, 2, 2)

        # Numerical from this software
        NUM_REF = [0.3045653798, -0.2328036389]
        for i in range(2):
            with self.subTest(state=i):
                de = mc_grad.kernel(state=i)[1, 0] / BOHR
                self.assertAlmostEqual(de, NUM_REF[i], 7)

    def test_grad_h2_lin3ftlda22_631g_slow(self):
        # orb:    yes
        # ci:     no
        mc, mc_grad = diatomic('H', 'H', 1.3, 'ftLDA,VWN3', '6-31G', 2, 2, 3)

        # Numerical from this software
        NUM_REF = [0.1892819283, -0.1459936283, -0.4812691849]
        for i in range(3):
            with self.subTest(state=i):
                de = mc_grad.kernel(state=i)[1, 0] / BOHR
                self.assertAlmostEqual(de, NUM_REF[i], 7)

    def test_grad_h2_lin2ftlda22_631g_slow(self):
        # orb:    yes
        # ci:     yes
        mc, mc_grad = diatomic('H', 'H', 1.3, 'ftLDA,VWN3', '6-31G', 2, 2, 2)

        # Numerical from this software
        NUM_REF = [0.2181136615, -0.1549280412]
        for i in range(2):
            with self.subTest(state=i):
                de = mc_grad.kernel(state=i)[1, 0] / BOHR
                self.assertAlmostEqual(de, NUM_REF[i], delta=1e-5)

    def test_grad_lih_lin2ftlda44_sto3g_slow(self):
        # z_orb:    no
        # z_ci:     yes
        mc, mc_grad = diatomic('Li', 'H', 1.8, 'ftLDA,VWN3', 'STO-3G', 4, 4, 2, symmetry=True, cas_irrep={'A1': 4})

        # gen(mc.as_scanner())

        # Numerical from this software
        NUM_REF = [0.0706137599, -0.0065958760]
        for i in range(2):
            with self.subTest(state=i):
                de = mc_grad.kernel(state=i)[1, 0] / BOHR
                self.assertAlmostEqual(de, NUM_REF[i], delta=1e-6)

    def test_grad_lih_lin3ftlda22_sto3g_slow(self):
        # z_orb:    yes
        # z_ci:     no
        mc, mc_grad = diatomic('Li', 'H', 2.4, 'ftLDA,VWN3', 'STO-3G', 2, 2, 3)

        # Numerical from this software
        NUM_REF = [0.1258966287, 0.0072754131, -0.1113474634]
        for i in range(3):
            with self.subTest(state=i):
                de = mc_grad.kernel(state=i)[1, 0] / BOHR
                self.assertAlmostEqual(de, NUM_REF[i], delta=1e-4)

    def test_grad_lih_lin2ftlda22_sto3g_slow(self):
        # z_orb:    yes
        # z_ci:     yes
        mc, mc_grad = diatomic('Li', 'H', 2.4, 'ftLDA,VWN3', 'STO-3G', 2, 2, 2)

        # Numerical from this software
        NUM_REF = [0.1072597373, 0.0235666209]
        for i in range(2):
            with self.subTest(state=i):
                de = mc_grad.kernel(state=i)[1, 0] / BOHR
                self.assertAlmostEqual(de, NUM_REF[i], delta=1e-4)

    def test_grad_lih_lin2ftpbe22_sto3g(self):
        # z_orb:    yes
        # z_ci:     yes
        mc, mc_grad = diatomic('Li', 'H', 2.4, 'ftPBE', 'STO-3G', 2, 2, 2)

        # Numerical from this software
        NUM_REF = [0.1024946074, 0.0190815023]
        for i in range(2):
            with self.subTest(state=i):
                de = mc_grad.kernel(state=i)[1, 0] / BOHR
                self.assertAlmostEqual(de, NUM_REF[i], delta=1e-4)

    # def test_grad_lih_lin2ftlda22_sto3g_df(self):
    #     # z_orb:    yes
    #     # z_ci:     yes
    #     mc, mc_grad = diatomic('Li', 'H', 2.4, 'ftLDA,VWN3', 'STO-3G', 2, 2, 2, density_fit=True)
    #
    #     #gen(mc.as_scanner(), start=2.4)
    #
    #     # Numerical from this software
    #     NUM_REF = [0.1074021770, 0.0234933387]
    #     for i in range(2):
    #         with self.subTest(state=i):
    #             de = mc_grad.kernel(state=i)[1, 0] / BOHR
    #             print(de)
    #             self.assertAlmostEqual(de, NUM_REF[i], 5)


if __name__ == "__main__":
    print("Full Tests for L-PDFT gradients API")
    unittest.main()
