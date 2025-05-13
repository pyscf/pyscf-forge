#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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
# Copied and modified by MRH 09/26/2023

import unittest
from functools import reduce
import numpy as np
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import fci
from pyscf import lib
from pyscf.fci import fci_slow
from pyscf.fci.spin_op import spin_square0
from pyscf.csf_fci import csf_solver
from pyscf.csf_fci.csfstring import CSFTransformer

def setUpModule():
    global mol, m, h1e, g2e, sol, h2mat
    global norb, nelec, neleci
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None#"out_h2o"
    mol.atom = [
        ['H', ( 1.,-1.    , 0.   )],
        ['H', ( 0.,-1.    ,-1.   )],
        ['H', ( 0.,-0.5   ,-0.   )],
        ['H', ( 0.,-0.    ,-1.   )],
        ['H', ( 1.,-0.5   , 0.   )],
        ['H', ( 0., 1.    , 1.   )],
    ]

    mol.basis = {'H': 'sto-3g'}
    mol.build()

    m = scf.RHF(mol)
    m.conv_tol = 1e-15
    ehf = m.scf()

    norb = m.mo_coeff.shape[1]
    nelec = (mol.nelectron//2, mol.nelectron//2)
    h1e = reduce(np.dot, (m.mo_coeff.T, m.get_hcore(), m.mo_coeff))
    g2e = ao2mo.incore.general(m._eri, (m.mo_coeff,)*4, compact=False)
    neleci = (mol.nelectron//2, mol.nelectron//2-1)
    sol = csf_solver (mol, smult=1)

    h2eff = sol.absorb_h1e (h1e, g2e, norb, nelec, .5)
    h2mat = np.zeros ((400,400))
    for i in range (400):
        c = np.zeros (400)
        c[i] = 1.0
        c = c.reshape (20,20)
        h2mat[i,:] = sol.contract_2e (h2eff, c, norb, nelec).ravel ()
    h2effi = sol.absorb_h1e (h1e, g2e, norb, neleci, .5)
    h2mati = np.zeros ((300,300))
    for i in range (300):
        c = np.zeros (300)
        c[i] = 1.0
        c = c.reshape (20,15)
        h2mati[i,:] = sol.contract_2e (h2effi, c, norb, neleci).ravel ()
    h2mat = [h2mati, h2mat]
    nel = (neleci, nelec)
    h2mat_csf = []
    for smult in range (1,8):
        h2mat_det = h2mat[smult % 2]
        ne = nel[smult % 2]
        t = CSFTransformer (norb, ne[0], ne[1], smult)
        mat = np.zeros ((h2mat_det.shape[0], t.ncsf))
        for i in range (h2mat_det.shape[0]):
            mat[i,:] = t.vec_det2csf (h2mat_det[i,:], normalize=False)
        h2mat_csf.append (np.zeros ((t.ncsf,t.ncsf)))
        for i in range (t.ncsf):
            h2mat_csf[-1][:,i] = t.vec_det2csf (mat[:,i], normalize=False)
    h2mat = h2mat_csf

def tearDownModule():
    global mol, m, h1e, g2e, sol, h2mat, norb, nelec, neleci
    del mol, m, h1e, g2e, sol, h2mat, norb, nelec, neleci

class KnownValues(unittest.TestCase):

    def test_kernel(self):
        nel = (neleci, nelec)
        refs = [-8.934702919292933, -8.749825398177125, -8.879204010931936,
                -8.407383592974286, -8.566577456561983, -7.8124466621492505,
                -7.484341852449313]
        for smult in range (1,8):
            with self.subTest (smult=smult):
                ne = nel[smult % 2]
                e, ci = sol.kernel (h1e, g2e, norb, ne, smult=smult)
                ss, smulttest = spin_square0 (ci, norb, ne)
                self.assertAlmostEqual (smulttest, smult, 8)
                self.assertAlmostEqual (e, refs[smult-1], 8)
        sol.davidson_only = True
        for smult in range (1,8):
            with self.subTest ("davidson only", smult=smult):
                ne = nel[smult % 2]
                e, ci = sol.kernel (h1e, g2e, norb, ne, smult=smult)
                ss, smulttest = spin_square0 (ci, norb, ne)
                self.assertAlmostEqual (smulttest, smult, 8)
                self.assertAlmostEqual (e, refs[smult-1], 8)

    def test_hdiag_csf (self):
        nel = (neleci, nelec)
        for smult in range (1,8):
            with self.subTest (smult=smult):
                ne = nel[smult % 2]
                hdiag = sol.make_hdiag_csf (h1e, g2e, norb, ne, smult=smult)
                hdiag_ref = h2mat[smult-1].diagonal ()
                self.assertAlmostEqual (lib.fp (hdiag), lib.fp (hdiag_ref), 8)

    def test_pspace(self):
        nel = (neleci, nelec)
        for smult in range (1,8):
            with self.subTest (smult=smult):
                ne = nel[smult % 2]
                addr, h0 = sol.pspace (h1e, g2e, norb, ne, smult=smult)
                h0_ref = h2mat[smult-1][addr,:][:,addr]
                self.assertAlmostEqual (lib.fp (h0), lib.fp (h0_ref), 8)

if __name__ == "__main__":
    print("Full Tests for csf_fci solver")
    unittest.main()

