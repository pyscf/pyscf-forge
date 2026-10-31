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
from pyscf.csf_fci.test.old_pspace import pspace as old_pspace

# TODO: test updating solver object in place
# 4 A1; 1 B1u, 1 B2u, 1 B3u
# Since smult > 5 has 5 unpaired electrons, A1 becomes impossible
# Test both B*u (smult=6) and B*g (smult=7)

def setUpModule():
    global mol, m, h1e, g2e, h2mat_det, h2mat_csf
    global norb, nelec, neleci, nel, orbsym, sol, sol_ref
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None#"out_h2o"
    mol.symmetry = True
    mol.spin = 1
    mol.atom = [
        ['H', ( 0. , 0. , 0. )],
        ['H', ( -0.5 , 0. , 0. )],
        ['H', ( +0.5 , 0. , 0. )],
        ['H', ( 0. , -1.0 , 0. )],
        ['H', ( 0. , 1.0 , 0. )],
        ['H', ( 0. , 0. , -1.5 )],
        ['H', ( 0. , 0. , 1.5 )]
    ]

    mol.basis = {'H': 'sto-3g'}
    mol.build()

    m = scf.RHF(mol)
    m.conv_tol = 1e-15
    ehf = m.scf()
    orbsym = m.get_orbsym (m.mo_coeff, m.get_ovlp ())

    norb = m.mo_coeff.shape[1]
    nelec = ((mol.nelectron + 1)//2, mol.nelectron//2)
    h1e = reduce(np.dot, (m.mo_coeff.T, m.get_hcore(), m.mo_coeff))
    g2e = ao2mo.incore.general(m._eri, (m.mo_coeff,)*4, compact=False)
    neleci = (mol.nelectron//2, mol.nelectron//2)
    sol_ref = csf_solver (mol, smult=1, symm=False)
    nel = (nelec, neleci)
    h2mat_det = {}
    h2mat_csf = {}

    sol = csf_solver (mol, smult=1).set (orbsym=orbsym)

def get_h2mat_ref (ne, smult):
    ne = tuple (ne)
    if (ne, smult) in h2mat_csf:
        return h2mat_csf[(ne, smult)]
    t = CSFTransformer (norb, ne[0], ne[1], smult, orbsym=orbsym, wfnsym=sol_ref.wfnsym)
    ndet = t.ndeta*t.ndetb
    if ne not in h2mat_det:
        h2eff = sol_ref.absorb_h1e (h1e, g2e, norb, ne, .5)
        h2mat_det[ne] = np.zeros ((ndet, ndet))
        for i in range (ndet):
            c = np.zeros (ndet)
            c[i] = 1.0
            c = c.reshape (t.ndeta,t.ndetb)
            h2mat_det[ne][i,:] = sol_ref.contract_2e (h2eff, c, norb, ne).ravel ()
    mat = np.zeros ((ndet, t.ncsf))
    for i in range (ndet):
        mat[i,:] = t.vec_det2csf (h2mat_det[ne][i,:], normalize=False)
    h2mat_csf[(ne, smult)] = np.zeros ((t.ncsf,t.ncsf))
    for i in range (t.ncsf):
        h2mat_csf[(ne, smult)][:,i] = t.vec_det2csf (mat[:,i], normalize=False)
    return h2mat_csf[(ne, smult)]

def tearDownModule():
    global mol, m, h1e, g2e, h2mat_det, h2mat_csf, norb, nelec, neleci, nel, orbsym, sol, sol_ref
    del mol, m, h1e, g2e, h2mat_det, h2mat_csf, norb, nelec, neleci, nel, orbsym, sol, sol_ref

class KnownValues(unittest.TestCase):

    @unittest.skip('debug')
    def test_kernel(self):
        refs = [-11.806515601408687, -11.840480580594795, -11.250779244019393, -10.717848638632578,
                -9.058272189032948, -9.60258614255864, -9.387425333405808]
        wfnsym = [0,]*5 + [5,1]
        for smult in range (1,8):
            with self.subTest (smult=smult):
                ne = nel[smult % 2]
                w = wfnsym[smult-1]
                e, ci = sol.kernel (h1e, g2e, norb, ne, smult=smult, wfnsym=w)
                ss, smulttest = spin_square0 (ci, norb, ne)
                self.assertAlmostEqual (smulttest, smult, 8)
                self.assertAlmostEqual (e, refs[smult-1], 8)
                self.assertTrue (sol.converged)
        sol.davidson_only = True
        for smult in range (1,6):
            with self.subTest ("davidson only", smult=smult):
                ne = nel[smult % 2]
                w = wfnsym[smult-1]
                e, ci = sol.kernel (h1e, g2e, norb, ne, smult=smult, wfnsym=w)
                ss, smulttest = spin_square0 (ci, norb, ne)
                self.assertAlmostEqual (smulttest, smult, 8)
                self.assertAlmostEqual (e, refs[smult-1], 8)
                self.assertTrue (sol.converged)

    def test_hdiag_csf (self):
        wfnsym = [0,]*5 + [5,1]
        for smult in range (1,8):
            n = sum (nel[smult % 2])
            s2 = smult - 1
            for m in range (-s2, s2+1, 2):
                ne = [(n+m) // 2, (n-m) // 2]
                with self.subTest (smult=smult, m=m):
                    w = wfnsym[smult-1]
                    sol.wfnsym = w
                    hdiag = sol.make_hdiag_csf (h1e, g2e, norb, ne, smult=smult)
                    hdiag_ref = get_h2mat_ref (ne, smult).diagonal ()
                    self.assertAlmostEqual (lib.fp (hdiag), lib.fp (hdiag_ref), 8)

    @unittest.skip('debug')
    def test_pspace(self):
        wfnsym = [0,]*5 + [5,1]
        for smult in range (1,8):
            n = sum (nel[smult % 2])
            s2 = smult - 1
            for m in range (-s2, s2+1, 2):
                ne = [(n+m) // 2, (n-m) // 2]
                with self.subTest (smult=smult, m=m):
                    w = wfnsym[smult-1]
                    sol.wfnsym = w
                    addr, h0 = sol.pspace (h1e, g2e, norb, ne, smult=smult)
                    h0_ref = get_h2mat_ref (ne, smult)[addr,:][:,addr]
                    self.assertAlmostEqual (lib.fp (h0), lib.fp (h0_ref), 8)

    #@unittest.skip('debug')
    def test_old_pspace(self):
        wfnsym = [0,]*5 + [5,1]
        for smult in range (1,8):
            n = sum (nel[smult % 2])
            s2 = smult - 1
            for m in range (-s2, s2+1, 2):
                ne = [(n+m) // 2, (n-m) // 2]
                with self.subTest (smult=smult, m=m):
                    w = wfnsym[smult-1]
                    sol.smult = smult
                    sol.norb = norb
                    sol.nelec = ne
                    sol.wfnsym = w
                    sol.check_transformer_cache ()
                    addr, h0 = old_pspace (sol, h1e, g2e, norb, ne, sol.transformer)
                    h0_ref = get_h2mat_ref (ne, smult)[addr,:][:,addr]
                    self.assertAlmostEqual (lib.fp (h0), lib.fp (h0_ref), 8)

if __name__ == "__main__":
    print("Full Tests for csf_fci solver with symmetry")
    unittest.main()

