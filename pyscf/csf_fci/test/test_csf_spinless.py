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

# This variant of test_csf.py is to check the trivial all-electrons-up case
# Although I guess hdiag also tests the all-ferromagnetic m<s case.

def setUpModule():
    global mol, m, h1e, g2e, sol
    global norb, rng
    rng = np.random.default_rng (1)
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None#"out_h2o"
    # To test different norb, comment out lines below and uncomment
    # the line skipping the kernel test
    mol.atom = [
        ['H', ( 1.,-1.    , 0.   )],
        ['H', ( 0.,-1.    ,-1.   )],
        ['H', ( 0.,-0.5   ,-0.   )],
        ['H', ( 0.,-0.    ,-1.   )],
        ['H', ( 1.,-0.5   , 0.   )],
        ['H', ( 0., 1.    , 1.   )],
    ]
    mol.spin = len (mol.atom) % 2

    mol.basis = {'H': 'sto-3g'}
    mol.build()

    m = scf.RHF(mol)
    m.conv_tol = 1e-15
    ehf = m.scf()

    norb = m.mo_coeff.shape[1]
    h1e = reduce(np.dot, (m.mo_coeff.T, m.get_hcore(), m.mo_coeff))
    #h1e[:] = 0
    h1e_s = (2 * rng.random (h1e.shape)) - 1
    h1e_s += h1e_s.conj ().T
    #h1e_s[:] = 0
    h1e = np.stack ([h1e+h1e_s, h1e-h1e_s], axis=0)
    g2e = ao2mo.incore.general(m._eri, (m.mo_coeff,)*4, compact=False)
    #g2e[:] = 1
    sol = csf_solver (mol, smult=1)

def get_h2mat_ref (ne, smult=None):
    if smult is None:
        smult = ne+1
        ne = (ne, 0)
    t = CSFTransformer (norb, ne[0], ne[1], smult)
    h2eff = sol.absorb_h1e (h1e, g2e, norb, ne, .5)
    ndet = t.ndeta*t.ndetb
    h2mat_det = np.zeros ((ndet, ndet))
    for i in range (ndet):
        c = np.zeros (ndet)
        c[i] = 1.0
        c = c.reshape (t.ndeta,t.ndetb)
        h2mat_det[i,:] = sol.contract_2e (h2eff, c, norb, ne).ravel ()
    mat = np.zeros ((h2mat_det.shape[0], t.ncsf))
    for i in range (h2mat_det.shape[0]):
        mat[i,:] = t.vec_det2csf (h2mat_det[i,:], normalize=False)
    h2mat_csf = np.zeros ((t.ncsf,t.ncsf))
    for i in range (t.ncsf):
        h2mat_csf[:,i] = t.vec_det2csf (mat[:,i], normalize=False)
    return h2mat_csf

def tearDownModule():
    global mol, m, h1e, g2e, sol, norb, rng
    del mol, m, h1e, g2e, sol, norb, rng

class KnownValues(unittest.TestCase):

    #@unittest.skip('debug')
    def test_kernel(self):
        refs = [-5.61216464986069, -9.755775510701424, -12.028799874041646,
                -13.191881433911803, -12.213871610466288, -9.154045604796789]
        for ne in range (1,7):
            with self.subTest (ne=ne):
                smult = ne+1
                e, ci = sol.kernel (h1e, g2e, norb, (ne,0), smult=smult)
                ss, smulttest = spin_square0 (ci, norb, (ne,0))
                self.assertAlmostEqual (smulttest, smult, 8)
                self.assertAlmostEqual (e, refs[ne-1], 8)
                self.assertTrue (sol.converged)
        sol.davidson_only = True
        for ne in range (1,7):
            with self.subTest ("davidson only", ne=ne):
                smult = ne+1
                e, ci = sol.kernel (h1e, g2e, norb, (ne,0), smult=smult)
                ss, smulttest = spin_square0 (ci, norb, (ne,0))
                self.assertAlmostEqual (smulttest, smult, 8)
                self.assertAlmostEqual (e, refs[ne-1], 8)
                self.assertTrue (sol.converged)

    #@unittest.skip('debug')
    def test_hdiag_csf (self):
        for n in range (1,7):
            s2 = n
            smult = n + 1
            for m in range (-s2, s2+1, 2):
                ne = [(n+m) // 2, (n-m) // 2]
                with self.subTest (smult=smult, m=m):
                    hdiag = sol.make_hdiag_csf (h1e, g2e, norb, ne, smult=smult)
                    hdiag_ref = get_h2mat_ref (ne, smult).diagonal ()
                    self.assertAlmostEqual (lib.fp (hdiag), lib.fp (hdiag_ref), 8)


    #@unittest.skip('debug')
    def test_pspace(self):
        for ne in range (1,7):
            with self.subTest (ne=ne):
                smult = ne+1
                addr, h0 = sol.pspace (h1e, g2e, norb, (ne,0), smult=smult)
                t = sol.transformer
                h0_ref = get_h2mat_ref (ne)[addr,:][:,addr]
                print (norb, ne, smult)
                for i in range (len (h0)):
                    for j in range (i):
                        if abs (h0[i,j] - h0_ref[i,j]) > 1e-8:
                            print (i, j, t.printable_csfstring (i), t.printable_csfstring (j),
                                   h0[i,j], h0_ref[i,j])
                self.assertAlmostEqual (lib.fp (h0), lib.fp (h0_ref), 8)

    @unittest.skip('debug')
    def test_pspace_single_element (self):
        i = 1 
        j = 0
        ne=1
        smult=ne+1
        addr = np.array ([i,j], dtype=np.int32)
        addr, h0 = sol.pspace (h1e, g2e, norb, (ne,0), smult=smult, csf_addr=addr)
        t = sol.transformer
        h0_ref = get_h2mat_ref (ne)[addr,:][:,addr]
        print (i, j, t.printable_csfstring (i), t.printable_csfstring (j),
            h0[0,1], h0_ref[0,1])
        self.assertAlmostEqual (lib.fp (h0), lib.fp (h0_ref), 8)

if __name__ == "__main__":
    print("Full Tests for csf_fci solver")
    unittest.main()

