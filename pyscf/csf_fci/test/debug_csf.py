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
from itertools import product
import numpy as np
from scipy import linalg
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import fci
from pyscf import lib
from pyscf.fci import fci_slow
from pyscf.fci.spin_op import spin_square0
from pyscf.fci.addons import des_a, des_b
from pyscf.csf_fci import csf_solver
from pyscf.csf_fci.csfstring import CSFTransformer


def setUpModule():
    global mol, m, h1e, g2e, sol
    global norb, nelec, neleci, rng, smult_lim
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
        #['H', ( 0.,-0.    ,-1.   )],
        #['H', ( 1.,-0.5   , 0.   )],
        #['H', ( 0., 1.    , 1.   )],
    ]
    mol.spin = len (mol.atom) % 2
    smult_lim = 3 #len (mol.atom) + 2

    mol.basis = {'H': 'sto-3g'}
    mol.build()

    m = scf.RHF(mol)
    m.conv_tol = 1e-15
    ehf = m.scf()

    neleca = (mol.nelectron+1)//2 # round up
    neleca = 2

    norb = m.mo_coeff.shape[1]
    nelec = (neleca, neleca)
    h1e = reduce(np.dot, (m.mo_coeff.T, m.get_hcore(), m.mo_coeff))
    h1e[:] = 1
    h1e_s = (2 * rng.random (h1e.shape)) - 1
    h1e_s += h1e_s.conj ().T
    h1e_s[:] = 0
    h1e = np.stack ([h1e+h1e_s, h1e-h1e_s], axis=0)
    g2e = ao2mo.incore.general(m._eri, (m.mo_coeff,)*4, compact=False)
    g2e[:] = 0
    neleci = (neleca, neleca-1)
    sol = csf_solver (mol, smult=1)
    nel = (neleci, nelec)

def get_h2mat_ref (ne, smult):
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
    global mol, m, h1e, g2e, sol, norb, nelec, neleci, rng, smult_lim
    del mol, m, h1e, g2e, sol, norb, nelec, neleci, rng, smult_lim

class KnownValues(unittest.TestCase):

    @unittest.skip('debug')
    def test_kernel(self):
        nel = (neleci, nelec)
        refs = [-8.934702919292933, -12.578019902416628, -8.879204010931936,
                -10.273273133118241, -8.566577456561983, -8.600167849055723,
                -7.484341852449313]
        for smult in range (1,8):
            with self.subTest (smult=smult):
                ne = nel[smult % 2]
                e, ci = sol.kernel (h1e, g2e, norb, ne, smult=smult)
                ss, smulttest = spin_square0 (ci, norb, ne)
                self.assertAlmostEqual (smulttest, smult, 8)
                self.assertAlmostEqual (e, refs[smult-1], 8)
        sol.davidson_only = True
        for smult in range (1,smult_lim):
            with self.subTest ("davidson only", smult=smult):
                ne = nel[smult % 2]
                e, ci = sol.kernel (h1e, g2e, norb, ne, smult=smult)
                ss, smulttest = spin_square0 (ci, norb, ne)
                self.assertAlmostEqual (smulttest, smult, 8)
                self.assertAlmostEqual (e, refs[smult-1], 8)

    @unittest.skip('debug')
    def test_hdiag_csf (self):
        nel = (neleci, nelec)
        for smult in range (1,smult_lim):
            n = sum (nel[smult % 2])
            s2 = smult - 1
            for m in range (-s2, s2+1, 2):
                ne = [(n+m) // 2, (n-m) // 2]
                with self.subTest (smult=smult, m=m):
                    hdiag = sol.make_hdiag_csf (h1e, g2e, norb, ne, smult=smult)
                    hdiag_ref = get_h2mat_ref (ne, smult).diagonal ()
                    self.assertAlmostEqual (lib.fp (hdiag), lib.fp (hdiag_ref), 8)


    @unittest.skip('debug')
    def test_pspace(self):
        nel = (neleci, nelec)
        for smult in range (1,smult_lim):
            with self.subTest (smult=smult):
                ne = nel[smult % 2]
                addr, h0 = sol.pspace (h1e, g2e, norb, ne, smult=smult)
                t = sol.transformer
                h0_ref = get_h2mat_ref (ne, smult)[addr,:][:,addr]
                print (norb, ne, smult)
                for i in range (len (h0)):
                    for j in range (i):
                        if abs (h0[i,j] - h0_ref[i,j]) > 1e-8:
                            print (t.printable_csfstring (i), t.printable_csfstring (j),
                                   h0[i,j], h0_ref[i,j])
                self.assertAlmostEqual (lib.fp (h0), lib.fp (h0_ref), 8)

    def test_csf_sign (self):
        rng = np.random.default_rng ()
        for smult, ndocc, nvirt in product (range (1,8), range(1,3), range(3)):
            with self.subTest (smult=smult, ndocc=ndocc, nvirt=nvirt):
              if smult==1 and ndocc==0: continue
              nelec = ((smult-1) + ndocc, ndocc)
              norb = (smult-1) + ndocc + nvirt
              trans_k = CSFTransformer (norb, nelec[0], nelec[1], smult)
              trans_b = CSFTransformer (norb, nelec[0]-1, nelec[1]-1, smult)
              dket, sket, tket = trans_k.csfaddrs2str (list (range (trans_k.ncsf)))
              dbra, sbra, tbra = trans_b.csfaddrs2str (list (range (trans_b.ncsf)))
              dket = np.maximum (dket, 0)
              dbra = np.maximum (dbra, 0)
              for iorb in range (norb):
                with self.subTest (iorb=iorb):
                  ikets = dket>=0
                  ikets = ikets & np.remainder (dket // (2**iorb), 2)
                  ikets = np.where (ikets)[0]
                  for iket in ikets:
                    with self.subTest (iket=iket):
                      dk, sk, tk = dket[iket], sket[iket], tket[iket]
                      ispin = iorb
                      for jorb in range (iorb):
                        if (dk & (1 << jorb)):
                            ispin -= 1
                      dconf = dk ^ (1 << iorb)
                      sconf_right = sk & ((1 << ispin)-1)
                      sconf_left = (sk >> ispin) << ispin
                      sconf = (sconf_left << 1) | sconf_right
                      ibra = (dbra==dconf) & (sbra==sconf) & (tbra==tket[iket])
                      ibra = np.where (ibra)[0]
                      assert (len (ibra) == 1)
                      ibra = ibra[0]
                      db, sb, tb = dbra[ibra], sbra[ibra], tbra[ibra]
                      ci_ket = np.zeros (trans_k.ncsf)
                      ci_ket[iket] = 1.0
                      ci_ket = trans_k.vec_csf2det (ci_ket)
                      ci_bra = np.zeros (trans_b.ncsf)
                      ci_bra[ibra] = 1.0
                      ci_bra = trans_b.vec_csf2det (ci_bra)
                      ci1 = des_b (ci_ket, norb, nelec, iorb)
                      ci1 = des_a (ci1, norb, (nelec[0], nelec[1]-1), iorb)
                      ovlp = np.dot (ci_bra.ravel ().conj (), ci1.ravel ())
                      msg = f'<{db},{sb},{tb}|a{iorb}b{iorb}|{dk},{sk},{tk}> = {ovlp}'
                      self.assertAlmostEqual (ovlp, 1.0, 9, msg=msg)

if __name__ == "__main__":
    print("Full Tests for csf_fci solver")
    unittest.main()

