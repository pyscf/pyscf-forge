#!/usr/bin/env python
# Copyright 2026 The PySCF Developers. All Rights Reserved.
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

import unittest
import numpy as np
from pyscf.scf import hf as mol_hf
from pyscf.pbc import gto, scf
from pyscf.pbc.lno.tools import k2s_scf


@unittest.skipIf(not hasattr(mol_hf.SCF, 'check_linear_dependency'),
                 'requires pyscf with canonical orthogonalization in the SCF eigensolver')
class LiFLinearDependency(unittest.TestCase):
    ''' Rocksalt LiF in def2-SVPD: the k-point overlaps, and hence the supercell overlap,
        are numerically singular.
    '''
    @classmethod
    def setUpClass(cls):
        a0 = 4.03
        cell = gto.Cell()
        cell.atom = [['Li', (0,0,0)], ['F', (a0/2,0,0)]]
        cell.a = np.array([[0,1,1],[1,0,1],[1,1,0]]) * a0/2
        cell.basis = 'def2-svpd'
        cell.verbose = 0
        cell.build()

        # one hcore diagonalization with the canonical orthogonalization of the k-point SCF
        # stands in for the SCF; k2s_scf does not build the DF object
        kmf = scf.KRHF(cell, kpts=cell.make_kpts([2,1,1])).density_fit(auxbasis='weigend')
        h1e = kmf.get_hcore()
        s1e = kmf.get_ovlp()
        x = kmf.check_linear_dependency(s1e)
        kmf.mo_energy, kmf.mo_coeff = kmf.eig(h1e, s1e, x=x)
        kmf.mo_occ = kmf.get_occ(kmf.mo_energy)
        kmf.e_tot = 0.

        cls.cell = cell
        cls.kmf = kmf
        cls.nmo_k = [xk.shape[1] for xk in x]

    @classmethod
    def tearDownClass(cls):
        del cls.cell, cls.kmf, cls.nmo_k

    def test_k2s_scf_lindep(self):
        kmf = self.kmf
        nao = self.cell.nao * len(kmf.kpts)
        nmo = sum(self.nmo_k)
        self.assertLess(nmo, nao)

        # canonical orthogonalization drops the same overlap eigenvectors as the k-point SCF
        smf = k2s_scf(kmf)
        c = smf.mo_coeff
        self.assertEqual(c.shape, (nao, nmo))
        self.assertAlmostEqual(abs(c.T.dot(smf.get_ovlp()).dot(c) - np.eye(nmo)).max(), 0, 8)
        e_k = np.sort(np.hstack([e[:n] for e,n in zip(kmf.mo_energy, self.nmo_k)]))
        self.assertAlmostEqual(abs(np.sort(smf.mo_energy) - e_k).max(), 0, 8)
        self.assertEqual(np.count_nonzero(smf.mo_occ), np.count_nonzero(np.asarray(kmf.mo_occ)))


if __name__ == "__main__":
    print("Tests for k2s_scf")
    unittest.main()
