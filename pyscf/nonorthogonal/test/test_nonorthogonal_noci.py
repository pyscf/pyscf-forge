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
import numpy
from pyscf import gto, scf, ci, fci
from pyscf import nonorthogonal


def setUpModule():
    global mol, mf, refs
    mol = gto.M(atom=[('H', (0, 0, 1.8 * i)) for i in range(6)], basis='sto-3g',
                verbose=5, output='/dev/null')
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    refs = [mf]
    # Antiferromagnetic UHF solutions with opposite spin patterns
    for pattern in ((1, -1, 1, -1, 1, -1), (-1, 1, -1, 1, -1, 1)):
        uhf = scf.UHF(mol)
        uhf.conv_tol = 1e-12
        dm = uhf.get_init_guess()
        for iatm, spin in enumerate(pattern):
            ao = mol.search_ao_label('%d H' % iatm)
            major = 0 if spin > 0 else 1
            dm[major][ao, ao] *= 1.8
            dm[1 - major][ao, ao] *= .2
        uhf.kernel(dm)
        refs.append(uhf)


def tearDownModule():
    global mol, mf, refs
    mol.stdout.close()
    del mol, mf, refs


class KnownValues(unittest.TestCase):
    def test_cisd(self):
        # NOCISD of a single RHF reference is CISD
        mynoci = nonorthogonal.NOCISD(mol, [mf]).run()
        myci = ci.CISD(mf).run()
        self.assertAlmostEqual(mynoci.e_tot, myci.e_tot, 9)

    def test_fci(self):
        # Every excitation of one reference spans the FCI space
        e_fci = fci.FCI(mf).kernel(nroots=3)[0]
        mynoci = nonorthogonal.NOCI(mol, refs[:2], level=6)
        mynoci.nroots = 3
        e, c = mynoci.kernel()
        self.assertAlmostEqual(abs(e - e_fci).max(), 0, 9)
        self.assertEqual(c.shape, (len(mynoci.dets), 3))

    def test_nocisd(self):
        e_ref = nonorthogonal.NOCI(mol, refs).run().e_tot
        mynoci = nonorthogonal.NOCISD(mol, refs).run()
        e_fci = fci.FCI(mf).kernel()[0]
        self.assertTrue(e_fci < mynoci.e_tot < e_ref < refs[1].e_tot)
        self.assertAlmostEqual(mynoci.e_tot, -2.8875402412, 7)
        self.assertEqual(mynoci.level, 2)
        # the coefficients are normalised in the overlap metric
        h, s = mynoci.wicks.build_matrices(mynoci.dets)
        c = mynoci.ci
        self.assertAlmostEqual(c.conj().dot(s).dot(c), 1, 9)
        self.assertAlmostEqual(c.conj().dot(h).dot(c), mynoci.e_tot, 9)

    def test_nocisdt(self):
        e1 = nonorthogonal.NOCISDT(mol, refs).run().e_tot
        e2 = nonorthogonal.NOCI(mol, refs, level=3).run().e_tot
        self.assertAlmostEqual(e1, e2, 12)

    def test_excitations(self):
        mynoci = nonorthogonal.NOCI(mol, refs[:1], level=2).build()
        dets = nonorthogonal.excitations(mynoci.wicks, 0, 2)
        # 1 + 2 * 3*3 singles + (3*3)**2 alpha-beta + 2 * 3*3 same-spin doubles
        self.assertEqual(len(dets), 1 + 18 + 81 + 18)
        self.assertEqual([sum(d.rank) for d in dets], sorted(sum(d.rank) for d in dets))
        doubles = nonorthogonal.excitations(mynoci.wicks, 0, 2, ranks=(2,))
        self.assertEqual(len(doubles), 99)
        self.assertRaises(ValueError, nonorthogonal.excitations, mynoci.wicks, 0, 2, (3,))

    def test_custom_space(self):
        mynoci = nonorthogonal.NOCI(mol, refs)
        mynoci.build()
        dets = [nonorthogonal.Determinant(k) for k in range(len(refs))]
        e, c = mynoci.kernel(dets=dets)
        self.assertAlmostEqual(e, nonorthogonal.NOCI(mol, refs).kernel()[0], 12)
        self.assertEqual(len(c), 3)

    def test_lindep(self):
        # Two copies of a reference are linearly dependent
        mynoci = nonorthogonal.NOCI(mol, [refs[1], refs[1], refs[2]])
        mynoci.nroots = 3
        e, c = mynoci.kernel()
        self.assertEqual(len(e), 2)
        self.assertAlmostEqual(e[0], nonorthogonal.NOCI(mol, refs[1:]).kernel()[0], 9)


if __name__ == "__main__":
    print("Full Tests for NOCI")
    unittest.main()
