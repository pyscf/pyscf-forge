#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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

# Author: Bhavnesh Jangid

'''
Test Description: SISO CI-vector contractions with spin operators. To test the
accuracy of the SISO CI-vector contractions, a reference python implementation
is also included in this test.

Test-1: Checking the accuracy of the same-spin contractions
Test-2: Checking spin-raising contractions
Test-3: Comparing spin-lowering contractions
Test-4: Check that the contraction drivers expose the expected names.
Test-5: Validate accepted input shapes for the contraction drivers.
'''


import numpy as np
import unittest
from numpy.testing import assert_allclose

from pyscf import fci
from pyscf.siso import siso


def contract_reference(h1e, ci0, link_indexa, link_indexb, spin_change):
    nroots = ci0.shape[0]
    nstra1 = link_indexa.shape[0]
    nstrb1 = link_indexb.shape[0]
    ci1 = np.empty((3, nroots, nstra1, nstrb1), dtype=np.complex128)

    for component in range(3):
        for root in range(nroots):
            for stra in range(nstra1):
                for strb in range(nstrb1):
                    value = 0j
                    if spin_change == 0:
                        for p, q, target, sign in link_indexa[stra]:
                            value += (sign * h1e[component, p, q]
                                      * ci0[root, target, strb])
                        for p, q, target, sign in link_indexb[strb]:
                            value -= (sign * h1e[component, p, q]
                                      * ci0[root, stra, target])
                    else:
                        for linka in link_indexa[stra]:
                            for linkb in link_indexb[strb]:
                                if spin_change > 0:
                                    p, q = linkb[0], linka[1]
                                else:
                                    p, q = linka[0], linkb[1]
                                value += (linka[3] * linkb[3]
                                          * h1e[component, p, q]
                                          * ci0[root, linka[2], linkb[2]])
                    ci1[component, root, stra, strb] = value
    return ci1


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(12)
        cls.norb = 4
        cls.h1e = (rng.standard_normal((3, cls.norb, cls.norb))
                   + 1j * rng.standard_normal((3, cls.norb, cls.norb)))

    @staticmethod
    def make_ci(shape, seed):
        rng = np.random.default_rng(seed)
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    def test_same_spin(self):
        link_indexa = fci.cistring.gen_linkstr_index(range(self.norb), 2)
        link_indexb = fci.cistring.gen_linkstr_index(range(self.norb), 1)
        ci0 = self.make_ci(
            (2, link_indexa.shape[0], link_indexb.shape[0]), 13)
        ci1_ref = contract_reference(
            self.h1e, ci0, link_indexa, link_indexb, 0)
        ci1 = siso.contract_same_spin(
            self.h1e, ci0, (link_indexa, link_indexb))
        assert_allclose(ci1, ci1_ref, atol=1e-14)

    def test_spin_plus(self):
        link_indexa = fci.cistring.gen_des_str_index(range(self.norb), 3)
        link_indexb = fci.cistring.gen_cre_str_index(range(self.norb), 0)
        ci0 = self.make_ci((2, 6, 4), 14)
        ci1_ref = contract_reference(
            self.h1e, ci0, link_indexa, link_indexb, 1)
        ci1 = siso.contract_spin_plus(
            self.h1e, ci0, (link_indexa, link_indexb))
        assert_allclose(ci1, ci1_ref, atol=1e-14)

    def test_spin_minus(self):
        link_indexa = fci.cistring.gen_cre_str_index(range(self.norb), 2)
        link_indexb = fci.cistring.gen_des_str_index(range(self.norb), 1)
        ci0 = self.make_ci((2, 4, 1), 15)
        ci1_ref = contract_reference(
            self.h1e, ci0, link_indexa, link_indexb, -1)
        ci1 = siso.contract_spin_minus(
            self.h1e, ci0, (link_indexa, link_indexb))
        assert_allclose(ci1, ci1_ref, atol=1e-14)

    def test_driver_names(self):
        self.assertTrue(hasattr(siso.libsiso, 'SISOcontract_same_spin'))
        self.assertTrue(hasattr(siso.libsiso, 'SISOcontract_spin_plus'))
        self.assertTrue(hasattr(siso.libsiso, 'SISOcontract_spin_minus'))

    def test_input_shapes(self):
        link_indexa = fci.cistring.gen_linkstr_index(range(self.norb), 2)
        link_indexb = fci.cistring.gen_linkstr_index(range(self.norb), 1)
        ci0 = self.make_ci(
            (1, link_indexa.shape[0], link_indexb.shape[0]), 16)

        with self.assertRaisesRegex(ValueError, 'SOC integrals'):
            siso.contract_same_spin(
                self.h1e[0], ci0, (link_indexa, link_indexb))
        with self.assertRaisesRegex(ValueError, 'CI vectors'):
            siso.contract_same_spin(
                self.h1e, ci0[0], (link_indexa, link_indexb))
        with self.assertRaisesRegex(ValueError, 'alpha link table'):
            siso.contract_same_spin(
                self.h1e, ci0, (link_indexa[..., :3], link_indexb))


if __name__ == '__main__':
    unittest.main()
