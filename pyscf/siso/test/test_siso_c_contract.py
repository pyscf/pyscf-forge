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
Test Description: SISO CI-vector contractions with spin operators. A Python
reference implementation is included to check the SISO contractions.

Test-1: Checking the accuracy of the same-spin contractions
Test-2: Checking spin-raising contractions
Test-3: Comparing spin-lowering contractions
Test-4: Validate accepted input shapes for the contraction drivers.
Test-5: Check spin-raising with an independent dummy-orbital contraction.
Test-6: Check spin-lowering with an independent dummy-orbital contraction.
Test-7: Check the spin-raising phase for an odd number of alpha electrons.
Test-8: Check the spin-lowering phase for an even number of alpha electrons.
'''

# Author: Bhavnesh Jangid

import numpy as np
import unittest
from numpy.testing import assert_allclose

from pyscf import fci
from pyscf.fci import direct_nosym
from pyscf.siso import siso


def contract_reference(h1e, ci0, link_indexa, link_indexb, nelec,
                       spin_change):
    nroots = ci0.shape[0]
    nstra1 = link_indexa.shape[0]
    nstrb1 = link_indexb.shape[0]
    ci1 = np.empty((3, nroots, nstra1, nstrb1), dtype=np.complex128)
    spin_phase = 1
    if spin_change > 0:
        spin_phase = (-1) ** nelec[0]
    elif spin_change < 0:
        spin_phase = (-1) ** (nelec[0] - 1)

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
                                value += (spin_phase * linka[3] * linkb[3]
                                          * h1e[component, p, q]
                                          * ci0[root, linka[2], linkb[2]])
                    ci1[component, root, stra, strb] = value
    return ci1


def _apply_ci_operator(operator, ci, *args):
    """Apply a PySCF CI operator without discarding complex coefficients.

    Some FCI creation and annihilation helpers allocate real output arrays.
    Applying them separately to the real and imaginary parts preserves a
    complex test vector without reproducing their fermionic sign logic.

    Returns:
        result: numpy.ndarray of complex, shape determined by ``operator``
            CI vector produced by the requested creation or annihilation.
    """
    if np.iscomplexobj(ci):
        return (operator(ci.real, *args)
                + 1j * operator(ci.imag, *args))
    return operator(ci, *args)


def _occupy_dummy_orbital(ci, norb, nelec, spin):
    """Append an orbital and occupy it with an alpha or beta electron.

    Returns:
        ci_dummy: numpy.ndarray, shape (nstra_dummy, nstrb_dummy)
            CI vector in ``norb + 1`` orbitals with the appended orbital
            occupied in the requested spin sector.
    """
    neleca, nelecb = nelec
    nstra = fci.cistring.num_strings(norb, neleca)
    nstrb = fci.cistring.num_strings(norb, nelecb)
    nstra_dummy = fci.cistring.num_strings(norb + 1, neleca)
    nstrb_dummy = fci.cistring.num_strings(norb + 1, nelecb)
    ci_unoccupied = np.zeros((nstra_dummy, nstrb_dummy), dtype=ci.dtype)
    ci_unoccupied[:nstra, :nstrb] = ci.reshape(nstra, nstrb)

    create = (fci.addons.cre_a, fci.addons.cre_b)[spin]
    return _apply_ci_operator(
        create, ci_unoccupied, norb + 1, nelec, norb)


def _project_occupied_dummy(ci, norb, nelec, spin):
    """Remove an occupied dummy orbital and return the physical CI block.

    Returns:
        ci_physical: numpy.ndarray, shape (nstra, nstrb)
            CI vector after annihilating the dummy electron and removing the
            appended orbital from the determinant space.
    """
    destroy = (fci.addons.des_a, fci.addons.des_b)[spin]
    ci_unoccupied = _apply_ci_operator(
        destroy, ci, norb + 1, nelec, norb)

    physical_nelec = list(nelec)
    physical_nelec[spin] -= 1
    nstra = fci.cistring.num_strings(norb, physical_nelec[0])
    nstrb = fci.cistring.num_strings(norb, physical_nelec[1])
    return ci_unoccupied[:nstra, :nstrb]


def contract_spin_flip_dummy_reference(h1e, ci0, norb, nelec,
                                       spin_change):
    r"""Contract S+ or S- using a dummy-orbital augmentation.

    One dummy orbital embeds the physical ket and bra sectors in a common
    fixed-(N_alpha, N_beta) Hilbert space. For S+, the ket carries a dummy
    alpha electron and the bra is projected onto a dummy beta electron; S-
    uses the reverse assignment. In the augmented space, the spin flip is
    represented by the ordinary two-electron operator

        ``-sum_pq h[q,p] E[p,d] E[d,q]``,

    where ``d`` is the dummy orbital. The minus sign follows from moving the
    dummy and physical fermion operators into normal order. This reference
    uses PySCF's general nonsymmetric two-electron contraction and does not
    reuse the SISO link tables or their sign/index logic.

    Args:
        h1e: numpy.ndarray of complex, shape (3, norb, norb)
            Three spherical SOC tensor components.
        ci0: numpy.ndarray of complex, shape (nroots, nstra, nstrb)
            Ket CI vectors in the physical orbital space.
        norb: int
            Number of physical spatial orbitals.
        nelec: tuple of two int
            Ket alpha and beta electron counts.
        spin_change: int
            ``+1`` applies beta-to-alpha S+; ``-1`` applies alpha-to-beta S-.

    Returns:
        ci1: numpy.ndarray of complex, shape (3, nroots, nstra_bra, nstrb_bra)
            Spin-flipped CI vectors after projecting out the dummy orbital.
    """
    if spin_change not in (-1, 1):
        raise ValueError("spin_change must be +1 or -1")

    neleca, nelecb = nelec
    bra_nelec = (neleca + spin_change, nelecb - spin_change)
    nstra_bra = fci.cistring.num_strings(norb, bra_nelec[0])
    nstrb_bra = fci.cistring.num_strings(norb, bra_nelec[1])
    ci1 = np.empty((h1e.shape[0], ci0.shape[0],
                    nstra_bra, nstrb_bra), dtype=np.complex128)

    ket_dummy_spin = 0 if spin_change > 0 else 1
    bra_dummy_spin = 1 - ket_dummy_spin
    augmented_nelec = list(nelec)
    augmented_nelec[ket_dummy_spin] += 1
    ci0_dummy = [
        _occupy_dummy_orbital(ci, norb, nelec, ket_dummy_spin)
        for ci in ci0
    ]

    dummy = norb
    for component, h1e_component in enumerate(h1e):
        eri2 = np.zeros((norb + 1,) * 4, dtype=np.complex128)
        eri2[:norb, dummy, dummy, :norb] = -h1e_component.T
        for root, ci_dummy in enumerate(ci0_dummy):
            contracted = direct_nosym.contract_2e(
                eri2, ci_dummy, norb + 1, augmented_nelec)
            ci1[component, root] = _project_occupied_dummy(
                contracted, norb, augmented_nelec, bra_dummy_spin)
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
            self.h1e, ci0, link_indexa, link_indexb, (2, 1), 0)
        ci1 = siso.contract_same_spin(
            self.h1e, ci0, (link_indexa, link_indexb))
        assert_allclose(ci1, ci1_ref, atol=1e-14)

    def test_spin_plus(self):
        link_indexa = fci.cistring.gen_des_str_index(range(self.norb), 3)
        link_indexb = fci.cistring.gen_cre_str_index(range(self.norb), 0)
        ci0 = self.make_ci((2, 6, 4), 14)
        ci1_ref = contract_reference(
            self.h1e, ci0, link_indexa, link_indexb, (2, 1), 1)
        ci1 = siso.contract_spin_plus(
            self.h1e, ci0, (link_indexa, link_indexb))
        assert_allclose(ci1, ci1_ref, atol=1e-14)

    def test_spin_minus(self):
        link_indexa = fci.cistring.gen_cre_str_index(range(self.norb), 2)
        link_indexb = fci.cistring.gen_des_str_index(range(self.norb), 1)
        ci0 = self.make_ci((2, 4, 1), 15)
        ci1_ref = contract_reference(
            self.h1e, ci0, link_indexa, link_indexb, (3, 0), -1)
        ci1 = siso.contract_spin_minus(
            self.h1e, ci0, (link_indexa, link_indexb))
        assert_allclose(ci1, ci1_ref, atol=1e-14)

    def test_spin_plus_with_dummy_orbital_reference(self):
        """Check S+ using an independent two-electron dummy-orbital path.

        The reference embeds the ket with an occupied dummy alpha orbital and
        projects the contracted vector onto an occupied dummy beta orbital.
        PySCF's generic two-electron driver supplies the fermionic signs, so a
        matching sign error in the SISO C and Python link loops cannot make
        this test pass spuriously.
        """
        nelec = (2, 1)
        link_indexa = fci.cistring.gen_des_str_index(range(self.norb), 3)
        link_indexb = fci.cistring.gen_cre_str_index(range(self.norb), 0)
        ci0 = self.make_ci((2, 6, 4), 17)

        ci1_ref = contract_spin_flip_dummy_reference(
            self.h1e, ci0, self.norb, nelec, spin_change=1)
        ci1 = siso.contract_spin_plus(
            self.h1e, ci0, (link_indexa, link_indexb))

        assert_allclose(ci1, ci1_ref, atol=1e-14)

    def test_spin_minus_with_dummy_orbital_reference(self):
        """Check S- using an independent two-electron dummy-orbital path.

        This is the reverse embedding of the S+ test: the ket carries a dummy
        beta electron and the bra is projected onto a dummy alpha electron.
        The resulting comparison tests the C contraction independently of its
        hand-written Python link-index analogue.
        """
        nelec = (3, 0)
        link_indexa = fci.cistring.gen_cre_str_index(range(self.norb), 2)
        link_indexb = fci.cistring.gen_des_str_index(range(self.norb), 1)
        ci0 = self.make_ci((2, 4, 1), 18)

        ci1_ref = contract_spin_flip_dummy_reference(
            self.h1e, ci0, self.norb, nelec, spin_change=-1)
        ci1 = siso.contract_spin_minus(
            self.h1e, ci0, (link_indexa, link_indexb))

        assert_allclose(ci1, ci1_ref, atol=1e-14)

    def test_spin_plus_dummy_orbital_odd_alpha_parity(self):
        """Check the canonical S+ phase for odd alpha-electron parity.

        The dummy-orbital contraction uses PySCF's full spin-orbital ordering
        and therefore provides an independent fermionic sign. This case
        requires the alpha-beta anticommutation phase that is absent from the
        separate alpha and beta link signs. Restoring it only rephases the
        adjacent-spin block and does not change the SISO energy spectrum.
        """
        nelec = (1, 1)
        bra_nelec = (2, 0)
        link_indexa = fci.cistring.gen_des_str_index(
            range(self.norb), bra_nelec[0])
        link_indexb = fci.cistring.gen_cre_str_index(
            range(self.norb), bra_nelec[1])
        ci0 = self.make_ci((2, 4, 4), 19)

        ci1_ref = contract_spin_flip_dummy_reference(
            self.h1e, ci0, self.norb, nelec, spin_change=1)
        ci1_link_ref = contract_reference(
            self.h1e, ci0, link_indexa, link_indexb, nelec, 1)
        ci1 = siso.contract_spin_plus(
            self.h1e, ci0, (link_indexa, link_indexb))

        assert_allclose(ci1_link_ref, ci1_ref, atol=1e-14)
        assert_allclose(ci1, ci1_ref, atol=1e-14)

    def test_spin_minus_dummy_orbital_even_alpha_parity(self):
        """Check the canonical S- phase for even alpha-electron parity.

        This is the reverse spin-flip check. With two alpha electrons in the
        ket, the beta creation operator crosses one remaining alpha electron,
        giving a minus sign that is independent of the determinant addresses.
        """
        nelec = (2, 0)
        bra_nelec = (1, 1)
        link_indexa = fci.cistring.gen_cre_str_index(
            range(self.norb), bra_nelec[0])
        link_indexb = fci.cistring.gen_des_str_index(
            range(self.norb), bra_nelec[1])
        ci0 = self.make_ci((2, 6, 1), 20)

        ci1_ref = contract_spin_flip_dummy_reference(
            self.h1e, ci0, self.norb, nelec, spin_change=-1)
        ci1_link_ref = contract_reference(
            self.h1e, ci0, link_indexa, link_indexb, nelec, -1)
        ci1 = siso.contract_spin_minus(
            self.h1e, ci0, (link_indexa, link_indexb))

        assert_allclose(ci1_link_ref, ci1_ref, atol=1e-14)
        assert_allclose(ci1, ci1_ref, atol=1e-14)

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
