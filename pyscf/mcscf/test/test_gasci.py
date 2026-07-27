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
# Author: Yi Deng <yideng@uchicago.edu>
#

"""Tests for GAS restriction handling, the GAS FCI solver, and GASCI."""

import unittest

import numpy

from pyscf import gto
from pyscf import mcscf
from pyscf import scf
from pyscf.fci import addons as fci_addons
from pyscf.fci import cistring
from pyscf.fci import direct_spin1
from pyscf.mcscf import addons_gas
from pyscf.mcscf import fci_gas
from pyscf.mcscf import gasci


def make_integrals(norb):
    """Return deterministic one- and two-electron test integrals."""

    h1e = numpy.empty((norb, norb))
    for p in range(norb):
        for q in range(norb):
            h1e[p, q] = (
                0.07 * (p + q + 2) +
                0.013 * ((p + 1) * (q + 3) % 7))
    h1e = (h1e + h1e.T) * 0.5

    npair = norb * (norb + 1) // 2
    eri = numpy.empty((npair, npair))
    for pq in range(npair):
        for rs in range(npair):
            lo, hi = min(pq, rs), max(pq, rs)
            eri[pq, rs] = 0.011 * (lo + 1) + 0.003 * (hi + 2)
    return h1e, eri


class TestGASRestrictions(unittest.TestCase):

    def test_nelec_spin_inference(self):
        cases = (
            (5, None, (3, 2)),
            (5, 1, (3, 2)),
            (5, 3, (4, 1)),
            (5, -1, (2, 3)),
            ((3, 2), None, (3, 2)),
            ((2, 2), 2, (3, 1)),
        )
        for nelec, spin, expected in cases:
            with self.subTest(nelec=nelec, spin=spin):
                self.assertEqual(
                    addons_gas.as_nelec_tuple(nelec, spin), expected)

        for nelec, spin in ((4, 1), (4, 6), (4, True), (4, 1.5)):
            with self.subTest(nelec=nelec, spin=spin):
                with self.assertRaises((TypeError, ValueError)):
                    addons_gas.as_nelec_tuple(nelec, spin)

        solver = fci_gas.FCISolver()
        solver.spin = 2
        self.assertEqual(solver._space_spec(4, 4)[1], (3, 1))
        self.assertEqual(solver.space_info(4, 4)["ndet_estimate"], 16)

    def test_equivalent_restriction_types(self):
        gas_orbs = (2, 2)
        nelec = (2, 2)
        expected = numpy.asarray([
            [0, 2, 2, 0],
            [1, 1, 1, 1],
            [2, 0, 0, 2],
        ], dtype=numpy.int32)
        explicit = numpy.asarray([
            [2, 0, 0, 2],
            [1, 1, 1, 1],
            [0, 2, 2, 0],
            [1, 1, 1, 1],
        ], dtype=numpy.int32)

        inputs = (
            ("spin-supergroup", explicit),
            ("supergroup", [[2, 2]]),
            ("cumulative-occ", [[2, 2], [4, 4]]),
        )
        for restriction_type, restriction in inputs:
            with self.subTest(restriction_type=restriction_type):
                blocks = addons_gas.normalize_gas_restr(
                    gas_orbs, nelec, restriction, restriction_type)
                numpy.testing.assert_array_equal(blocks, expected)

        self.assertTrue(
            addons_gas.is_spin_complete(gas_orbs, nelec, expected))

    def test_ras_matches_cumulative_bounds(self):
        gas_orbs = (1, 2, 1)
        nelec = (2, 2)
        ras = {"max_holes": 1, "max_particles": 1}
        cumulative = [[1, 2], [3, 4], [4, 4]]

        kernel_orbs, ras_blocks = addons_gas.normalize_gas_spec(
            gas_orbs, nelec, ras, "ras")
        cumulative_blocks = addons_gas.normalize_gas_restr(
            gas_orbs, nelec, cumulative, "cumulative-occ")

        self.assertEqual(kernel_orbs, gas_orbs)
        numpy.testing.assert_array_equal(ras_blocks, cumulative_blocks)


class TestGASFCISolver(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.norb = 4
        cls.nelec = (2, 2)
        cls.h1e, cls.eri = make_integrals(cls.norb)
        cls.nstr = cistring.num_strings(cls.norb, cls.nelec[0])

        rng = numpy.random.default_rng(12)
        cls.ci = rng.normal(size=cls.nstr**2)
        cls.ci /= numpy.linalg.norm(cls.ci)
        cls.ket = rng.normal(size=cls.nstr**2)
        cls.ket /= numpy.linalg.norm(cls.ket)

    def test_gas_as_cas_hamiltonian(self):
        solver = fci_gas.FCISolver()
        hdiag = solver.make_hdiag(
            self.h1e, self.eri, self.norb, self.nelec)
        reference_hdiag = direct_spin1.make_hdiag(
            self.h1e, self.eri, self.norb, self.nelec)
        numpy.testing.assert_allclose(
            hdiag, reference_hdiag, atol=1e-12, rtol=0)

        h2e = fci_gas.absorb_h1e(
            self.h1e, self.eri, self.norb, self.nelec, fac=0.5)
        hc = solver.contract_2e(
            h2e, self.ci, self.norb, self.nelec)
        reference_hc = direct_spin1.contract_2e(
            h2e, self.ci.reshape(self.nstr, self.nstr),
            self.norb, self.nelec).reshape(-1)
        numpy.testing.assert_allclose(
            hc, reference_hc, atol=1e-11, rtol=0)

        energy = solver.energy(
            self.h1e, self.eri, self.ci, self.norb, self.nelec)
        self.assertAlmostEqual(energy, numpy.dot(self.ci, hc), places=11)

    def test_gas_as_cas_rdms(self):
        solver = fci_gas.FCISolver()
        ci = self.ci.reshape(self.nstr, self.nstr)
        ket = self.ket.reshape(self.nstr, self.nstr)

        dm1, dm2 = solver.make_rdm12(
            self.ci, self.norb, self.nelec)
        reference_dm1, reference_dm2 = direct_spin1.make_rdm12(
            ci, self.norb, self.nelec)
        numpy.testing.assert_allclose(
            dm1, reference_dm1, atol=1e-12, rtol=0)
        numpy.testing.assert_allclose(
            dm2, reference_dm2, atol=1e-12, rtol=0)

        tdm1, tdm2 = solver.trans_rdm12(
            self.ci, self.ket, self.norb, self.nelec)
        reference_tdm1, reference_tdm2 = direct_spin1.trans_rdm12(
            ci, ket, self.norb, self.nelec)
        numpy.testing.assert_allclose(
            tdm1, reference_tdm1, atol=1e-12, rtol=0)
        numpy.testing.assert_allclose(
            tdm2, reference_tdm2, atol=1e-12, rtol=0)

    def test_restricted_space_and_spin_guards(self):
        gas_orbs = (2, 2)
        nelec = (2, 1)
        blocks = numpy.asarray([[1, 1, 1, 0]], dtype=numpy.int32)
        solver = fci_gas.FCISolver(
            gas_orbs=gas_orbs, gas_restr=blocks)
        h1e = numpy.diag((-1.0, -0.6, 0.2, 0.7))
        eri = numpy.zeros((10, 10))

        info = solver.space_info(4, nelec)
        self.assertEqual(info["ndet_estimate"], 8)
        energies, roots = solver.kernel(
            h1e, eri, 4, nelec, nroots=2, tol=1e-12)
        self.assertEqual(numpy.asarray(energies).shape, (2,))
        self.assertEqual(len(roots), 2)

        root = numpy.asarray(roots[0]).reshape(-1)
        (dm1a, dm1b), _ = solver.make_rdm12s(root, 4, nelec)
        self.assertAlmostEqual(numpy.trace(dm1a), nelec[0], places=11)
        self.assertAlmostEqual(numpy.trace(dm1b), nelec[1], places=11)
        with self.assertRaises(ValueError):
            solver.contract_ss(root, 4, nelec)

        fci_addons.fix_spin_(solver, shift=0.2, ss=0.75)
        with self.assertRaises(ValueError):
            solver.kernel(h1e, eri, 4, nelec)


class TestGASCI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(
            atom="N 0 0 0; N 0 0 1.10",
            basis="sto-3g",
            spin=0,
            verbose=0,
        )
        cls.mf = scf.RHF(cls.mol).run()

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf

    def test_gas_as_cas(self):
        reference = mcscf.CASCI(
            self.mf, 4, (2, 2), ncore=5)
        reference.verbose = 0
        reference.canonicalization = False
        reference.kernel()

        mc = gasci.GASCI(
            self.mf, 4, (2, 2), ncore=5)
        mc.verbose = 0
        mc.canonicalization = False
        mc.kernel()

        self.assertAlmostEqual(mc.e_tot, reference.e_tot, places=9)
        self.assertEqual(mc.gas_space_info()["core"]["ndet"], 36)
        numpy.testing.assert_allclose(
            mc.make_gasdm1(), reference.fcisolver.make_rdm1(
                reference.ci, 4, (2, 2)),
            atol=1e-9, rtol=0)

    def test_restricted_multiroot_rdms(self):
        mc = gasci.GASCI(
            self.mf, 4, (2, 2), ncore=5,
            gas_orbs=(2, 2),
            gas_restr=[[2, 2]],
            gas_restr_type="supergroup",
        )
        mc.verbose = 0
        mc.canonicalization = False
        mc.fcisolver.nroots = 3
        mc.kernel()

        energies = numpy.asarray(mc.e_tot).reshape(-1)
        self.assertEqual(energies.shape, (3,))
        self.assertEqual(len(mc.ci), 3)
        self.assertEqual(mc.gas_space_info()["core"]["ndet"], 18)

        for state in range(3):
            dm1, dm2 = mc.make_gasdm12(state=state)
            self.assertAlmostEqual(numpy.trace(dm1), 4.0, places=9)
            self.assertAlmostEqual(
                numpy.einsum("pprr", dm2), 12.0, places=8)
            ss, multiplicity = mc.spin_square(state=state)
            self.assertTrue(numpy.isfinite(ss))
            self.assertTrue(numpy.isfinite(multiplicity))

        tdm1, tdm2 = mc.trans_gasdm12(
            bra_state=0, ket_state=1)
        reverse1, reverse2 = mc.trans_gasdm12(
            bra_state=1, ket_state=0)
        numpy.testing.assert_allclose(
            tdm1, reverse1.T, atol=1e-10, rtol=0)
        numpy.testing.assert_allclose(
            tdm2, reverse2.transpose(1, 0, 3, 2),
            atol=1e-10, rtol=0)

    def test_state_average(self):
        weights = numpy.asarray([0.6, 0.4])
        mc = gasci.GASCI(
            self.mf, 4, (2, 2), ncore=5,
            gas_orbs=(2, 2),
            gas_restr=[[2, 2]],
            gas_restr_type="supergroup",
        ).state_average_(weights)
        mc.verbose = 0
        mc.canonicalization = False
        mc.kernel()

        numpy.testing.assert_allclose(
            mc.e_tot, numpy.dot(weights, mc.e_states),
            atol=1e-10, rtol=0)
        reference_dm1 = sum(
            weights[state] * mc.make_gasdm1(state=state)
            for state in range(2))
        numpy.testing.assert_allclose(
            mc.make_gasdm1(), reference_dm1, atol=1e-10, rtol=0)


if __name__ == "__main__":
    unittest.main()
