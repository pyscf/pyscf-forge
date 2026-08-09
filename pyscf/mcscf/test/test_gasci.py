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

import io
import unittest

import numpy

from pyscf import gto
from pyscf import mcscf
from pyscf import scf
from pyscf.fci import addons as fci_addons
from pyscf.fci import cistring
from pyscf.fci import direct_spin1
from pyscf.fci import spin_op
from pyscf.lib import logger
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
        solver = fci_gas.FCISolver()
        for nelec, spin, expected in cases:
            with self.subTest(nelec=nelec, spin=spin):
                solver.spin = spin
                unpacked = fci_addons._unpack_nelec(nelec, spin)
                self.assertEqual(solver._space_spec(4, nelec)[1], unpacked)
                self.assertEqual(unpacked, expected)

        solver.spin = 2
        self.assertEqual(solver._space_spec(4, 4)[1], (3, 1))
        self.assertEqual(solver.space_info(4, 4)["ndet_estimate"], 16)

    def test_integer_inputs_reject_floats(self):
        nelec = (2, 2)

        invalid = (
            ((2.0, 2), [[2, 2]], "supergroup"),
            ((2, 2), [[2.0, 2]], "supergroup"),
            ((2, 2), [[2, 2], [4, 4.0]], "cumulative-occ"),
            ((1, 2, 1), {"max_holes": 1.0, "max_particles": 1}, "ras"),
            ((2, True), [[2, 2]], "supergroup"),
        )
        for gas_orbs, gas_restr, gas_restr_type in invalid:
            with self.subTest(
                    gas_orbs=gas_orbs, gas_restr_type=gas_restr_type):
                with self.assertRaises(TypeError):
                    addons_gas.normalize_gas_restr(
                        gas_orbs, nelec, gas_restr, gas_restr_type)

        blocks = addons_gas.normalize_gas_restr(
            numpy.asarray([2, 2], dtype=numpy.int64),
            nelec,
            numpy.asarray([[2, 2]], dtype=numpy.int64),
            "supergroup",
        )
        numpy.testing.assert_array_equal(
            blocks,
            numpy.asarray([[0, 2, 2, 0],
                           [1, 1, 1, 1],
                           [2, 0, 0, 2]], dtype=numpy.int32),
        )

        for base in (0.0, True):
            with self.subTest(base=base):
                with self.assertRaises(TypeError):
                    addons_gas._normalize_gaslst(
                        [[0], [1]], (1, 1), 2, base)

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

    def test_init_guess_reuses_existing_gas_space(self):
        solver = fci_gas.FCISolver(
            gas_orbs=(1, 2, 1),
            gas_restr=[[1, 2, 1], [2, 1, 1], [1, 1, 2]],
            gas_restr_type="supergroup")
        hdiag = solver.make_hdiag(
            self.h1e, self.eri, self.norb, self.nelec)

        reference = solver.get_init_guess(
            self.norb, self.nelec, 2, hdiag)
        with solver.make_space(
                self.norb, self.nelec, compress_links=False) as gas:
            reused = solver.get_init_guess(
                self.norb, self.nelec, 2, hdiag, gas=gas)

        self.assertEqual(len(reference), len(reused))
        for expected, actual in zip(reference, reused):
            numpy.testing.assert_allclose(actual, expected, atol=0, rtol=0)

    def test_contract_plan_reuse(self):
        solver = fci_gas.FCISolver(
            gas_orbs=(1, 2, 1),
            gas_restr=[[1, 2, 1], [2, 1, 1], [1, 1, 2]],
            gas_restr_type="supergroup")
        h2e = fci_gas.absorb_h1e(
            self.h1e, self.eri, self.norb, self.nelec, fac=0.5)

        with solver.make_space(
                self.norb, self.nelec, compress_links=True) as gas:
            rng = numpy.random.default_rng(31)
            vectors = (
                rng.normal(size=gas.ndet),
                rng.normal(size=gas.ndet),
            )
            with fci_gas.GasContractPlan(gas, h2e) as plan:
                for ci in vectors:
                    planned = plan.contract(ci)
                    through_solver = solver.contract_2e(
                        h2e, ci, self.norb, self.nelec, plan=plan)
                    one_shot = solver.contract_2e(
                        h2e, ci, self.norb, self.nelec)
                    numpy.testing.assert_allclose(
                        planned, through_solver, atol=1e-12, rtol=0)
                    numpy.testing.assert_allclose(
                        planned, one_shot, atol=1e-12, rtol=0)

    def test_gas_fci_vector_converters(self):
        gas_orbs = (2, 2)
        nelec = (2, 1)
        blocks = numpy.asarray([[1, 1, 1, 0]], dtype=numpy.int32)
        solver = fci_gas.FCISolver(
            gas_orbs=gas_orbs,
            gas_restr=blocks,
            gas_restr_type="spin-supergroup")

        with solver.make_space(self.norb, nelec) as gas:
            nstr_alpha = cistring.num_strings(self.norb, nelec[0])
            nstr_beta = cistring.num_strings(self.norb, nelec[1])
            self.assertLess(gas.ndet, nstr_alpha * nstr_beta)

            rng = numpy.random.default_rng(19)
            gas_ci = rng.normal(size=gas.ndet)
            fci_ci = fci_gas.gas2fci(gas_ci, gas)
            self.assertEqual(fci_ci.shape, (nstr_alpha, nstr_beta))
            numpy.testing.assert_array_equal(
                fci_gas.fci2gas(fci_ci, gas), gas_ci)

            full_ci = numpy.arange(
                1, nstr_alpha * nstr_beta + 1, dtype=numpy.float64
            ).reshape(nstr_alpha, nstr_beta)
            projected = fci_gas.gas2fci(
                fci_gas.fci2gas(full_ci, gas), gas)
            projected_twice = fci_gas.gas2fci(
                fci_gas.fci2gas(projected, gas), gas)
            numpy.testing.assert_array_equal(projected_twice, projected)
            self.assertEqual(numpy.count_nonzero(projected), gas.ndet)

    def test_restricted_hamiltonian_projection(self):
        cases = (
            {
                "name": "spin-supergroup",
                "gas_orbs": (2, 2),
                "nelec": (2, 1),
                "gas_restr": numpy.asarray(
                    [[1, 1, 1, 0]], dtype=numpy.int32),
                "gas_restr_type": "spin-supergroup",
            },
            {
                "name": "supergroup",
                "gas_orbs": (1, 2, 1),
                "nelec": (2, 2),
                "gas_restr": [[1, 2, 1], [2, 1, 1], [1, 1, 2]],
                "gas_restr_type": "supergroup",
            },
            {
                "name": "ras",
                "gas_orbs": (1, 2, 1),
                "nelec": (2, 2),
                "gas_restr": {"max_holes": 1, "max_particles": 1},
                "gas_restr_type": "ras",
            },
        )

        rng = numpy.random.default_rng(23)
        for case in cases:
            with self.subTest(case=case["name"]):
                nelec = case["nelec"]
                solver = fci_gas.FCISolver(
                    gas_orbs=case["gas_orbs"],
                    gas_restr=case["gas_restr"],
                    gas_restr_type=case["gas_restr_type"])
                h1e, eri = make_integrals(self.norb)
                h2e = fci_gas.absorb_h1e(
                    h1e, eri, self.norb, nelec, fac=0.5)

                with solver.make_space(self.norb, nelec) as gas:
                    nstr_alpha = cistring.num_strings(self.norb, nelec[0])
                    nstr_beta = cistring.num_strings(self.norb, nelec[1])
                    self.assertLess(gas.ndet, nstr_alpha * nstr_beta)

                    gas_ci = rng.normal(size=gas.ndet)
                    gas_ci /= numpy.linalg.norm(gas_ci)
                    fci_ci = fci_gas.gas2fci(gas_ci, gas)

                    gas_hc = solver.contract_2e(
                        h2e, gas_ci, self.norb, nelec)
                    fci_hc = direct_spin1.contract_2e(
                        h2e, fci_ci, self.norb, nelec)
                    projected_hc = fci_gas.fci2gas(fci_hc, gas)
                    numpy.testing.assert_allclose(
                        gas_hc, projected_hc, atol=1e-11, rtol=0)

    def test_spin_square_projection(self):
        cases = (
            {
                "name": "supergroup",
                "gas_orbs": (1, 2, 1),
                "nelec": (2, 2),
                "gas_restr": [[1, 2, 1], [2, 1, 1], [1, 1, 2]],
                "gas_restr_type": "supergroup",
            },
            {
                "name": "ras",
                "gas_orbs": (1, 2, 1),
                "nelec": (2, 2),
                "gas_restr": {"max_holes": 1, "max_particles": 1},
                "gas_restr_type": "ras",
            },
        )

        rng = numpy.random.default_rng(29)
        for case in cases:
            with self.subTest(case=case["name"]):
                nelec = case["nelec"]
                solver = fci_gas.FCISolver(
                    gas_orbs=case["gas_orbs"],
                    gas_restr=case["gas_restr"],
                    gas_restr_type=case["gas_restr_type"])

                with solver.make_space(self.norb, nelec) as gas:
                    self.assertTrue(addons_gas.is_spin_complete(
                        gas.norb, nelec, gas.blocks))

                    gas_ci = rng.normal(size=gas.ndet)
                    gas_ci /= numpy.linalg.norm(gas_ci)
                    fci_ci = fci_gas.gas2fci(gas_ci, gas)

                    gas_sc = solver.contract_ss(
                        gas_ci, self.norb, nelec)
                    fci_sc = spin_op.contract_ss(
                        fci_ci, self.norb, nelec)
                    projected_sc = fci_gas.fci2gas(fci_sc, gas)
                    numpy.testing.assert_allclose(
                        gas_sc, projected_sc, atol=1e-12, rtol=0)

                    embedded_sc = fci_gas.gas2fci(gas_sc, gas)
                    numpy.testing.assert_allclose(
                        embedded_sc, fci_sc, atol=1e-12, rtol=0)

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

    def test_spin_free_rdm_assembly(self):
        solver = fci_gas.FCISolver()
        with solver.make_rdm_plan(self.norb, self.nelec) as plan:
            dm1a, dm1b = plan.make_rdm1s(self.ci, self.ket)
            dm1 = plan.make_rdm1(self.ci, self.ket)
            numpy.testing.assert_allclose(
                dm1, dm1a + dm1b, atol=1e-12, rtol=0)

            (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = plan.make_rdm12s(
                self.ci, self.ket)
            dm1, dm2 = plan.make_rdm12(self.ci, self.ket)
            expected_dm1 = dm1a + dm1b
            expected_dm2 = dm2aa + dm2bb
            expected_dm2 += dm2ab
            expected_dm2 += dm2ab.transpose(2, 3, 0, 1)
            numpy.testing.assert_allclose(
                dm1, expected_dm1, atol=1e-12, rtol=0)
            numpy.testing.assert_allclose(
                dm2, expected_dm2, atol=1e-12, rtol=0)

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


# Fixed-orbital NO+ data imported from an OpenMolcas HDF5 file.

no_plus_mol = (
    '{"atom": "[[\'N1\', [0.0, 0.0, 0.0]], [\'O2\', [0.0, 0.0, 2.1746968256946757]]]"'
    ', "basis": "{\'N1\': [[0, [19730.800647, 0.00021887984991, 0.0, 0.0, 0.0, 0.0]'
    ', [2957.8958745, 0.0016960708803, 0.0, 0.0, 0.0, 0.0], [673.22133595, 0.0087'
    '954603538, 0.0, 0.0, 0.0, 0.0], [190.68249494, 0.035359382605, 0.0, 0.0, 0.0'
    ', 0.0], [62.295441898, 0.11095789217, 0.0, 0.0, 0.0, 0.0], [22.654161182, 0.'
    '24982972552, 0.0, 0.0, 0.0, 0.0], [8.9791477428, 0.0, 0.40623896148, 0.0, 0.'
    '0, 0.0], [3.686300237, 0.0, 0.24338217176, 0.0, 0.0, 0.0], [0.84660076805, 0'
    '.0, 0.0, 1.0, 0.0, 0.0], [0.33647133771, 0.0, 0.0, 0.0, 1.0, 0.0], [0.136476'
    '53675, 0.0, 0.0, 0.0, 0.0, 1.0]], [1, [49.20038051, 0.0055552416751, 0.0, 0.'
    '0], [11.346790537, 0.038052379723, 0.0, 0.0], [3.4273972411, 0.14953671029, '
    '0.0, 0.0], [1.1785525134, 0.3494930523, 0.0, 0.0], [0.41642204972, 0.0, 0.45'
    '843153697, 0.0], [0.14260826011, 0.0, 0.0, 0.24428771672]], [2, [1.654, 1.0,'
    " 0.0], [0.469, 0.0, 1.0]], [3, [1.093, 1.0]]], 'O2': [[0, [27032.382631, 0.0"
    '0021726302465, 0.0, 0.0, 0.0, 0.0], [4052.3871392, 0.0016838662199, 0.0, 0.0'
    ', 0.0, 0.0], [922.3272271, 0.0087395616265, 0.0, 0.0, 0.0, 0.0], [261.240709'
    '89, 0.035239968808, 0.0, 0.0, 0.0, 0.0], [85.354641351, 0.11153519115, 0.0, '
    '0.0, 0.0, 0.0], [31.035035245, 0.25588953961, 0.0, 0.0, 0.0, 0.0], [12.26086'
    '0728, 0.0, 0.39768730901, 0.0, 0.0, 0.0], [4.9987076005, 0.0, 0.2462784943, '
    '0.0, 0.0, 0.0], [1.1703108158, 0.0, 0.0, 1.0, 0.0, 0.0], [0.46474740994, 0.0'
    ', 0.0, 0.0, 1.0, 0.0], [0.18504536357, 0.0, 0.0, 0.0, 0.0, 1.0]], [1, [63.27'
    '4954801, 0.0060685103418, 0.0, 0.0], [14.627049379, 0.041912575824, 0.0, 0.0'
    '], [4.4501223456, 0.16153841088, 0.0, 0.0], [1.5275799647, 0.35706951311, 0.'
    '0, 0.0], [0.52935117943, 0.0, 0.44794207502, 0.0], [0.1747842127, 0.0, 0.0, '
    '0.24446069663]], [2, [2.314, 1.0, 0.0], [0.645, 0.0, 1.0]], [3, [1.428, 1.0]'
    ']]}", "nucmod": {}, "ecp": "{}", "nucprop": {}, "magmom": [0, 0], "pseudo": '
    '"None", "_atm": [[7, 20, 1, 23, 0, 0], [8, 24, 1, 27, 0, 0]], "_bas": [[0, 0'
    ', 11, 5, 0, 28, 39, 0], [0, 1, 6, 3, 0, 94, 100, 0], [0, 2, 2, 2, 0, 118, 12'
    '0, 0], [0, 3, 1, 1, 0, 124, 125, 0], [1, 0, 11, 5, 0, 126, 137, 0], [1, 1, 6'
    ', 3, 0, 192, 198, 0], [1, 2, 2, 2, 0, 216, 218, 0], [1, 3, 1, 1, 0, 222, 223'
    ', 0]], "_env": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, '
    '0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.1746'
    '968256946757, 0.0, 19730.800647, 2957.8958745, 673.22133595, 190.68249494, 6'
    '2.295441898, 22.654161182, 8.9791477428, 3.686300237, 0.84660076805, 0.33647'
    '133771, 0.13647653675, 2.4658508374156667, 4.6034522992919324, 7.86646232650'
    '279, 12.278357689028272, 16.649564930304884, 17.555273048650747, 0.0, 0.0, 0'
    '.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.465479929725005, 2.60121028934'
    '62696, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.229841622743'
    '3675, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1161590381537'
    '6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5672944296303319'
    ', 49.20038051, 11.346790537, 3.4273972411, 1.1785525134, 0.41642204972, 0.14'
    '260826011, 4.352170395048921, 4.764479639730067, 4.192718762589188, 2.580281'
    '4243677257, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9758915092644845, 0.0, 0.0, 0.0,'
    ' 0.0, 0.0, 0.0, 0.25566144791664475, 1.654, 0.469, 6.294576559932722, 0.0, 0'
    '.0, 0.693556523160707, 1.093, 2.4093823583303675, 27032.382631, 4052.3871392'
    ', 922.3272271, 261.24070989, 85.354641351, 31.035035245, 12.260860728, 4.998'
    '7076005, 1.1703108158, 0.46474740994, 0.18504536357, 3.048118116984593, 5.69'
    '1457632864213, 9.733883574443253, 15.238733819733028, 20.843228934131737, 22'
    '.39104905999299, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.5'
    '68131135849375, 3.3391469496791393, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, '
    '0.0, 0.0, 0.0, 2.8427648592056753, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0'
    '.0, 0.0, 0.0, 1.422092211265869, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0'
    ', 0.0, 0.0, 0.7128098301044613, 63.274954801, 14.627049379, 4.4501223456, 1.'
    '5275799647, 0.52935117943, 0.1747842127, 6.257032374789428, 6.92686562359984'
    '2, 6.032359926541528, 3.5035168827833356, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.31'
    '7237993956345, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3296948367394935, 2.314, 0.64'
    '5, 11.32831343293501, 0.0, 0.0, 1.2113199965714336, 1.428, 4.396922678265652'
    '], "_ecpbas": [], "groupname": "C1", "topgroup": "C1", "_symm_orig": null, "'
    '_symm_axes": null, "_nelectron": null, "_nao": null, "_enuc": null, "_atom":'
    ' [["N1", [0.0, 0.0, 0.0]], ["O2", [0.0, 0.0, 2.1746968256946757]]], "_basis"'
    ': {"N1": [[0, [19730.800647, 0.00021887984991, 0.0, 0.0, 0.0, 0.0], [2957.89'
    '58745, 0.0016960708803, 0.0, 0.0, 0.0, 0.0], [673.22133595, 0.0087954603538,'
    ' 0.0, 0.0, 0.0, 0.0], [190.68249494, 0.035359382605, 0.0, 0.0, 0.0, 0.0], [6'
    '2.295441898, 0.11095789217, 0.0, 0.0, 0.0, 0.0], [22.654161182, 0.2498297255'
    '2, 0.0, 0.0, 0.0, 0.0], [8.9791477428, 0.0, 0.40623896148, 0.0, 0.0, 0.0], ['
    '3.686300237, 0.0, 0.24338217176, 0.0, 0.0, 0.0], [0.84660076805, 0.0, 0.0, 1'
    '.0, 0.0, 0.0], [0.33647133771, 0.0, 0.0, 0.0, 1.0, 0.0], [0.13647653675, 0.0'
    ', 0.0, 0.0, 0.0, 1.0]], [1, [49.20038051, 0.0055552416751, 0.0, 0.0], [11.34'
    '6790537, 0.038052379723, 0.0, 0.0], [3.4273972411, 0.14953671029, 0.0, 0.0],'
    ' [1.1785525134, 0.3494930523, 0.0, 0.0], [0.41642204972, 0.0, 0.45843153697,'
    ' 0.0], [0.14260826011, 0.0, 0.0, 0.24428771672]], [2, [1.654, 1.0, 0.0], [0.'
    '469, 0.0, 1.0]], [3, [1.093, 1.0]]], "O2": [[0, [27032.382631, 0.00021726302'
    '465, 0.0, 0.0, 0.0, 0.0], [4052.3871392, 0.0016838662199, 0.0, 0.0, 0.0, 0.0'
    '], [922.3272271, 0.0087395616265, 0.0, 0.0, 0.0, 0.0], [261.24070989, 0.0352'
    '39968808, 0.0, 0.0, 0.0, 0.0], [85.354641351, 0.11153519115, 0.0, 0.0, 0.0, '
    '0.0], [31.035035245, 0.25588953961, 0.0, 0.0, 0.0, 0.0], [12.260860728, 0.0,'
    ' 0.39768730901, 0.0, 0.0, 0.0], [4.9987076005, 0.0, 0.2462784943, 0.0, 0.0, '
    '0.0], [1.1703108158, 0.0, 0.0, 1.0, 0.0, 0.0], [0.46474740994, 0.0, 0.0, 0.0'
    ', 1.0, 0.0], [0.18504536357, 0.0, 0.0, 0.0, 0.0, 1.0]], [1, [63.274954801, 0'
    '.0060685103418, 0.0, 0.0], [14.627049379, 0.041912575824, 0.0, 0.0], [4.4501'
    '223456, 0.16153841088, 0.0, 0.0], [1.5275799647, 0.35706951311, 0.0, 0.0], ['
    '0.52935117943, 0.0, 0.44794207502, 0.0], [0.1747842127, 0.0, 0.0, 0.24446069'
    '663]], [2, [2.314, 1.0, 0.0], [0.645, 0.0, 1.0]], [3, [1.428, 1.0]]]}, "_ecp'
    '": {}, "_pseudo": {}, "_built": true, "unit": "Bohr", "charge": 1, "symmetry'
    '": false}'
)

no_plus_mo_coeff = numpy.asarray([
    -0.00030993465811149129, 0.45199552674204058, -0.028904704553936636, -0.090403382320240686,
    3.3099317312990211e-23, -5.4836216379542063e-23, 2.2319201719783142e-23, 1.5234367861943547e-23,
    -0.014919945575310562, -0.050585226876045802, 0.00021964788007901282, 0.63737873963625524,
    -0.075317996175523913, -0.22391930706443322, 7.7663101496108472e-23, -1.2866581498817644e-22,
    6.134007397957805e-23, 4.1868757819906538e-23, -0.031952211830968344, -0.12800752151058567,
    -0.0032609643711730505, 0.0068800847859985248, 0.13963789699680393, 0.3910739921457736,
    -1.2545999583162285e-22, 2.0785176354004214e-22, -1.2107811154499295e-22, -8.2644017195444379e-23,
    0.051249641321393952, 0.28818667646763757, 0.0033075416811898057, -0.019087360189030803,
    0.11625833200242082, 0.52831848220213351, -1.5014963062947533e-22, 2.487555121767254e-22,
    -1.4634497672650712e-22, -9.9890365142258758e-23, -0.085174620576668386, 0.41636853967767923,
    -0.0031469672611320427, -0.0031292027582865865, 0.038604836398930767, 0.15078923105718686,
    -4.4348773230544274e-23, 7.3473386202630555e-23, -4.3092200287570662e-23, -2.9413347268918058e-23,
    -0.098222788203448541, 0.024944319955640809, 0, 0,
    0, 0, -0.067945434982550693, -0.27486984552486465,
    0.48235372755507899, -0.091000564454398664, -8.5687039038947365e-22, -2.4242021424911906e-22,
    0, 0, 0, 0,
    0.27486984552486465, -0.067945434982550706, 0.091000564454398664, 0.48235372755507905,
    -8.5687039077997024e-22, -2.4242021424027878e-22, 0.0012639557204546764, 0.0080067496754596592,
    -0.018257071700353976, -0.14240824605537175, 3.7000633254809653e-23, -6.1299594527524831e-23,
    3.6390182472685632e-23, 2.4838765880972595e-23, 0.35818949693892382, 0.59404179492175535,
    0, 0, 0, 0,
    -0.061104899094375123, -0.24719680057269985, 0.39108520443955269, -0.073781899737669826,
    -7.3177996867130668e-22, -2.1453274242608933e-22, 0, 0,
    0, 0, 0.24719680057269985, -0.06110489909437513,
    0.073781899737669812, 0.39108520443955275, -7.3177996938849402e-22, -2.145327425795134e-22,
    0.0057906197720830306, 0.0044905244275705147, -0.050359547498062866, -0.10989774503971134,
    3.8395986659881985e-23, -6.3611300856164208e-23, 3.6587167858830604e-23, 2.4973221758275792e-23,
    0.22119527201835215, 0.47949477769533205, 0, 0,
    0, 0, -0.01396644117585785, -0.056500536402334142,
    0.040432602332096335, -0.0076279904457113544, -1.2275500964325165e-22, -4.5043903833080602e-23,
    0, 0, 0, 0,
    0.056500536402334142, -0.013966441175857852, 0.0076279904457113553, 0.040432602332096348,
    -1.2275500971741671e-22, -4.5043903840723138e-23, -0.00038238083713755197, 0.0013279845592333254,
    -0.017755908800136749, -0.022999209207775772, 9.967481539395441e-24, -1.6513300533565073e-23,
    9.3188838241300475e-24, 6.3607697972387392e-24, 0.035002096717448826, 0.028040985470694852,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0.018989939867174316, -0.0046941479818857768,
    0.0014968545370685424, 0.0079341635928896221, -3.6117117263261199e-23, -1.4678334348277796e-23,
    -2.3551743008472153e-05, 0.00086403660450635534, 0.0056433849698386855, -0.0030629827416551736,
    -8.2746920683314983e-25, 1.3708826692233768e-24, -6.3044707156074036e-25, -4.3032285495950224e-25,
    0.027148172581965901, 0.0034952577378645397, 0, 0,
    0, 0, -0.0046941479818857759, -0.018989939867174316,
    0.0079341635928896186, -0.0014968545370685419, -3.6117117187018993e-23, -1.46783343256146e-23,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0.0411345053288956, -0.010168091982423612, 0.0070672418789917996, 0.037460322234573212,
    -9.6664512026001079e-23, -3.3447737295251316e-23, 0.0014997602905395015, 0.00067502139964927693,
    -0.0011449273695446154, -0.0049406028118178973, 1.4457980820968987e-24, -2.3952789022211201e-24,
    1.3900830830707213e-24, 9.4882591704464335e-25, 0.01520903339622085, 0.048103483584991717,
    0, 0, 0, 0,
    -0.01016809198242361, -0.0411345053288956, 0.037460322234573212, -0.0070672418789917996,
    -9.6664511844836758e-23, -3.3447737231663719e-23, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    7.2499999874677741e-10, -1.7899999969058371e-10, 4.8100000048012137e-10, 2.5480000032457044e-09,
    -3.4197542448540226e-30, -7.4324802158081137e-31, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0.0083670497255368612, -0.002068261916424838, 0.00054552165354917993, 0.0028915689204292929,
    -1.5364044981505871e-23, -6.4180798320080482e-24, -2.2886869008232982e-05, 0.00026753548184679175,
    0.0013612316299479799, 0.0016529014805612883, -7.3599720742662531e-25, 1.2193394118588906e-24,
    -6.9389502040660553e-25, -4.7363038015345652e-25, 0.0072369016334001219, -0.0030670077221483846,
    0, 0, 0, 0,
    -0.002068261916424838, -0.0083670497255368612, 0.0028915689204292924, -0.00054552165354917971,
    -1.5364044975957611e-23, -6.4180798343126379e-24, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    1.7899999969058368e-10, 7.2499999874677741e-10, -2.5480000032457044e-09, 4.8100000048012137e-10,
    3.4198478942988598e-30, 7.4349905418819973e-31, 0.4393247431580361, 0.0018457249590554949,
    -0.082440307245391961, 0.032835096634947859, 2.0821619123975467e-23, -3.4495539612889851e-23,
    1.5598619460356965e-23, 1.0647114980004755e-23, -0.020336271781835273, 0.047334385730994001,
    0.65162213123440482, 0.0037255278528676517, -0.21656746376996816, 0.084096230873614689,
    4.8416450422560335e-23, -8.021237798474233e-23, 3.7070735687412094e-23, 2.530328957971206e-23,
    -0.046972380543379266, 0.12297622487225722, 0.0079436071128575151, -0.0030155181580962718,
    0.35839646499999284, -0.14176954708538833, -6.4904984018073025e-23, 1.075292192122642e-22,
    -5.126638960110416e-23, -3.4992785486762102e-23, 0.055790043610147701, -0.28073838895907421,
    -0.033200168311942932, -0.0068520008268535201, 0.51531686098463858, -0.22432121813260097,
    -8.9353987271549006e-23, 1.4803430938855026e-22, -6.9444510230380106e-23, -4.7400584840645831e-23,
    0.065018982623952684, -0.39655542026124913, -0.0025375225209128102, -0.00097308982350304918,
    0.15173392999789087, -0.11397918105798752, -1.5443590729720988e-23, 2.5585666163230348e-23,
    -9.5534902377339056e-24, -6.5209045775134434e-24, -0.04924140514115187, -0.12183400846770261,
    0, 0, 0, 0,
    -0.094587070236498413, -0.38264724233856257, -0.48671419684477685, 0.091823208129428818,
    -1.3996591993569832e-22, -2.4305995029394422e-22, 0, 0,
    0, 0, 0.38264724233856257, -0.094587070236498427,
    -0.091823208129428791, -0.48671419684477685, -1.3996592014462721e-22, -2.4305995020900676e-22,
    -0.0088521748631843478, -0.0021438770389801847, 0.12232729499607477, -0.11144240005486751,
    -8.1684090422108868e-24, 1.3532745755891694e-23, -3.3239534345426224e-24, -2.2688235022350226e-24,
    -0.39294798183193108, 0.61598417392035687, 0, 0,
    0, 0, -0.081681162058807349, -0.3304370394288122,
    -0.29688857455715278, 0.056010820065607377, -2.3306222063716283e-22, -2.1995631270757339e-22,
    0, 0, 0, 0,
    0.3304370394288122, -0.081681162058807363, -0.056010820065607356, -0.29688857455715278,
    -2.3306222087937335e-22, -2.1995631253857797e-22, -0.00026648872209586265, 8.1205679947524303e-05,
    0.083600597499161003, -0.07641327573735239, -5.4311033514729559e-24, 8.9978036665685639e-24,
    -2.1806279240103904e-24, -1.4884263516833888e-24, -0.32850953285686413, 0.38438219416587477,
    0, 0, 0, 0,
    -0.025415706556066845, -0.10281796482227062, 0.01206057577249472, -0.0022753409559873156,
    -1.6746212967910739e-22, -7.6954759609250789e-23, 0, 0,
    0, 0, 0.10281796482227062, -0.025415706556066849,
    0.0022753409559873182, 0.012060575772494728, -1.6746212959324084e-22, -7.6954759506770098e-23,
    -0.0026062364409375283, -0.0013393631192955995, 0.03701392549892029, 0.00104693344732496,
    -1.0281903630394735e-23, 1.7034209098145332e-23, -8.8625483178806873e-24, -6.0492898864790742e-24,
    -0.070911916200646075, 0.015732860878165628, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    -0.012426798778519248, 0.0030717965746901456, -0.00024128447605068648, -0.0012789422564260531,
    2.0077357189300258e-23, 9.2863473077724753e-24, 0.00042642603815339612, -2.840236988567041e-06,
    0.00081366179114275724, 0.0052067273720378575, -1.3827261220261426e-24, 2.2907864857010504e-24,
    -1.3634921867485244e-24, -9.3067582809012728e-25, 0.016842225986242795, 0.0090004167732186668,
    0, 0, 0, 0,
    0.0030717965746901452, 0.012426798778519248, -0.001278942256426052, 0.00024128447605068615,
    2.0077357170642162e-23, 9.2863473068702584e-24, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, -0.034208236740868231, 0.0084559785853831424,
    -0.0010117741623665965, -0.005362967222734636, 5.6943337445289288e-23, 2.5713448479830125e-23,
    -0.0014007224605038749, -0.0012961610192925158, 0.009343509189581517, 0.010273146103284694,
    -4.8625970199805678e-24, 8.0559493035791074e-24, -4.4975543354603261e-24, -3.0698856556828762e-24,
    0.042769660961498243, 0.010560637948058595, 0, 0,
    0, 0, 0.0084559785853831407, 0.034208236740868231,
    -0.0053629672227346334, 0.0010117741623665957, 5.6943337426602712e-23, 2.5713448488114281e-23,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 1.0739999981435021e-09, -2.6499999954192564e-10,
    -8.1700000142478931e-10, -4.3290000065090597e-09, 2.3018139448893698e-30, -4.4023421173914863e-31,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0.0067681971283006107, -0.001673039457108013,
    0.0011507847598172889, 0.0060998008355994402, -1.5846957237402312e-23, -5.4982250554805852e-24,
    0.00016707332506010045, 0.00042762110775980243, -0.0024446931399710443, -0.0015034305903884981,
    1.0047151352617039e-24, -1.6645290903649695e-24, 9.0611785430358747e-25, 6.1848684770916688e-25,
    -0.0079573813825088695, -0.010525235841618416, 0, 0,
    0, 0, -0.0016730394571080128, -0.0067681971283006107,
    0.0060998008355994384, -0.0011507847598172889, -1.5846957241604296e-23, -5.4982250616607101e-24,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 2.6499999954192559e-10, 1.0739999981435021e-09,
    4.3290000065090605e-09, -8.1700000142478942e-10, -2.2999117424193358e-30, 4.4095372155272348e-31,
], dtype=numpy.float64).reshape((62, 10))

no_plus_openmolcas_energies = numpy.asarray([
    -129.12863949,
    -128.72314673,
    -128.72314673,
    -128.72083180,
    -128.70307916,
    -128.70307916,
    -128.50342069,
    -128.50342069,
    -128.46044322,
    -128.39715780,
])


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

    def test_o2_triplet_energy(self):
        mol = gto.M(
            atom="O 0 0 0; O 0 0 1.21",
            basis="sto-3g",
            spin=2,
            symmetry=False,
            verbose=0,
        )

        mf = scf.ROHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()
        self.assertTrue(mf.converged)

        mc = gasci.GASCI(
            mf,
            8,
            (7, 5),
            ncore=2,
            gas_orbs=(2, 4, 2),
            gas_restr=(
                (2, 4),
                (8, 10),
                (12, 12),
            ),
            gas_restr_type="cumulative-occ",
        )
        mc.verbose = 0
        mc.canonicalization = False
        mc.fcisolver.spin = 2
        mc.fcisolver.nroots = 1
        mc.fcisolver.max_cycle = 300
        mc.fcisolver.conv_tol = 1e-12
        mc.kernel()
        self.assertTrue(mc.converged)

        numpy.testing.assert_allclose(
            mc.e_tot,
            -147.5163002382864,
            atol=1e-7,
            rtol=0,
        )
        ss, multiplicity = mc.spin_square()
        self.assertAlmostEqual(ss, 2.0, places=9)
        self.assertAlmostEqual(multiplicity, 3.0, places=9)
        self.assertEqual(mc.gas_space_info()["core"]["ndet"], 424)

    def test_no_plus_openmolcas_multiroot(self):
        mol = gto.loads(no_plus_mol)
        mo_coeff = no_plus_mo_coeff.copy()

        overlap = mol.intor_symmetric("int1e_ovlp")
        numpy.testing.assert_allclose(
            mo_coeff.T @ overlap @ mo_coeff,
            numpy.eye(mo_coeff.shape[1]),
            atol=1e-10,
            rtol=0,
        )

        mol.verbose = 0
        mf = scf.RHF(mol)
        mf.max_cycle = 1
        mf.kernel()
        mf.mo_coeff = mo_coeff

        mc = gasci.GASCI(
            mf,
            8,
            (5, 5),
            ncore=2,
            gas_orbs=(2, 4, 2),
            gas_restr=(
                (2, 4),
                (7, 9),
                (10, 10),
            ),
            gas_restr_type="cumulative-occ",
        )
        mc.verbose = 0
        mc.canonicalization = False
        mc.fcisolver.spin = 0
        mc.fcisolver.nroots = 30
        mc.fcisolver.max_space = 30
        mc.fcisolver.max_cycle = 300
        mc.fcisolver.conv_tol = 1e-10
        mc.kernel()

        self.assertTrue(numpy.all(numpy.asarray(mc.converged)))

        singlet_energies = []
        for state, energy in enumerate(
                numpy.asarray(mc.e_tot).reshape(-1)):
            ss, _ = mc.spin_square(state=state)
            if abs(ss) < 1e-6:
                singlet_energies.append(energy)

        self.assertEqual(
            len(singlet_energies),
            len(no_plus_openmolcas_energies),
        )
        numpy.testing.assert_allclose(
            numpy.asarray(singlet_energies),
            no_plus_openmolcas_energies,
            atol=1e-7,
            rtol=0,
        )

    def test_kernel_log_label(self):
        output = io.StringIO()
        mc = gasci.GASCI(
            self.mf, 4, (2, 2), ncore=5)
        mc.stdout = output
        mc.verbose = logger.DEBUG
        mc.canonicalization = False
        mc.kernel()

        text = output.getvalue()
        self.assertIn("Start GASCI", text)
        self.assertNotIn("Start CASCI", text)

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

    def test_fcisolver_spin_propagation(self):
        mc = gasci.GASCI(
            self.mf, 4, 4, ncore=5)
        mc.verbose = 0
        mc.canonicalization = False
        mc.fcisolver.spin = 2
        mc.kernel()

        info = mc.gas_space_info()["core"]
        self.assertEqual(info["nelec"], (3, 1))
        self.assertEqual(info["ndet"], 16)
        self.assertEqual(mc._gas_problem_signature()[1], (3, 1))

        dm1a, dm1b = mc.make_gasdm1s()
        self.assertAlmostEqual(numpy.trace(dm1a), 3.0, places=10)
        self.assertAlmostEqual(numpy.trace(dm1b), 1.0, places=10)
        self.assertAlmostEqual(
            mc.spin_square()[0],
            mc.fcisolver.spin_square(mc.ci, 4, mc.nelecas)[0],
            places=10)

        spin_signature = mc._gas_problem_signature()
        mc.fcisolver.spin = 0
        self.assertNotEqual(spin_signature, mc._gas_problem_signature())

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
