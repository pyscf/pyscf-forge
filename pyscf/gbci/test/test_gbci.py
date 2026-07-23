#!/usr/bin/env python
#
# Copyright 2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0

import unittest

import numpy as np
from pyscf import gto, scf
from pyscf import gbci
from pyscf.gbci import gbpdft


def first_energy(e_tot):
    return float(np.asarray(e_tot, dtype=float).reshape(-1)[0])


def build_mf():
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        verbose=0,
    )
    return scf.RHF(mol).run()


GROUP_CASES = (
    ("none", None),
    ("mo", {"mo": [[0], [1]]}),
    ("atom", {"atom": [[0], [1]]}),
    ("occ", {"occ": [[0], [1], [2]]}),
)


class KnownValues(unittest.TestCase):
    def test_gbci(self):
        mc = gbci.gbci(build_mf(), 2, (1, 1))
        e_tot, e_cas, ci = mc.kernel()

        self.assertAlmostEqual(first_energy(e_tot), -1.137283834488503, 9)
        self.assertTrue(np.all(np.isfinite(e_cas)))
        self.assertIsNotNone(ci)

    def test_gbci_grouping_modes(self):
        for label, group_a in GROUP_CASES:
            with self.subTest(group_a=label):
                mc = gbci.gbci(build_mf(), 2, (1, 1), group_a=group_a)
                e_tot, e_cas, ci = mc.kernel()

                self.assertAlmostEqual(
                    first_energy(e_tot), -1.137283834488503, 9)
                self.assertTrue(np.all(np.isfinite(e_cas)))
                self.assertIsNotNone(ci)

    def test_gbpdft_grouping_modes(self):
        for label, group_a in GROUP_CASES:
            with self.subTest(group_a=label):
                mc = gbpdft.GBCI(
                    build_mf(), "tPBE", 2, (1, 1),
                    group_a=group_a, grids_level=0)
                e_tot, e_ot, e_gbci, e_cas, ci = mc.kernel()

                self.assertTrue(np.isfinite(first_energy(e_tot)))
                self.assertTrue(np.isfinite(first_energy(e_ot)))
                self.assertAlmostEqual(
                    first_energy(e_gbci), -1.137283834488503, 9)
                self.assertTrue(np.all(np.isfinite(e_cas)))
                self.assertIsNotNone(ci)

    def test_xms_gbpdft_grouping_modes(self):
        for label, group_a in GROUP_CASES:
            with self.subTest(group_a=label):
                mc = gbpdft.GBCI(
                    build_mf(), "tPBE", 2, (1, 1),
                    group_a=group_a, grids_level=0)
                xms = mc.multi_state([0.5, 0.5], "xms")
                e_tot, e_ot, e_gbci, e_cas, ci, mo_coeff, mo_energy = (
                    xms.kernel(debug=True)
                )

                self.assertTrue(np.isfinite(e_tot))
                self.assertTrue(np.all(np.isfinite(e_ot)))
                self.assertTrue(np.all(np.isfinite(e_gbci)))
                self.assertEqual(len(ci), 2)
                self.assertEqual(xms.heff_gbci.shape, (2, 2))
                self.assertEqual(xms.get_heff_pdft().shape, (2, 2))
                np.testing.assert_allclose(
                    xms.heff_gbci, xms.heff_gbci.conj().T, atol=1e-10)
                np.testing.assert_allclose(
                    xms.get_heff_pdft(),
                    xms.get_heff_pdft().conj().T, atol=1e-10)
                self.assertEqual(len(xms.get_ci_adiabats(uci="MSGBPDFT")), 2)
                self.assertIs(mo_coeff, xms.mo_coeff)
                self.assertTrue(
                    mo_energy is None or np.asarray(mo_energy).ndim >= 1)
                self.assertIsNotNone(e_cas)


if __name__ == "__main__":
    print("Full Tests for GBCI")
    unittest.main()
