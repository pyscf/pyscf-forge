#!/usr/bin/env python
#
# Copyright 2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0

import sys
import unittest

import numpy as np
from pyscf import gto, mcpdft, scf
from pyscf.gbci import gbpdft


OTXCS = ("tPBE", "tPBE0")

LIH_XMS_GROUP_CASES = (
    ("none", None, -7.82530158785605),
    ("mo", {"mo": [[0], [1]]}, -7.825301573668855),
    ("atom", {"atom": [[0], [1]]}, -7.825301573668856),
    ("occ", {"occ": [[0], [1], [2]]}, -7.825301573668856),
)

_MCS = []
_MFS = []


def close_scf_resources(mf):
    chkfile = getattr(mf, "_chkfile", None)
    if chkfile is not None:
        chkfile.close()
        mf._chkfile = None
    if hasattr(mf, "chkfile"):
        mf.chkfile = None


def close_gbci_resources(mc):
    close_scf_resources(mc)
    fasscf = getattr(mc, "_fasscf", None)
    if fasscf is not None:
        close_scf_resources(fasscf)


def close_mol_stdout(mol):
    stdout = getattr(mol, "stdout", None)
    if stdout in (None, sys.stdout, sys.stderr):
        return
    try:
        if not stdout.closed:
            stdout.close()
    except AttributeError:
        pass


def build_mf():
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        output="/dev/null",
        verbose=0,
    )
    mf = scf.RHF(mol)
    close_scf_resources(mf)
    _MFS.append(mf)
    return mf.run()


def build_lih_mf():
    mol = gto.M(
        atom="Li 0 0 0; H 0 0 1.6",
        basis="sto-3g",
        output="/dev/null",
        verbose=0,
    )
    mf = scf.RHF(mol)
    close_scf_resources(mf)
    _MFS.append(mf)
    return mf.run()


def track_mc(mc):
    _MCS.append(mc)
    return mc


def tearDownModule():
    while _MCS:
        mc = _MCS.pop()
        close_gbci_resources(mc)
        close_mol_stdout(mc.mol)
    while _MFS:
        mf = _MFS.pop()
        close_scf_resources(mf)
        close_mol_stdout(mf.mol)


def check_msgbpdft_kernel(testcase, xms, result):
    e_tot, e_ot, e_gbci, e_cas, ci, mo_coeff, mo_energy = result
    testcase.assertTrue(np.isfinite(e_tot))
    testcase.assertTrue(np.all(np.isfinite(e_ot)))
    testcase.assertTrue(np.all(np.isfinite(e_gbci)))
    testcase.assertEqual(len(ci), 2)
    testcase.assertEqual(xms.heff_gbci.shape, (2, 2))
    testcase.assertEqual(xms.get_heff_pdft().shape, (2, 2))
    np.testing.assert_allclose(
        xms.heff_gbci, xms.heff_gbci.conj().T, atol=1e-10)
    np.testing.assert_allclose(
        xms.get_heff_pdft(), xms.get_heff_pdft().conj().T, atol=1e-10)
    np.testing.assert_allclose(xms.heff_gbci, xms.heff_mcscf, atol=1e-10)
    np.testing.assert_allclose(xms.si_gbci, xms.si_mcscf, atol=1e-10)
    testcase.assertEqual(len(xms.get_ci_adiabats(uci="GBCI")), 2)
    testcase.assertEqual(len(xms.get_ci_adiabats(uci="MSGBPDFT")), 2)
    testcase.assertIs(mo_coeff, xms.mo_coeff)
    testcase.assertTrue(mo_energy is None or np.asarray(mo_energy).ndim >= 1)
    testcase.assertIsNotNone(e_cas)


def check_h2_mspdft(testcase, mf, otxc, result, xms):
    mc = track_mc(mcpdft.CASCI(mf, otxc, 2, (1, 1), grids_level=0))
    ref_xms = track_mc(mc.multi_state([0.5, 0.5], method="xms"))
    ref = ref_xms.kernel(mf.mo_coeff)
    np.testing.assert_allclose(result[0], ref[0], atol=1e-9)
    np.testing.assert_allclose(result[1], ref[1], atol=1e-9)
    np.testing.assert_allclose(result[2], ref[2], atol=1e-9)
    np.testing.assert_allclose(xms.e_states, ref_xms.e_states, atol=1e-9)


class KnownValues(unittest.TestCase):
    def test_h2_xms_gbpdft_parity(self):
        for otxc in OTXCS:
            with self.subTest(otxc=otxc):
                mf = build_mf()
                mc = track_mc(
                    gbpdft.GBCI(mf, otxc, 2, (1, 1), grids_level=0))
                xms = track_mc(mc.multi_state([0.5, 0.5], "xms"))
                result = xms.kernel(mf.mo_coeff)
                check_msgbpdft_kernel(self, xms, result)
                check_h2_mspdft(self, mf, otxc, result, xms)

    def test_lih_xms_gbpdft(self):
        mf = build_lih_mf()
        for label, group_a, expected in LIH_XMS_GROUP_CASES:
            with self.subTest(group_a=label):
                mc = track_mc(
                    gbpdft.GBCI(
                        mf, "tPBE0", 2, (1, 1),
                        group_a=group_a, grids_level=0))
                xms = track_mc(mc.multi_state([0.5, 0.5], "xms"))
                result = xms.kernel(mf.mo_coeff)
                check_msgbpdft_kernel(self, xms, result)
                self.assertAlmostEqual(float(result[0]), expected, 9)


if __name__ == "__main__":
    print("Full Tests for MS-GBPDFT")
    unittest.main()
