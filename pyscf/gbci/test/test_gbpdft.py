#!/usr/bin/env python
#
# Copyright 2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0

import sys
import unittest

import numpy as np
from pyscf import ao2mo, gto, mcpdft, scf
from pyscf.gbci import gbpdft


H2_GBCI_ENERGY = -1.137283834488503
LIH_GBCI_CORRELATION_ENERGY = -0.000582570524654713
OTXCS = ("tPBE", "tPBE0")

LIH_GROUP_CASES = (
    ("none", None, -7.862129687670069),
    ("mo", {"mo": [[0], [1]]}, -7.862129687582131),
    ("atom", {"atom": [[0], [1]]}, -7.862129687582131),
    ("occ", {"occ": [[0], [1], [2]]}, -7.862129687582131),
)

_MCS = []
_MFS = []


def first_energy(e_tot):
    return float(np.asarray(e_tot, dtype=float).reshape(-1)[0])


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


def build_h2_mf():
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


def check_gbpdft_kernel(testcase, mc, result, expected_e_gbci):
    e_tot, e_ot, e_gbci, e_cas, ci, mo_coeff, mo_energy = result
    testcase.assertTrue(np.isfinite(first_energy(e_tot)))
    testcase.assertTrue(np.isfinite(first_energy(e_ot)))
    testcase.assertAlmostEqual(first_energy(e_gbci), expected_e_gbci, 9)
    testcase.assertTrue(np.all(np.isfinite(e_cas)))
    testcase.assertIsNotNone(ci)
    testcase.assertIs(mo_coeff, mc.mo_coeff)
    testcase.assertTrue(mo_energy is None or np.asarray(mo_energy).ndim >= 1)


def check_h2_mcpdft(testcase, mf, otxc, result):
    mc = track_mc(mcpdft.CASCI(mf, otxc, 2, (1, 1), grids_level=0))
    ref = mc.kernel(mf.mo_coeff)
    np.testing.assert_allclose(first_energy(result[0]), first_energy(ref[0]),
                               atol=1e-9)
    np.testing.assert_allclose(first_energy(result[1]), first_energy(ref[1]),
                               atol=1e-9)
    np.testing.assert_allclose(first_energy(result[2]), first_energy(ref[2]),
                               atol=1e-9)


def check_energy_components(testcase, mc):
    dm1s = mc.make_rdm1s(mc.ci, mo_coeff=mc.mo_coeff)
    e_mcwfn = mc.energy_mcwfn(
        dm1s=dm1s, e_gbci=first_energy(mc.e_gbci))
    e_dft = mc.energy_dft(dm1s=dm1s)

    np.testing.assert_allclose(e_dft, first_energy(mc.e_ot), atol=1e-9)
    np.testing.assert_allclose(
        e_mcwfn + e_dft, first_energy(mc.e_tot), atol=1e-9)
    testcase.assertTrue(np.isfinite(e_mcwfn))


def make_gbci_cumulant(dm2, dm1s):
    dm1 = dm1s[0] + dm1s[1]
    cm2 = dm2.copy()
    cm2 -= np.multiply.outer(dm1, dm1)
    cm2 += np.multiply.outer(dm1s[0], dm1s[0]).transpose(0, 3, 2, 1)
    cm2 += np.multiply.outer(dm1s[1], dm1s[1]).transpose(0, 3, 2, 1)
    return cm2


def check_correlation_energy(testcase, mc, expected):
    dm1s_ao = mc.make_rdm1s(mc.ci, mo_coeff=mc.mo_coeff)
    dm1_ao = dm1s_ao[0] + dm1s_ao[1]
    vj, vk = mc._scf.get_jk(dm=dm1s_ao)
    vj = vj[0] + vj[1]
    e_classical = (
        mc._scf.energy_nuc()
        + np.tensordot(mc._scf.get_hcore(), dm1_ao)
        + 0.5 * np.tensordot(vj, dm1_ao)
    )
    e_x = -0.5 * (
        np.tensordot(vk[0], dm1s_ao[0])
        + np.tensordot(vk[1], dm1s_ao[1]))
    e_corr_residual = first_energy(mc.e_gbci) - e_classical - e_x

    dm1s_mo = np.asarray(mc.make_rdm1s_mo(mc.ci, mo_coeff=mc.mo_coeff))
    dm2_mo = mc.make_rdm2_mo(mc.ci, mo_coeff=mc.mo_coeff)
    cumulant = make_gbci_cumulant(dm2_mo, dm1s_mo)
    nmo = mc.mo_coeff.shape[1]
    eri_mo = ao2mo.full(
        mc.mol, mc.mo_coeff, compact=False).reshape((nmo,) * 4)
    e_corr_rdm = 0.5 * np.tensordot(eri_mo, cumulant, axes=4)

    testcase.assertAlmostEqual(e_corr_rdm, expected, 9)
    np.testing.assert_allclose(e_corr_residual, e_corr_rdm, atol=1e-8)


class KnownValues(unittest.TestCase):
    def test_h2_gbpdft_parity(self):
        for otxc in OTXCS:
            with self.subTest(otxc=otxc):
                mf = build_h2_mf()
                mc = track_mc(
                    gbpdft.GBCI(mf, otxc, 2, (1, 1),
                                grids_level=0))
                result = mc.kernel(mf.mo_coeff)
                check_gbpdft_kernel(
                    self, mc, result, H2_GBCI_ENERGY)
                check_h2_mcpdft(self, mf, otxc, result)

    def test_lih_gbpdft(self):
        mf = build_lih_mf()
        for label, group_a, expected in LIH_GROUP_CASES:
            with self.subTest(group_a=label):
                mc = track_mc(
                    gbpdft.GBCI(
                        mf, "tPBE0", 2, (1, 1),
                        group_a=group_a, grids_level=0))
                check_gbpdft_kernel(
                    self, mc, mc.kernel(mf.mo_coeff), expected)

    def test_energy_components(self):
        for otxc in OTXCS:
            with self.subTest(otxc=otxc):
                mc = track_mc(
                    gbpdft.GBCI(
                        build_h2_mf(), otxc, 2, (1, 1), grids_level=0))
                mc.kernel()
                check_energy_components(self, mc)

    def test_lih_correlation_energy(self):
        mc = track_mc(
            gbpdft.GBCI(
                build_lih_mf(), "tPBE0", 2, (1, 1),
                group_a={"mo": [[0], [1]]}, grids_level=0))
        mc.kernel()
        check_correlation_energy(self, mc, LIH_GBCI_CORRELATION_ENERGY)

    def test_tpbe0(self):
        mf = build_h2_mf()
        pdft = track_mc(gbpdft.GBCI(mf, "tPBE", 2, (1, 1), grids_level=0))
        hyb = track_mc(gbpdft.GBCI(mf, "tPBE0", 2, (1, 1), grids_level=0))

        e_pdft = first_energy(pdft.kernel()[0])
        e_gbci = first_energy(pdft.e_gbci)
        e_hyb = first_energy(hyb.kernel()[0])
        hyb_x, hyb_c = hyb.otfnal._numint.rsh_and_hybrid_coeff(
            hyb.otfnal.otxc, hyb.mol.spin)[2]

        self.assertAlmostEqual(hyb_x, 0.25, 12)
        self.assertAlmostEqual(hyb_c, 0.25, 12)
        np.testing.assert_allclose(
            e_hyb, 0.75 * e_pdft + 0.25 * e_gbci, atol=1e-9)


if __name__ == "__main__":
    print("Full Tests for GBPDFT")
    unittest.main()
