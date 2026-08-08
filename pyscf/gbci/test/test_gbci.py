#!/usr/bin/env python
#
# Copyright 2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0

import importlib
import sys
import unittest
import warnings

import numpy as np
from pyscf import gto, mcscf, scf
from pyscf import gbci
from pyscf.gbci import direct_gbci


H2_CASCI_ENERGY = -1.137283834488503

LIH_GROUP_CASES = (
    ("none", None, -7.862129687670069),
    ("mo", {"mo": [[0], [1]]}, -7.862129687582131),
    ("atom", {"atom": [[0], [1]]}, -7.862129687582131),
    ("occ", {"occ": [[0], [1], [2]]}, -7.862129687582131),
)

RDM_CASES = (
    ("none", None),
    ("mo", {"mo": [[0], [1]]}),
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


def build_rhf(mol):
    mf = scf.RHF(mol)
    close_scf_resources(mf)
    _MFS.append(mf)
    return mf.run()


def build_h2_mf():
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        output="/dev/null",
        verbose=0,
    )
    return build_rhf(mol)


def build_lih_mf():
    mol = gto.M(
        atom="Li 0 0 0; H 0 0 1.6",
        basis="sto-3g",
        output="/dev/null",
        verbose=0,
    )
    return build_rhf(mol)


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


def check_gbci_kernel(testcase, e_tot, e_cas, ci, expected_e_tot):
    testcase.assertAlmostEqual(first_energy(e_tot), expected_e_tot, 9)
    testcase.assertTrue(np.all(np.isfinite(e_cas)))
    testcase.assertIsNotNone(ci)


def check_h2_casci(testcase, mf, e_tot):
    mc = mcscf.CASCI(mf, 2, (1, 1))
    e_casci = first_energy(mc.kernel(mf.mo_coeff)[0])
    testcase.assertAlmostEqual(e_casci, H2_CASCI_ENERGY, 9)
    testcase.assertAlmostEqual(first_energy(e_tot), e_casci, 9)


def check_deprecation(testcase, caught, text):
    messages = [str(w.message) for w in caught]
    testcase.assertTrue(any(text in msg for msg in messages))


def make_rdm_kwargs(mc):
    intermediates = mc.get_gbci_intermediates()
    return {
        "mo_coeff": mc.mo_coeff,
        "dmet_core_list": intermediates["dmet_core_list"],
        "conf_info_list": intermediates["conf_info_list"],
        "ov_list": intermediates["ov_list"],
    }


def make_contract_h_args(mc):
    intermediates = mc.get_gbci_intermediates()
    dmet_act_list = mc.get_active_dm(mc.mo_coeff)
    h1e, ecore_list = mc.get_h1cas(
        dmet_act_list=dmet_act_list,
        mo_list=intermediates["mo_list"],
        dmet_core_list=intermediates["dmet_core_list"])
    eri = mc.get_h2eff(mc.mo_coeff)
    erieff = mc.fcisolver.absorb_h1e(
        h1e, eri, mc.ncas, mc.nelecas, fac=.5)
    return (
        erieff,
        intermediates["conf_info_list"],
        intermediates["ov_list"],
        ecore_list,
    )


def make_transition_ci(ci):
    ci1 = np.asarray(ci, dtype=float)
    ci2 = np.roll(ci1.ravel(), 1).reshape(ci1.shape)
    return ci1, ci2 / np.linalg.norm(ci2)


def check_rdm1(testcase, mc, ci, kwargs):
    data = mc.precompute_rdm1s(**kwargs)
    rdm1a, rdm1b = mc.make_rdm1s(ci, **kwargs)
    rdm1a_pre, rdm1b_pre = mc.make_rdm1s(ci, data=data)
    rdm1 = mc.make_rdm1(ci, data=data)
    trans_rdm1 = mc.trans_rdm1(ci, ci, data=data)
    rdm1a_mo, rdm1b_mo = mc.make_rdm1s_mo(ci, data=data)
    rdm1_mo = mc.make_rdm1_mo(ci, data=data)
    trans_rdm1_mo = mc.trans_rdm1_mo(ci, ci, data=data)

    np.testing.assert_allclose(rdm1a, rdm1a_pre, atol=1e-10)
    np.testing.assert_allclose(rdm1b, rdm1b_pre, atol=1e-10)
    np.testing.assert_allclose(rdm1, rdm1a + rdm1b, atol=1e-10)
    np.testing.assert_allclose(trans_rdm1, rdm1, atol=1e-10)
    np.testing.assert_allclose(rdm1_mo, rdm1a_mo + rdm1b_mo, atol=1e-10)
    np.testing.assert_allclose(trans_rdm1_mo, rdm1_mo, atol=1e-10)

    s1e = mc._scf.get_ovlp(mc.mol)
    testcase.assertAlmostEqual(
        float(np.einsum("ij,ji", s1e, rdm1)), mc.mol.nelectron, 9)
    testcase.assertAlmostEqual(float(np.trace(rdm1_mo)), mc.mol.nelectron, 9)


def check_transition_rdm1s(testcase, mc, ci, kwargs):
    data = mc.precompute_rdm1s(**kwargs)
    ci_bra, ci_ket = make_transition_ci(ci)
    rdm1a, rdm1b = mc.trans_rdm1s(ci_bra, ci_ket, data=data)
    rdm1 = mc.trans_rdm1(ci_bra, ci_ket, data=data)
    rdm1a_mo, rdm1b_mo = mc.trans_rdm1s_mo(ci_bra, ci_ket, data=data)
    rdm1_mo = mc.trans_rdm1_mo(ci_bra, ci_ket, data=data)

    np.testing.assert_allclose(rdm1, rdm1a + rdm1b, atol=1e-10)
    np.testing.assert_allclose(rdm1_mo, rdm1a_mo + rdm1b_mo, atol=1e-10)
    testcase.assertEqual(rdm1.shape, (mc.mo_coeff.shape[0],) * 2)
    testcase.assertEqual(rdm1_mo.shape, (mc.mo_coeff.shape[1],) * 2)
    testcase.assertTrue(np.all(np.isfinite(rdm1)))
    testcase.assertTrue(np.all(np.isfinite(rdm1_mo)))


def check_rdm2(testcase, mc, ci, kwargs):
    rdm2s = mc.make_rdm2s(ci, **kwargs)
    rdm2 = mc.make_rdm2(ci, **kwargs)
    np.testing.assert_allclose(rdm2, sum(rdm2s), atol=1e-8)

    data = mc.precompute_rdm2s_mo(**kwargs)
    rdm2s_mo = mc.make_rdm2s_mo(ci, data=data)
    rdm2s_mo_slow = mc.make_rdm2s_mo_slow(ci, **kwargs)
    rdm2_mo = mc.make_rdm2_mo(ci, data=data)
    rdm2_mo_slow = mc.make_rdm2_mo_slow(ci, **kwargs)

    for fast, slow in zip(rdm2s_mo, rdm2s_mo_slow):
        np.testing.assert_allclose(fast, slow, atol=1e-8)
    np.testing.assert_allclose(rdm2_mo, sum(rdm2s_mo), atol=1e-8)
    np.testing.assert_allclose(rdm2_mo, rdm2_mo_slow, atol=1e-8)
    testcase.assertEqual(rdm2_mo.shape, (mc.mo_coeff.shape[1],) * 4)


class KnownValues(unittest.TestCase):
    def test_h2_gbci_parity(self):
        mf = build_h2_mf()
        mc = track_mc(gbci.gbci(mf, 2, (1, 1)))
        e_tot, e_cas, ci = mc.kernel(mf.mo_coeff)
        check_gbci_kernel(self, e_tot, e_cas, ci, H2_CASCI_ENERGY)
        check_h2_casci(self, mf, e_tot)

    def test_lih_gbci(self):
        mf = build_lih_mf()
        for label, group_a, expected in LIH_GROUP_CASES:
            with self.subTest(group_a=label):
                mc = track_mc(
                    gbci.gbci(mf, 2, (1, 1), group_a=group_a))
                check_gbci_kernel(self, *mc.kernel(mf.mo_coeff), expected)

    def test_lih_rdm(self):
        for label, group_a in RDM_CASES:
            with self.subTest(group_a=label):
                mc = track_mc(
                    gbci.gbci(build_lih_mf(), 2, (1, 1), group_a=group_a))
                _e_tot, _e_cas, ci = mc.kernel()
                kwargs = make_rdm_kwargs(mc)
                check_rdm1(self, mc, ci, kwargs)
                check_transition_rdm1s(self, mc, ci, kwargs)
                check_rdm2(self, mc, ci, kwargs)

    def test_contract_h_matches_slow(self):
        for label, group_a in RDM_CASES:
            with self.subTest(group_a=label):
                mc = track_mc(
                    gbci.gbci(build_lih_mf(), 2, (1, 1), group_a=group_a))
                _e_tot, _e_cas, ci = mc.kernel()
                erieff, conf_info_list, ov_list, ecore_list = (
                    make_contract_h_args(mc))
                fast = direct_gbci.contract_h(
                    erieff, ci, mc.ncas, mc.nelecas, conf_info_list,
                    ov_list, ecore_list)
                slow = direct_gbci.contract_h_slow(
                    erieff, ci, mc.ncas, mc.nelecas, conf_info_list,
                    ov_list, ecore_list)
                np.testing.assert_allclose(fast, slow, atol=1e-10)

    def test_sfnoci_aliases(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from pyscf import sfnoci as sfnoci_pkg
            sfnoci_mod = importlib.import_module("pyscf.sfnoci.sfnoci")
            direct_mod = importlib.import_module("pyscf.sfnoci.direct_sfnoci")
            sfnoci_pkg = importlib.reload(sfnoci_pkg)
            sfnoci_mod = importlib.reload(sfnoci_mod)
            direct_mod = importlib.reload(direct_mod)
            legacy_mc = track_mc(sfnoci_pkg.sfnoci(
                build_h2_mf(), 2, (1, 1)))

        check_deprecation(self, caught, "pyscf.sfnoci is deprecated")
        check_deprecation(self, caught, "pyscf.sfnoci.sfnoci is deprecated")
        check_deprecation(
            self, caught, "pyscf.sfnoci.direct_sfnoci is deprecated")
        self.assertIs(sfnoci_pkg.SFNOCI, gbci.GBCI)
        self.assertIs(sfnoci_mod.SFNOCI, gbci.GBCI)
        self.assertIsInstance(legacy_mc, gbci.GBCI)
        self.assertIs(direct_mod.SFNOCISolver, direct_gbci.GBCISolver)
        self.assertIs(direct_mod.contract_H, direct_gbci.contract_h)


if __name__ == "__main__":
    print("Full Tests for GBCI")
    unittest.main()
