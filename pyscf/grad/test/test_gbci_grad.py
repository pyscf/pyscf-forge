#!/usr/bin/env python
#
# Copyright 2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0

import unittest
from unittest import mock

from pyscf import gbci, gto, lib, scf
from pyscf.grad import gbci as gbci_grad
from pyscf.grad import rohf_gbci as rohf_gbci_grad
from pyscf.grad.gbci import _bath_rotation_zvec
from pyscf.grad.rohf_casci import _solve_rohf_adjoint


lib.num_threads(1)

BASIS = "cc-pvdz"
BOND_LENGTH = 1.5
GROUP_A = {"atom": [0]}

LIH_GRAD_Z = 0.01830102797453953
LIF_GRAD_Z = 0.03548531842959424
LIH_TRIPLET_ROHF_GRAD_Z = 0.0294477687822165
LIF_TRIPLET_ROHF_GRAD_Z = 0.04667256084318572
LIH_BATH_RESPONSE_FP = (
    -1.627492829776010e-05,
    1.119788583093171e-04,
    2.120511245200006e-04,
)
LIF_BATH_RESPONSE_FP = (
    0.0,
    2.216177220066382e-10,
    2.631660466514099e-05,
    -1.611782286360561e-03,
    -4.668488171487795e-04,
)
LIH_TRIPLET_ROHF_BATH_RESPONSE_FP = (
    -1.934943396922866e-03,
    8.244476143925915e-03,
    2.983600786582193e-02,
)
LIF_TRIPLET_ROHF_BATH_RESPONSE_FP = (
    0.0,
    6.210682700180350e-09,
    2.158303995930926e-05,
    -4.669944849630451e-03,
    -5.302987569270201e-03,
)
LIH_TRIPLET_ROHF_RESPONSE_FP = 0.12102737231463209
LIF_TRIPLET_ROHF_RESPONSE_FP = 0.367922010414346


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


def get_gbci_grad(atom, ncas, nelecas, spin=0, mf_cls=scf.RHF):
    mol = gto.M(
        atom=atom,
        basis=BASIS,
        spin=spin,
        verbose=0,
    )
    mf = mf_cls(mol)
    mc = None
    try:
        mf.conv_tol = 1e-12
        mf.run()

        mc = gbci.gbci(mf, ncas, nelecas, group_a=GROUP_A)
        mc.fcisolver.conv_tol = 1e-10
        mc.run()

        bath_response_fp = []
        rohf_response_fp = []

        def record_bath_response(*args, **kwargs):
            response = _bath_rotation_zvec(*args, **kwargs)
            bath_response_fp.extend(float(lib.fp(x)) for x in response[1])
            return response

        def record_rohf_response(*args, **kwargs):
            response = _solve_rohf_adjoint(*args, **kwargs)
            rohf_response_fp.append(float(lib.fp(response[0])))
            return response

        with mock.patch.object(
                gbci_grad, "_bath_rotation_zvec",
                side_effect=record_bath_response), mock.patch.object(
                    rohf_gbci_grad, "_bath_rotation_zvec",
                    side_effect=record_bath_response), mock.patch.object(
                    rohf_gbci_grad, "_solve_rohf_adjoint",
                    side_effect=record_rohf_response):
            grad = mc.nuc_grad_method().kernel()
        return grad, bath_response_fp, rohf_response_fp
    finally:
        if mc is not None:
            close_gbci_resources(mc)
        close_scf_resources(mf)


class KnownValues(unittest.TestCase):
    def assert_response_fp(self, response_fp, reference, places=10):
        self.assertEqual(len(response_fp), len(reference))
        for value, ref in zip(response_fp, reference):
            self.assertAlmostEqual(value, ref, places)

    def test_lih_2o2e(self):
        grad, bath_response_fp, rohf_response_fp = get_gbci_grad(
            f"Li 0 0 0; H 0 0 {BOND_LENGTH}", 2, (1, 1))
        self.assertAlmostEqual(float(grad[0, 2]), LIH_GRAD_Z, 9)
        self.assert_response_fp(
            bath_response_fp, LIH_BATH_RESPONSE_FP, places=6)
        self.assertEqual(rohf_response_fp, [])

    def test_lif_4o4e(self):
        grad, bath_response_fp, rohf_response_fp = get_gbci_grad(
            f"Li 0 0 0; F 0 0 {BOND_LENGTH}", 4, (2, 2))
        self.assertAlmostEqual(float(grad[0, 2]), LIF_GRAD_Z, 9)
        self.assert_response_fp(
            bath_response_fp, LIF_BATH_RESPONSE_FP, places=6)
        self.assertEqual(rohf_response_fp, [])

    def test_lih_triplet_rohf_2o2e(self):
        grad, bath_response_fp, rohf_response_fp = get_gbci_grad(
            f"Li 0 0 0; H 0 0 {BOND_LENGTH}", 2, (1, 1),
            spin=2, mf_cls=scf.ROHF)
        self.assertAlmostEqual(float(grad[0, 2]), LIH_TRIPLET_ROHF_GRAD_Z, 9)
        self.assert_response_fp(
            bath_response_fp, LIH_TRIPLET_ROHF_BATH_RESPONSE_FP, places=6)
        self.assert_response_fp(
            rohf_response_fp, (LIH_TRIPLET_ROHF_RESPONSE_FP,))

    def test_lif_triplet_rohf_4o4e(self):
        grad, bath_response_fp, rohf_response_fp = get_gbci_grad(
            f"Li 0 0 0; F 0 0 {BOND_LENGTH}", 4, (2, 2),
            spin=2, mf_cls=scf.ROHF)
        self.assertAlmostEqual(
            float(grad[0, 2]), LIF_TRIPLET_ROHF_GRAD_Z, 9)
        self.assert_response_fp(
            bath_response_fp, LIF_TRIPLET_ROHF_BATH_RESPONSE_FP, places=6)
        self.assert_response_fp(
            rohf_response_fp, (LIF_TRIPLET_ROHF_RESPONSE_FP,))


if __name__ == "__main__":
    print("Full Tests for GBCI nuclear gradients")
    unittest.main()
