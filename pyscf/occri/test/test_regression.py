#!/usr/bin/env python
"""
Regression tests for OCCRI using the diamond test case.

These tests verify that OCCRI maintains the same accuracy as the
existing test case in test_get_k.py.
"""

import unittest

import numpy
from pyscf.occri import OCCRI
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf

# Tolerance for energy comparison (Hartree per atom)
TOL = 1.0e-8


class TestRegression(unittest.TestCase):

    def setUp(self):
        """Setup diamond structure test system"""
        self.cell = pbcgto.Cell()
        self.cell.atom = """
            C 0.000000 0.000000 1.780373
            C 0.890186 0.890186 2.670559
            C 0.000000 1.780373 0.000000
            C 0.890186 2.670559 0.890186
            C 1.780373 0.000000 0.000000
            C 2.670559 0.890186 0.890186
            C 1.780373 1.780373 1.780373
            C 2.670559 2.670559 2.670559
        """
        self.cell.a = numpy.array(
            [
                [3.560745, 0.000000, 0.000000],
                [0.000000, 3.560745, 0.000000],
                [0.000000, 0.000000, 3.560745],
            ]
        )
        self.cell.basis = "gth-cc-dzvp"
        self.cell.pseudo = "gth-pbe"
        self.cell.ke_cutoff = 70
        self.cell.verbose = 0  # Suppress output for tests
        self.cell.build()

    def test_rhf_regression(self):
        """Test RHF against known reference energy"""
        # Reference energy from standard FFTDF calculation
        en_fftdf = -43.9399339901445

        mf = scf.RHF(self.cell)
        mf.with_df = OCCRI.from_mf(mf)
        en = mf.kernel()

        en_diff = abs(en - en_fftdf) / self.cell.natm
        self.assertLess(
            en_diff, TOL, f"RHF energy difference {en_diff} exceeds tolerance {TOL}"
        )

    def test_uhf_regression(self):
        """Test UHF against known reference energy"""
        # For closed-shell diamond, UHF should give same energy as RHF
        en_fftdf = -43.9399339901445

        mf = scf.UHF(self.cell)
        mf.with_df = OCCRI.from_mf(mf)
        en = mf.kernel()

        en_diff = abs(en - en_fftdf) / self.cell.natm
        self.assertLess(
            en_diff, TOL, f"UHF energy difference {en_diff} exceeds tolerance {TOL}"
        )

    def test_rks_pbe0_regression(self):
        """Test RKS with PBE0 hybrid functional"""
        # PBE0 contains 25% exact exchange
        en_fftdf = -45.0265010261793

        mf = scf.RKS(self.cell)
        mf.xc = "pbe0"
        mf.with_df = OCCRI.from_mf(mf)
        en = mf.kernel()

        en_diff = abs(en - en_fftdf) / self.cell.natm
        self.assertLess(
            en_diff, TOL, f"RKS energy difference {en_diff} exceeds tolerance {TOL}"
        )

    def test_uks_pbe0_regression(self):
        """Test UKS with PBE0 hybrid functional"""
        # Should match RKS for closed shell
        en_fftdf = -45.0265010261753

        mf = scf.UKS(self.cell)
        mf.xc = "pbe0"
        mf.with_df = OCCRI.from_mf(mf)
        en = mf.kernel()

        en_diff = abs(en - en_fftdf) / self.cell.natm
        self.assertLess(
            en_diff, TOL, f"UKS energy difference {en_diff} exceeds tolerance {TOL}"
        )

    def test_h2_small_system(self):
        """Test on H2 system for basic functionality"""
        h2_cell = pbcgto.Cell()
        h2_cell.atom = """
            H 3.00   3.00   2.10
            H 3.00   3.00   3.90
            """
        h2_cell.a = """
            6.0   0.0   0.0
            0.0   6.0   0.0
            0.0   0.0   6.0
            """
        h2_cell.unit = "B"
        h2_cell.basis = "gth-szv"
        h2_cell.pseudo = "gth-pbe"
        h2_cell.verbose = 0
        h2_cell.build()

        # Reference calculation
        mf_ref = scf.RHF(h2_cell)
        en_ref = mf_ref.kernel()

        # OCCRI calculation
        mf = scf.RHF(h2_cell)
        mf.with_df = OCCRI.from_mf(mf)
        en = mf.kernel()

        # Should be very close for small system
        en_diff = abs(en - en_ref) / h2_cell.natm
        self.assertLess(en_diff, 1e-10, f"H2 energy difference {en_diff} too large")

    def test_convergence_properties(self):
        """Test that calculations properly converge"""
        mf = scf.RHF(self.cell)
        mf.with_df = OCCRI.from_mf(mf)
        mf.kernel()

        # Check convergence
        self.assertTrue(mf.converged, "SCF calculation did not converge")

        # Check energy is reasonable (should be negative for stable system)
        self.assertLess(mf.e_tot, 0, "Total energy should be negative")

        # Check that exchange energy is significant
        vj, vk = mf.get_jk()
        dm = mf.make_rdm1()
        e_exchange = -0.5 * numpy.einsum("ij,ji->", vk, dm)
        self.assertLess(
            e_exchange, -1.0, "Exchange energy should be significant and negative"
        )


if __name__ == "__main__":
    print("Running OCCRI regression tests...")
    unittest.main()
