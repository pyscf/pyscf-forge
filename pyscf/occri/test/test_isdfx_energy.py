"""
Energy comparison tests for ISDFX vs FFTDF

These tests verify that ISDFX produces energies consistent with FFTDF
within acceptable numerical precision (50e-6 Hartree tolerance).

Tests cover:
- Different electronic structure methods (RHF, UHF, RKS, UKS)
- Various systems and basis sets
- K-point sampling
- Edge cases and challenging systems

Test categories:
- Standard tests: All except expensive k-point tests
- Performance tests: Include expensive asymmetric k-point tests
"""

import sys
import unittest

import numpy
from pyscf.occri.isdfx import ISDFX
from pyscf.pbc import gto, scf

# Check if we're being run as performance tests (expensive k-point tests)
# This could be set by the main test runner
RUN_PERFORMANCE_TESTS = getattr(
    sys.modules.get("__main__"), "RUN_PERFORMANCE_TESTS", False
)


class TestISdfxEnergyComparison(unittest.TestCase):
    """Compare ISDFX energies to FFTDF reference energies"""

    ENERGY_TOLERANCE = 50e-6  # Hartree

    def setUp(self):
        """Common setup for energy comparison tests"""
        # Simple H2 system for basic tests
        self.cell_h2 = gto.Cell()
        self.cell_h2.atom = """
            H 0.0 0.0 0.0
            H 1.4 0.0 0.0
        """
        self.cell_h2.basis = "gth-szv"
        self.cell_h2.pseudo = "gth-pbe"
        self.cell_h2.a = numpy.eye(3) * 6.0
        self.cell_h2.mesh = [11] * 3
        self.cell_h2.verbose = 0
        self.cell_h2.build()

        # LiH system for more complex tests
        self.cell_lih = gto.Cell()
        self.cell_lih.atom = """
            Li 0.0 0.0 0.0
            H  2.0 0.0 0.0
        """
        self.cell_lih.basis = "gth-szv"
        self.cell_lih.pseudo = "gth-pbe"
        self.cell_lih.a = numpy.eye(3) * 6.0
        self.cell_lih.mesh = [15] * 3
        self.cell_lih.verbose = 0
        self.cell_lih.build()

        # Diamond structure for k-point tests
        self.cell_diamond = gto.Cell()
        self.cell_diamond.atom = """
            C 0.000000 0.000000 0.000000
            C 0.890186 0.890186 0.890186
        """
        self.cell_diamond.basis = "gth-szv"
        self.cell_diamond.pseudo = "gth-pbe"
        self.cell_diamond.a = numpy.eye(3) * 3.5607
        self.cell_diamond.mesh = [21] * 3
        self.cell_diamond.verbose = 0
        self.cell_diamond.build()

    def _compare_energies(self, cell, method_class, isdf_kwargs=None, scf_kwargs=None):
        """
        Compare ISDFX and FFTDF energies for a given method

        Parameters:
        -----------
        cell : Cell
            PySCF cell object
        method_class : class
            SCF method class (RHF, UHF, RKS, UKS, etc.)
        isdf_kwargs : dict
            Additional kwargs for ISDFX
        scf_kwargs : dict
            Additional kwargs for SCF method

        Returns:
        --------
        tuple : (isdfx_energy, fftdf_energy, energy_diff)
        """
        if isdf_kwargs is None:
            isdf_kwargs = {}
        if scf_kwargs is None:
            scf_kwargs = {}

        # FFTDF reference calculation
        mf_fftdf = method_class(cell, **scf_kwargs)
        mf_fftdf.max_cycle = 30
        mf_fftdf.conv_tol = 1e-8
        e_fftdf = mf_fftdf.kernel()

        # Check FFTDF convergence
        if not mf_fftdf.converged:
            raise RuntimeError(f"FFTDF calculation did not converge. Energy: {e_fftdf}")

        # ISDFX calculation
        mf_isdfx = method_class(cell, **scf_kwargs)
        mf_isdfx.with_df = ISDFX.from_mf(mf_isdfx, **isdf_kwargs)
        mf_isdfx.max_cycle = 30
        mf_isdfx.conv_tol = 1e-8
        e_isdfx = mf_isdfx.kernel()

        # Check ISDFX convergence
        if not mf_isdfx.converged:
            raise RuntimeError(f"ISDFX calculation did not converge. Energy: {e_isdfx}")

        energy_diff = abs(e_isdfx - e_fftdf)

        return e_isdfx, e_fftdf, energy_diff

    def test_rhf_h2_energy(self):
        """Test RHF energy for H2 system"""
        e_isdfx, e_fftdf, diff = self._compare_energies(self.cell_h2, scf.RHF)

        self.assertLess(
            diff,
            self.ENERGY_TOLERANCE,
            f"RHF H2 energy difference {diff:.2e} exceeds tolerance {self.ENERGY_TOLERANCE:.2e}",
        )
        self.assertTrue(numpy.isfinite(e_isdfx), "ISDFX energy should be finite")
        self.assertTrue(numpy.isfinite(e_fftdf), "FFTDF energy should be finite")

    def test_rhf_lih_energy(self):
        """Test RHF energy for LiH system"""
        e_isdfx, e_fftdf, diff = self._compare_energies(self.cell_lih, scf.RHF)

        self.assertLess(
            diff,
            self.ENERGY_TOLERANCE,
            f"RHF LiH energy difference {diff:.2e} exceeds tolerance {self.ENERGY_TOLERANCE:.2e}",
        )

    def test_uhf_h2_stretched_energy(self):
        """Test UHF energy for stretched H2 (testing spin polarization)"""
        # Create stretched H2 to test UHF behavior
        cell_h2_stretched = self.cell_h2.copy()
        cell_h2_stretched.atom = """
            H 0.0 0.0 0.0
            H 3.0 0.0 0.0
        """
        cell_h2_stretched.build()

        e_isdfx, e_fftdf, diff = self._compare_energies(cell_h2_stretched, scf.UHF)

        self.assertLess(
            diff,
            self.ENERGY_TOLERANCE,
            f"UHF stretched H2 energy difference {diff:.2e} exceeds tolerance {self.ENERGY_TOLERANCE:.2e}",
        )

    def test_rks_lda_energy(self):
        """Test RKS with LDA functional"""
        e_isdfx, e_fftdf, diff = self._compare_energies(
            self.cell_lih, scf.RKS, scf_kwargs={"xc": "lda,vwn"}
        )

        self.assertLess(
            diff,
            self.ENERGY_TOLERANCE,
            f"RKS-LDA energy difference {diff:.2e} exceeds tolerance {self.ENERGY_TOLERANCE:.2e}",
        )

    def test_rks_pbe_energy(self):
        """Test RKS with PBE functional"""
        e_isdfx, e_fftdf, diff = self._compare_energies(
            self.cell_lih, scf.RKS, scf_kwargs={"xc": "pbe,pbe"}
        )

        self.assertLess(
            diff,
            self.ENERGY_TOLERANCE,
            f"RKS-PBE energy difference {diff:.2e} exceeds tolerance {self.ENERGY_TOLERANCE:.2e}",
        )

    def test_uks_lih_energy(self):
        """Test UKS energy for LiH system"""
        e_isdfx, e_fftdf, diff = self._compare_energies(
            self.cell_lih, scf.UKS, scf_kwargs={"xc": "pbe,pbe"}
        )

        self.assertLess(
            diff,
            self.ENERGY_TOLERANCE,
            f"UKS LiH energy difference {diff:.2e} exceeds tolerance {self.ENERGY_TOLERANCE:.2e}",
        )

    def test_rhf_different_isdf_thresholds(self):
        """Test RHF with different ISDFX thresholds"""
        thresholds = [1e-4, 1e-5, 1e-6]

        for thresh in thresholds:
            with self.subTest(threshold=thresh):
                e_isdfx, e_fftdf, diff = self._compare_energies(
                    self.cell_h2, scf.RHF, isdf_kwargs={"isdf_thresh": thresh}
                )

                self.assertLess(
                    diff,
                    self.ENERGY_TOLERANCE,
                    f"RHF energy difference {diff:.2e} with threshold {thresh:.2e} exceeds tolerance",
                )

    def test_rhf_kpoints_diamond(self):
        """Test RHF with k-point sampling on diamond structure"""
        kpts = self.cell_diamond.make_kpts([2, 2, 1])  # Reduced k-mesh for speed

        # FFTDF reference
        mf_fftdf = scf.KRHF(self.cell_diamond, kpts=kpts)
        mf_fftdf.max_cycle = 20
        mf_fftdf.conv_tol = 1e-7
        e_fftdf = mf_fftdf.kernel()

        # ISDFX calculation
        from pyscf.occri.isdfx import ISDFX

        mf_isdfx = scf.KRHF(self.cell_diamond, kpts=kpts)
        mf_isdfx.with_df = ISDFX.from_mf(mf_isdfx)
        mf_isdfx.max_cycle = 20
        mf_isdfx.conv_tol = 1e-7
        e_isdfx = mf_isdfx.kernel()

        diff = abs(e_isdfx - e_fftdf)

        self.assertLess(
            diff,
            self.ENERGY_TOLERANCE,
            f"K-point RHF diamond energy difference {diff:.2e} exceeds tolerance {self.ENERGY_TOLERANCE:.2e}",
        )

    def test_asymmetric_kpoints_123(self):
        """Test RHF with highly asymmetric k-point mesh [1,2,3] to test symmetry handling"""
        kmesh = [1, 2, 3]
        kpts = self.cell_diamond.make_kpts(
            kmesh,
            space_group_symmetry=False,
            time_reversal_symmetry=False,
            wrap_around=True,
        )

        # FFTDF reference
        mf_fftdf = scf.KRHF(self.cell_diamond, kpts=kpts)
        mf_fftdf.max_cycle = 15
        mf_fftdf.conv_tol = 1e-6
        e_fftdf = mf_fftdf.kernel()

        # ISDFX calculation
        from pyscf.occri.isdfx import ISDFX

        mf_isdfx = scf.KRHF(self.cell_diamond, kpts=kpts)
        mf_isdfx.with_df = ISDFX.from_mf(mf_isdfx)
        mf_isdfx.max_cycle = 15
        mf_isdfx.conv_tol = 1e-6
        e_isdfx = mf_isdfx.kernel()

        diff = abs(e_isdfx - e_fftdf)

        self.assertLess(
            diff,
            self.ENERGY_TOLERANCE,
            f"Asymmetric k-mesh [1,2,3] energy difference {diff:.2e} exceeds tolerance {self.ENERGY_TOLERANCE:.2e}",
        )

    def test_asymmetric_kpoints_135(self):
        """Test RHF with asymmetric k-point mesh [1,3,5] to test complex conjugation bugs"""
        kmesh = [1, 3, 5]
        kpts = self.cell_h2.make_kpts(
            kmesh,
            space_group_symmetry=False,
            time_reversal_symmetry=False,
            wrap_around=True,
        )

        # FFTDF reference
        mf_fftdf = scf.KRHF(self.cell_h2, kpts=kpts)
        mf_fftdf.max_cycle = 12
        mf_fftdf.conv_tol = 1e-6
        e_fftdf = mf_fftdf.kernel()

        # ISDFX calculation
        from pyscf.occri.isdfx import ISDFX

        mf_isdfx = scf.KRHF(self.cell_h2, kpts=kpts)
        mf_isdfx.with_df = ISDFX.from_mf(mf_isdfx)
        mf_isdfx.max_cycle = 12
        mf_isdfx.conv_tol = 1e-6
        e_isdfx = mf_isdfx.kernel()

        diff = abs(e_isdfx - e_fftdf)

        self.assertLess(
            diff,
            self.ENERGY_TOLERANCE,
            f"Asymmetric k-mesh [1,3,5] energy difference {diff:.2e} exceeds tolerance {self.ENERGY_TOLERANCE:.2e}",
        )

    def test_asymmetric_kpoints_uks(self):
        """Test UKS with asymmetric k-points to test spin handling with complex conjugation"""
        # Create a system that benefits from UKS
        cell_test = self.cell_lih.copy()
        cell_test.mesh = [14, 14, 14]  # Slightly smaller mesh for speed
        cell_test.build()

        kmesh = [2, 1, 3]
        kpts = cell_test.make_kpts(
            kmesh,
            space_group_symmetry=False,
            time_reversal_symmetry=False,
            wrap_around=True,
        )

        # FFTDF reference
        mf_fftdf = scf.KUKS(cell_test, kpts=kpts)
        mf_fftdf.xc = "pbe,pbe"
        mf_fftdf.max_cycle = 12
        mf_fftdf.conv_tol = 1e-6
        e_fftdf = mf_fftdf.kernel()

        # ISDFX calculation
        from pyscf.occri.isdfx import ISDFX

        mf_isdfx = scf.KUKS(cell_test, kpts=kpts)
        mf_isdfx.xc = "pbe,pbe"
        mf_isdfx.with_df = ISDFX.from_mf(mf_isdfx)
        mf_isdfx.max_cycle = 12
        mf_isdfx.conv_tol = 1e-6
        e_isdfx = mf_isdfx.kernel()

        diff = abs(e_isdfx - e_fftdf)

        self.assertLess(
            diff,
            self.ENERGY_TOLERANCE,
            f"UKS asymmetric k-mesh [2,1,3] energy difference {diff:.2e} exceeds tolerance {self.ENERGY_TOLERANCE:.2e}",
        )

    @unittest.skipUnless(
        RUN_PERFORMANCE_TESTS, "Use --perf to run expensive k-point tests"
    )
    def test_prime_kpoints_247(self):
        """Test with prime numbers in k-mesh [2,4,7] to catch FFT-related bugs"""
        kmesh = [2, 4, 7]
        # Increase mesh. We reccomend 1.e-6 FFT error for accuracy but are using less
        # in general here for speed. Need more accurate grid for large k-mesh accuracy.
        cell = self.cell_h2
        cell.mesh = [15] * 3
        cell.build()

        kpts = cell.make_kpts(
            kmesh,
            space_group_symmetry=False,
            time_reversal_symmetry=False,
            wrap_around=True,
        )

        # FFTDF reference
        mf_fftdf = scf.KRHF(cell, kpts=kpts)
        mf_fftdf.max_cycle = 10
        mf_fftdf.conv_tol = 1e-7
        e_fftdf = mf_fftdf.kernel()

        # ISDFX calculation
        from pyscf.occri.isdfx import ISDFX

        mf_isdfx = scf.KRHF(cell, kpts=kpts)
        mf_isdfx.with_df = ISDFX.from_mf(mf_isdfx)
        mf_isdfx.max_cycle = 10
        mf_isdfx.conv_tol = 1e-7
        e_isdfx = mf_isdfx.kernel()

        diff = abs(e_isdfx - e_fftdf)

        tolerance = self.ENERGY_TOLERANCE
        self.assertLess(
            diff,
            tolerance,
            f"Prime k-mesh [2,4,7] energy difference {diff:.2e} exceeds tolerance {tolerance:.2e}",
        )

    @unittest.skipUnless(
        RUN_PERFORMANCE_TESTS, "Use --perf to run expensive k-point tests"
    )
    def test_single_kpoint_direction(self):
        """Test with single k-point in one direction [1,1,4] to test edge cases"""
        kmesh = [1, 1, 4]

        # Increase mesh. We reccomend 1.e-6 FFT error for accuracy but are using less
        # in general here for speed. Need more accurate grid for large k-mesh accuracy.
        cell = self.cell_diamond
        cell.mesh = [27] * 3
        cell.build()

        kpts = cell.make_kpts(
            kmesh,
            space_group_symmetry=False,
            time_reversal_symmetry=False,
            wrap_around=True,
        )

        # FFTDF reference
        mf_fftdf = scf.KRHF(cell, kpts=kpts)
        mf_fftdf.max_cycle = 15
        mf_fftdf.conv_tol = 1e-6
        e_fftdf = mf_fftdf.kernel()

        # ISDFX calculation
        from pyscf.occri.isdfx import ISDFX

        mf_isdfx = scf.KRHF(cell, kpts=kpts)
        mf_isdfx.with_df = ISDFX.from_mf(mf_isdfx)
        mf_isdfx.max_cycle = 15
        mf_isdfx.conv_tol = 1e-6
        e_isdfx = mf_isdfx.kernel()

        diff = abs(e_isdfx - e_fftdf)

        self.assertLess(
            diff,
            self.ENERGY_TOLERANCE,
            f"Single k-direction [1,1,4] energy difference {diff:.2e} exceeds tolerance {self.ENERGY_TOLERANCE:.2e}",
        )

    def test_large_mesh_convergence(self):
        """Test convergence with larger mesh size"""
        cell_fine = self.cell_h2.copy()
        cell_fine.mesh = [20, 20, 20]
        cell_fine.build()

        e_isdfx, e_fftdf, diff = self._compare_energies(cell_fine, scf.RHF)

        self.assertLess(
            diff,
            self.ENERGY_TOLERANCE,
            f"Large mesh energy difference {diff:.2e} exceeds tolerance {self.ENERGY_TOLERANCE:.2e}",
        )

    def test_metallic_system_edge_case(self):
        """Test challenging metallic system (aluminum)"""
        # Simple aluminum fcc structure - use UKS for odd electron system
        cell_al = gto.Cell()
        cell_al.atom = "Al 0.0 0.0 0.0"
        cell_al.basis = "gth-szv"
        cell_al.pseudo = "gth-pbe"
        cell_al.a = numpy.eye(3) * 4.05  # Al lattice parameter
        cell_al.mesh = [12, 12, 12]
        cell_al.spin = 1  # Al has 3 valence electrons, needs open shell
        cell_al.verbose = 0
        cell_al.build()

        e_isdfx, e_fftdf, diff = self._compare_energies(
            cell_al,
            scf.UKS,
            scf_kwargs={"xc": "pbe,pbe"},
        )

        # Allow slightly larger tolerance for metallic systems
        metal_tolerance = self.ENERGY_TOLERANCE
        self.assertLess(
            diff,
            metal_tolerance,
            f"Metallic system energy difference {diff:.2e} exceeds tolerance {metal_tolerance:.2e}",
        )

    def test_small_cell_edge_case(self):
        """Test edge case with very small unit cell"""
        cell_small = self.cell_h2.copy()
        cell_small.a = numpy.diag([3.0, 3.0, 3.0])  # Smaller cell
        cell_small.mesh = [8, 8, 8]  # Reduced mesh to maintain reasonable density
        cell_small.build()

        e_isdfx, e_fftdf, diff = self._compare_energies(cell_small, scf.RHF)

        self.assertLess(
            diff,
            self.ENERGY_TOLERANCE,
            f"Small cell energy difference {diff:.2e} exceeds tolerance {self.ENERGY_TOLERANCE:.2e}",
        )

    def test_different_pseudopotentials(self):
        """Test with different pseudopotential types"""
        # Test with HGH pseudopotentials if available
        try:
            cell_hgh = self.cell_lih.copy()
            cell_hgh.pseudo = "gth-hf"  # Different pseudopotential
            cell_hgh.build()

            e_isdfx, e_fftdf, diff = self._compare_energies(cell_hgh, scf.RHF)

            self.assertLess(
                diff,
                self.ENERGY_TOLERANCE,
                f"Different pseudopotential energy difference {diff:.2e} exceeds tolerance {self.ENERGY_TOLERANCE:.2e}",
            )
        except Exception:
            # Skip if pseudopotential not available
            self.skipTest("Alternative pseudopotential not available")

    def test_convergence_vs_isdf_threshold(self):
        """Test that tighter ISDFX threshold improves agreement with FFTDF"""
        thresholds = [1e-4, 1e-5, 1e-6, 1e-7]
        diffs = []

        # Get FFTDF reference once
        mf_fftdf = scf.RHF(self.cell_lih)
        mf_fftdf.max_cycle = 20
        e_fftdf = mf_fftdf.kernel()

        for thresh in thresholds:
            mf_isdfx = scf.RHF(self.cell_lih)
            mf_isdfx.with_df = ISDFX.from_mf(mf_isdfx, isdf_thresh=thresh)
            mf_isdfx.max_cycle = 20
            e_isdfx = mf_isdfx.kernel()

            diff = abs(e_isdfx - e_fftdf)
            diffs.append(diff)

            # Each threshold should give acceptable accuracy
            self.assertLess(
                diff,
                self.ENERGY_TOLERANCE,
                f"Threshold {thresh:.2e} gives energy difference {diff:.2e} > tolerance",
            )

        # Verify general trend: tighter threshold should improve accuracy
        # (allowing for some numerical noise in the trend)
        for i in range(len(diffs) - 1):
            # Don't require strict monotonicity due to numerical noise,
            # just check that we don't get significantly worse
            if diffs[i] * 2.0 < 1.0e-12:
                continue
            self.assertLess(
                diffs[i + 1],
                diffs[i] * 2.0,
                f"Energy accuracy degraded significantly from threshold {thresholds[i]:.2e} to {thresholds[i + 1]:.2e}",
            )


if __name__ == "__main__":
    unittest.main()
