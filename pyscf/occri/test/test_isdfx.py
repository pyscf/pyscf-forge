"""
Test suite for ISDFX (Interpolative Separable Density Fitting eXchange)

Tests the ISDFX implementation including:
- ISDFX interpolation functions
- THC potential calculation
- Exchange matrix evaluation
- Integration with PySCF workflow
"""

import unittest

import numpy
from pyscf.occri.isdfx import ISDFX
from pyscf.occri.isdfx.utils import (ao_indices_by_atom,
                                     pivoted_cholesky_decomposition,
                                     voronoi_partition)
from pyscf.pbc import gto, scf


def setUpModule():
    """Setup test cells for all tests"""
    global cell_h2, cell_diamond, kpts_2x2x2

    # Simple H2 cell for fast unit tests
    cell_h2 = gto.Cell()
    cell_h2.atom = """
        H 0.000000 0.000000 0.000000
        H 1.500000 0.000000 0.000000
    """
    cell_h2.basis = "gth-szv"
    cell_h2.pseudo = "gth-pbe"
    cell_h2.a = numpy.array([[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]])
    cell_h2.mesh = [12, 12, 12]
    cell_h2.verbose = 0
    cell_h2.build()

    # Diamond structure for more realistic tests
    cell_diamond = gto.Cell()
    cell_diamond.atom = """
        C 0.000000 0.000000 0.000000
        C 0.890186 0.890186 0.890186
    """
    cell_diamond.basis = "gth-szv"
    cell_diamond.pseudo = "gth-pbe"
    cell_diamond.a = numpy.eye(3) * 3.5607
    cell_diamond.mesh = [16, 16, 16]
    cell_diamond.verbose = 0
    cell_diamond.build()

    # k-points for testing
    kpts_2x2x2 = cell_diamond.make_kpts(
        [2, 2, 2],
        space_group_symmetry=False,
        time_reversal_symmetry=False,
        wrap_around=True,
    )


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions in ISDFX interpolation"""

    def test_voronoi_partition(self):
        """Test Voronoi partitioning of grid points"""
        mf = scf.RHF(cell_h2)
        mydf = ISDFX.from_mf(mf)

        # Test basic partitioning
        coords_by_atom = voronoi_partition(mydf)

        # Should have one list per atom
        self.assertEqual(len(coords_by_atom), cell_h2.natm)

        # Each should be numpy array of grid indices
        for coord_list in coords_by_atom:
            self.assertIsInstance(coord_list, numpy.ndarray)
            self.assertEqual(coord_list.dtype, numpy.int32)

        # Total grid points should equal mesh size
        total_points = sum(len(coord_list) for coord_list in coords_by_atom)
        expected_total = numpy.prod(mydf.mesh)
        self.assertEqual(total_points, expected_total)

        # No duplicate indices across atoms
        all_indices = numpy.concatenate(coords_by_atom)
        self.assertEqual(len(all_indices), len(numpy.unique(all_indices)))

    def test_init_ao_indices(self):
        """Test AO index initialization by atom"""
        mf = scf.RHF(cell_diamond)
        mydf = ISDFX.from_mf(mf)

        ao_indices = ao_indices_by_atom(mydf.cell)

        # Should have one array per atom
        self.assertEqual(len(ao_indices), cell_diamond.natm)

        # Check data types and ranges
        for atom_aos in ao_indices:
            self.assertIsInstance(atom_aos, numpy.ndarray)
            self.assertEqual(atom_aos.dtype, numpy.int32)
            self.assertTrue(numpy.all(atom_aos >= 0))
            self.assertTrue(numpy.all(atom_aos < cell_diamond.nao))

        # Total AOs should match cell
        total_aos = sum(len(atom_aos) for atom_aos in ao_indices)
        self.assertEqual(total_aos, cell_diamond.nao)


class TestCholeskyDecomposition(unittest.TestCase):
    """Test pivoted Cholesky decomposition function"""

    def setUp(self):
        """Setup ISDFX object for testing"""
        mf = scf.RHF(cell_h2)
        self.mydf = ISDFX.from_mf(mf, isdf_thresh=1e-4)  # Looser threshold for testing

    def test_cholesky_basic(self):
        """Test basic Cholesky decomposition functionality"""
        # Create test AO values (small for testing)
        nao, ngrids = 4, 10

        # Mock AO values
        aovals = [
            numpy.random.rand(nao, ngrids) + 0.1j * numpy.random.rand(nao, ngrids)
        ]

        # Test without AO restriction
        pivots = pivoted_cholesky_decomposition(self.mydf, aovals)

        self.assertIsInstance(pivots, numpy.ndarray)
        self.assertEqual(pivots.dtype, numpy.int32)
        self.assertTrue(len(pivots) > 0)
        self.assertTrue(numpy.all(pivots >= 0))
        self.assertTrue(numpy.all(pivots < ngrids))

    def test_cholesky_with_ao_subset(self):
        """Test Cholesky with AO index restriction"""
        nao, ngrids = 6, 15

        # Mock AO values
        aovals = [
            numpy.random.rand(nao, ngrids) + 0.1j * numpy.random.rand(nao, ngrids)
        ]

        # Test with AO subset
        ao_subset = numpy.array([0, 2, 4], dtype=numpy.int32)
        pivots = pivoted_cholesky_decomposition(self.mydf, aovals, ao_subset)

        self.assertIsInstance(pivots, numpy.ndarray)
        self.assertEqual(pivots.dtype, numpy.int32)
        self.assertTrue(len(pivots) >= 0)  # Could be empty with small test case


class TestISDFX(unittest.TestCase):
    """Test main ISDFX class functionality"""

    def test_isdf_initialization(self):
        """Test ISDFX object initialization"""
        mf = scf.RHF(cell_h2)
        mydf = ISDFX.from_mf(mf, isdf_thresh=1e-6)

        # Check basic attributes
        self.assertEqual(mydf.cell, mf.mol)
        self.assertEqual(mydf.isdf_thresh, 1e-6)

        # Check that build was called
        self.assertTrue(hasattr(mydf, "pivots"))
        self.assertTrue(hasattr(mydf, "aovals"))
        self.assertTrue(hasattr(mydf, "W"))

    def test_isdf_with_kpoints(self):
        """Test ISDFX with k-point sampling"""
        mf = scf.KRHF(cell_diamond, kpts_2x2x2)
        mydf = ISDFX.from_mf(mf, isdf_thresh=1e-5)

        # Check k-point handling
        self.assertEqual(len(mydf.kpts), len(kpts_2x2x2))
        self.assertTrue(numpy.allclose(mydf.kpts, kpts_2x2x2))

        # Check ISDFX data structures
        self.assertIsInstance(mydf.pivots, numpy.ndarray)
        self.assertIsInstance(mydf.aovals, list)
        self.assertEqual(len(mydf.aovals), len(kpts_2x2x2))

        # Check THC potential
        self.assertTrue(hasattr(mydf, "W"))
        self.assertIsInstance(mydf.W, numpy.ndarray)

    def test_isdf_threshold_validation(self):
        """Test ISDFX threshold parameter validation"""
        mf = scf.RHF(cell_h2)

        # Valid thresholds should work
        mydf1 = ISDFX.from_mf(mf, isdf_thresh=1e-6)
        self.assertEqual(mydf1.isdf_thresh, 1e-6)

        mydf2 = ISDFX.from_mf(mf, isdf_thresh=1e-3)
        self.assertEqual(mydf2.isdf_thresh, 1e-3)


class TestPivotSelection(unittest.TestCase):
    """Test ISDFX pivot point selection"""

    def setUp(self):
        """Setup for pivot selection tests"""
        mf = scf.RHF(cell_h2)
        self.mydf = ISDFX.from_mf(mf, isdf_thresh=1e-4)

    def test_get_pivots_execution(self):
        """Test that get_pivots executes without error"""
        # This function is called during ISDFX initialization,
        # so we test that it completed successfully
        self.assertTrue(hasattr(self.mydf, "pivots"))
        self.assertTrue(hasattr(self.mydf, "aovals"))

        # Check pivot properties
        self.assertIsInstance(self.mydf.pivots, numpy.ndarray)
        self.assertEqual(self.mydf.pivots.dtype, numpy.int32)
        self.assertTrue(len(self.mydf.pivots) > 0)

        # Pivots should be valid grid indices
        max_grid_idx = numpy.prod(self.mydf.mesh) - 1
        self.assertTrue(numpy.all(self.mydf.pivots >= 0))
        self.assertTrue(numpy.all(self.mydf.pivots <= max_grid_idx))

        # Pivots should be sorted and unique
        self.assertTrue(
            numpy.array_equal(self.mydf.pivots, numpy.sort(self.mydf.pivots))
        )
        self.assertEqual(len(self.mydf.pivots), len(numpy.unique(self.mydf.pivots)))

    def test_pivot_compression(self):
        """Test that pivot selection achieves compression"""
        total_grid_points = numpy.prod(self.mydf.mesh)
        selected_pivots = len(self.mydf.pivots)

        # Should select significantly fewer points than total grid
        compression_ratio = selected_pivots / total_grid_points
        self.assertLess(compression_ratio, 0.5)  # At least 50% compression
        self.assertGreater(selected_pivots, 0)  # But not zero points


class TestFittingFunctions(unittest.TestCase):
    """Test ISDFX fitting function construction"""

    def setUp(self):
        """Setup for fitting function tests"""
        mf = scf.RHF(cell_h2)
        self.mydf = ISDFX.from_mf(mf, isdf_thresh=1e-4)

    def test_get_fitting_functions_execution(self):
        """Test fitting function construction"""
        # Functions are built during initialization, test they exist
        self.assertTrue(hasattr(self.mydf, "aovals"))

        # AO values should be updated to pivot points only
        for ao_k in self.mydf.aovals:
            self.assertIsInstance(ao_k, numpy.ndarray)
            self.assertEqual(ao_k.shape[1], len(self.mydf.pivots))


class TestTHCPotential(unittest.TestCase):
    """Test THC potential calculation"""

    def setUp(self):
        """Setup for THC potential tests"""
        mf = scf.RHF(cell_h2)
        self.mydf = ISDFX.from_mf(mf, isdf_thresh=1e-4)

    def test_thc_potential_exists(self):
        """Test that THC potential was computed"""
        self.assertTrue(hasattr(self.mydf, "W"))
        self.assertIsInstance(self.mydf.W, numpy.ndarray)

    def test_thc_potential_shape(self):
        """Test THC potential tensor shape"""
        W = self.mydf.W
        npivots = len(self.mydf.pivots)

        # Should have shape (*kmesh, npivots, npivots)
        expected_shape = tuple(self.mydf.kmesh) + (npivots, npivots)
        self.assertEqual(W.shape, expected_shape)


class TestExchangeMatrixEvaluation(unittest.TestCase):
    """Test ISDFX exchange matrix computation"""

    def setUp(self):
        """Setup for exchange matrix tests"""
        mf = self.mf = scf.RHF(cell_h2)
        self.mydf = ISDFX.from_mf(mf, isdf_thresh=1e-4)

        # Run SCF to get density matrices with orbitals
        mf.with_df = self.mydf
        mf.kernel()
        self.dm = mf.make_rdm1()

    def test_exchange_matrix_computation(self):
        """Test basic exchange matrix evaluation"""
        # Test that exchange matrix computation works
        _, vk = self.mf.get_jk(
            dm=self.dm, with_j=False, with_k=True, kpts=self.mydf.kpts
        )

        self.assertIsInstance(vk, numpy.ndarray)

        # Should be real for gamma-point calculation
        if len(self.mydf.kpts) == 1 and numpy.allclose(self.mydf.kpts[0], 0):
            self.assertTrue(numpy.allclose(vk.imag, 0, atol=1e-10))

    def test_exchange_matrix_hermiticity(self):
        """Test exchange matrix Hermiticity"""
        _, vk = self.mydf.get_jk(
            dm=self.dm, with_j=False, with_k=True, kpts=numpy.zeros(3)
        )

        # Exchange matrix should be Hermitian for each k-point
        for n in range(vk.shape[0]):  # spin sets
            for k in range(vk.shape[1]):  # k-points
                vk_nk = vk[n, k]
                vk_herm = numpy.conj(vk_nk.T)

                self.assertTrue(numpy.allclose(vk_nk, vk_herm, rtol=1e-10, atol=1e-12))


class TestISdfxKpoints(unittest.TestCase):
    """Test ISDFX with k-point calculations"""

    def test_kpoint_exchange_evaluation(self):
        """Test exchange matrix evaluation with k-points"""
        # Use small k-point set for testing
        kmesh = [2, 2, 1]
        kpts = cell_diamond.make_kpts(kmesh)  # Reduced for testing
        mf = scf.KRHF(cell_diamond, kpts=kpts)
        mydf = ISDFX.from_mf(mf, isdf_thresh=1e-3)  # Looser threshold for speed

        mf.with_df = mydf
        mf.max_cycle = 3  # Just a few iterations for testing
        mf.kernel()

        dm = mf.make_rdm1()
        _, vk = mf.get_jk(dm=dm, with_j=False, with_k=True, kpts=kpts)

        # Check output shape
        expected_shape = (kpts.shape[0], cell_diamond.nao, cell_diamond.nao)
        self.assertEqual(vk.shape, expected_shape)

        # Check that matrices are finite
        self.assertTrue(numpy.all(numpy.isfinite(vk)))


class TestISdfxIntegration(unittest.TestCase):
    """Test ISDFX integration with PySCF workflow"""

    def test_scf_convergence(self):
        """Test that SCF converges with ISDFX"""
        mf = scf.RHF(cell_h2)
        mydf = ISDFX.from_mf(mf, isdf_thresh=1e-4)

        mf.with_df = mydf
        mf.max_cycle = 10
        mf.conv_tol = 1e-6

        # Should converge without errors
        energy = mf.kernel()

        self.assertTrue(mf.converged)
        self.assertIsInstance(energy, float)
        self.assertTrue(numpy.isfinite(energy))

    def test_energy_consistency(self):
        """Test energy consistency between runs"""
        mf1 = scf.RHF(cell_h2)
        mydf1 = ISDFX.from_mf(mf1, isdf_thresh=1e-5)
        mf1.with_df = mydf1
        mf1.max_cycle = 8
        e1 = mf1.kernel()

        mf2 = scf.RHF(cell_h2)
        mydf2 = ISDFX.from_mf(mf2, isdf_thresh=1e-5)
        mf2.with_df = mydf2
        mf2.max_cycle = 8
        e2 = mf2.kernel()

        # Energies should be very close (within numerical precision)
        self.assertAlmostEqual(e1, e2, places=10)


if __name__ == "__main__":
    unittest.main()
