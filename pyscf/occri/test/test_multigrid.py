"""
Test suite for multigrid OCCRI implementation

Tests for multigrid components including grid hierarchy,
interpolation operators, solvers, and k-point exchange
matrix evaluation.
"""

import unittest
import numpy
from pyscf.pbc import gto, scf
from pyscf.pbc.tools import cutoff_to_mesh

from pyscf.occri.multigrid import MultigridOccRI
from pyscf.occri.multigrid.mg_grids import (
    UniformGrid, AtomGrid, Atoms,
    list_to_slices, _initialize_atoms_and_centers,
    _create_atom_grids, _create_universal_grids,
    _get_atom_centers_with_pbc, _compute_distances_to_atom
)


def setUpModule():
    """Setup test cells for all tests"""
    global cell_small, cell_medium
    
    # Small cell for fast unit tests
    cell_small = gto.Cell()
    cell_small.atom = """
        H 0.000000 0.000000 0.000000
        H 1.200000 0.000000 0.000000
    """
    cell_small.basis = "gth-szv"
    cell_small.pseudo = "gth-pbe"
    cell_small.a = numpy.array([[4.0, 0.0, 0.0],
                                [0.0, 4.0, 0.0],
                                [0.0, 0.0, 4.0]])
    cell_small.mesh = [12, 12, 12]
    cell_small.verbose = 0
    cell_small.build()
    
    # Medium cell for more comprehensive tests
    cell_medium = gto.Cell()
    cell_medium.atom = """
        C 0.000000 0.000000 0.000000
        C 0.890186 0.890186 0.890186
    """
    cell_medium.basis = "gth-szv"
    cell_medium.pseudo = "gth-pbe"
    cell_medium.a = numpy.eye(3) * 3.5607
    cell_medium.mesh = [16, 16, 16]
    cell_medium.verbose = 0
    cell_medium.build()


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions in mg_grids"""
    
    def test_list_to_slices(self):
        """Test conversion of index lists to slices"""
        # Test simple consecutive sequence
        arr = [0, 1, 2, 5, 6, 9]
        slices = list_to_slices(arr)
        expected = [[0, 3], [5, 7], [9, 10]]
        self.assertEqual(slices, expected)
        
        # Test single element
        arr = [5]
        slices = list_to_slices(arr)
        expected = [[5, 6]]
        self.assertEqual(slices, expected)
        
        # Test unsorted input (should be sorted internally)
        arr = [5, 1, 3, 2]
        slices = list_to_slices(arr)
        expected = [[1, 4], [5, 6]]
        self.assertEqual(slices, expected)


class TestUniformGrid(unittest.TestCase):
    """Test UniformGrid class"""
    
    def setUp(self):
        """Setup multigrid OCCRI object for testing"""
        mf = scf.RHF(cell_small)
        self.mydf = MultigridOccRI(mf)
    
    def test_init_with_mesh(self):
        """Test UniformGrid initialization with explicit mesh"""
        mesh = [8, 8, 8]
        ke_cutoff = 10.0
        
        grid = UniformGrid(self.mydf, ke_cutoff, mesh)
        
        self.assertTrue(numpy.array_equal(grid.mesh, numpy.array(mesh, numpy.int32)))
        self.assertEqual(grid.ngrids, numpy.prod(mesh))
        self.assertEqual(grid.coords.shape[0], grid.ngrids)
        self.assertEqual(grid.coords.shape[1], 3)
    
    def test_init_without_mesh(self):
        """Test UniformGrid initialization with mesh calculated from cutoff"""
        ke_cutoff = 20.0
        
        grid = UniformGrid(self.mydf, ke_cutoff)
        
        # Verify mesh was calculated from cutoff
        expected_mesh = cutoff_to_mesh(self.mydf.cell.lattice_vectors(), ke_cutoff)
        self.assertTrue(numpy.array_equal(grid.mesh, expected_mesh))
        self.assertEqual(grid.ngrids, numpy.prod(expected_mesh))
        
        # Check coordinates have correct shape
        self.assertEqual(grid.coords.shape[0], grid.ngrids)
        self.assertEqual(grid.coords.shape[1], 3)


class TestAtomGrid(unittest.TestCase):
    """Test AtomGrid class"""
    
    def setUp(self):
        """Setup test data"""
        self.shell_idx_sharp = numpy.array([0, 2], dtype=numpy.int32)
        self.alpha = numpy.array([1.5, 0.8, 2.1, 0.5], dtype=numpy.float64)
        self.l = numpy.array([0, 1, 0, 1], dtype=numpy.int32)
        self.rcut = numpy.array([2.0, 3.0, 1.8, 3.5], dtype=numpy.float64)
        self.shell_idx = numpy.array([0, 1, 2, 3], dtype=numpy.int32)
        self.ao_index = numpy.arange(8, dtype=numpy.int32)  # 1+3+1+3 = 8 AOs
    
    def test_init_with_sharp_functions(self):
        """Test AtomGrid with sharp functions present"""
        atom_grid = AtomGrid(
            self.shell_idx_sharp, self.alpha, self.l, 
            self.rcut, self.shell_idx, self.ao_index
        )
        
        # Should have sharp functions
        self.assertEqual(len(atom_grid.shell_index_sharp), 2)
        self.assertTrue(numpy.array_equal(atom_grid.shell_index_sharp, [0, 2]))
        
        # Check exponents
        self.assertAlmostEqual(atom_grid.max_exp, 2.1)  # max of shells 0,2
        self.assertAlmostEqual(atom_grid.min_exp, 1.5)  # min of shells 0,2
        
        # Check AO count (shell 0: 1 AO, shell 2: 1 AO)
        self.assertEqual(atom_grid.n_sharp, 2)
        self.assertEqual(atom_grid.nao, 2)
    
    def test_init_no_sharp_functions(self):
        """Test AtomGrid with no sharp functions"""
        empty_sharp = numpy.array([], dtype=numpy.int32)
        
        atom_grid = AtomGrid(
            empty_sharp, self.alpha, self.l, 
            self.rcut, self.shell_idx, self.ao_index
        )
        
        # Should have no functions
        self.assertEqual(atom_grid.rcut[0], -1.0)
        self.assertEqual(atom_grid.max_exp, 0.0)
        self.assertEqual(atom_grid.min_exp, 1.e6)
        self.assertEqual(atom_grid.ngrids, 0)
        self.assertEqual(atom_grid.nao, 0)
        self.assertEqual(atom_grid.n_sharp, 0)


class TestAtoms(unittest.TestCase):
    """Test Atoms class"""
    
    def setUp(self):
        """Setup multigrid OCCRI object"""
        mf = scf.RHF(cell_medium)
        self.mydf = MultigridOccRI(mf)
    
    def test_init_atom_0(self):
        """Test Atoms initialization for first carbon atom"""
        atom = Atoms(self.mydf, 0)
        
        # Check basic properties
        self.assertIsInstance(atom.shell_index, numpy.ndarray)
        self.assertEqual(atom.shell_index.dtype, numpy.int32)
        
        self.assertIsInstance(atom.l, numpy.ndarray)
        self.assertEqual(atom.l.dtype, numpy.int32)
        
        self.assertIsInstance(atom.exponents, numpy.ndarray)
        self.assertEqual(atom.exponents.dtype, numpy.float64)
        
        self.assertIsInstance(atom.ao_index, numpy.ndarray)
        self.assertEqual(atom.ao_index.dtype, numpy.int32)
        
        self.assertIsInstance(atom.rcut, numpy.ndarray)
        self.assertEqual(atom.rcut.dtype, numpy.float64)
        
        # Check atom center
        expected_center = self.mydf.cell_unc.atom_coords()[0]
        self.assertTrue(numpy.allclose(atom.atom_center, expected_center))
        
        # Verify exponents and rcut have same length
        self.assertEqual(len(atom.exponents), len(atom.rcut))
    
    def test_init_with_specific_shells(self):
        """Test Atoms initialization with specific shell indices"""
        # Get first two shells of atom 0
        all_shells = self.mydf.cell_unc.atom_shell_ids(0)
        shell_subset = all_shells[:2] if len(all_shells) >= 2 else all_shells
        
        atom = Atoms(self.mydf, 0, shell_subset)
        
        self.assertTrue(numpy.array_equal(atom.shell_index, shell_subset))
        self.assertEqual(len(atom.l), len(shell_subset))


class TestGridInitialization(unittest.TestCase):
    """Test grid initialization functions"""
    
    def setUp(self):
        """Setup multigrid OCCRI object"""
        mf = scf.RHF(cell_medium)
        self.mydf = MultigridOccRI(mf)
    
    def test_initialize_atoms_and_centers(self):
        """Test _initialize_atoms_and_centers function"""
        atoms, mesh, ke_grid, max_exp = _initialize_atoms_and_centers(self.mydf)
        
        # Check atoms
        self.assertEqual(len(atoms), self.mydf.cell_unc.natm)
        self.assertIsInstance(atoms[0], Atoms)
        
        # Check mesh (should match cell.mesh or be calculated)
        if self.mydf.cell_unc.mesh is not None:
            self.assertTrue(numpy.array_equal(mesh, self.mydf.cell_unc.mesh))
        
        # Check ke_grid is positive
        self.assertGreater(ke_grid, 0)
        
        # Check max_exp is positive and reasonable
        self.assertGreater(max_exp, 0)
        self.assertLess(max_exp, 1000)  # Sanity check
        
        # Verify alpha_cutoff was adjusted if needed
        self.assertLessEqual(self.mydf.alpha_cutoff, max_exp)
    
    def test_create_atom_grids(self):
        """Test _create_atom_grids function"""
        atoms, _, _, _ = _initialize_atoms_and_centers(self.mydf)
        atom_grids, diffuse_atoms = _create_atom_grids(self.mydf, atoms)
        
        # Check output lengths match input
        self.assertEqual(len(atom_grids), len(atoms))
        self.assertEqual(len(diffuse_atoms), len(atoms))
        
        # Check types
        self.assertIsInstance(atom_grids[0], AtomGrid)
        self.assertIsInstance(diffuse_atoms[0], Atoms)
    
    def test_create_universal_grids(self):
        """Test _create_universal_grids function"""
        atoms, mesh, ke_grid, _ = _initialize_atoms_and_centers(self.mydf)
        atom_grids, _ = _create_atom_grids(self.mydf, atoms)
        
        universal_grids = _create_universal_grids(self.mydf, atom_grids, ke_grid, mesh)
        
        # Should create 2 universal grids (dense and sparse)
        self.assertEqual(len(universal_grids), 2)
        
        # Both should be UniformGrid instances
        self.assertIsInstance(universal_grids[0], UniformGrid)
        self.assertIsInstance(universal_grids[1], UniformGrid)
        
        # Dense grid should have larger or equal mesh
        self.assertGreaterEqual(numpy.prod(universal_grids[0].mesh), 
                               numpy.prod(universal_grids[1].mesh))
    
    def test_get_atom_centers_with_pbc(self):
        """Test _get_atom_centers_with_pbc function"""
        centers = _get_atom_centers_with_pbc(self.mydf.cell_unc)
        
        # Should have 27 images per atom (3x3x3)
        expected_total = 27 * self.mydf.cell_unc.natm
        self.assertEqual(centers.shape[0], expected_total)
        self.assertEqual(centers.shape[1], 3)
        
        # Check data type
        self.assertEqual(centers.dtype, numpy.float64)
    
    def test_compute_distances_to_atom(self):
        """Test _compute_distances_to_atom function"""
        # Create test grid points
        grid_coords = numpy.array([[0.0, 0.0, 0.0],
                                  [1.0, 0.0, 0.0],
                                  [2.0, 1.0, 0.0]])
        
        # Create test atom centers (including PBC images)
        atom_centers = numpy.array([[0.0, 0.0, 0.0],
                                   [0.5, 0.0, 0.0]])
        
        distances = _compute_distances_to_atom(grid_coords, atom_centers)
        
        # Check shape
        self.assertEqual(distances.shape[0], grid_coords.shape[0])
        
        # Check that distances are non-negative
        self.assertTrue(numpy.all(distances >= 0))
        
        # Check specific values
        self.assertAlmostEqual(distances[0], 0.0)  # First point at origin
        self.assertAlmostEqual(distances[1], 0.5)  # Second point closer to second center


class TestMultigridOccRI(unittest.TestCase):
    """Test MultigridOccRI class"""
    
    def test_init_basic(self):
        """Test basic MultigridOccRI initialization"""
        mf = scf.RHF(cell_small)
        mg_occri = MultigridOccRI(mf)
        
        # Check inheritance
        self.assertIsInstance(mg_occri.cell, gto.Cell)
        self.assertEqual(mg_occri.cell, mf.mol)
        
        # Check multigrid-specific attributes
        self.assertIsInstance(mg_occri.alpha_cutoff, float)
        self.assertIsInstance(mg_occri.rcut_epsilon, float)
        self.assertIsInstance(mg_occri.ke_epsilon, float)
        self.assertIsInstance(mg_occri.incore, bool)
        
        # Check that grids were built
        self.assertTrue(hasattr(mg_occri, 'cell_unc'))
        self.assertTrue(hasattr(mg_occri, 'c'))
        self.assertTrue(hasattr(mg_occri, 'atom_grids'))
        self.assertTrue(hasattr(mg_occri, 'universal_grids'))
    
    def test_to_uncontracted_basis(self):
        """Test basis uncontraction"""
        mf = scf.RHF(cell_small)
        mg_occri = MultigridOccRI(mf)
        
        # Check that uncontracted cell was created
        self.assertIsInstance(mg_occri.cell_unc, gto.Cell)
        self.assertGreaterEqual(mg_occri.cell_unc.nbas, mg_occri.cell.nbas)
        
        # Check contraction coefficients
        self.assertIsInstance(mg_occri.c, numpy.ndarray)
        self.assertEqual(mg_occri.c.ndim, 2)
    
    def test_primitive_gto_cutoff(self):
        """Test primitive GTO cutoff calculation"""
        mf = scf.RHF(cell_small)
        mg_occri = MultigridOccRI(mf)
        
        # Test with first shell
        shell_idx = numpy.array([0])
        rcut = mg_occri.primitive_gto_cutoff(shell_idx)
        
        self.assertIsInstance(rcut, numpy.ndarray)
        self.assertEqual(rcut.dtype, numpy.float64)
        self.assertTrue(numpy.all(rcut > 0))
    
    def test_primitive_gto_exponent(self):
        """Test primitive GTO exponent calculation"""
        mf = scf.RHF(cell_small)
        mg_occri = MultigridOccRI(mf)
        
        rmin = 1.0
        exponent = mg_occri.primitive_gto_exponent(rmin)
        
        self.assertIsInstance(exponent, float)
        self.assertGreater(exponent, 0)
        
        # Test with very small rmin (should use minimum threshold)
        exponent_small = mg_occri.primitive_gto_exponent(1e-15)
        self.assertGreater(exponent_small, 0)
        self.assertTrue(numpy.isfinite(exponent_small))


class TestGridBuilding(unittest.TestCase):
    """Test complete grid building process"""
    
    def test_build_grids_complete(self):
        """Test complete grid building process"""
        mf = scf.RHF(cell_medium)
        mg_occri = MultigridOccRI(mf)  # This calls build_grids internally
        
        # Check that all grid components were created
        self.assertTrue(hasattr(mg_occri, 'atom_grids'))
        self.assertTrue(hasattr(mg_occri, 'universal_grids'))
        
        # Check atom grids
        self.assertIsInstance(mg_occri.atom_grids, list)
        self.assertGreater(len(mg_occri.atom_grids), 0)
        
        # Check universal grids
        self.assertIsInstance(mg_occri.universal_grids, list)
        self.assertEqual(len(mg_occri.universal_grids), 2)  # Dense and sparse
        
        # Verify last atom grid is the sparse grid
        sparse_grid = mg_occri.atom_grids[-1]
        self.assertIsInstance(sparse_grid, AtomGrid)
        
        # Check that grid coordinates exist
        for ag in mg_occri.atom_grids[:-1]:  # Exclude sparse grid
            if ag.nao > 0:  # Only check grids with basis functions
                self.assertGreater(ag.ng, 0)
                self.assertIsInstance(ag.coord_idx, numpy.ndarray)


class TestGridAssignment(unittest.TestCase):
    """Test basis function assignment to grids"""
    
    def setUp(self):
        """Setup test environment"""
        mf = scf.RHF(cell_small)
        self.mydf = MultigridOccRI(mf)
    
    def test_basis_function_assignment(self):
        """Test that basis functions are properly assigned to grids"""
        # Check that each atom grid has appropriate basis function assignments
        for ag in self.mydf.atom_grids[:-1]:  # Exclude sparse grid
            if ag.nao > 0:
                # Check AO indices are valid
                self.assertTrue(numpy.all(ag.ao_index >= 0))
                self.assertTrue(numpy.all(ag.ao_index < self.mydf.cell_unc.nao))
                
                # Check that sharp functions are subset of all functions
                self.assertTrue(numpy.all(numpy.isin(ag.ao_index_sharp, ag.ao_index)))
                
                # Check metadata consistency
                if hasattr(ag, 'exponents') and len(ag.exponents) > 0:
                    self.assertEqual(len(ag.exponents), ag.nexp)
                    self.assertTrue(numpy.all(ag.exponents > 0))


if __name__ == '__main__':
    unittest.main()
