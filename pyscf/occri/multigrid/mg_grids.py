"""
Grid management for multigrid OCCRI

This module handles the creation and management of exchange grids
for multigrid methods.
"""

import numpy
from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.pbc.gto.cell import estimate_ke_cutoff

class UniformGrid:
    """
    Represents a uniform grid for electronic structure calculations.
    """
    
    def __init__(self, mydf, ke_cutoff, mesh=None):
        """
        Initialize grid with specified kinetic energy cutoff.
        
        Parameters:
        -----------
        mydf : object
            ISDF object containing cell information
        ke_cutoff : float
            Kinetic energy cutoff in Hartree
        mesh : tuple, optional
            Grid dimensions; if None, calculated from cutoff
        """
        cell = mydf.cell
        
        # Calculate mesh from cutoff if not provided
        if mesh is None:
            mesh = tools.cutoff_to_mesh(cell.lattice_vectors(), ke_cutoff)

        self.mesh = numpy.array(mesh, numpy.int32)
        self.ngrids = numpy.prod(mesh)  # Total number of grid points
        
        # Generate uniform grid coordinates
        self.coords = cell.get_uniform_grids(mesh=self.mesh, wrap_around=True)


class AtomGrid:
    """
    Represents a grid associated with a specific atom and its basis functions.
    """
    
    def __init__(self, shell_idx_sharp, alpha, l, rcut, shell_idx, ao_index):
        """
        Initialize atom-specific grid information.
        
        Parameters:
        -----------
        shell_idx_sharp : array_like
            Shell indices that are included on this grid
        alpha : array_like
            Gaussian exponents for each shell
        l : array_like
            Angular momentum quantum numbers
        rcut : array_like
            Cutoff radii for each shell
        shell_idx : array_like
            Global shell indices
        ao_index : array_like
            Atomic orbital indices
        """
        # Find which shells are actually present on this grid
        sharp_idx = numpy.isin(shell_idx, shell_idx_sharp)

        # Handle case where no shells are on this grid
        if sum(sharp_idx) == 0:
            self.rcut = numpy.asarray([-1.0], numpy.float64)
            self.max_exp = 0.0
            self.min_exp = 1.e6
            self.ngrids = 0
            self.nao = 0
            self.nthc = 0
            self.n_sharp = 0
            self.n_diffuse = 0
            return

        # Store information for shells present on this grid
        self.shell_index_sharp = shell_idx[sharp_idx]
        self.max_exp = max(alpha[sharp_idx])  # Maximum exponent
        self.min_exp = min(alpha[sharp_idx])  # Minimum exponent
        self.rcut = rcut[sharp_idx]           # Cutoff radii
        self.l = l[sharp_idx]                 # Angular momenta
        
        # Account for angular momentum multiplicity (2l+1 orbitals per shell)
        multiplicity = 2 * l + 1
        sharp_idx = numpy.repeat(sharp_idx, multiplicity)
        
        # Map to actual atomic orbital indices
        self.ao_index_sharp = ao_index[sharp_idx]
        self.n_sharp = len(self.ao_index_sharp)  # Number of sharp orbitals
        self.nao = len(self.ao_index_sharp)      # Total number of AOs
        self.nthc = 0                            # Initialize counter


class Atoms:
    """
    Represents atomic information for grid-based calculations.
    """
    
    def __init__(self, mydf, atom_index, shell_index=None):
        """
        Initialize atomic information for a specific atom.
        
        Parameters:
        -----------
        atom_index : int
            Index of the atom
        shell_index : array_like
            Indices of shells belonging to this atom
        """
        
        cell = mydf.cell_unc

        if shell_index is None:
            shell_index = cell.atom_shell_ids(atom_index)

        self.shell_index = numpy.asarray(shell_index, dtype=numpy.int32)

        # Extract angular momentum quantum numbers for each shell
        self.l = numpy.asarray([cell.bas_angular(j) for j in shell_index], numpy.int32)
        
        # Extract and concatenate Gaussian exponents
        self.exponents = numpy.concatenate([cell.bas_exp(j) for j in shell_index], dtype=numpy.float64)
        
        # Get atomic orbital indices for this atom
        ao_slice = cell.aoslice_by_atom()[atom_index]
        ao_index = numpy.arange(ao_slice[2], ao_slice[3], dtype=numpy.int32)
        
        # Get angular momenta for all shells on this atom
        atom_shells = cell.atom_shell_ids(atom_index)
        l = numpy.asarray([cell.bas_angular(j) for j in atom_shells], dtype=numpy.int32)
        
        # Account for angular momentum multiplicity
        multiplicity = 2 * l + 1
        shell_idx = numpy.isin(atom_shells, shell_index)
        shell_idx = numpy.repeat(shell_idx, multiplicity)
        
        # Filter AO indices to include only relevant shells
        self.ao_index = ao_index[shell_idx]
        
        # Calculate cutoff radii for primitive Gaussians
        self.rcut = mydf.primitive_gto_cutoff(shell_index)
        
        # Store atomic center coordinates
        self.atom_center = cell.atom_coords()[atom_index]

def list_to_slices(arr):
    arr = numpy.sort(arr)
    it = iter(arr)
    start = next(it)
    slices = []
    for i, x in enumerate(it):
        if x - arr[i] != 1:
            end = arr[i]
            slices.append([start, end + 1])
            start = x
    slices.append([start, arr[-1] + 1])

    return slices

def _initialize_atoms_and_centers(mydf):
    """
    Setup atoms and determine mesh specifications.
    """
    cell = mydf.cell_unc
    atoms = [Atoms(mydf, i) for i in range(cell.natm)]
    
    # Determine mesh specifications
    # Priority: k_grid_mesh > cell.mesh > default from ke_cutoff
    mesh = None
    if mydf.k_grid_mesh is not None:
        mesh = mydf.k_grid_mesh[0]
    elif cell.mesh is not None:
        mesh = cell.mesh
    
    # Calculate kinetic energy cutoff
    ke_grid = (
        tools.mesh_to_cutoff(cell.lattice_vectors(), mesh)[0] 
        if mesh is not None 
        else cell.ke_cutoff
    )
    
    # Find maximum exponent and adjust alpha_cutoff if needed
    max_exp = max(atm.exponents.max() for atm in atoms)
    if mydf.alpha_cutoff > max_exp:
        mydf.alpha_cutoff = max_exp
        
    return atoms, mesh, ke_grid, max_exp


def _create_atom_grids(mydf, atoms):
    """
    Create atom-centered grids based on exponent thresholds.
    """
    cell = mydf.cell_unc
    atom_grids = []
    diffuse_atoms = []
    
    for i, atom in enumerate(atoms):
        # Separate sharp and diffuse exponents using alpha_cutoff
        is_sharp = numpy.greater(atom.exponents, mydf.alpha_cutoff - cell.precision)
        is_diffuse = ~is_sharp
        
        # Create atom grid for sharp functions
        atom_grid = AtomGrid(
            atom.shell_index[is_sharp],  # atom-centered sharp shells
            atom.exponents,
            atom.l,
            atom.rcut,
            atom.shell_index,
            atom.ao_index,
        )
        atom_grids.append(atom_grid)
        
        # Create diffuse atom object for sparse grid
        diffuse_atom = Atoms(mydf, i, atom.shell_index[is_diffuse])
        diffuse_atoms.append(diffuse_atom)
    
    return atom_grids, diffuse_atoms


def build_grids(mydf):
    """
    Create multi-grid system optimized for exchange (K) matrix calculations.
    """
    # Initialize basic structures
    atoms, mesh, ke_grid, max_exp = _initialize_atoms_and_centers(mydf)
    
    # Create atom-centered grids
    atom_grids, diffuse_atoms = _create_atom_grids(mydf, atoms)
    
    # Store in mydf for later access
    mydf.atom_grids = atom_grids
    mydf.universal_grids = []
    
    # Create universal grids
    universal_grids = _create_universal_grids(mydf, atom_grids, ke_grid, mesh)
    
    # Assign basis functions to grids
    _assign_basis_functions_to_grids(mydf, atom_grids, universal_grids, atoms)
    
    # Finalize grid setup
    _finalize_grid_setup(mydf, atom_grids, universal_grids, atoms, diffuse_atoms)

    if mydf.cell.verbose > 3:
        print_grids(atom_grids, universal_grids, atoms, mydf.cell_unc.nao, mydf.cell_unc.nbas)


def _create_universal_grids(mydf, atom_grids, ke_grid, mesh):
    """
    Create universal grids for different resolution levels.
    """
    universal_grids = []
    
    # Create dense universal grid
    dense_grid = UniformGrid(mydf, ke_grid, mesh)
    dense_grid.ke_min = 0.0  # Will be updated later
    dense_grid.ke_max = 0.0  # Will be updated later  
    dense_grid.ke_grid = ke_grid
    universal_grids.append(dense_grid)
    
    # Calculate minimum exponent from atom grids for sparse grid
    grid_min_exp = min(ag.min_exp for ag in atom_grids if ag.nao > 0)
    
    # Create sparse universal grid
    cell = mydf.cell_unc
    if mydf.k_grid_mesh is not None:
        ke_max = max(tools.mesh_to_cutoff(cell.lattice_vectors(), mydf.k_grid_mesh[-1]))
        sparse_mesh = mydf.k_grid_mesh[-1]
    else:
        # Create basis with only diffuse functions for sparse grid sizing
        sparse_basis = {}
        for symb, basis_funcs in cell._basis.items():
            sparse_basis[symb] = [func for func in basis_funcs if func[1][0] < grid_min_exp]
        
        # Create temporary cell for sparse grid estimation
        sparse_cell = cell.copy()
        sparse_cell.basis = sparse_basis
        sparse_cell.build(dump_input=False)
        
        sparse_grid_ke = estimate_ke_cutoff(sparse_cell)
        dense_grid_ke = estimate_ke_cutoff(cell)
        ke_max = (sparse_grid_ke / dense_grid_ke * 
                 max(tools.mesh_to_cutoff(cell.lattice_vectors(), cell.mesh)))
        
        sparse_mesh = tools.cutoff_to_mesh(cell.lattice_vectors(), ke_max)
        
        # Ensure sparse mesh is actually sparser than dense mesh
        if (sparse_mesh >= universal_grids[0].mesh).all():
            sparse_mesh = [1, 1, 1]
            sparse_mesh = sparse_mesh + (sparse_mesh + 1) % 2  # Make odd
            ke_max = tools.mesh_to_cutoff(cell.lattice_vectors(), sparse_mesh)
    
    ke_grid = ke_max
    sparse_grid = UniformGrid(mydf, ke_max, sparse_mesh)
    sparse_grid.ke_min = 0.0  # Will be updated later
    sparse_grid.ke_max = ke_max
    sparse_grid.ke_grid = ke_max
    universal_grids.append(sparse_grid)
    
    return universal_grids


def _get_atom_centers_with_pbc(cell):
    """Get atom centers including periodic boundary conditions."""
    return tools.pbc.cell_plus_imgs(cell, [1, 1, 1]).atom_coords().astype(numpy.float64)


PERIODIC_IMAGES = 27  # 3x3x3 neighboring cells for PBC


def _assign_basis_functions_to_grids(mydf, atom_grids, universal_grids, atoms):
    """
    Assign basis functions to appropriate grids based on locality.
    
    This is the most complex part of grid construction, where we determine
    which basis functions should be evaluated on which grids based on
    their spatial extent and the grid resolution.
    
    Parameters:
    -----------
    mydf : ISDF object
        Contains computational parameters
    atom_grids : list
        Atom-centered grids to populate
    universal_grids : list  
        Universal grids for reference
    atoms : list
        Atomic information objects
    atom_centers : ndarray
        Atom coordinates including PBC images
    """
    grid_coordinates = universal_grids[0].coords.round(10)
    natm = len(atoms)
    grid_min_exp = min(ag.min_exp for ag in atom_grids if ag.nao > 0)
    # Get atom centers for basis function assignment
    atom_centers = _get_atom_centers_with_pbc(mydf.cell_unc)
    
    for i, atom_grid in enumerate(atom_grids):
        if atom_grid.nao == 0:
            continue
            
        # Find grid points within cutoff radius of this atom
        atom_centers_i = atom_centers[i : PERIODIC_IMAGES * natm : natm]
        distances = _compute_distances_to_atom(grid_coordinates, atom_centers_i)
        
        # Select grid points within cutoff radius
        within_cutoff = distances < atom_grid.rcut.max()
        atom_grid.coord_idx = numpy.flatnonzero(within_cutoff)
        atom_grid.ng = len(atom_grid.coord_idx)
        
        if atom_grid.ng == 0:
            continue
            
        # Get grid coordinates for this atom
        atom_grid_coords = grid_coordinates[atom_grid.coord_idx]
        
        # Assign basis functions from all atoms to this grid
        _assign_functions_from_all_atoms(
            mydf, atom_grid, atoms, atom_centers, atom_grid_coords, 
            i, natm, grid_min_exp
        )


def _compute_distances_to_atom(grid_coords, atom_centers):
    """Compute minimum distances from grid points to atom centers (including PBC)."""
    distances_to_all_images = numpy.linalg.norm(
        grid_coords[:, None, :] - atom_centers[None, :, :], 
        axis=-1
    )
    return distances_to_all_images.min(axis=1)


def _assign_functions_from_all_atoms(mydf, atom_grid, atoms, atom_centers, 
                                   atom_grid_coords, center_atom_idx, natm, grid_min_exp):
    """
    Assign basis functions from all atoms to the current atom grid.
    
    This determines which basis functions from each atom should be 
    evaluated on the current atom-centered grid based on their
    spatial extent and decay properties.
    """
    # Storage for basis function assignments
    diffuse_indices = []
    ao_on_grid_indices = []
    atoms_on_grid = []
    shells_on_grid = []
    exponents_on_grid = []
    angular_momenta = []
    
    for j in range(natm):
        if j == center_atom_idx:
            # For the central atom, include all sharp + diffuse functions
            # This prevents issues when ke_cutoff is too low
            max_allowed_exponent = atom_grid.max_exp
        else:
            # For other atoms, calculate maximum exponent that gives
            # non-negligible contribution at the closest grid point
            atom_centers_j = atom_centers[j : PERIODIC_IMAGES * natm : natm]
            min_distance = _compute_distances_to_atom(atom_grid_coords, atom_centers_j).min()
            max_allowed_exponent = mydf.primitive_gto_exponent(min_distance)
        
        # Select basis functions that can contribute to this grid
        can_contribute = numpy.less_equal(atoms[j].exponents, max_allowed_exponent)
        
        # Store information for contributing functions
        shells_on_grid.extend(atoms[j].shell_index[can_contribute])
        exponents_on_grid.extend(atoms[j].exponents[can_contribute])
        angular_momenta.extend(atoms[j].l[can_contribute])
        atoms_on_grid.extend(numpy.repeat(j, sum(can_contribute)))
        
        # Map to atomic orbital indices (accounting for angular momentum multiplicity)
        multiplicity = 2 * atoms[j].l + 1
        ao_mask = numpy.repeat(can_contribute, multiplicity)
        contributing_aos = atoms[j].ao_index[ao_mask]
        ao_on_grid_indices.extend(contributing_aos)
        
        # Identify diffuse functions for sparse grid
        is_diffuse_and_contributing = (
            numpy.less(atoms[j].exponents, grid_min_exp) & can_contribute
        )
        diffuse_ao_mask = numpy.repeat(is_diffuse_and_contributing, multiplicity)
        diffuse_aos = atoms[j].ao_index[diffuse_ao_mask]
        diffuse_indices.extend(diffuse_aos)
    
    # Store assignments in atom grid
    _store_basis_assignments(atom_grid, ao_on_grid_indices, diffuse_indices,
                           angular_momenta, exponents_on_grid, shells_on_grid,
                           atoms_on_grid)


def _store_basis_assignments(atom_grid, ao_indices, diffuse_indices, 
                           angular_momenta, exponents, shells, atoms_on_grid):
    """Store the computed basis function assignments in the atom grid."""
    # Sort and store AO indices
    atom_grid.ao_index = numpy.sort(ao_indices)
    atom_grid.nao = len(atom_grid.ao_index)
    
    # Map sharp functions to positions in full AO list
    atom_grid.ao_index_sharp_on_grid = numpy.flatnonzero(
        numpy.isin(atom_grid.ao_index, atom_grid.ao_index_sharp)
    )
    
    # Store diffuse function information
    atom_grid.ao_index_diffuse = numpy.sort(diffuse_indices)
    atom_grid.ao_index_diffuse_on_grid = numpy.flatnonzero(
        numpy.isin(atom_grid.ao_index, atom_grid.ao_index_diffuse)
    )
    atom_grid.n_diffuse = len(atom_grid.ao_index_diffuse)
    
    # Store basis function metadata
    atom_grid.l = numpy.asarray(angular_momenta, numpy.int32)
    atom_grid.exponents = numpy.asarray(exponents, numpy.float64)
    atom_grid.shells = list_to_slices(numpy.asarray(shells))
    atom_grid.nexp = len(exponents)
    atom_grid.atoms = numpy.asarray(atoms_on_grid, numpy.int32)


def _finalize_grid_setup(mydf, atom_grids, universal_grids, atoms, diffuse_atoms):
    """
    Final configuration and optimization of grid structures.
    
    This function creates the final sparse grid for diffuse functions
    and performs any remaining setup tasks.
    
    Parameters:
    -----------
    mydf : ISDF object
        Contains computational parameters
    atom_grids : list
        Atom-centered grids
    universal_grids : list
        Universal grids
    atoms : list
        Original atomic information
    diffuse_atoms : list
        Atoms objects for diffuse functions
    """
    cell = mydf.cell_unc
    natm = len(atoms)
    
    # Collect all atomic information for sparse grid
    all_exponents = numpy.concatenate([atm.exponents for atm in atoms])
    all_angular_momenta = numpy.concatenate([atm.l for atm in atoms])
    all_rcut = numpy.concatenate([atm.rcut for atm in atoms])
    
    # Calculate total number of AOs and basis functions
    total_nao = int(sum(2 * all_angular_momenta + 1))
    total_nbas = len(all_exponents)
    
    # Create indices for all shells and AOs
    all_shell_indices = numpy.arange(total_nbas, dtype=numpy.int32)
    all_ao_indices = numpy.arange(total_nao, dtype=numpy.int32)
    
    # Collect shell indices for diffuse functions
    sparse_shell_indices = numpy.concatenate([
        diffuse_atom.shell_index for diffuse_atom in diffuse_atoms
        if len(diffuse_atom.shell_index) > 0
    ], dtype=numpy.int32)
    
    # Create final atom grid for sparse/diffuse functions
    sparse_atom_grid = AtomGrid(
        sparse_shell_indices, all_exponents, all_angular_momenta, 
        all_rcut, all_shell_indices, all_ao_indices
    )

    sparse_atom_grid.ao_index = sparse_atom_grid.ao_index_sharp
    
    # Configure sparse atom grid
    if len(sparse_shell_indices) > 0:
        sparse_atom_grid.l = all_angular_momenta[sparse_atom_grid.shell_index_sharp]
        sparse_atom_grid.exponents = all_exponents[sparse_atom_grid.shell_index_sharp]
        sparse_atom_grid.nexp = len(sparse_atom_grid.exponents)
        
        # Create atom assignment array
        atom_assignments = []
        for atom_idx in range(natm):
            atom_assignments.extend([atom_idx] * len(atoms[atom_idx].exponents))
        atom_assignments = numpy.asarray(atom_assignments, numpy.int32)
        sparse_atom_grid.atoms = atom_assignments[sparse_atom_grid.shell_index_sharp]
        
        sparse_atom_grid.shells = list_to_slices(sparse_atom_grid.shell_index_sharp)
        sparse_atom_grid.ng = universal_grids[-1].ngrids
        sparse_atom_grid.coord_idx = numpy.arange(sparse_atom_grid.ng)
    else:
        # Handle case with no diffuse functions
        sparse_atom_grid.ng = 0
        sparse_atom_grid.coord_idx = numpy.array([], dtype=numpy.int32)
    
    # Add sparse atom grid to the list
    atom_grids.append(sparse_atom_grid)
    
    # Store grids in mydf object
    mydf.universal_grids = universal_grids

def print_grids(atom_grids, universal_grids, atoms, nao, nbas):
    print()
    print(
        "{0:8s} {1:8s} {2:8s} {3:10s} {4:6s} {5:6s} {6:12s} {7:6s} {8:10s} {9:6s} {10:12s}".format(
            "Grid",
            "exp_max",
            "exp_min",
            "rcut_max",
            "lmax",
            "nexp",
            "nexp_sharp",
            "nao",
            "nao_sharp",
            "PWs",
            "mesh",
        ),
        flush=True,
    )

    universal_grids[0].max_exp = max(ag.max_exp for ag in atom_grids[:-1])
    universal_grids[0].min_exp = min(ag.min_exp for ag in atom_grids[:-1])
    universal_grids[-1].max_exp = atom_grids[-1].max_exp
    universal_grids[-1].min_exp = atom_grids[-1].min_exp

    universal_grids[0].max_rcut = max(max(ag.rcut) for ag in atom_grids[:-1])
    universal_grids[-1].max_rcut = max(atom_grids[-1].rcut)

    universal_grids[0].lmax = max(max(ag.l) for ag in atom_grids[:-1])
    universal_grids[-1].lmax = max(atom_grids[-1].l)

    all_exponents = numpy.concatenate([atm.exponents for atm in atoms])

    universal_grids[0].nexp_sharp = sum(all_exponents >= universal_grids[0].min_exp)
    universal_grids[0].nexp = nbas
    universal_grids[-1].nexp = universal_grids[-1].nexp_sharp = atom_grids[-1].exponents.shape[0]
    universal_grids[0].nao_sharp = sum(ag.ao_index_sharp.shape[0] for ag in atom_grids[:-1])
    universal_grids[0].nao = nao
    universal_grids[-1].nao = universal_grids[-1].nao_sharp = atom_grids[-1].ao_index.shape[0]


    for i, grid in enumerate(universal_grids):
        print(
        "{0:<8d} {1:<8.1f} {2:<8.1f} {3:<10.1f} {4:<6d} {5:<6d} {6:<12d} {7:<6d} {8:<10d} {9:<6d} {10:<10s}".format(
            i,
            grid.max_exp,
            grid.min_exp,
            grid.max_rcut,
            grid.lmax,
            grid.nexp,
            grid.nexp_sharp,
            grid.nao,
            grid.nao_sharp,
            grid.ngrids,
            numpy.array2string(grid.mesh),
        ),
        flush=True,
        )
    print()