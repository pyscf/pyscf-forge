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
            self.min_exp = 0.0
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
    
    def __init__(self, cell, rcut_epsilon, atom_index, shell_index=None):
        """
        Initialize atomic information for a specific atom.
        
        Parameters:
        -----------
        cell : pyscf.pbc.gto.Cell
            Unit cell object
        rcut_epsilon : float
            Cutoff threshold for basis function truncation
        atom_index : int
            Index of the atom
        shell_index : array_like
            Indices of shells belonging to this atom
        """
        
        if shell_index is None:
            shell_index = numpy.ascontiguousarray(cell.atom_shell_ids(atom_index), dtype=numpy.int32)

        # Extract angular momentum quantum numbers for each shell
        self.l = numpy.ascontiguousarray(
            [cell.bas_angular(j) for j in shell_index], 
            dtype=numpy.int32
        )
        
        # Extract and concatenate Gaussian exponents
        self.exponents = numpy.ascontiguousarray(
            numpy.concatenate([cell.bas_exp(j) for j in shell_index]),
            dtype=numpy.float64
        )
        
        # Get atomic orbital indices for this atom
        ao_slice = cell.aoslice_by_atom()[atom_index]
        ao_index = numpy.arange(ao_slice[2], ao_slice[3], dtype=numpy.int32)
        
        # Get angular momenta for all shells on this atom
        atom_shells = cell.atom_shell_ids(atom_index)
        l = numpy.ascontiguousarray(
            [cell.bas_angular(j) for j in atom_shells], 
            dtype=numpy.int32
        )
        
        # Account for angular momentum multiplicity
        multiplicity = 2 * l + 1
        shell_idx = numpy.isin(atom_shells, shell_index)
        shell_idx = numpy.repeat(shell_idx, multiplicity)
        
        # Filter AO indices to include only relevant shells
        self.ao_index = ao_index[shell_idx]
        
        # Calculate cutoff radii for primitive Gaussians
        self.rcut = numpy.ascontiguousarray(
            numpy.concatenate(self.primitive_gto_cutoff(cell, rcut_epsilon, shell_index)),
            dtype=numpy.float64,
        )
        
        # Store atomic center coordinates
        self.atom_center = cell.atom_coords()[atom_index]

def make_exchange_lists(mydf):
    """
    Create multi-grid system optimized for exchange (K) matrix calculations.
    
    This function constructs grids specifically optimized for exchange integral
    evaluation. Unlike Coulomb grids, exchange grids use different criteria for
    basis function assignment based on the local nature of exchange interactions.
    
    Parameters:
    -----------
    mydf : ISDF
        ISDF object containing cell and computational parameters
        
    Returns:
    --------
    tuple
        (mg, atomgrids) where:
        - mg: list of Grid objects optimized for exchange
        - atomgrids: list of AtomGrid objects for each grid level and atom
        
    Algorithm:
    ----------
    1. Determine kinetic energy cutoffs from ke_epsilon parameter
    2. Create atom-centered grids based on exponent thresholds
    3. Assign basis functions using alpha_cutoff criteria
    4. Add universal sparse grid for diffuse functions
    5. Optimize grid spacing for exchange integral locality
    
    Key Differences from Coulomb Grids:
    -----------------------------------
    - Uses alpha_cutoff to separate sharp/diffuse functions
    - Optimized for the locality of exchange interactions
    - Typically requires fewer grid levels than Coulomb
    - Grid construction based on ke_epsilon precision parameter
    """

    cell = mydf.cell_unc
    atoms = [Atoms(cell, mydf.rcut_epsilon, i) for i in range(cell.natm)]
    natm = cell.natm
    atomgrids = []
    mg = []

    ##### Alternate mesh specifications ######
    # (1) k_grid_mesh can store a list of user-given meshes.
    # (2) cell.mesh
    #     When calling tools.pbc.super_cell, the mesh is not from cell.ke_cutoff
    #     i.e., cutoff_to_mesh(supcell.ke_cutoff) != supcell.mesh
    #     For do_real_isdfx, set supcell.mesh = primitive_cell.mesh * ncopy.

    mesh = None
    if mydf.k_grid_mesh is not None:
        mesh = mydf.k_grid_mesh[0]
    if cell.mesh is not None:
        mesh = cell.mesh

    # Put exponents from all atoms on the dense grid.
    ke_max = 0.0
    ke_min = 1.0e8
    grid_max_exp = 0.0
    grid_min_exp = 1.0e8    
    # If a user specified mesh is used, take the ke_cutoff from it.
    ke_grid = (
        tools.mesh_to_cutoff(cell.lattice_vectors(), mesh)[0] if mesh is not None else cell.ke_cutoff
    )

    exp_list = []
    atoms2 = [None] * natm
    max_exp = max([atoms[i].exponents.max() for i in range(natm)])
    if mydf.alpha_cutoff > max_exp:
        mydf.alpha_cutoff = max_exp

    for i, atomi in enumerate(atoms):
        idx = numpy.greater(atomi.exponents, mydf.alpha_cutoff - cell.precision)
        diffuse_exp = (1 - idx).astype(numpy.bool_)
        sharp_exp = atomi.exponents[idx]
        exp_list.extend(sharp_exp)
        atoms2[i] = Atoms(cell, mydf.rcut_epsilon, i, atomi.shell_index[diffuse_exp])
        atomgrids.append(
            AtomGrid(
                atomi.shell_index[idx],
                atomi.exponents,
                atomi.l,
                atomi.rcut,
                atomi.shell_index,
                atomi.ao_index,
            ),
        )

    
    # Based on the range of exponents on it, we build the dense grid.
    mg.append(UniformGrid(mydf, ke_grid, mesh))
    mg[-1].ke_min = ke_min
    mg[-1].ke_max = ke_max
    mg[-1].ke_grid = ke_grid

    # If a user specified mesh is used, take the ke_cutoff from it.
    if mydf.k_grid_mesh is not None:
        ke_max = max(tools.mesh_to_cutoff(cell.lattice_vectors(), mydf.k_grid_mesh[-1]))
        mesh = mydf.k_grid_mesh[-1]
    else:
        new_bs = {}
        for symb, bs in cell._basis.items():
            new_bs[symb] = [itm for itm in bs if itm[1][0] < grid_min_exp]
        cell2 = cell.copy()
        cell2.basis = new_bs
        cell2.build(dump_input=False)
        sparse_grid_ke = estimate_ke_cutoff(cell2)
        dense_grid_ke = estimate_ke_cutoff(cell)
        ke_max = sparse_grid_ke/dense_grid_ke * max(tools.mesh_to_cutoff(cell.lattice_vectors(), cell.mesh))
        mesh = tools.cutoff_to_mesh(cell.lattice_vectors(), ke_grid)
        if (mesh >= mg[-1].mesh).all():
            mesh = [1, 1, 1]
            mesh = mesh  + (mesh  + 1) % 2
            ke_max = tools.mesh_to_cutoff(cell.lattice_vectors(), mesh)
    ke_grid = ke_max
    mg.append(UniformGrid(mydf, ke_grid, mesh))
    mg[-1].ke_min = ke_min
    mg[-1].ke_max = ke_max
    mg[-1].ke_grid = ke_grid

    exponents = numpy.asarray(numpy.concatenate([atoms[i].exponents for i in range(natm)]), numpy.float64)
    la = numpy.asarray(numpy.concatenate([atoms[i].l for i in range(natm)]), numpy.int32)
    rcut = numpy.asarray(numpy.concatenate([atoms[i].rcut for i in range(natm)]), numpy.float64)
    shell_index = numpy.asarray(numpy.arange(cell.nbas), numpy.int32)
    sparse_shell_idx = numpy.concatenate([a2.shell_index for a2 in atoms2])
    atomgrids.append(AtomGrid(sparse_shell_idx, exponents, la, rcut, shell_index, sparse_shell_idx))

