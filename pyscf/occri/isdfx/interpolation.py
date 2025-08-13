"""
ISDF Interpolation Functions

This module contains the core interpolation functions for the ISDF method:
- Voronoi partitioning of grid points
- Pivot point selection via Cholesky decomposition  
- Fitting function construction
"""

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import tools
from scipy.spatial.distance import cdist
from scipy.linalg.lapack import dpstrf
from scipy.linalg import solve

def _pivoted_cholesky_decomposition(mydf, aovals, ao_indices=None):
    """
    Select interpolation points using pivoted Cholesky decomposition.
    
    This function performs pivoted Cholesky decomposition on the product of
    overlap matrices to select the most important grid points for ISDF interpolation.
    
    Parameters:
    -----------
    mydf : ISDF
        ISDF object containing threshold settings
    aovals : list of ndarray
        AO values on grid for each k-point (shape: nao x ngrids for each k-point)
    ao_indices : ndarray, optional
        Subset of AO indices to consider for local decomposition
        
    Returns:
    --------
    ndarray
        Selected pivot point indices
        
    Notes:
    ------
    The Cholesky matrix is constructed as Z = (L†L ⊙ G†G) / nk²
    """
    nk = mydf.kpts.shape[0]
    
    # Compute G†G = Σ_k φ_k† φ_k
    GG1 = sum(numpy.conj(ao_k.T) @ ao_k for ao_k in aovals)
    
    if ao_indices is None:
        # No AO restriction - use full overlap
        LL1 = GG1
    else:
        # Compute restricted overlap L†L for selected AOs
        local_aos = [ao_k[ao_indices] for ao_k in aovals]
        LL1 = sum(numpy.conj(ao_k.T) @ ao_k for ao_k in local_aos)
    
    # Construct Cholesky matrix: Z = (L†L ⊙ G†G) / nk²
    Z = ((LL1 * GG1) / nk**2).real
    Z = numpy.asarray(Z, dtype=numpy.float64)
    
    # Pivoted Cholesky decomposition with threshold
    permutation, nfit = dpstrf(Z, tol=mydf.isdf_thresh**2, 
                                overwrite_a=True)[1:3]
    pivot_indices = permutation[:nfit] - 1  # Convert to 0-based indexing
    
    logger.debug1(mydf, 'Cholesky selected %d/%d points (thresh=%.2e)', 
                    nfit, Z.shape[0], mydf.isdf_thresh)
        
    return pivot_indices.astype(numpy.int32)
        

def get_fitting_functions(mydf, ao_indices=None):
    """
    Construct ISDF fitting functions at selected pivot points.
    
    This function solves the linear system χ_g = (X^(Rg,Rg))^(-1) X^(Rg,R)
    to obtain interpolation coefficients that allow reconstruction of AO
    products at arbitrary grid points from values at pivot points only.
    
    Parameters:
    -----------
    mydf : ISDF
        ISDF object with pivot points already determined
    ao_indices : ndarray, optional
        Subset of AO indices for restricted fitting (default: None, use all AOs)
        
    Returns:
    --------
        
    Notes:
    ------
    The overlap matrices are:
    - X^(Rg,Rg) = Σ_k φ_μ^k(Rg)* φ_ν^k(Rg) : overlap at pivot points
    - X^(Rg,R) = Σ_k φ_μ^k(Rg)* φ_ν^k(R) : cross-overlap pivot to all points
    """
    aovals = mydf.aovals
    Rg = mydf.pivots
    
    phiLocalR = aovals if ao_indices is None else [ao[ao_indices] for ao in aovals]
    X_Rg_Rg = numpy.sum(numpy.matmul(aok[:, Rg].T, aok[:, Rg].conj()) for aok in phiLocalR)
    if ao_indices is None:
        X_Rg_Rg *= X_Rg_Rg
    else:
        X_Rg_Rg *= numpy.sum(numpy.matmul(aok[:, Rg].T.conj(), aok[:, Rg]) for aok in aovals)

    X_Rg_R = numpy.sum(numpy.matmul(aok[:, Rg].T, aok.conj()) for aok in phiLocalR)
    if ao_indices is None:
        X_Rg_R *= X_Rg_R
    else:
        X_Rg_R *= numpy.sum(numpy.matmul(aok[:, Rg].T.conj(), aok) for aok in aovals)

    mydf.aovals = phiLocalR
    # Give expected symmetry??
    return solve( X_Rg_Rg, X_Rg_R, overwrite_a=True, overwrite_b=True, check_finite=False).real


def _voronoi_partition(mydf):
    """
    Partition universal grid points using Voronoi tessellation.

    Each grid point is assigned to the nearest atom (including periodic images).
    This creates atom-centered regions for efficient local ISDF operations.

    Parameters:
    -----------
    mydf : ISDF
        ISDF object to be modified in-place

    Returns:
    --------
    None
        Modifies mydf.coords_by_atom in-place with coordinate indices for each atom
    """
    cell = mydf.cell
    logger.debug1(mydf, 'Starting Voronoi partitioning for %d atoms', cell.natm)
    
    # Get atom positions including periodic boundary conditions (3x3x3 = 27 images)
    atom_coords_pbc = tools.pbc.cell_plus_imgs(cell, [1, 1, 1]).atom_coords()
    atom_coords_pbc = numpy.asarray(atom_coords_pbc, dtype=numpy.float64)
    
    # Generate uniform grid coordinates
    coords = cell.gen_uniform_grids(mydf.mesh)
    
    # Find nearest atom (including PBC images) for each grid point
    distances = cdist(coords, atom_coords_pbc, metric='euclidean')
    nearest_atom_indices = numpy.argmin(distances, axis=1)
    
    # Group grid points by atom (accounting for 27 periodic images per atom)
    natm = cell.natm
    coords_by_atom = []
    
    for atom_id in range(natm):
        # Find grid points belonging to this atom (any of its 27 images)
        atom_image_indices = numpy.arange(atom_id, 27 * natm, natm)
        mask = numpy.isin(nearest_atom_indices, atom_image_indices)
        grid_indices = numpy.flatnonzero(mask).astype(numpy.int32)
        coords_by_atom.append(grid_indices)
        logger.debug1(mydf, 'Atom %d assigned %d grid points', atom_id, len(grid_indices))
    
    return coords_by_atom

def _init_ao_indices(mydf):
    """AO indices grouped by atom."""
    cell = mydf.cell
    ao_slices = cell.aoslice_by_atom()

    ao_index_by_atom = []
    for atm_id in range(cell.natm):
        start_ao, end_ao = ao_slices[atm_id, 2], ao_slices[atm_id, 3]
        ao_index_by_atom.append(numpy.arange(start_ao, end_ao, dtype=numpy.int32))
    
    return ao_index_by_atom

def get_pivots(mydf):
    """
    Determine ISDF pivot points using hierarchical Cholesky decomposition.
    
    This function performs a two-stage process:
    1. Local pivot selection within each Voronoi region  
    2. Global pivot refinement on the universal grid
    
    The hierarchical approach reduces computational cost while maintaining
    accuracy by first selecting locally important points, then globally
    refining the selection.
    
    Parameters:
    -----------
    mydf : ISDF
        ISDF object to be modified in-place
        
    Returns:
    --------
    None
        Modifies mydf.pivots and mydf.aovals in-place
    """
    cell = mydf.cell
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(mydf.stdout, mydf.verbose)
    
    # Evaluate AOs on grid
    coords = cell.gen_uniform_grids(mydf.mesh)
    ngrids = coords.shape[0]
    volume_element = (cell.vol / ngrids) ** 0.5
    kpts = mydf.kpts
    aovals = []
    for ao_k in mydf._numint.eval_ao(cell, coords, kpts=kpts):
        aovals.append(numpy.asarray(ao_k.T * volume_element, order='C'))
        
    local_pivots = []
    coords_by_atom = _voronoi_partition(mydf)
    ao_index_by_atom = _init_ao_indices(mydf)
    for atom_id in range(cell.natm):
        ao_indices = ao_index_by_atom[atom_id]
        grid_indices = coords_by_atom[atom_id]
            
        # Extract local AO values for this atom's grid region
        local_aovals = [ao_k[:, grid_indices] for ao_k in aovals]
        
        # Select local pivots using Cholesky decomposition
        local_pivot_indices = grid_indices[_pivoted_cholesky_decomposition(mydf, local_aovals, ao_indices)]
        local_pivots.extend( local_pivot_indices )
        
        logger.debug1(mydf, 'Atom %d: selected %d/%d local pivots', 
                        atom_id, len(local_pivot_indices), len(grid_indices))
        
    local_pivots = numpy.array(local_pivots, dtype=numpy.int32)
    
    logger.info(mydf, '  Partitioned ISDF: %d candidate pivots from %d grid points (%.2f%% compression)',
                len(local_pivots), ngrids, 100 * len(local_pivots) / ngrids)
        
    aovals_on_pivots = [ao_k[:, local_pivots] for ao_k in aovals]
    global_pivots = _pivoted_cholesky_decomposition(mydf, aovals_on_pivots)
    
    # Store results
    mydf.pivots = numpy.sort(local_pivots[global_pivots])
    mydf.aovals = aovals
    
    logger.info(mydf, '  ISDF selected %d/%d grid points (%.2f%% compression)', 
                len(mydf.pivots), ngrids, 100 * len(mydf.pivots) / ngrids)
    cput0 = log.timer('Pivot selection', *cput0)

