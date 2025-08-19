"""
Utility functions for ISDFX

This module contains pure mathematical operations and algorithms that are
reused across different contexts.
"""

import numpy
from pyscf.lib import logger
from pyscf.pbc import tools
from scipy.linalg import solve
from scipy.linalg.lapack import dpstrf
from scipy.spatial.distance import cdist


def pivoted_cholesky_decomposition(mydf, aovals, ao_indices=None):
    """
    Select interpolation points using pivoted Cholesky decomposition.

    This function performs pivoted Cholesky decomposition on the product of
    overlap matrices to select the most important grid points for ISDFX interpolation.

    Parameters:
    -----------
    mydf : ISDFX
        ISDFX object containing threshold settings
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
    GRR = sum(ao_k.conj().T @ ao_k for ao_k in aovals)

    if ao_indices is None:
        # No AO restriction - use full overlap
        LRR = GRR
    else:
        # Compute restricted overlap L†L for selected AOs
        local_aos = [ao_k[ao_indices] for ao_k in aovals]
        LRR = sum(ao_k.T @ ao_k.conj() for ao_k in local_aos)

    # Construct Cholesky matrix: Z = (L†L ⊙ G†G) / nk²
    Z = ((LRR * GRR) / nk**2).real
    Z = numpy.asarray(Z, dtype=numpy.float64)

    # Pivoted Cholesky decomposition with threshold
    permutation, nfit = dpstrf(Z, tol=mydf.isdf_thresh**2, overwrite_a=True)[1:3]
    pivot_indices = permutation[:nfit] - 1  # Convert to 0-based indexing

    logger.debug1(
        mydf,
        'Cholesky selected %d/%d points (thresh=%.2e)',
        nfit,
        Z.shape[0],
        mydf.isdf_thresh,
    )

    return pivot_indices.astype(numpy.int32)


def voronoi_partition(mydf):
    """
    Partition universal grid points using Voronoi tessellation.

    Each grid point is assigned to the nearest atom (including periodic images).
    This creates atom-centered regions for efficient local ISDFX operations.

    Parameters:
    -----------
    mydf : ISDFX
        ISDFX object to be modified in-place

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


def ao_indices_by_atom(cell):
    """
    Return AO indices grouped by atom.
    """
    ao_slices = cell.aoslice_by_atom()
    ao_index_by_atom = []

    for atm_id in range(cell.natm):
        start_ao, end_ao = ao_slices[atm_id, 2], ao_slices[atm_id, 3]
        ao_index_by_atom.append(numpy.arange(start_ao, end_ao, dtype=numpy.int32))

    return ao_index_by_atom


def get_fitting_functions(mydf, aovals, ao_indices=None):
    """
    Construct ISDFX fitting functions at selected pivot points.

    This function solves the linear system χ_g = (X^(Rg,Rg))^(-1) X^(Rg,R)
    to obtain interpolation coefficients that allow reconstruction of AO
    products at arbitrary grid points from values at pivot points only.

    Parameters:
    -----------
    mydf : ISDFX
        ISDFX object with pivot points already determined
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
    Rg = mydf.pivots

    phiLocalR = aovals if ao_indices is None else [ao[ao_indices] for ao in aovals]
    X_Rg_Rg = sum(numpy.matmul(aok[:, Rg].T, aok[:, Rg].conj()) for aok in phiLocalR)
    if ao_indices is None:
        X_Rg_Rg *= X_Rg_Rg
    else:
        X_Rg_Rg *= sum(numpy.matmul(aok[:, Rg].T.conj(), aok[:, Rg]) for aok in aovals)

    X_Rg_R = sum(numpy.matmul(aok[:, Rg].T, aok.conj()) for aok in phiLocalR)
    if ao_indices is None:
        X_Rg_R *= X_Rg_R
    else:
        X_Rg_R *= sum(numpy.matmul(aok[:, Rg].T.conj(), aok) for aok in aovals)

    # Give expected symmetry??
    return solve(X_Rg_Rg, X_Rg_R, overwrite_a=True, overwrite_b=True, check_finite=False).real.astype(numpy.float64)
