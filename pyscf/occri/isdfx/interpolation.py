"""
ISDFX Interpolation Functions

This module contains the core interpolation functions for the ISDFX method:
- Voronoi partitioning of grid points
- Pivot point selection via Cholesky decomposition
- Fitting function construction
"""

import numpy
from pyscf.lib import logger
from pyscf.pbc import tools
from scipy.fft import hfftn

from .utils import ao_indices_by_atom, pivoted_cholesky_decomposition, voronoi_partition


def get_pivots(mydf):
    """
    Determine ISDFX pivot points using hierarchical Cholesky decomposition.

    This function performs a two-stage process:
    1. Local pivot selection within each Voronoi region
    2. Global pivot refinement on the universal grid

    The hierarchical approach reduces computational cost while maintaining
    accuracy by first selecting locally important points, then globally
    refining the selection.

    Parameters:
    -----------
    mydf : ISDFX
        ISDFX object to be modified in-place

    Returns:
    --------
    None
        Modifies mydf.pivots and mydf.aovals in-place
    """
    cell = mydf.cell

    # Evaluate AOs on grid
    coords = cell.gen_uniform_grids(mydf.mesh)
    ngrids = coords.shape[0]
    volume_element = (cell.vol / ngrids) ** 0.5
    kpts = mydf.kpts
    aovals = [numpy.asarray(ao.T * volume_element, order='C') for ao in mydf._numint.eval_ao(cell, coords, kpts=kpts)]

    local_pivots = []
    coords_by_atom = voronoi_partition(mydf)
    ao_index_by_atom = ao_indices_by_atom(cell)
    for atom_id in range(cell.natm):
        ao_indices = ao_index_by_atom[atom_id]
        grid_indices = coords_by_atom[atom_id]

        # Extract local AO values for this atom's grid region
        local_aovals = [ao_k[:, grid_indices] for ao_k in aovals]

        # Select local pivots using Cholesky decomposition
        local_pivot_indices = grid_indices[pivoted_cholesky_decomposition(mydf, local_aovals, ao_indices)]
        local_pivots.extend(local_pivot_indices)

        logger.debug1(
            mydf,
            'Atom %d: selected %d/%d local pivots',
            atom_id,
            len(local_pivot_indices),
            len(grid_indices),
        )

    local_pivots = numpy.array(local_pivots, dtype=numpy.int32)

    logger.info(
        mydf,
        '  Partitioned ISDFX: %d candidate pivots from %d grid points (%.2f%% compression)',
        len(local_pivots),
        ngrids,
        100 * len(local_pivots) / ngrids,
    )

    aovals_on_pivots = [ao_k[:, local_pivots] for ao_k in aovals]
    global_pivots = pivoted_cholesky_decomposition(mydf, aovals_on_pivots)
    return numpy.sort(local_pivots[global_pivots]), aovals


def get_thc_potential(mydf, fitting_functions):
    """
    Calculate the THC (Tensor Hypercontraction) potential for ISDFX exchange evaluation.

    This function computes the potential W_μν(k) from the fitting functions
    that appears in the tensor hypercontraction representation of the exchange matrix.

    Parameters:
    -----------
    mydf : ISDFX
        ISDFX object with pivot points and mesh information
    fitting_functions : ndarray
        ISDFX fitting functions χ_g with shape (npivots, ngrids)

    Returns:
    --------
    None
        Modifies mydf.W in-place with the THC potential

    Notes:
    ------
    The THC potential is computed as:
    W_μν(k) = ∫ χ_μ(r) v(r-r') χ_ν(r') e^{ik·(r-r')} dr dr'
    where v(r) is the Coulomb interaction and χ_μ are fitting functions.

    The calculation involves:
    1. Phase factor application: χ_μ(r) e^{-ik·r}
    2. Forward FFT to k-space
    3. Multiplication by Coulomb kernel G(k)
    4. Inverse FFT back to r-space
    5. Integration with conjugate phase factors
    """
    cell = mydf.cell
    npivots, ngrids = fitting_functions.shape
    nk = mydf.kpts.shape[0]

    # Physical constants and normalization
    volume_factor = ngrids / cell.vol / nk
    kpts = mydf.kpts
    mesh = mydf.mesh
    coords = cell.gen_uniform_grids(mesh)

    # Initialize THC potential tensor
    thc_potential = numpy.empty((nk, npivots, npivots), dtype=numpy.complex128)

    # Compute THC potential for each k-point
    for k, kpt in enumerate(kpts):
        # Step 1: Apply phase factors e^{-ik·r}
        phase_factors = numpy.exp(-1.0j * numpy.dot(coords, kpt))
        modulated_functions = fitting_functions * phase_factors[numpy.newaxis, :]

        # Step 2: Forward FFT to momentum space
        vG = tools.fft(modulated_functions, mesh)

        # Step 3: Apply Coulomb kernel G(k) = 4π/|k|²
        coulG = tools.get_coulG(cell, kpt, mesh=mesh).reshape(1, -1)
        vG *= coulG

        # Step 4: Inverse FFT back to position space
        vR = tools.ifft(vG, mesh)

        # Step 5: Apply conjugate phase factors and integrate
        vR *= phase_factors.conj()

        # Matrix multiplication: W_μν = ∫ χ_μ(r) O(r) χ_ν*(r) dr
        numpy.matmul(vR, fitting_functions.conj().T, out=thc_potential[k])

    # Apply normalization and complex conjugation
    thc_potential = thc_potential.conj()
    thc_potential *= volume_factor

    # Transform to k-space representation for efficient convolution
    thc_potential = hfftn(
        thc_potential.reshape(*mydf.kmesh, npivots, npivots),
        s=tuple(mydf.kmesh),
        axes=[0, 1, 2],
        overwrite_x=True,
    )

    return thc_potential
