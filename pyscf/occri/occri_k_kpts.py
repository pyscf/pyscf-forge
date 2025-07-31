"""
Python-side integrals and exchange matrix evaluation for OCCRI

This file defines:
    - fallback Python-only routines for reference/validation (e.g., integrals_uu)
    - orbital transformation and natural orbital construction
    - wrapper functions to call the C-extension occri_vR
    - utilities to build the full AO exchange matrix

Key Functions:
    - occri_get_k_opt: Calls the C implementation of OCCRI
    - occri_get_k:      Calls the reference Python implementation
    - occri.build_full_exchange: Contracts exchange contributions from AO basis

Used internally by the OCCRI class defined in __init__.py
"""

import numpy
import scipy
from pyscf import lib
from pyscf.pbc import tools
from functools import reduce
from pyscf import occri
from pyscf.lib import logger


def make_natural_orbitals_kpts(cell, dms, kpts=numpy.zeros((1,3))):
    """
    Parameters:
    -----------
    cell : pyscf.pbc.gto.Cell
        Unit cell object containing atomic and basis set information
    dms : ndarray
        Density matrix or matrices in AO basis, shape (..., nao, nao)
    kpts : ndarray, optional
        k-point coordinates. If None, assumes Gamma point calculation
        
    Returns:
    --------
    tuple of ndarray
        (mo_coeff, mo_occ) where:
        - mo_coeff: Natural orbital coefficients, same shape as dms
                  mo_occ[n, k, i] = occupation of orbital i at k-point k, spin n
                  Real values ordered from highest to lowest occupation
        
    k-point Specific Features:
    --------------------------
    - Handles complex overlap matrices for general k-points
    - Automatic dtype detection: uses real arithmetic when |Im(dm)| < 1e-6
    - Independent natural orbital construction at each k-point
    - Preserves k-point symmetry and Bloch function properties
    - Supports both collinear and non-collinear spin systems
    """
    # Compute k-point dependent overlap matrices
    sk = numpy.asarray(cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts))
    if abs(dms.imag).max() < 1.e-6:
        sk = sk.real
        
    # Ensure consistent dtype (PySCF sometimes returns float128)
    sk = numpy.asarray(sk, dtype=dms.dtype)
    mo_coeff = numpy.zeros_like(dms)
    nao = cell.nao
    nset = dms.shape[0]
    nK = kpts.shape[0]
    mo_occ = numpy.zeros((nset, nK, nao), numpy.float64 )
    for i, dm in enumerate(dms):
        for k, s in enumerate(sk):
            # Diagonalize the DM in AO
            A = lib.reduce(numpy.dot, (s, dm[k], s))
            w, v = scipy.linalg.eigh(A, b=s)

            # Flip since they're in increasing order
            mo_occ[i][k] = numpy.flip(w)
            mo_coeff[i][k] = numpy.flip(v, axis=1)

    return mo_coeff, mo_occ


def integrals_uu_kpts(j, k, k_prim, ao_mos, vR_dm, coulG, mo_occ, mesh, expmikr):
    """
    Compute k-point exchange integrals between occupied orbitals using complex FFT.
    
    This function evaluates exchange integrals for periodic systems with k-point sampling,
    handling complex Bloch functions and k-point phase factors. It implements the core
    computational kernel for the OCCRI method in momentum space.
    
    Algorithm:
    ----------
    For each orbital i at k-point k_prim:
        1. Form complex orbital pair density: ρ_ij(r) = conj(φ_i^{k'}(r)) * exp(-i(k-k')·r) * φ_j^k(r)
        2. Transform to reciprocal space: ρ̃_ij(G) = FFT[ρ_ij(r)]
        3. Apply Coulomb kernel: Ṽ_ij(G) = ρ̃_ij(G) * v_C(|G+k-k'|) / sqrt(N_grid)
        4. Transform back to real space: V_ij(r) = IFFT[Ṽ_ij(G)]
        5. Contract and accumulate: vR_dm[j] += V_ij(r) * conj(φ_i^{k'}(r)) * exp(+i(k-k')·r) * n_i * sqrt(N_grid)
    
    Parameters:
    -----------
    j : int
        Orbital index j at k-point k
    k : int
        k-point index for orbital j
    k_prim : int 
        k-point index for orbital i (k')
    ao_mos : list of ndarray
        Molecular orbitals for all k-points, each shape (nmo, ngrids)
        ao_mos[k][i] contains orbital i at k-point k evaluated on real-space grid
    vR_dm : ndarray
        Output exchange potential array, shape (nmo, ngrids), complex dtype
        Modified in-place to accumulate contributions
    coulG : ndarray
        Coulomb interaction kernel in G-space for momentum transfer k-k'
        Shape: (ngrids,), real values: v_C(|G+k-k'|) = 4π/|G+k-k'|^2
    mo_occ : list of ndarray
        Orbital occupation numbers for each k-point
        mo_occ[k_prim][i] = occupation of orbital i at k-point k_prim
    mesh : ndarray
        FFT mesh dimensions [nx, ny, nz], determines grid resolution
    expmikr : ndarray
        k-point phase factors exp(-i(k-k')·r) on real-space grid
        Shape: (ngrids,), complex values
        
    Notes:
    ------
    - Uses complex FFT to handle k-point phase factors from Bloch functions
    - Applies sqrt(N_grid) normalization to match PySCF FFT conventions
    - Phase factors expmikr handle momentum conservation: G -> G + k - k'
    - The complex conjugate operations ensure proper Hermiticity of exchange matrix
    - For Γ-point only (k=k'=0), this reduces to the standard real-space algorithm
    
    Performance:
    ------------
    - O(N_occ * N_grid * log(N_grid)) complexity per k-point pair
    - Memory usage: O(N_grid) for temporary arrays
    - Can be called in parallel for different (j,k,k') combinations
    """
    ngrids = ao_mos[0].shape[1]
    nmo = ao_mos[k_prim].shape[0]
    sqrt_ngrids = ngrids ** 0.5
    inv_sqrt_ngrids = 1.0 / sqrt_ngrids
    
    rho1 = numpy.empty(ngrids, dtype=numpy.complex128)
    
    for i in range(nmo):
        i_Rg_exp = ao_mos[k_prim][i].conj() * expmikr
        numpy.multiply(i_Rg_exp, ao_mos[k][j], out=rho1)
        vG = tools.fft(rho1, mesh)
        vG *= inv_sqrt_ngrids * coulG
        vR = tools.ifft(vG, mesh)
        vR_dm[j] += vR * i_Rg_exp.conj() * (mo_occ[k_prim][i] * sqrt_ngrids)


def occri_get_k_kpts(mydf, dms, exxdiv=None):
    """
    Reference Python implementation of OCCRI exchange matrix evaluation.
    
    This function provides a pure Python implementation of the occupied orbital
    resolution of identity method for computing exact exchange matrices. It serves
    as a reference for validation and fallback when the optimized C implementation
    is unavailable.
    
    Parameters:
    -----------
    mydf : OCCRI object
        Density fitting object containing cell and grid information
    dms : ndarray or list of ndarray
        Density matrix or matrices in AO basis
        Shape: (..., nao, nao) for each spin component
    exxdiv : str, optional
        Exchange divergence treatment. Options:
        - 'ewald': Apply Ewald probe charge correction (recommended for 3D)
        - None: No correction applied
        
    Returns:
    --------
    ndarray
        Exchange matrix or matrices in AO basis, same shape as input dms
        
    Notes:
    ------
    This implementation:
    1. Constructs natural orbitals from density matrices
    2. Evaluates orbitals on real-space FFT grid
    3. Computes exchange integrals using FFT-based Coulomb evaluation
    4. Contracts results back to AO basis using occri.build_full_exchange
    5. Applies Ewald correction if requested for periodic boundary conditions
    
    The method scales as O(N_k^2 * N_occ^2 * N_grid * log(N_grid)) where:
    - N_k = number of k-points
    - N_occ = average number of occupied orbitals per k-point  
    - N_grid = number of FFT grid points
    The log(N_grid) factor comes from FFT operations, and N_k^2 from all k-point pair interactions.
    
    Raises:
    -------
    AssertionError
        If cell.low_dim_ft_type == 'inf_vacuum' or cell.dimension == 1
        (not supported in current implementation)
    """
    cell = mydf.cell
    mesh = mydf.mesh
    assert cell.low_dim_ft_type != 'inf_vacuum'
    assert cell.dimension != 1
    coords = cell.gen_uniform_grids(mesh)
    ngrids = coords.shape[0]
    
    if getattr(dms, "mo_coeff", None) is not None:
        mo_coeff = numpy.asarray(dms.mo_coeff)
        mo_occ = numpy.asarray(dms.mo_occ)
    else:
        mo_coeff, mo_occ = make_natural_orbitals_kpts(cell, dms, mydf.kpts)
    tol = 1.0e-6
    is_occ = mo_occ > tol
    nset = dms.shape[0]
    kpts = mydf.kpts
    nk = mydf.Nk
    mo_coeff = [[numpy.asarray(mo_coeff[n][k][:, is_occ[n][k]].T, order='C') for k in range(nk)] for n in range(nset)]
    mo_occ = [[numpy.ascontiguousarray(mo_occ[n][k][is_occ[n][k]]) for k in range(nk)] for n in range(nset)]
    
    mesh = cell.mesh
    weight = ngrids/ cell.vol  / nk
    nao = cell.nao
    vk = numpy.empty((nset, nk, nao, nao), dms.dtype)
    s = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)

    aovals = mydf._numint.eval_ao(cell, coords, kpts=kpts)
    aovals = [numpy.asarray(ao.T * (cell.vol / ngrids) ** 0.5, order='C') for ao in aovals]
    ao_mos = [[numpy.matmul( mo_coeff[n][k], aovals[k], order='C') for k in range(nk)] for n in range(nset)]

    occri.log_mem(mydf)
    t1 = (logger.process_clock(), logger.perf_counter())

    for n in range(nset):
        for k in range(nk):
            nmo = mo_coeff[n][k].shape[0]
            vR_dm = numpy.zeros((nmo, ngrids), numpy.complex128)
            for j in range(nmo):
                for k_prim in range(nk):
                    coulG = tools.get_coulG(cell, kpts[k] - kpts[k_prim], False, mesh=mesh)
                    expmikr = numpy.exp(-1j * (coords @ (kpts[k] - kpts[k_prim])))
                    integrals_uu_kpts(j, k, k_prim, ao_mos[n], vR_dm, coulG, mo_occ[n], mesh, expmikr)

            vR_dm *= weight
            vk_j = aovals[k].conj() @ vR_dm.T
            vk[n][k] = occri.build_full_exchange(s[k], vk_j, mo_coeff[n][k]).real

            t1 = logger.timer_debug1(mydf, 'get_k_kpts: make_kpt (%d,*)'%k, *t1)

    if exxdiv == 'ewald' and cell.dimension != 0:
        _ewald_exxdiv_for_G0(mydf, s, dms, vk)

    return vk


def occri_get_k_opt_kpts(mydf, dms, exxdiv=None):
    """
    Production C-accelerated implementation of k-point OCCRI exchange matrix evaluation.
    
    This function provides the high-performance implementation of the occupied orbital
    resolution of identity method for k-point calculations using an optimized C extension
    with FFTW and OpenMP. It handles complex Bloch functions, k-point phase factors,
    and multiple k-point interactions with maximum computational efficiency.
    
    Performance Features:
    ---------------------
    - Native C implementation with FFTW for optimal FFT performance
    - OpenMP parallelization over orbital indices for multi-core scaling
    - Optimized memory layout with padding for vectorization
    - Complex-to-complex FFTs for proper k-point phase handling
    - Thread-safe operation with per-thread FFTW buffers
    
    Parameters:
    -----------
    mydf : OCCRI object
        Density fitting object containing cell, k-points, and grid information
        Must have attributes: cell, kpts, mesh, Nk (number of k-points)
    dms : ndarray
        Density matrices in AO basis for all k-points  
        Shape: (nset, nk, nao, nao) where nset is number of spin components
        Complex dtype supported for non-collinear systems
    exxdiv : str, optional
        Exchange divergence treatment for periodic systems. Options:
        - 'ewald': Apply Ewald probe charge correction (recommended for 3D)
        - None: No divergence correction applied
        
    Returns:
    --------
    ndarray
        Exchange matrices in AO basis for all k-points
        Shape: (nset, nk, nao, nao), same as input dms
        Real parts taken for final result (imaginary parts should be negligible)
        
    Algorithm (C Implementation):
    -----------------------------
    1. Flatten and pad orbital data with nmo_max for consistent indexing
    2. For each k-point and spin component:
       a. Pre-compute all Coulomb kernels v_C(|G+k-k'|) for k-point differences
       b. Pre-compute phase factors exp(-i(k-k')·r) for all k-point pairs
       c. Call optimized C function occri_vR_kpts:
          - Parallel loop over orbital j at target k-point
          - For each j, loop over all k-points k'
          - For each (j,k,k'), loop over orbitals i at k'
          - Compute exchange integral using complex FFT with phase factors
       d. Contract results back to AO basis using occri.build_full_exchange
    3. Apply Ewald correction if requested
    
    Memory Management:
    ------------------
    - Uses nmo_max padding for consistent array indexing across k-points
    - Separate arrays for real/imaginary parts to interface with C
    - Pre-allocates all temporary arrays to avoid memory fragmentation
    - FFTW buffers allocated per thread for optimal performance
    
    Performance Notes:
    ------------------
    - Typical speedup: 5-10x over Python reference implementation
    - Scales well with number of CPU cores via OpenMP
    - Memory usage: O(N_k * N_occ_max * N_grid)
    - Best performance with FFTW_PATIENT planning (done once per mesh size)
    
    
    Requirements:
    -------------
    - Compiled C extension with FFTW and OpenMP support
    - Compatible with both real and complex density matrices
    - Requires periodic boundary conditions (cell.dimension > 0)
    
    Limitations:
    ------------
    - 1D systems with inf_vacuum not supported
    - Very large k-point meshes may exhaust memory
    - C extension must be properly compiled and linked
    
    Raises:
    -------
    RuntimeError
        If C extension is not available or fails to load
    AssertionError  
        If cell.low_dim_ft_type == 'inf_vacuum' or cell.dimension == 1
    """
    cell = mydf.cell
    mesh = mydf.mesh
    assert cell.low_dim_ft_type != 'inf_vacuum'
    assert cell.dimension != 1
    coords = cell.gen_uniform_grids(mesh)
    ngrids = coords.shape[0]
    
    if getattr(dms, "mo_coeff", None) is not None:
        mo_coeff = numpy.asarray(dms.mo_coeff)
        mo_occ = numpy.asarray(dms.mo_occ)
    else:
        mo_coeff, mo_occ = make_natural_orbitals_kpts(cell, dms, mydf.kpts)
    
    # Optimize occupation number filtering
    tol = 1.0e-6
    is_occ = mo_occ > tol
    nset = dms.shape[0]
    kpts = mydf.kpts
    nk = mydf.Nk
    mo_coeff = [[numpy.ascontiguousarray(mo_coeff[n][k][:, is_occ[n][k]].T) 
                for k in range(nk)] for n in range(nset)]
    mo_occ = [[numpy.ascontiguousarray(mo_occ[n][k][is_occ[n][k]]) 
                for k in range(nk)] for n in range(nset)]

    # Pre-allocate output arrays
    mesh = numpy.asarray(cell.mesh, numpy.int32)
    weight = ngrids / cell.vol / nk
    nao = cell.nao
    vk = numpy.zeros((nset, nk, nao, nao), numpy.complex128, order='C')
    s = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)
    
    # Evaluate AOs on the grid for each k-point
    aovals = mydf._numint.eval_ao(cell, coords, kpts=kpts)
    aovals = [numpy.ascontiguousarray(ao.T * (cell.vol / ngrids) ** 0.5) for ao in aovals]
    
    # Transform to MO basis for each k-point and spin
    ao_mos = [[numpy.matmul( mo_coeff[n][k], aovals[k], order='C') for k in range(nk)] for n in range(nset)]
    
    occri.log_mem(mydf)
    t1 = (logger.process_clock(), logger.perf_counter())
    
    # Import C extension components
    from pyscf.occri import occri_vR_kpts, _OCCRI_C_AVAILABLE
    
    for n in range(nset):

        nmo_max = max(mo_coeff[n][k].shape[0] for k in range(nk))

        # Flatten AO data for all k-points - pad to nmo_max for consistent indexing
        ao_mos_real = numpy.zeros((nk, nmo_max, ngrids), dtype=numpy.float64, order='C')
        ao_mos_imag = numpy.zeros((nk, nmo_max, ngrids), dtype=numpy.float64, order='C')
        mo_occ_flat = numpy.zeros((nk, nmo_max), dtype=numpy.float64, order='C')
  
        nmo = []
        for k in range(nk):
            nmo.append(ao_mos[n][k].shape[0])
            
            ao_mos_real[k][:nmo[k]] = ao_mos[n][k].real
            ao_mos_imag[k][:nmo[k]] = ao_mos[n][k].imag
            
            mo_occ_flat[k][:nmo[k]] = mo_occ[n][k]
        nmo = numpy.asarray(nmo, numpy.int32)

        for k in range(nk):
            # Prepare arrays for C function
            vR_dm_real = numpy.zeros(nmo[k] * ngrids, dtype=numpy.float64, order='C')
            vR_dm_imag = numpy.zeros(nmo[k] * ngrids, dtype=numpy.float64, order='C')
            
            # Prepare all Coulomb kernels for this k-point against all k_prim
            coulG_all = numpy.empty((nk, ngrids), dtype=numpy.float64, order='C')
            expmikr_all_real = numpy.empty((nk, ngrids), dtype=numpy.float64, order='C')
            expmikr_all_imag = numpy.empty((nk, ngrids), dtype=numpy.float64, order='C')            
            for k_prim in range(nk):
                coulG_all[k_prim] = tools.get_coulG(cell, kpts[k] - kpts[k_prim], False, mesh=mesh)
                expmikr = numpy.exp(-1j * (coords @ (kpts[k] - kpts[k_prim])))
                expmikr_all_real[k_prim] = expmikr.real
                expmikr_all_imag[k_prim] = expmikr.imag
            
            # Call optimized C function
            if _OCCRI_C_AVAILABLE and occri_vR_kpts is not None:
                occri_vR_kpts(
                    vR_dm_real, vR_dm_imag,
                    mo_occ_flat, coulG_all.ravel(), mesh, 
                    expmikr_all_real.ravel(), expmikr_all_imag.ravel(), kpts.ravel(),
                    ao_mos_real.ravel(), ao_mos_imag.ravel(),
                    nmo, ngrids, nk, k, 
                )
            else:
                raise RuntimeError("occri_get_k_opt_kpts called but C extension not available.")
            
            # Reshape and apply weight - only take the first nmo orbitals
            vR_dm = (vR_dm_real + 1j * vR_dm_imag).reshape(nmo[k], ngrids)
            
            # Contract back to AO basis
            vR_dm *= weight
            vk_j = aovals[k].conj() @ vR_dm.T
            vk[n][k] = occri.build_full_exchange(s[k], vk_j, mo_coeff[n][k]).real
            
            t1 = logger.timer_debug1(mydf, f'get_k_kpts_opt: k-point {k}', *t1)
    
    # Apply Ewald correction if requested
    if exxdiv == 'ewald' and cell.dimension != 0:
        _ewald_exxdiv_for_G0(mydf, s, dms, vk)
    
    return vk

def _ewald_exxdiv_for_G0(mydf, s, dms, vk):
    # Function _ewald_exxdiv_for_G0 to add back in the G=0 component to vk_kpts
    # Note in the _ewald_exxdiv_for_G0 implementation, the G=0 treatments are
    # different for 1D/2D and 3D systems.  The special treatments for 1D and 2D
    # can only be used with AFTDF/GDF/MDF method.  In the FFTDF method, 1D, 2D
    # and 3D should use the ewald probe charge correction.
    cell = mydf.cell
    madelung = tools.pbc.madelung(cell, mydf.kpts)
    nset = dms.shape[0]
    nk = dms.shape[1]
    for n in range(nset):
        for k in range(nk):
            vk[n][k] += madelung * reduce(numpy.dot, (s[k], dms[n][k], s[k]))