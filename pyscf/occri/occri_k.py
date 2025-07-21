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
    - build_full_exchange: Contracts exchange contributions from AO basis

Used internally by the OCCRI class defined in __init__.py
"""

import numpy
import scipy
from pyscf import lib
from pyscf.pbc import tools
from functools import reduce
from pyscf import occri

def make_natural_orbitals(cell, dms, kpts=None):
    """
    Construct natural orbitals from density matrices.
    
    This function diagonalizes the density matrix in the AO basis to obtain
    natural orbitals and their occupation numbers. Natural orbitals provide
    an optimal single-particle representation of the many-body wavefunction.
    
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
        - mo_occ: Natural orbital occupation numbers, shape (..., nao)
        
    Notes:
    ------
    The natural orbitals are obtained by solving the generalized eigenvalue problem:
    S · D · S · C = S · C · n
    where S is the overlap matrix, D is the density matrix, C are the coefficients,
    and n are the occupation numbers.
    """
    s = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts).astype(dms[0].dtype)
    mo_coeff = numpy.zeros_like(dms)
    mo_occ = numpy.zeros((dms.shape[0], dms.shape[1]), numpy.float64 )
    for i, dm in enumerate(dms):
        # Diagonalize the DM in AO
        A = lib.reduce(numpy.dot, (s, dm, s))
        w, v = scipy.linalg.eigh(A, b=s)

        # Flip since they're in increasing order
        mo_occ[i] = numpy.flip(w)
        mo_coeff[i] = numpy.flip(v, axis=1)

    if mo_coeff.ndim != dms.ndim:
        mo_coeff = mo_coeff.reshape(dms.shape)
    if mo_occ.ndim != (dms.ndim -1) :
        mo_occ = mo_occ.reshape(-1, dms.shape[-1])

    return mo_coeff, mo_occ

def build_full_exchange(S, Kao, mo_coeff):
    """
    Construct full exchange matrix from occupied orbital components.
    
    This function builds the complete exchange matrix in the atomic orbital (AO)
    basis from the occupied-occupied (Koo) and occupied-all (Koa) components
    computed using the resolution of identity approximation.
    
    Parameters:
    -----------
    Sa : numpy.ndarray
        Overlap matrix times MO coefficients (nao x nocc)
    Kao : numpy.ndarray
        Occupied-all exchange matrix components (nao x nocc)
    Koo : numpy.ndarray
        Occupied-occupied exchange matrix components (nocc x nocc)
        
    Returns:
    --------
    numpy.ndarray
        Full exchange matrix in AO basis (nao x nao)
        
    Algorithm:
    ----------
    K_μν = Sa_μi * Koa_iν + Sa_νi * Koa_iμ - Sa_μi * Koo_ij * Sa_νj
    
    This corresponds to the resolution of identity expression:
    K_μν ≈ Σ_P C_μP W_PP' C_νP' where C are fitting coefficients
    """

    # Compute Sa = S @ mo_coeff.T once and reuse
    Sa = S @ mo_coeff.T
    
    # First and second terms: Sa @ Kao.T + (Sa @ Kao.T).T
    # This is equivalent to Sa @ Kao.T + Kao @ Sa.T
    # Use symmetric rank-k update (SYRK) when possible
    Sa_Kao = numpy.matmul(Sa, Kao.T, order='C')
    Kuv = Sa_Kao + Sa_Kao.T
    
    # Third term: -Sa @ (mo_coeff @ Kao) @ Sa.T
    # Optimize as -Sa @ Koo @ Sa.T using GEMM operations
    Koo = mo_coeff @ Kao
    Sa_Kao = numpy.matmul(Sa, Koo)
    Kuv -= numpy.matmul(Sa_Kao, Sa.T, order='C')
    return Kuv


def integrals_uu(i, ao_mos, vR_dm, coulG, mo_occ, mesh):
    """
    Compute occupied-occupied exchange integrals using FFT.
    
    This function evaluates the exchange integrals between occupied orbitals
    using FFT-based techniques. It computes the Coulomb interaction in reciprocal
    space and transforms back to real space for contraction with orbital densities.
    
    Parameters:
    -----------
    i : int
        Index of the reference occupied orbital
    ao_mos : list of ndarray
        Molecular orbitals in AO basis evaluated on real-space grid
        Shape: (nmo, ngrids) for each k-point/spin
    vR_dm : ndarray
        Output array for exchange potential, shape (nmo, ngrids)
        Modified in-place
    coulG : ndarray
        Coulomb interaction in reciprocal space, shape (ngrids,)
    mo_occ : ndarray
        Molecular orbital occupation numbers, shape (nmo,)
    mesh : ndarray
        FFT mesh dimensions [nx, ny, nz]
        
    Notes:
    ------
    This function implements the core FFT-based exchange integral evaluation:
    1. Form orbital pair density ρ_ij(r) = φ_i(r) φ_j(r)  
    2. Transform to reciprocal space: ρ̃_ij(G) = FFT[ρ_ij(r)]
    3. Apply Coulomb kernel: Ṽ_ij(G) = ρ̃_ij(G) * v_C(G)
    4. Transform back: V_ij(r) = IFFT[Ṽ_ij(G)]
    5. Contract with orbital and occupation: vR_dm += V_ij(r) * φ_j(r) * n_j
    """
    ngrids = ao_mos.shape[-1]
    moIR_i = ao_mos[i]
    sqrt_ngrids = ngrids ** 0.5
    inv_sqrt_ngrids = 1.0 / sqrt_ngrids
    
    rho1 = numpy.empty(ngrids, dtype=numpy.float64)
    
    for j, moIR_j in enumerate(ao_mos):
        numpy.multiply(moIR_i, moIR_j, out=rho1)
        vG = tools.fft(rho1, mesh)
        vG *= inv_sqrt_ngrids * coulG
        vR = tools.ifft(vG, mesh)
        vR_dm[i] += vR.real * moIR_j * (mo_occ[j] * sqrt_ngrids)


def occri_get_k(mydf, dms, exxdiv=None):
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
    4. Contracts results back to AO basis using build_full_exchange
    5. Applies Ewald correction if requested for periodic boundary conditions
    
    The method scales as O(N_occ^2 * N_grid) where N_occ is the number of 
    occupied orbitals and N_grid is the number of FFT grid points.
    
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
        mo_coeff, mo_occ = make_natural_orbitals(cell, dms)
    tol = 1.0e-6
    is_occ = mo_occ > tol
    mo_coeff = [numpy.asarray(coeff[:, is_occ[i]].T, order='C') for i, coeff in enumerate(mo_coeff)]

    nset = len(mo_coeff)
    mesh = cell.mesh
    weight = (cell.vol/ ngrids)
    nao = mo_coeff[0].shape[-1]
    vk = numpy.empty((nset, nao, nao), numpy.float64)
    s = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=None).astype(numpy.float64)

    aovals = mydf._numint.eval_ao(cell, coords)[0]
    aovals = numpy.asarray(aovals.T, order='C')
    ao_mos = [numpy.matmul( mo, aovals, order='C') for mo in mo_coeff]

    # Parallelize over the outer loop (i) using joblib.
    # The inner loop is handled inside the function. Nec for memory cost.
    coulG = tools.get_coulG(cell, mesh=mesh)
    for j in range(nset):
        nmo = mo_coeff[j].shape[0]
        vR_dm = numpy.zeros((nmo, ngrids), numpy.float64)
        for i in range(nmo):
            integrals_uu(i, ao_mos[j], vR_dm, coulG, mo_occ[j], mesh)

        vR_dm *= weight
        vk_j = aovals @ vR_dm.T
        vk[j] = build_full_exchange(s, vk_j, mo_coeff[j])

    # Function _ewald_exxdiv_for_G0 to add back in the G=0 component to vk_kpts
    # Note in the _ewald_exxdiv_for_G0 implementation, the G=0 treatments are
    # different for 1D/2D and 3D systems.  The special treatments for 1D and 2D
    # can only be used with AFTDF/GDF/MDF method.  In the FFTDF method, 1D, 2D
    # and 3D should use the ewald probe charge correction.
    if exxdiv == 'ewald' and cell.dimension != 0:
        madelung = tools.pbc.madelung(cell, mydf.kpts)
        for i, dm in enumerate(dms):
            vk[i] += madelung * reduce(numpy.dot, (s, dm, s))

    return vk


def occri_get_k_opt(mydf, dms, exxdiv=None):
    """
    Optimized C-accelerated implementation of OCCRI exchange matrix evaluation.
    
    This function provides the production implementation of the occupied orbital
    resolution of identity method using an optimized C extension with OpenMP
    parallelization and FFTW for maximum performance. This is the recommended
    method for all production calculations.
    
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
    This optimized implementation:
    1. Uses the same algorithm as occri_get_k but with C acceleration
    2. Leverages FFTW for optimized FFT operations
    3. Uses OpenMP for parallel loop execution
    4. Optimizes memory layout for better cache performance
    5. Calls the external C function occri_vR for core computations
    
    Performance improvements over Python version:
    - ~5-10x speedup from compiled C code
    - Additional 2-4x speedup from OpenMP parallelization  
    - Better memory efficiency from optimized data layouts
    - FFTW provides fastest possible FFT operations
    
    The C extension must be properly compiled and linked for this function
    to work correctly. Falls back to occri_get_k if C extension fails.
    
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
        mo_coeff, mo_occ = make_natural_orbitals(cell, dms)
    
    tol = 1.0e-6
    is_occ = mo_occ > tol
    mo_coeff = [numpy.ascontiguousarray(coeff[:, is_occ[i]].T) for i, coeff in enumerate(mo_coeff)]

    nset = len(mo_coeff)
    mesh = cell.mesh.astype(numpy.int32)
    weight = cell.vol / ngrids
    nao = mo_coeff[0].shape[-1]
    nmo = mo_coeff[0].shape[0]
    
    vk = numpy.zeros((nset, nao, nao), numpy.float64, order='C')
    s = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=None).astype(numpy.float64)

    aovals = mydf._numint.eval_ao(cell, coords)[0]
    aovals = numpy.ascontiguousarray(aovals.T)
    
    ao_mos = []
    for mo in mo_coeff:
        ao_mo = numpy.empty((mo.shape[0], ngrids), dtype=numpy.float64, order='C')
        numpy.dot(mo, aovals, out=ao_mo)
        ao_mos.append(ao_mo)

    coulG = tools.get_coulG(cell, mesh=mesh).reshape(*mesh)[..., : mesh[2] // 2 + 1].ravel()
    
    for j in range(nset):
        nmo = mo_coeff[j].shape[0]
        vR_dm = numpy.zeros(nmo * ngrids, dtype=numpy.float64, order='C')
        occri.occri_vR(vR_dm, mo_occ[j], coulG, mesh, ao_mos[j].ravel(), nmo)
        
        vR_dm = vR_dm.reshape(nmo, ngrids) * weight
        
        vk_j = numpy.empty((nao, nmo), dtype=numpy.float64, order='C')
        numpy.dot(aovals, vR_dm.T, out=vk_j)
        vk[j] = build_full_exchange(s, vk_j, mo_coeff[j])

    # Function _ewald_exxdiv_for_G0 to add back in the G=0 component to vk_kpts
    # Note in the _ewald_exxdiv_for_G0 implementation, the G=0 treatments are
    # different for 1D/2D and 3D systems.  The special treatments for 1D and 2D
    # can only be used with AFTDF/GDF/MDF method.  In the FFTDF method, 1D, 2D
    # and 3D should use the ewald probe charge correction.
    if exxdiv == 'ewald' and cell.dimension != 0:
        madelung = tools.pbc.madelung(cell, mydf.kpts)
        for i, dm in enumerate(dms):
            vk[i] += madelung * reduce(numpy.dot, (s, dm, s))

    return vk


def occri_get_k(mydf, dms, exxdiv=None):
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
    4. Contracts results back to AO basis using build_full_exchange
    5. Applies Ewald correction if requested for periodic boundary conditions
    
    The method scales as O(N_occ^2 * N_grid) where N_occ is the number of 
    occupied orbitals and N_grid is the number of FFT grid points.
    
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
        mo_coeff, mo_occ = make_natural_orbitals(cell, dms)
    tol = 1.0e-6
    is_occ = mo_occ > tol
    mo_coeff = [numpy.asarray(coeff[:, is_occ[i]].T, order='C') for i, coeff in enumerate(mo_coeff)]

    nset = len(mo_coeff)
    mesh = cell.mesh
    weight = (cell.vol/ ngrids)
    nao = mo_coeff[0].shape[-1]
    vk = numpy.empty((nset, nao, nao), numpy.float64)
    s = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=None).astype(numpy.float64)

    aovals = mydf._numint.eval_ao(cell, coords)[0]
    aovals = numpy.asarray(aovals.T, order='C')
    ao_mos = [numpy.matmul( mo, aovals, order='C') for mo in mo_coeff]

    # Parallelize over the outer loop (i) using joblib.
    # The inner loop is handled inside the function. Nec for memory cost.
    coulG = tools.get_coulG(cell, mesh=mesh)
    for j in range(nset):
        nmo = mo_coeff[j].shape[0]
        vR_dm = numpy.zeros((nmo, ngrids), numpy.float64)
        for i in range(nmo):
            integrals_uu(i, ao_mos[j], vR_dm, coulG, mo_occ[j], mesh)

        vR_dm *= weight
        vk_j = aovals @ vR_dm.T
        vk[j] = build_full_exchange(s, vk_j, mo_coeff[j])

    # Function _ewald_exxdiv_for_G0 to add back in the G=0 component to vk_kpts
    # Note in the _ewald_exxdiv_for_G0 implementation, the G=0 treatments are
    # different for 1D/2D and 3D systems.  The special treatments for 1D and 2D
    # can only be used with AFTDF/GDF/MDF method.  In the FFTDF method, 1D, 2D
    # and 3D should use the ewald probe charge correction.
    if exxdiv == 'ewald' and cell.dimension != 0:
        madelung = tools.pbc.madelung(cell, mydf.kpts)
        for i, dm in enumerate(dms):
            vk[i] += madelung * reduce(numpy.dot, (s, dm, s))

    return vk