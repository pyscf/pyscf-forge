"""
Python-side integrals and exchange matrix evaluation for OCCRI

This file defines:
    - fallback Python-only routines for reference/validation (e.g., integrals_uu)
    - wrapper functions to call the C-extension occri_vR
    - utilities to build the full AO exchange matrix

Key Functions:
    - occri_get_k_kpts_opt: Calls the C implementation of OCCRI
    - occri_get_k_kpts: Calls the reference Python implementation
    - build_full_exchange: Contracts exchange contributions from AO basis

Used internally by the OCCRI class defined in __init__.py
"""

from functools import reduce

import numpy
from pyscf.pbc.df.df_jk import _ewald_exxdiv_for_G0

from pyscf import lib, occri
from pyscf.lib import logger
from pyscf.pbc import tools


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
    Sa_Kao = numpy.matmul(Sa, Kao.T.conj(), order="C")
    Kuv = Sa_Kao + Sa_Kao.T.conj()

    # Third term: -Sa @ (mo_coeff @ Kao) @ Sa.T
    Koo = mo_coeff.conj() @ Kao
    Sa_Koo = numpy.matmul(Sa, Koo)
    Kuv -= numpy.matmul(Sa_Koo, Sa.T.conj(), order="C")
    return Kuv


def integrals_uu(j, k, k_prim, ao_mos, vR_dm, coulG, mo_occ, mesh, expmikr):
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

    """
    ngrids = ao_mos[0].shape[1]
    nmo = ao_mos[k_prim].shape[0]
    rho1 = numpy.empty(ngrids, dtype=vR_dm.dtype)

    for i in range(nmo):
        i_Rg_exp = ao_mos[k_prim][i].conj() * expmikr
        numpy.multiply(i_Rg_exp, ao_mos[k][j], out=rho1)
        vG = tools.fft(rho1, mesh)
        vG *= coulG
        vR = tools.ifft(vG, mesh)
        vR_dm[j] += vR * i_Rg_exp.conj() * mo_occ[k_prim][i]


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
    1. Evaluates orbitals on real-space FFT grid
    2. Computes exchange integrals using FFT-based Coulomb evaluation
    3. Contracts results back to AO basis using occri.build_full_exchange
    4. Applies Ewald correction if requested for periodic boundary conditions

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
    assert cell.low_dim_ft_type != "inf_vacuum"
    assert cell.dimension != 1
    coords = cell.gen_uniform_grids(mesh)
    ngrids = coords.shape[0]
    nset, nk, nao = dms.shape[:3]
    mesh = cell.mesh
    kpts = mydf.kpts
    weight = cell.vol / ngrids**0.5 / nk
    mo_coeff = dms.mo_coeff
    mo_occ = dms.mo_occ

    # Evaluate AOs on the grid for each k-point
    aovals = mydf._numint.eval_ao(cell, coords, kpts=kpts)
    aovals = [numpy.asarray(ao.T, order="C") for ao in aovals]

    # Transform to MO basis for each k-point and spin
    ao_mos = [[mo_coeff[n][k] @ aovals[k] for k in range(nk)] for n in range(nset)]
    out_type = (
        numpy.complex128
        if [abs(ao.imag).max() > 1.0e-6 for ao in aovals]
        else numpy.float64
    )
    aovals = [ao * weight for ao in aovals]

    # Pre-allocate output arrays
    vk = numpy.empty((nset, nk, nao, nao), out_type, order="C")
    s = cell.pbc_intor("int1e_ovlp", hermi=1, kpts=kpts)

    occri.log_mem(mydf)
    t1 = (logger.process_clock(), logger.perf_counter())

    coulG_cache = {}
    expmikr_cache = {}
    inv_sqrt = 1.0 / ngrids**0.5
    for k in range(nk):
        coulG_cache[k] = {}
        expmikr_cache[k] = {}
        for k_prim in range(nk):
            dk = kpts[k] - kpts[k_prim]
            coulG_cache[k][k_prim] = (
                tools.get_coulG(cell, dk, False, mesh=mesh) * inv_sqrt
            )
            if numpy.allclose(dk, 0):
                expmikr_cache[k][k_prim] = numpy.ones(1, dtype=out_type)
            else:
                expmikr_cache[k][k_prim] = numpy.exp(-1j * (coords @ dk))

    for n in range(nset):
        for k in range(nk):
            nmo = mo_coeff[n][k].shape[0]
            vR_dm = numpy.zeros((nmo, ngrids), out_type)
            for j in range(nmo):
                for k_prim in range(nk):
                    coulG = coulG_cache[k][k_prim]
                    expmikr = expmikr_cache[k][k_prim]
                    integrals_uu(
                        j, k, k_prim, ao_mos[n], vR_dm, coulG, mo_occ[n], mesh, expmikr
                    )

            vk_j = numpy.matmul(aovals[k].conj(), vR_dm.T, order="C")
            vk[n][k] = build_full_exchange(s[k], vk_j, mo_coeff[n][k])

            t1 = logger.timer_debug1(mydf, "get_k_kpts: make_kpt (%d,*)" % k, *t1)

    if exxdiv == "ewald" and cell.dimension != 0:
        _ewald_exxdiv_for_G0(cell, kpts, dms, vk)

    return vk


def occri_get_k_kpts_opt(mydf, dms, exxdiv=None):
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

    Raises:
    -------
    AssertionError
        If cell.low_dim_ft_type == 'inf_vacuum' or cell.dimension == 1
    """
    cell = mydf.cell
    mesh = mydf.mesh
    assert cell.low_dim_ft_type != "inf_vacuum"
    assert cell.dimension != 1
    coords = cell.gen_uniform_grids(mesh)
    ngrids = coords.shape[0]
    nset, nk, nao = dms.shape[:3]
    mesh = numpy.asarray(cell.mesh, numpy.int32)
    kpts = mydf.kpts
    weight = cell.vol / ngrids / ngrids**0.5 / nk
    mo_coeff = dms.mo_coeff
    mo_occ = dms.mo_occ

    # Pre-allocate output arrays
    vk = numpy.empty((nset, nk, nao, nao), numpy.complex128, order="C")
    s = cell.pbc_intor("int1e_ovlp", hermi=1, kpts=kpts)

    # Evaluate AOs on the grid for each k-point
    aovals = mydf._numint.eval_ao(cell, coords, kpts=kpts)
    aovals = [numpy.ascontiguousarray(ao.T) for ao in aovals]

    # Transform to MO basis for each k-point and spin
    ao_mos = [[mo_coeff[n][k] @ aovals[k] for k in range(nk)] for n in range(nset)]
    aovals = [ao * weight for ao in aovals]

    occri.log_mem(mydf)
    t1 = (logger.process_clock(), logger.perf_counter())

    # Import C extension components
    from pyscf.occri import _OCCRI_C_AVAILABLE, occri_vR_kpts

    inv_sqrt = 1.0 / ngrids**0.5
    coulG_all = numpy.empty((nk, ngrids), dtype=numpy.float64, order="C")
    expmikr_all_real = numpy.empty((nk, ngrids), dtype=numpy.float64, order="C")
    expmikr_all_imag = numpy.empty((nk, ngrids), dtype=numpy.float64, order="C")
    for n in range(nset):

        nmo_max = max(mo_coeff[n][k].shape[0] for k in range(nk))

        # Flatten AO data for all k-points - pad to nmo_max for consistent indexing
        ao_mos_real = numpy.zeros((nk, nmo_max, ngrids), dtype=numpy.float64, order="C")
        ao_mos_imag = numpy.zeros((nk, nmo_max, ngrids), dtype=numpy.float64, order="C")
        mo_occ_flat = numpy.zeros((nk, nmo_max), dtype=numpy.float64, order="C")

        nmo = []
        for k in range(nk):
            nmo.append(ao_mos[n][k].shape[0])

            ao_mos_real[k][: nmo[k]] = ao_mos[n][k].real
            ao_mos_imag[k][: nmo[k]] = ao_mos[n][k].imag

            mo_occ_flat[k][: nmo[k]] = mo_occ[n][k]
        nmo = numpy.asarray(nmo, numpy.int32)

        for k in range(nk):
            # Prepare arrays for C function
            vR_dm_real = numpy.zeros(nmo[k] * ngrids, dtype=numpy.float64, order="C")
            vR_dm_imag = numpy.zeros(nmo[k] * ngrids, dtype=numpy.float64, order="C")

            # Prepare all Coulomb kernels for this k-point against all k_prim
            for k_prim in range(nk):
                coulG_all[k_prim] = (
                    tools.get_coulG(cell, kpts[k] - kpts[k_prim], False, mesh=mesh)
                    * inv_sqrt
                )
                expmikr = numpy.exp(-1j * coords @ (kpts[k] - kpts[k_prim]))
                expmikr_all_real[k_prim] = expmikr.real
                expmikr_all_imag[k_prim] = expmikr.imag

            # Call optimized C function
            if _OCCRI_C_AVAILABLE and occri_vR_kpts is not None:
                occri_vR_kpts(
                    vR_dm_real,
                    vR_dm_imag,
                    mo_occ_flat,
                    coulG_all.ravel(),
                    mesh,
                    expmikr_all_real.ravel(),
                    expmikr_all_imag.ravel(),
                    kpts.ravel(),
                    ao_mos_real.ravel(),
                    ao_mos_imag.ravel(),
                    nmo,
                    ngrids,
                    nk,
                    k,
                )
            else:
                raise RuntimeError(
                    "occri_get_k_opt_kpts called but C extension not available."
                )

            # Reshape and apply weight - only take the first nmo orbitals
            vR_dm = (vR_dm_real + 1j * vR_dm_imag).reshape(nmo[k], ngrids)

            # Contract back to AO basis
            vk_j = numpy.matmul(aovals[k].conj(), vR_dm.T, order="C")
            vk[n][k] = build_full_exchange(s[k], vk_j, mo_coeff[n][k])

            t1 = logger.timer_debug1(mydf, f"get_k_kpts_opt: k-point {k}", *t1)

    # Apply Ewald correction if requested
    if exxdiv == "ewald" and cell.dimension != 0:
        _ewald_exxdiv_for_G0(cell, kpts, dms, vk)

    return vk
