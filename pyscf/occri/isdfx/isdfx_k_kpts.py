"""
ISDFX K-point Exchange Matrix Evaluation

This module implements k-point exchange matrix calculation using the ISDFX
(Interpolative Separable Density Fitting eXchange) method with tensor
hypercontraction for efficient evaluation in periodic systems.

Main functions:
    isdfx_get_k_kpts: Compute exchange matrices for all k-points using ISDFX
"""

import numpy
from pyscf.lib import logger
from pyscf.occri.utils import build_full_exchange
from pyscf.pbc.df.df_jk import _ewald_exxdiv_for_G0


def isdfx_get_k_kpts(mydf, dms, exxdiv=None):
    """
    Compute exchange matrices for all k-points using ISDFX method.

    This function evaluates the exact exchange contribution to the Kohn-Sham
    potential using the Interpolative Separable Density Fitting (ISDFX) method
    with tensor hypercontraction. The method reduces computational cost by
    representing exchange integrals in terms of auxiliary functions at
    carefully selected grid points.

    Parameters:
    -----------
    mydf : ISDFX
        ISDFX object containing precomputed interpolation data:
        - pivots: Selected interpolation grid points
        - aovals: AO values at interpolation points
        - W: THC potential tensor
    dms : ndarray
        Density matrices with orbital information
        Shape: (nset, nk, nao, nao) where nset is number of spin sets
        Must have attributes: mo_coeff, mo_occ (molecular orbital data)
    exxdiv : str, optional
        Ewald divergence treatment method (default: None)
        'ewald' applies Ewald correction for periodic boundary conditions

    Returns:
    --------
    ndarray
        Exchange matrix for each k-point and spin
        Shape: (nset, nk, nao, nao)

    Notes:
    ------
    The ISDFX exchange matrix is computed as:
    K_μν^k = Σ_ij^occ C_μi^k C_νj^k* ∫∫ φ_i^k(r) φ_j^k*(r') v(r-r') φ_μ^k(r') φ_ν^k*(r) dr dr'

    Using tensor hypercontraction:
    K = Σ_g,h χ_g W_gh χ_h^†
    where χ_g are ISDFX fitting functions and W_gh is the THC potential.

    The algorithm:
    1. Transform MOs to interpolation basis: U = C^† @ φ(Rg)
    2. Construct interaction tensor: UU = U^† O U
    3. Apply THC potential via convolution: UU ← W * UU
    4. Back-transform to AO basis: K = φ(Rg) @ UU @ U^†
    5. Apply Ewald correction if requested
    """
    cell = mydf.cell
    assert cell.low_dim_ft_type != 'inf_vacuum'
    assert cell.dimension != 1
    nao = cell.nao
    kpts = mydf.kpts
    nset, nk = dms.shape[:2]
    mo_coeff = dms.mo_coeff
    mo_occ = dms.mo_occ
    aovals = mydf.aovals  # AO values at ISDFX interpolation points

    s = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)
    out_type = numpy.complex128 if any(sk.dtype == numpy.complex128 for sk in s) else numpy.float64
    vk = numpy.empty((nset, nk, nao, nao), dtype=out_type, order='C')
    for n in range(nset):
        ao_mos = [mo_coeff[n][k] @ aovals[k] for k in range(nk)]
        rho1 = [(ao_mos[k].T * mo_occ[n][k]) @ ao_mos[k].conj() for k in range(nk)]
        rho1 = numpy.asarray(rho1, dtype=numpy.complex128, order='C')
        mydf.convolve_with_W(rho1)
        vR = rho1
        if out_type == numpy.float64:
            vR = vR.real

        t1 = (logger.process_clock(), logger.perf_counter())
        for k in range(nk):
            vR_dm = numpy.matmul(vR[k], ao_mos[k].T, order='C')
            vkao = numpy.matmul(aovals[k].conj(), vR_dm, order='C')
            vk[n][k] = build_full_exchange(s[k], vkao, mo_coeff[n][k])
            t1 = logger.timer_debug1(mydf, 'get_k_kpts: make_kpt (%d,*)' % k, *t1)

    if exxdiv == 'ewald' and cell.dimension != 0:
        _ewald_exxdiv_for_G0(cell, kpts, dms, vk)

    return vk
