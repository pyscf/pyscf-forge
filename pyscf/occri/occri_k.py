import numpy
import scipy
from pyscf import lib
from pyscf.pbc import tools
from functools import reduce
from pyscf import occri 
# import joblib

def make_natural_orbitals(cell, dms, kpts=None):
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

    # First term: Sa @ Koa  
    Sa = S @ mo_coeff.T
    tmp = numpy.matmul(Sa, Kao.T, order='C')  # Ensure C-contiguous for better cache performance
    
    # Initialize with first term
    Kuv = tmp.copy()
    
    # Second term: add transpose in-place (leverages symmetry)
    Kuv += tmp.T
    
    # Third term: use temporary for the inner multiplication to avoid double allocation
    Koo = mo_coeff @ Kao
    Sa_Koo = numpy.matmul(Sa, Koo, order='C')
    Kuv -= numpy.matmul(Sa_Koo, Sa.T)
    return Kuv


def integrals_uu(i, ao_mos, vR_dm, coulG, mo_occ, mesh):
    ngrids = ao_mos.shape[-1]
    moIR_i = ao_mos[i]
    for j, moIR_j in enumerate(ao_mos):
        rho1 = moIR_i * moIR_j
        vG = tools.fft(rho1, mesh)
        vG *= 1.0 / ngrids ** 0.5
        vG *= coulG
        vR = tools.ifft(vG, mesh)
        vR *= ngrids ** 0.5
        vR_dm[i] += vR.real * moIR_j * mo_occ[j]


def occRI_get_k(mydf, dms, exxdiv=None):
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
        vR_dm2 = numpy.zeros((nmo, ngrids), numpy.float64)
        for i in range(nmo):
            integrals_uu(i, ao_mos[j], vR_dm2, coulG, mo_occ[j], mesh)

        vR_dm *= weight
        vk_j = aovals @ vR_dm.T
        vk[j] = build_full_exchange(s, vk_j, mo_coeff[j])

    # Function _ewald_exxdiv_for_G0 to add back in the G=0 component to vk_kpts
    # Note in the _ewald_exxdiv_for_G0 implementation, the G=0 treatments are
    # different for 1D/2D and 3D systems.  The special treatments for 1D and 2D
    # can only be used with AFTDF/GDF/MDF method.  In the FFTDF method, 1D, 2D
    # and 3D should use the ewald probe charge correction.
    if exxdiv == 'ewald' and cell.dimension != 0:
        s = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=None).astype(numpy.float64)
        madelung = tools.pbc.madelung(cell, mydf.kpts)
        for i, dm in enumerate(dms):
            vk[i] += madelung * reduce(numpy.dot, (s, dm, s))

    return vk


def occRI_get_k_opt(mydf, dms, exxdiv=None):
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
    mesh = cell.mesh.astype(numpy.int32)
    weight = (cell.vol / ngrids)
    nao = mo_coeff[0].shape[-1]
    nmo = mo_coeff[0].shape[0]
    vk = numpy.empty((nset, nao, nao), numpy.float64)
    s = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=None).astype(numpy.float64)

    aovals = mydf._numint.eval_ao(cell, coords)[0]
    aovals = numpy.asarray(aovals.T, order='C')
    ao_mos = [numpy.matmul( mo, aovals, order='C') for mo in mo_coeff]

    # Parallelize over the outer loop (i) using joblib.
    # The inner loop is handled inside the function. Nec for memory cost.
    coulG = tools.get_coulG(cell, mesh=mesh).reshape(*mesh)[..., : mesh[2] // 2 + 1].ravel()

    for j in range(nset):
        nmo = mo_coeff[j].shape[0]
        vR_dm = numpy.zeros((nmo* ngrids), numpy.float64)    
        occri.occRI_vR(vR_dm, mo_occ[j], coulG, mesh, ao_mos[j].ravel(), nmo)
        vR_dm = vR_dm.reshape(nmo, ngrids)
        vR_dm *= weight
        vk_j = aovals @ vR_dm.T
        vk[j] = build_full_exchange(s, vk_j, mo_coeff[j])

    # Function _ewald_exxdiv_for_G0 to add back in the G=0 component to vk_kpts
    # Note in the _ewald_exxdiv_for_G0 implementation, the G=0 treatments are
    # different for 1D/2D and 3D systems.  The special treatments for 1D and 2D
    # can only be used with AFTDF/GDF/MDF method.  In the FFTDF method, 1D, 2D
    # and 3D should use the ewald probe charge correction.
    if exxdiv == 'ewald' and cell.dimension != 0:
        s = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=None).astype(numpy.float64)
        madelung = tools.pbc.madelung(cell, mydf.kpts)
        for i, dm in enumerate(dms):
            vk[i] += madelung * reduce(numpy.dot, (s, dm, s))

    return vk