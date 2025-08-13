import numpy
from pyscf.pbc.df.df_jk import _ewald_exxdiv_for_G0

def isdfx_get_k_kpts(mydf, dms, exxdiv=None):
    cell = mydf.cell
    nao = cell.nao
    kpts = mydf.kpts
    kmesh = mydf.kmesh
    nset, nk, nao = dms.shape[:3]
    mo_coeff = dms.mo_coeff[0]
    mo_occ = dms.mo_occ
    aovals = mydf.aovals
    
    UX = [mo_coeff[k] @ ao_k for k, ao_k in enumerate(aovals)]
    UU = numpy.asarray([numpy.matmul(ux.T * mo_occ[k], ux.conj()) for k, ux in enumerate(UX)], dtype=numpy.complex128, order='C')
    mydf.convolve_with_W(UU)

    # Pre-allocate output arrays
    out_type = (
        numpy.complex128
        if any(abs(ao.imag).max() > 1.0e-6 for ao in aovals)
        else numpy.float64
    )    
    vk = numpy.empty((nset, nk, nao, nao), out_type, order="C")
    s = cell.pbc_intor("int1e_ovlp", hermi=1, kpts=kpts)

    n = 0
    if out_type == numpy.double:
        UU = UU.real
    for k in range(nk):
        XUU = UU[k] @ UX[k].T
        vkao = aovals[k].conj() @ XUU
        vk[n][k] = mydf.build_full_exchange(s[k], vkao, mo_coeff[k])

    if exxdiv == "ewald" and cell.dimension != 0:
        _ewald_exxdiv_for_G0(cell, kpts, dms, vk)

    return vk