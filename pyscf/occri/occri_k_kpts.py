"""
OCCRI exchange matrix evaluation for k-point calculations
"""

import numpy
from pyscf import lib, occri
from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.pbc.df.df_jk import _ewald_exxdiv_for_G0

from .utils import build_full_exchange


def integrals_uu(j, k, k_prim, ao_mos, vR_dm, coulG, mo_occ, mesh, expmikr):
    """Compute k-point exchange integrals using complex FFT"""
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
    Reference Python implementation of k-point exchange matrix evaluation with MO blocking
    """
    cell = mydf.cell
    mesh = mydf.mesh
    assert cell.low_dim_ft_type != 'inf_vacuum'
    assert cell.dimension != 1
    coords = cell.gen_uniform_grids(mesh)
    ngrids = coords.shape[0]
    nset, nk, nao = dms.shape[:3]
    mesh = cell.mesh
    kpts = mydf.kpts
    weight = cell.vol / ngrids**0.5 / nk
    mo_coeff = dms.mo_coeff
    mo_occ = dms.mo_occ

    # Calculate MO block size for memory management (following PySCF pattern)
    mem_now = lib.current_memory()[0]
    max_memory = mydf.max_memory - mem_now
    # Memory estimate: blksize * max_nmo * ngrids * 16 bytes (complex128) for main arrays
    max_nmo = max(mo_coeff[n][k].shape[0] for n in range(nset) for k in range(nk))
    mo_blksize = int(min(max_nmo, max(1, (max_memory - mem_now) * 1e6 / 16 / 4 / ngrids / max_nmo)))

    logger.debug1(
        mydf,
        'OCCRI MO blocking: max_memory %d MB, mo_blksize %d',
        max_memory,
        mo_blksize,
    )

    # Evaluate AOs on the grid for each k-point
    aovals = [numpy.asarray(ao.T, order='C') for ao in mydf._numint.eval_ao(cell, coords, kpts=kpts)]

    # Pre-allocate output arrays
    s = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)
    out_type = numpy.complex128 if any(sk.dtype == numpy.complex128 for sk in s) else numpy.float64
    vk = numpy.empty((nset, nk, nao, nao), out_type, order='C')

    occri.log_mem(mydf)
    t1 = (logger.process_clock(), logger.perf_counter())

    inv_sqrt = 1.0 / ngrids**0.5

    for n in range(nset):
        for k in range(nk):
            nmo_k = mo_coeff[n][k].shape[0]
            vR_dm = numpy.zeros((nmo_k, ngrids), out_type)

            for k_prim in range(nk):
                nmo_kprim = mo_coeff[n][k_prim].shape[0]
                dk = kpts[k] - kpts[k_prim]
                coulG = tools.get_coulG(cell, dk, False, mesh=mesh) * inv_sqrt

                if numpy.allclose(dk, 0):
                    expmikr = numpy.ones(1, dtype=out_type)
                else:
                    expmikr = numpy.exp(-1j * (coords @ dk))

                # Transform AOs to MO basis for k_prim (full set needed for contraction)
                ao_mo_kprim = mo_coeff[n][k_prim] @ aovals[k_prim]
                ao_phase = ao_mo_kprim.conj() * expmikr

                # Block over MOs at k-point k (following PySCF pattern)
                for p0, p1 in lib.prange(0, nmo_k, mo_blksize):
                    # Transform AOs to MO basis for this MO block
                    ao_mo_k_blk = mo_coeff[n][k][p0:p1] @ aovals[k]

                    # Compute density and exchange for this MO block
                    rho1 = numpy.einsum('ig,jg->ijg', ao_phase, ao_mo_k_blk)
                    vG = tools.fft(rho1.reshape(-1, ngrids), mesh)
                    rho1 = None  # Free memory like PySCF
                    vG *= coulG
                    vR = tools.ifft(vG, mesh).reshape(nmo_kprim, p1 - p0, ngrids)
                    vG = None  # Free memory like PySCF

                    if vR_dm.dtype == numpy.double:
                        vR = vR.real

                    # Accumulate into the appropriate slice of vR_dm
                    vR_dm[p0:p1] += numpy.einsum('ijg,ig->jg', vR, ao_phase.conj() * mo_occ[n][k_prim][:, None])
                    vR = None  # Free memory like PySCF

            vR_dm *= weight
            vkao = numpy.matmul(aovals[k].conj(), vR_dm.T, order='C')
            vk[n][k] = build_full_exchange(s[k], vkao, mo_coeff[n][k])

            t1 = logger.timer_debug1(mydf, 'get_k_kpts: make_kpt (%d,*)' % k, *t1)

    if exxdiv == 'ewald' and cell.dimension != 0:
        _ewald_exxdiv_for_G0(cell, kpts, dms, vk)

    return vk


def occri_get_k_kpts_opt(mydf, dms, exxdiv=None):
    """Optimized C implementation of k-point exchange matrix evaluation"""
    cell = mydf.cell
    mesh = mydf.mesh
    assert cell.low_dim_ft_type != 'inf_vacuum'
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
    vk = numpy.empty((nset, nk, nao, nao), numpy.complex128, order='C')
    s = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)

    # Evaluate AOs on the grid for each k-point
    aovals = [numpy.asarray(ao.T, order='C') for ao in mydf._numint.eval_ao(cell, coords, kpts=kpts)]

    # Transform to MO basis for each k-point and spin
    ao_mos = [[mo_coeff[n][k] @ aovals[k] for k in range(nk)] for n in range(nset)]
    aovals = [ao * weight for ao in aovals]

    occri.log_mem(mydf)
    t1 = (logger.process_clock(), logger.perf_counter())

    from pyscf.occri import occri_vR_kpts

    inv_sqrt = 1.0 / ngrids**0.5
    coulG_all = numpy.empty((nk, ngrids), dtype=numpy.float64, order='C')
    expmikr_all_real = numpy.empty((nk, ngrids), dtype=numpy.float64, order='C')
    expmikr_all_imag = numpy.empty((nk, ngrids), dtype=numpy.float64, order='C')
    for n in range(nset):
        nmo_max = max(mo_coeff[n][k].shape[0] for k in range(nk))

        # Flatten AO data for all k-points - pad to nmo_max for consistent indexing
        ao_mos_real = numpy.zeros((nk, nmo_max, ngrids), dtype=numpy.float64, order='C')
        ao_mos_imag = numpy.zeros((nk, nmo_max, ngrids), dtype=numpy.float64, order='C')
        mo_occ_flat = numpy.zeros((nk, nmo_max), dtype=numpy.float64, order='C')

        nmo = []
        for k in range(nk):
            nmo.append(ao_mos[n][k].shape[0])

            ao_mos_real[k][: nmo[k]] = ao_mos[n][k].real
            ao_mos_imag[k][: nmo[k]] = ao_mos[n][k].imag

            mo_occ_flat[k][: nmo[k]] = mo_occ[n][k]
        nmo = numpy.asarray(nmo, numpy.int32)

        for k in range(nk):
            # Prepare arrays for C function
            vR_dm_real = numpy.zeros(nmo[k] * ngrids, dtype=numpy.float64, order='C')
            vR_dm_imag = numpy.zeros(nmo[k] * ngrids, dtype=numpy.float64, order='C')

            # Prepare all Coulomb kernels for this k-point against all k_prim
            for k_prim in range(nk):
                coulG_all[k_prim] = tools.get_coulG(cell, kpts[k] - kpts[k_prim], False, mesh=mesh) * inv_sqrt
                expmikr = numpy.exp(-1j * coords @ (kpts[k] - kpts[k_prim]))
                expmikr_all_real[k_prim] = expmikr.real
                expmikr_all_imag[k_prim] = expmikr.imag

            # Call optimized C function
            error_code = occri_vR_kpts(
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
            if error_code != 0:
                if error_code == -1:
                    raise RuntimeError('FFTW thread initialization failed')
                else:
                    raise RuntimeError(f'OCCRI computation failed with error code {error_code}')

            # Reshape and apply weight - only take the first nmo orbitals
            vR_dm = (vR_dm_real + 1j * vR_dm_imag).reshape(nmo[k], ngrids)

            # Contract back to AO basis
            vk_j = numpy.matmul(aovals[k].conj(), vR_dm.T, order='C')
            vk[n][k] = build_full_exchange(s[k], vk_j, mo_coeff[n][k])

            t1 = logger.timer_debug1(mydf, f'get_k_kpts_opt: k-point {k}', *t1)

    # Apply Ewald correction if requested
    if exxdiv == 'ewald' and cell.dimension != 0:
        _ewald_exxdiv_for_G0(cell, kpts, dms, vk)

    return vk
