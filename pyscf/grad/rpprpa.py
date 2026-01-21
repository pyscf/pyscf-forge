import numpy as np
from functools import reduce
from pyscf import lib, scf
from pyscf.df.df_jk import _DFHF
from pyscf.lib import logger
from pyscf.df.grad import rhf as rhf_grad
from pyscf.grad import rks as rks_grad


def grad_elec(pprpa_grad, xy, mult, atmlst=None):
    mf = pprpa_grad.base._scf
    pprpa = pprpa_grad.base
    mol = mf.mol
    mf_grad = mf.nuc_grad_method()
    if atmlst is None:
        atmlst = range(mol.natm)
    assert mult in ['t', 's'], 'mult = {}. is not valid in grad_elec'.format(mult)

    nocc_all = mf.mol.nelectron // 2
    nocc = pprpa.nocc_act
    nvir = pprpa.nvir_act
    nfrozen_occ = nocc_all - nocc

    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)
    dm0, i_int = make_rdm1_relaxed_rhf_pprpa(
        pprpa, mf, xy=xy, mult=mult, cphf_max_cycle=pprpa_grad.cphf_max_cycle, cphf_conv_tol=pprpa_grad.cphf_conv_tol
    )
    dm0 = np.einsum('pi,ij,qj->pq', mf.mo_coeff, dm0, mf.mo_coeff, optimize=True)
    pprpa_grad.rdm1e = dm0
    dm0_hf = mf.make_rdm1()
    i_int = np.einsum('pi,ij,qj->pq', mf.mo_coeff, i_int, mf.mo_coeff, optimize=True)
    i_int -= mf_grad.make_rdm1e(mf.mo_energy, mf.mo_coeff, mf.mo_occ)

    occ_y_mat, vir_x_mat = get_xy_full(xy, pprpa.oo_dim, mult)
    coeff_occ = mf.mo_coeff[:, nfrozen_occ : nfrozen_occ + nocc]
    coeff_vir = mf.mo_coeff[:, nfrozen_occ + nocc : nfrozen_occ + nocc + nvir]
    xy_ao = np.einsum('pi,ij,qj->pq', coeff_vir, vir_x_mat, coeff_vir, optimize=True) + np.einsum(
        'pi,ij,qj->pq', coeff_occ, occ_y_mat, coeff_occ, optimize=True
    )

    aux_response = False
    if isinstance(mf, _DFHF):
        # aux_response is Ture by default in DFHF
        # To my opinion, aux_response should always be True for DFHF
        aux_response = mf_grad.auxbasis_response
    else:
        logger.warn(pprpa, 
                    'The analytical gradient of the ppRPA must be used with the density\n\
                    fitting mean-field method. The calculation will proceed but the analytical\n\
                    gradient is no longer exact (does NOT agree with numerical gradients).')

    if not hasattr(mf, 'xc'):  # RHF
        vj, vk = mf_grad.get_jk(mol, (dm0_hf, dm0, xy_ao), hermi=0)
        vhf = np.zeros_like(vj)

        vhf[:2] = vj[:2] - 0.5 * vk[:2]
        vhf[2] = vk[2]
        if aux_response:
            vhf_aux = np.zeros_like(vj.aux)
            vhf_aux[:2, :2] = vj.aux[:2, :2] - 0.5 * vk.aux[:2, :2]
            if mult == 's':
                vhf_aux[2, 2] = vk.aux[2, 2]
            else:
                vhf_aux[2, 2] = -vk.aux[2, 2]
            vhf = lib.tag_array(vhf, aux=vhf_aux)

        aoslices = mol.aoslice_by_atom()
        de = np.zeros((len(atmlst), 3))
        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia, 2:]
            h1ao = hcore_deriv(ia)
            h1ao[:,p0:p1]   += vhf[0,:,p0:p1]
            h1ao[:,:,p0:p1] += vhf[0,:,p0:p1].transpose(0,2,1)
            de[k] += np.einsum('xij,ij->x', h1ao, dm0+dm0_hf)
            # nabla was applied on bra in s1, *2 for the contributions of nabla|ket>
            de[k] += np.einsum('xij,ij->x', vhf[1, :, p0:p1], dm0_hf[p0:p1, :]) * 2
            de[k] += np.einsum('xij,ij->x', vhf[2, :, p0:p1], xy_ao[p0:p1, :]) * 2

            de[k] += np.einsum('xij,ji->x', s1[:, p0:p1], i_int[:, p0:p1]) * 2

            if aux_response:
                de[k] += vhf.aux[0, 1, ia] + 0.5*vhf.aux[0, 0, ia]
                de[k] += vhf.aux[1, 0, ia] + 0.5*vhf.aux[0, 0, ia]
                de[k] += vhf.aux[2, 2, ia]
    else:  # RKS
        # The grid response by default is not included.
        # Even if grid_response is set to True, the response is not complete.
        # It will include the response of the Vxc but NOT the fxc.
        # For benchmarking, one can use high grid level to avoid this error.
        # mf_grad.grid_response = True

        vj, vk = mf_grad.get_jk(mol, xy_ao, hermi=0)
        vhf = vk
        if aux_response:
            vxc, vjk = get_veff_df_rks(mf_grad, mol, (dm0_hf, dm0))
            if mult == 's':
                vhf_aux = vk.aux[0, 0]
            else:
                vhf_aux = -vk.aux[0, 0]
            vhf = lib.tag_array(vhf, aux=vhf_aux)
        else:
            vxc, vjk = get_veff_rks(mf_grad, mol, (dm0_hf, dm0))
        
        vjk[1] += _contract_xc_kernel(mf, mf.xc, dm0, None, False, False, True)[0][1:]*0.5

        aoslices = mol.aoslice_by_atom()
        de = np.zeros((len(atmlst), 3))
        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia, 2:]
            h1ao = hcore_deriv(ia)
            h1ao[:, p0:p1] += vxc[0, :, p0:p1] + vjk[0, :, p0:p1]
            h1ao[:, :, p0:p1] += vxc[0, :, p0:p1].transpose(0, 2, 1) + vjk[0, :, p0:p1].transpose(0, 2, 1)
            de[k] += np.einsum('xij,ij->x', h1ao, dm0 + dm0_hf)
            # nabla was applied on bra in s1, *2 for the contributions of nabla|ket>
            de[k] += np.einsum('xij,ij->x', vjk[1, :, p0:p1], dm0_hf[p0:p1, :]) * 2
            de[k] += np.einsum('xij,ij->x', vhf[:, p0:p1], xy_ao[p0:p1, :]) * 2

            de[k] += np.einsum('xij,ji->x', s1[:, p0:p1], i_int[:, p0:p1]) * 2

            if aux_response:
                de[k] += vjk.aux[0, 1, ia] + 0.5*vjk.aux[0, 0, ia]
                de[k] += vjk.aux[1, 0, ia] + 0.5*vjk.aux[0, 0, ia]
                de[k] += vhf.aux[ia]
            if mf_grad.grid_response:
                de[k] += vxc.exc1_grid[0, ia]

    return de


def get_xy_full(xy, oo_dim, mult='t'):
    """Expand the lower triangular xy matrix to the full matrix."""
    vv_dim = len(xy) - oo_dim
    if mult == 't':
        ndim_v = round(0.5 * (np.sqrt(8 * vv_dim + 1) + 1)) if vv_dim > 0 else 0
        ndim_o = round(0.5 * (np.sqrt(8 * oo_dim + 1) + 1)) if oo_dim > 0 else 0
        occ_y_mat = np.zeros((ndim_o, ndim_o), dtype=xy.dtype)
        vir_x_mat = np.zeros((ndim_v, ndim_v), dtype=xy.dtype)
        occ_y_mat[np.tril_indices(ndim_o, -1)] = xy[:oo_dim]
        vir_x_mat[np.tril_indices(ndim_v, -1)] = xy[oo_dim:]
        occ_y_mat = occ_y_mat - occ_y_mat.T
        vir_x_mat = vir_x_mat - vir_x_mat.T
    else:
        ndim_v = round(0.5 * (np.sqrt(8 * vv_dim + 1) - 1)) if vv_dim > 0 else 0
        ndim_o = round(0.5 * (np.sqrt(8 * oo_dim + 1) - 1)) if oo_dim > 0 else 0
        occ_y_mat = np.zeros((ndim_o, ndim_o), dtype=xy.dtype)
        vir_x_mat = np.zeros((ndim_v, ndim_v), dtype=xy.dtype)
        occ_y_mat[np.tril_indices(ndim_o)] = xy[:oo_dim]
        vir_x_mat[np.tril_indices(ndim_v)] = xy[oo_dim:]
        occ_y_mat = occ_y_mat + occ_y_mat.T
        vir_x_mat = vir_x_mat + vir_x_mat.T
        np.fill_diagonal(occ_y_mat, 1.0 / np.sqrt(2.0) * occ_y_mat.diagonal())
        np.fill_diagonal(vir_x_mat, 1.0 / np.sqrt(2.0) * vir_x_mat.diagonal())

    return occ_y_mat, vir_x_mat


def make_rdm1_unrelaxed_from_xy_full(occ_y_mat, vir_x_mat, diag=True):
    """Make unrelaxed one-particle density matrix from the full X and Y matrices."""
    if diag:
        di = -np.einsum('ij,ij->i', occ_y_mat.conj(), occ_y_mat)
        da = np.einsum('ab,ab->a', vir_x_mat.conj(), vir_x_mat)
        # combine the two parts
        den = np.concatenate((di, da))
    else:
        print('Warning: non-diagonal 1e-RDM is not well-defined for pp-RPA with both DEA and DIP blocks.')
        den_v = np.einsum('ac,bc->ba', vir_x_mat.conj(), vir_x_mat)
        den_o = -np.einsum('ik,jk->ij', occ_y_mat.conj(), occ_y_mat)
        den = np.zeros((den_v.shape[0] + den_o.shape[0], den_v.shape[1] + den_o.shape[1]), dtype=den_v.dtype)
        den[: den_o.shape[0], : den_o.shape[1]] = den_o
        den[den_o.shape[0] :, den_o.shape[1] :] = den_v
    return den


def make_rdm1_relaxed_rhf_pprpa(pprpa, mf, xy=None, mult='t', istate=0, cphf_max_cycle=20, cphf_conv_tol=1.0e-8):
    r"""Calculate relaxed density matrix (and the I intermediates)
        for given pprpa and mean-field object.
    Args:
        pprpa: a pprpa object.
        mf: a mean-field RHF/RKS object.
    Returns:
        den_relaxed: the relaxed one-particle density matrix (nmo_full, nmo_full)
        i_int: the I intermediates (nmo_full, nmo_full)
        Both are in the MO basis.
    """
    assert mult in ['t', 's'], 'mult = {}. is not valid in make_rdm1_relaxed_pprpa'.format(mult)

    if xy is None:
        if mult == 's':
            xy = pprpa.xy_s[istate]
        else:
            xy = pprpa.xy_t[istate]
    nocc_all = mf.mol.nelectron // 2
    nvir_all = mf.mol.nao - nocc_all
    nocc = pprpa.nocc_act
    nvir = pprpa.nvir_act
    nfrozen_occ = nocc_all - nocc
    nfrozen_vir = nvir_all - nvir
    if mult == 's':
        oo_dim = (nocc + 1) * nocc // 2
    else:
        oo_dim = (nocc - 1) * nocc // 2

    # create slices
    slice_p = choose_slice('p', nfrozen_occ, nocc, nvir, nfrozen_vir)  # all active
    slice_i = choose_slice('i', nfrozen_occ, nocc, nvir, nfrozen_vir)  # active occupied
    slice_a = choose_slice('a', nfrozen_occ, nocc, nvir, nfrozen_vir)  # active virtual
    slice_ip = choose_slice('ip', nfrozen_occ, nocc, nvir, nfrozen_vir)  # frozen occupied
    slice_ap = choose_slice('ap', nfrozen_occ, nocc, nvir, nfrozen_vir)  # frozen virtual
    slice_I = choose_slice('I', nfrozen_occ, nocc, nvir, nfrozen_vir)  # all occupied
    slice_A = choose_slice('A', nfrozen_occ, nocc, nvir, nfrozen_vir)  # all virtual

    orbA = mf.mo_coeff[:, slice_A]
    orbI = mf.mo_coeff[:, slice_I]
    orbp = mf.mo_coeff[:, slice_p]
    orbi = mf.mo_coeff[:, slice_i]
    orba = mf.mo_coeff[:, slice_a]
    occ_y_mat, vir_x_mat = get_xy_full(xy, oo_dim, mult)
    if nfrozen_occ > 0 or nfrozen_vir > 0:
        mo_ene_full = mf.mo_energy
        Lpq_full = pprpa.ao2mo(full_mo=True)
    else:
        mo_ene_full = pprpa.mo_energy
        if pprpa.Lpq is not None:
            Lpq_full = pprpa.Lpq
        elif pprpa.Lpi is not None and pprpa.Lpa is not None:
            Lpq_full = np.concatenate((pprpa.Lpi, pprpa.Lpa), axis=2)
        else:
            raise RuntimeError('Lpq or Lpi/Lpa is required in make_rdm1_relaxed_rhf_pprpa')

    # set singlet=None, generate function for CPHF type response kernel
    vresp = mf.gen_response(singlet=None, hermi=1)
    den_u = make_rdm1_unrelaxed_from_xy_full(occ_y_mat, vir_x_mat)
    den_u_ao = np.einsum('pi,i,qi->pq', orbp, den_u, orbp, optimize=True)
    veff_den_u = reduce(np.dot, (mf.mo_coeff.T, vresp(den_u_ao) * 2, mf.mo_coeff))

    # calculate I' first
    i_prime = np.zeros((len(mo_ene_full), len(mo_ene_full)), dtype=occ_y_mat.dtype)
    # I' active-active block
    i_prime[slice_p, slice_p] += contraction_2rdm_Lpq(
        occ_y_mat, vir_x_mat, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, 'p', 'p'
    )
    i_prime[slice_a, slice_i] += veff_den_u[slice_a, slice_i]
    for p in choose_range('p', nfrozen_occ, nocc, nvir, nfrozen_vir):
        i_prime[p, p] += mo_ene_full[p] * den_u[p - nfrozen_occ]

    if nfrozen_vir > 0:
        # I' frozen virtual-active block
        i_prime[slice_ap, slice_p] += contraction_2rdm_Lpq(
            occ_y_mat, vir_x_mat, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, 'ap', 'p'
        )
        i_prime[slice_ap, slice_i] += veff_den_u[slice_ap, slice_i]
    if nfrozen_occ > 0:
        # I' frozen occupied-active block
        i_prime[slice_ip, slice_p] += contraction_2rdm_Lpq(
            occ_y_mat, vir_x_mat, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, 'ip', 'p'
        )
        # I' all virtual-frozen occupied block
        i_prime[slice_A, slice_ip] += veff_den_u[slice_A, slice_ip]

    # calculate I'' next
    i_prime_prime = np.zeros_like(i_prime)
    # I'' active virtual-all occupied block
    i_prime_prime[slice_a, slice_I] = i_prime[slice_a, slice_I] - i_prime[slice_I, slice_a].T
    # I'' = I' blocks
    i_prime_prime[slice_A, slice_a] = i_prime[slice_A, slice_a]
    i_prime_prime[slice_I, slice_i] = i_prime[slice_I, slice_i]
    i_prime_prime[slice_ap, slice_I] = i_prime[slice_ap, slice_I]

    d_prime = np.zeros_like(i_prime_prime)
    threshold = 1.0e-6
    # D' all occupied-active occupied block
    for i in choose_range('I', nfrozen_occ, nocc, nvir, nfrozen_vir):
        for j in choose_range('i', nfrozen_occ, nocc, nvir, nfrozen_vir):
            denorm = mo_ene_full[j] - mo_ene_full[i]
            factor = 1.0 / denorm if abs(denorm) >= threshold else 0.0
            d_prime[i, j] = factor * i_prime_prime[i, j]

    # D' all virtual-active virtual block
    for a in choose_range('A', nfrozen_occ, nocc, nvir, nfrozen_vir):
        for b in choose_range('a', nfrozen_occ, nocc, nvir, nfrozen_vir):
            denorm = mo_ene_full[b] - mo_ene_full[a]
            factor = 1.0 / denorm if abs(denorm) >= threshold else 0.0
            d_prime[a, b] = factor * i_prime_prime[a, b]

    x_int = i_prime_prime[slice_A, slice_I].copy()
    d_ao = reduce(np.dot, (orbI, d_prime[slice_I, slice_i], orbi.T))
    d_ao += reduce(np.dot, (orbA, d_prime[slice_A, slice_a], orba.T))
    d_ao += d_ao.T
    x_int += reduce(np.dot, (orbA.T, vresp(d_ao) * 2, orbI))

    def fvind(x):
        dm = reduce(np.dot, (orbA, x.reshape(nvir + nfrozen_vir, nocc + nfrozen_occ) * 2, orbI.T))
        v1ao = vresp(dm + dm.T)
        return reduce(np.dot, (orbA.T, v1ao, orbI)).ravel()

    from pyscf.scf import cphf

    d_prime[slice_A, slice_I] = cphf.solve(
        fvind, mo_ene_full, mf.mo_occ, x_int, max_cycle=cphf_max_cycle, tol=cphf_conv_tol
    )[0].reshape(nvir + nfrozen_vir, nocc + nfrozen_occ)

    i_int = -np.einsum('qp,p->qp', d_prime, mo_ene_full)
    # I all occupied-all occupied block
    dp_ao = reduce(np.dot, (mf.mo_coeff, d_prime, mf.mo_coeff.T))
    veff_dp_II = reduce(np.dot, (orbI.T, vresp(dp_ao + dp_ao.T), orbI))
    i_int[slice_I, slice_I] -= 0.5 * veff_den_u[slice_I, slice_I]
    i_int[slice_I, slice_I] -= veff_dp_II
    # I active virtual-all occupied block
    i_int[slice_I, slice_a] -= i_prime[slice_I, slice_a]

    # I active-active block extra term
    for i in choose_range('p', nfrozen_occ, nocc, nvir, nfrozen_vir):
        for j in choose_range('p', nfrozen_occ, nocc, nvir, nfrozen_vir):
            denorm = mo_ene_full[j] - mo_ene_full[i]
            if abs(denorm) < threshold:
                i_int[i, j] -= 0.5 * i_prime[i, j]

    den_relaxed = d_prime
    # active-active block
    for p in choose_range('p', nfrozen_occ, nocc, nvir, nfrozen_vir):
        den_relaxed[p, p] += 0.5 * den_u[p - nfrozen_occ]
    den_relaxed = den_relaxed + den_relaxed.T
    i_int = i_int + i_int.T

    return den_relaxed, i_int


def choose_slice(label, nfrozen_occ, nocc, nvir, nfrozen_vir):
    """Choose the slice for the given label.
    "i" for active occupied orbitals;
    "a" for active virtual orbitals;
    "p" for all active orbitals;
    "ip" for frozen occupied orbitals;
    "ap" for frozen virtual orbitals;
    "I" for all occupied orbitals;
    "A" for all virtual orbitals;
    "P" for all orbitals;

    In the energy ordering, frozen occ -> active occ -> active vir -> frozen vir.
    """
    if label == 'i':
        return slice(nfrozen_occ, nfrozen_occ + nocc)
    elif label == 'a':
        return slice(nfrozen_occ + nocc, nfrozen_occ + nocc + nvir)
    elif label == 'p':
        return slice(nfrozen_occ, nfrozen_occ + nocc + nvir)
    elif label == 'ip':
        return slice(0, nfrozen_occ)
    elif label == 'ap':
        return slice(nfrozen_occ + nocc + nvir, nfrozen_occ + nocc + nvir + nfrozen_vir)
    elif label == 'I':
        return slice(0, nfrozen_occ + nocc)
    elif label == 'A':
        return slice(nfrozen_occ + nocc, nfrozen_occ + nocc + nvir + nfrozen_vir)
    elif label == 'P':
        return slice(0, nfrozen_occ + nocc + nvir + nfrozen_vir)
    else:
        raise ValueError('label = {}. is not valid in choose_slice'.format(label))


def choose_range(label, nfrozen_occ, nocc, nvir, nfrozen_vir):
    """Choose the range list for the given label.
    "i" for active occupied orbitals;
    "a" for active virtual orbitals;
    "p" for all active orbitals;
    "ip" for frozen occupied orbitals;
    "ap" for frozen virtual orbitals;
    "I" for all occupied orbitals;
    "A" for all virtual orbitals;
    "P" for all orbitals;

    In the energy ordering, frozen occ -> active occ -> active vir -> frozen vir.
    """
    if label == 'i':
        return range(nfrozen_occ, nfrozen_occ + nocc)
    elif label == 'a':
        return range(nfrozen_occ + nocc, nfrozen_occ + nocc + nvir)
    elif label == 'p':
        return range(nfrozen_occ, nfrozen_occ + nocc + nvir)
    elif label == 'ip':
        return range(0, nfrozen_occ)
    elif label == 'ap':
        return range(nfrozen_occ + nocc + nvir, nfrozen_occ + nocc + nvir + nfrozen_vir)
    elif label == 'I':
        return range(0, nfrozen_occ + nocc)
    elif label == 'A':
        return range(nfrozen_occ + nocc, nfrozen_occ + nocc + nvir + nfrozen_vir)
    elif label == 'P':
        return range(0, nfrozen_occ + nocc + nvir + nfrozen_vir)
    else:
        raise ValueError('label = {}. is not valid in choose_slice'.format(label))


def contraction_2rdm_Lpq(occ_y_mat, vir_x_mat, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, label1, label2):
    r"""
    Contraction in the form of (anti-symmetrized or symmetrized)
        I_{tp} = \frac{1}{2} \sum_{qrs} \Gamma_{pq,rs} \langle tq||rs \rangle
               = \frac{1}{2} \sum_{qrsP} XY_{pq}^* XY_{rs}
                  (L^P_{tr} L^P_{qs} \pm L^P_{ts} L^P_{qr})
               = \sum_{qrsP} XY_{pq}^* XY_{rs} L^P_{tr} L^P_{qs}
        I_{ti} = \sum_{jklP} Y_{ij}^* Y_{kl} L^P_{tk} L^P_{jl}
               + \sum_{jcdP} Y_{ij}^* X_{cd} L^P_{tc} L^P_{jd}
        I_{ta} = \sum_{bklP} X_{ab}^* Y_{kl} L^P_{tk} L^P_{bl}
               + \sum_{bcdP} X_{ab}^* X_{cd} L^P_{tc} L^P_{bd}
    Args:
        occ_y_mat: coefficients for occupied orbitals Y
        vir_x_mat: coefficients for virtual orbitals X
        Lpq_full: full Lpq matrix including frozen orbitals.
        nocc: number of active occupied orbitals.
        nvir: number of active virtual orbitals.
        nfrozen_occ: number of frozen occupied orbitals.
        nfrozen_vir: number of frozen virtual orbitals.
        label1: label for the first index t
        label2: label for the second index p
    Returns:
        out: contracted intermediates.
    """
    # qrs are all active space indices.
    slice1 = choose_slice(label1, nfrozen_occ, nocc, nvir, nfrozen_vir)
    slice_i = choose_slice('i', nfrozen_occ, nocc, nvir, nfrozen_vir)
    slice_a = choose_slice('a', nfrozen_occ, nocc, nvir, nfrozen_vir)
    naux = Lpq_full.shape[0]
    
    # Special cases for TDA
    if label1 == 'p':
        if nocc == 0:
            label1 = 'a'
        elif nvir == 0:
            label1 = 'i'
    if label2 == 'p':
        if nocc == 0:
            label2 = 'a'
        elif nvir == 0:
            label2 = 'i'

    if label1 == 'i':
        n1 = nocc
    elif label1 == 'a':
        n1 = nvir
    elif label1 == 'ip':
        n1 = nfrozen_occ
    elif label1 == 'ap':
        n1 = nfrozen_vir
    else:
        n1 = nocc + nvir
    if label2 == 'i':
        # Slow but more readable version
        # out = np.einsum("ij,kl,Ptk,Pjl->ti", occ_y_mat.conj(), occ_y_mat,
        #                 Lpq_full[:,slice1,slice_i],
        #                 Lpq_full[:,slice_i,slice_i],
        #                 optimize=True)
        # out+= np.einsum("ij,cd,Ptc,Pjd->ti", occ_y_mat.conj(), vir_x_mat,
        #                 Lpq_full[:,slice1,slice_a],
        #                 Lpq_full[:,slice_i,slice_a],
        #                 optimize=True)
        L1i = np.ascontiguousarray(Lpq_full[:, slice1, slice_i]).reshape(-1, nocc)  # (Pt,k)
        Lij = np.ascontiguousarray(Lpq_full[:, slice_i, slice_i]).reshape(-1, nocc).conj()  # (P,j,l)*->(Pl,j)
        L1i = np.matmul(L1i, occ_y_mat).reshape(naux, n1, nocc)  # (Pt,k)(k,l)->(P,t,l)
        L1i = L1i.transpose(1, 0, 2).reshape(n1, -1)  # (P,t,l)->(t,Pl)
        tmp = np.matmul(L1i, Lij)  # (t,Pl)(Pl,j)->(t,j)
        out = np.matmul(tmp, occ_y_mat.T.conj())  # (t,j)(j,i) -> (t,i)

        if nvir > 0:
            L1a = np.ascontiguousarray(Lpq_full[:, slice1, slice_a]).reshape(-1, nvir)  # (Pt,c)
            Lai = np.ascontiguousarray(Lpq_full[:, slice_a, slice_i]).reshape(-1, nocc).conj()  # (P,d,j)*->(Pd,j)
            L1a = np.matmul(L1a, vir_x_mat).reshape(naux, n1, nvir)  # (Pt,c)(c,d)->(P,t,d)
            L1a = L1a.transpose(1, 0, 2).reshape(n1, -1)  # (P,t,d)->(t,Pd)
            tmp = np.matmul(L1a, Lai)  # (t,Pd)(Pd,j)->(t,j)
            out += np.matmul(tmp, occ_y_mat.T.conj())  # (t,j)(j,i) -> (t,i)
    elif label2 == 'a':
        # Slow but more readable version
        # out = np.einsum("ab,cd,Ptc,Pbd->ta", vir_x_mat.conj(), vir_x_mat,
        #                 Lpq_full[:,slice1,slice_a],
        #                 Lpq_full[:,slice_a,slice_a],
        #                 optimize=True)
        # out+= np.einsum("ab,kl,Ptk,Pbl->ta", vir_x_mat.conj(), occ_y_mat,
        #                 Lpq_full[:,slice1,slice_i],
        #                 Lpq_full[:,slice_a,slice_i],
        #                 optimize=True)
        L1a = np.ascontiguousarray(Lpq_full[:, slice1, slice_a]).reshape(-1, nvir)  # (Pt,c)
        Lab = np.ascontiguousarray(Lpq_full[:, slice_a, slice_a]).reshape(-1, nvir).conj()  # (P,b,d)*->(Pd,b)
        L1a = np.matmul(L1a, vir_x_mat).reshape(naux, n1, nvir)  # (Pt,c)(c,d)->(P,t,d)
        L1a = L1a.transpose(1, 0, 2).reshape(n1, -1)  # (P,t,d)->(t,Pd)
        tmp = np.matmul(L1a, Lab)  # (t,Pd)(Pd,b)->(t,b)
        out = np.matmul(tmp, vir_x_mat.T.conj())  # (t,b)(b,a) -> (t,a)

        if nocc > 0:
            L1i = np.ascontiguousarray(Lpq_full[:, slice1, slice_i]).reshape(-1, nocc)  # (Pt,k)
            Lia = np.ascontiguousarray(Lpq_full[:, slice_i, slice_a]).reshape(-1, nvir).conj()  # (P,l,b)*->(Pl,b)
            L1i = np.matmul(L1i, occ_y_mat).reshape(naux, n1, nocc)  # (Pt,k)(k,l)->(P,t,l)
            L1i = L1i.transpose(1, 0, 2).reshape(n1, -1)  # (P,t,l)->(t,Pl)
            tmp = np.matmul(L1i, Lia)  # (t,Pl)(Pl,b)->(t,b)
            out += np.matmul(tmp, vir_x_mat.T.conj())  # (t,b)(b,a) -> (t,a)
    elif label2 == 'p':
        # slow (more copies) but more readable version
        # out = np.concatenate((
        # contraction_2rdm_Lpq(occ_y_mat, vir_x_mat, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, label1, "i"),
        # contraction_2rdm_Lpq(occ_y_mat, vir_x_mat, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, label1, "a")), axis=1)
        out = np.zeros((n1, nocc + nvir), dtype=occ_y_mat.dtype)
        L1i = np.ascontiguousarray(Lpq_full[:, slice1, slice_i]).reshape(-1, nocc)  # (Pt,k)
        Lia = np.ascontiguousarray(Lpq_full[:, slice_i, slice_a]).reshape(-1, nvir).conj()  # (P,l,b)*->(Pl,b)
        Lij = np.ascontiguousarray(Lpq_full[:, slice_i, slice_i]).reshape(-1, nocc).conj()  # (P,j,l)*->(Pl,j)
        L1i = np.matmul(L1i, occ_y_mat).reshape(naux, n1, nocc)  # (Pt,k)(k,l)->(P,t,l)
        L1i = L1i.transpose(1, 0, 2).reshape(n1, -1)  # (P,t,l)->(t,Pl)
        tmp = np.matmul(L1i, Lia)  # (t,Pl)(Pl,b)->(t,b)
        out[:, nocc:] = np.matmul(tmp, vir_x_mat.T.conj())  # (t,b)(b,a) -> (t,a)
        tmp = np.matmul(L1i, Lij)  # (t,Pl)(Pl,j)->(t,j)
        out[:, :nocc] = np.matmul(tmp, occ_y_mat.T.conj())  # (t,j)(j,i) -> (t,i)

        L1a = np.ascontiguousarray(Lpq_full[:, slice1, slice_a]).reshape(-1, nvir)  # (Pt,c)
        Lab = np.ascontiguousarray(Lpq_full[:, slice_a, slice_a]).reshape(-1, nvir).conj()  # (P,b,d)*->(Pd,b)
        Lai = np.ascontiguousarray(Lpq_full[:, slice_a, slice_i]).reshape(-1, nocc).conj()  # (P,d,j)*->(Pd,j)
        L1a = np.matmul(L1a, vir_x_mat).reshape(naux, n1, nvir)  # (Pt,c)(c,d)->(P,t,d)
        L1a = L1a.transpose(1, 0, 2).reshape(n1, -1)  # (P,t,d)->(t,Pd)
        tmp = np.matmul(L1a, Lab)  # (t,Pd)(Pd,b)->(t,b)
        out[:, nocc:] += np.matmul(tmp, vir_x_mat.T.conj())  # (t,b)(b,a) -> (t,a)
        tmp = np.matmul(L1a, Lai)  # (t,Pd)(Pd,j)->(t,j)
        out[:, :nocc] += np.matmul(tmp, occ_y_mat.T.conj())  # (t,j)(j,i) -> (t,i)
    else:
        raise ValueError('label2 = {}. is not valid in contraction_2rdm_Lpq'.format(label2))

    return out


'''
The following functions are modified from pyscf functions
to accommodate auxbasis_response in pprpa.
'''
def get_veff_df_rks(ks_grad, mol=None, dm=None):
    """Coulomb + XC functional response
    Modified from pyscf.df.grad.rks.get_veff
    """
    if mol is None:
        mol = ks_grad.mol
    if dm is None:
        dm = ks_grad.base.make_rdm1()
    t0 = (logger.process_clock(), logger.perf_counter())

    mf = ks_grad.base
    ni = mf._numint
    grids, nlcgrids = rks_grad._initialize_grids(ks_grad)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, ks_grad.max_memory * 0.9 - mem_now)
    if ks_grad.grid_response:
        exc = []
        vxc = []
        for dmi in dm:
            exci, vxci = rks_grad.get_vxc_full_response(
                ni, mol, grids, mf.xc, dmi, max_memory=max_memory, verbose=ks_grad.verbose
            )
            exc.append(exci)
            vxc.append(vxci)
        exc = np.asarray(exc)
        vxc = np.asarray(vxc)
        if mf.do_nlc():
            if ni.libxc.is_nlc(mf.xc):
                xc = mf.xc
            else:
                xc = mf.nlc
            enlc, vnlc = rks_grad.get_nlc_vxc_full_response(
                ni, mol, nlcgrids, xc, dm, max_memory=max_memory, verbose=ks_grad.verbose
            )
            exc += enlc
            vxc += vnlc
        logger.debug1(ks_grad, 'sum(grids response) %s', exc.sum(axis=0))
    else:
        exc, vxc = rks_grad.get_vxc(ni, mol, grids, mf.xc, dm, max_memory=max_memory, verbose=ks_grad.verbose)
        if mf.do_nlc():
            if ni.libxc.is_nlc(mf.xc):
                xc = mf.xc
            else:
                xc = mf.nlc
            enlc, vnlc = rks_grad.get_nlc_vxc(ni, mol, nlcgrids, xc, dm, max_memory=max_memory, verbose=ks_grad.verbose)
            vxc += vnlc
    t0 = logger.timer(ks_grad, 'vxc', *t0)

    vjk = np.zeros_like(vxc)
    if not ni.libxc.is_hybrid_xc(mf.xc):
        vj = ks_grad.get_j(mol, dm)
        vjk += vj
        if ks_grad.auxbasis_response:
            e1_aux = vj.aux
    else:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
        vj, vk = ks_grad.get_jk(mol, dm)
        if ks_grad.auxbasis_response:
            vk.aux *= hyb
        vk[:] *= hyb  # Don't erase the .aux tags!
        if omega != 0:  # For range separated Coulomb operator
            # TODO: replaced with vk_sr which is numerically more stable for
            # inv(int2c2e)
            vk_lr = ks_grad.get_k(mol, dm, omega=omega)
            vk[:] += vk_lr * (alpha - hyb)
            if ks_grad.auxbasis_response:
                vk.aux[:] += vk_lr.aux * (alpha - hyb)
        vjk += vj - vk * 0.5
        if ks_grad.auxbasis_response:
            e1_aux = vj.aux - vk.aux * 0.5

    if ks_grad.auxbasis_response:
        vjk = lib.tag_array(vjk, aux=e1_aux)
    if ks_grad.grid_response:
        vxc = lib.tag_array(vxc, exc1_grid=exc)
    return vxc, vjk


def get_veff_rks(ks_grad, mol=None, dm=None):
    """
    First order derivative of DFT effective potential matrix (wrt electron coordinates)

    Args:
        ks_grad : grad.uhf.Gradients or grad.uks.Gradients object
    """
    if mol is None:
        mol = ks_grad.mol
    if dm is None:
        dm = ks_grad.base.make_rdm1()
    t0 = (logger.process_clock(), logger.perf_counter())

    mf = ks_grad.base
    ni = mf._numint
    grids, nlcgrids = rks_grad._initialize_grids(ks_grad)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, ks_grad.max_memory * 0.9 - mem_now)
    if ks_grad.grid_response:
        exc, vxc = rks_grad.get_vxc_full_response(
            ni, mol, grids, mf.xc, dm, max_memory=max_memory, verbose=ks_grad.verbose
        )
        if mf.do_nlc():
            if ni.libxc.is_nlc(mf.xc):
                xc = mf.xc
            else:
                xc = mf.nlc
            enlc, vnlc = rks_grad.get_nlc_vxc_full_response(
                ni, mol, nlcgrids, xc, dm, max_memory=max_memory, verbose=ks_grad.verbose
            )
            exc += enlc
            vxc += vnlc
        logger.debug1(ks_grad, 'sum(grids response) %s', exc.sum(axis=0))
    else:
        exc, vxc = rks_grad.get_vxc(ni, mol, grids, mf.xc, dm, max_memory=max_memory, verbose=ks_grad.verbose)
        if mf.do_nlc():
            if ni.libxc.is_nlc(mf.xc):
                xc = mf.xc
            else:
                xc = mf.nlc
            enlc, vnlc = rks_grad.get_nlc_vxc(ni, mol, nlcgrids, xc, dm, max_memory=max_memory, verbose=ks_grad.verbose)
            vxc += vnlc
    t0 = logger.timer(ks_grad, 'vxc', *t0)

    vjk = np.zeros_like(vxc)
    if not ni.libxc.is_hybrid_xc(mf.xc):
        vj = ks_grad.get_j(mol, dm)
        vjk += vj
    else:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
        vj, vk = ks_grad.get_jk(mol, dm)
        vk *= hyb
        if omega != 0:
            vk += ks_grad.get_k(mol, dm, omega=omega) * (alpha - hyb)
        vjk += vj - vk * 0.5

    vxc = lib.tag_array(vxc, exc1_grid=exc)
    return vxc, vjk


def _contract_xc_kernel(mf, xc_code, dmvo, dmoo=None, with_vxc=True, with_kxc=True, singlet=True, max_memory=2000):
    from pyscf import lib
    from pyscf.lib import logger
    from pyscf.grad.tdrks import _lda_eval_mat_, _gga_eval_mat_, _mgga_eval_mat_

    mol = mf.mol
    grids = mf.grids

    ni = mf._numint
    xctype = ni._xc_type(xc_code)

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    # dmvo ~ reduce(np.dot, (orbv, Xai, orbo.T))
    dmvo = (dmvo + dmvo.T) * 0.5  # because K_{ia,jb} == K_{ia,bj}

    f1vo = np.zeros((4, nao, nao))  # 0th-order, d/dx, d/dy, d/dz
    deriv = 2
    if dmoo is not None:
        f1oo = np.zeros((4, nao, nao))
    else:
        f1oo = None
    if with_vxc:
        v1ao = np.zeros((4, nao, nao))
    else:
        v1ao = None
    if with_kxc:
        k1ao = np.zeros((4, nao, nao))
        deriv = 3
    else:
        k1ao = None

    if xctype == 'HF':
        return f1vo, f1oo, v1ao, k1ao
    elif xctype == 'LDA':
        fmat_, ao_deriv = _lda_eval_mat_, 1
    elif xctype == 'GGA':
        fmat_, ao_deriv = _gga_eval_mat_, 2
    elif xctype == 'MGGA':
        fmat_, ao_deriv = _mgga_eval_mat_, 2
        logger.warn(mf, 'PPRPA-MGGA Gradients may be inaccurate due to grids response')
    else:
        raise NotImplementedError(f'td-rks for functional {xc_code}')

    if singlet:
        for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            if xctype == 'LDA':
                ao0 = ao[0]
            else:
                ao0 = ao
            rho = ni.eval_rho2(mol, ao0, mo_coeff, mo_occ, mask, xctype, with_lapl=False)
            vxc, fxc, kxc = ni.eval_xc_eff(xc_code, rho, deriv, xctype=xctype)[1:]

            rho1 = ni.eval_rho(mol, ao0, dmvo, mask, xctype, hermi=1, with_lapl=False) * 2  # *2 for alpha + beta
            if xctype == 'LDA':
                rho1 = rho1[np.newaxis]
            wv = np.einsum('yg,xyg,g->xg', rho1, fxc, weight)
            fmat_(mol, f1vo, ao, wv, mask, shls_slice, ao_loc)

            if dmoo is not None:
                rho2 = ni.eval_rho(mol, ao0, dmoo, mask, xctype, hermi=1, with_lapl=False) * 2
                if xctype == 'LDA':
                    rho2 = rho2[np.newaxis]
                wv = np.einsum('yg,xyg,g->xg', rho2, fxc, weight)
                fmat_(mol, f1oo, ao, wv, mask, shls_slice, ao_loc)
            if with_vxc:
                fmat_(mol, v1ao, ao, vxc * weight, mask, shls_slice, ao_loc)
            if with_kxc:
                wv = np.einsum('yg,zg,xyzg,g->xg', rho1, rho1, kxc, weight)
                fmat_(mol, k1ao, ao, wv, mask, shls_slice, ao_loc)
    else:
        for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            if xctype == 'LDA':
                ao0 = ao[0]
            else:
                ao0 = ao
            rho = ni.eval_rho2(mol, ao0, mo_coeff, mo_occ, mask, xctype, with_lapl=False)
            rho *= 0.5
            rho = np.repeat(rho[np.newaxis], 2, axis=0)
            vxc, fxc, kxc = ni.eval_xc_eff(xc_code, rho, deriv, xctype=xctype)[1:]
            # fxc_t couples triplet excitation amplitudes
            # 1/2 int (tia - tIA) fxc (tjb - tJB) = tia fxc_t tjb
            fxc_t = fxc[:, :, 0] - fxc[:, :, 1]
            fxc_t = fxc_t[0] - fxc_t[1]

            rho1 = ni.eval_rho(mol, ao0, dmvo, mask, xctype, hermi=1, with_lapl=False)
            if xctype == 'LDA':
                rho1 = rho1[np.newaxis]
            wv = np.einsum('yg,xyg,g->xg', rho1, fxc_t, weight)
            fmat_(mol, f1vo, ao, wv, mask, shls_slice, ao_loc)

            if dmoo is not None:
                # fxc_s == 2 * fxc of spin restricted xc kernel
                # provides f1oo to couple the interaction between first order MO
                # and density response of tddft amplitudes, which is described by dmoo
                fxc_s = fxc[0, :, 0] + fxc[0, :, 1]
                rho2 = ni.eval_rho(mol, ao0, dmoo, mask, xctype, hermi=1, with_lapl=False)
                if xctype == 'LDA':
                    rho2 = rho2[np.newaxis]
                wv = np.einsum('yg,xyg,g->xg', rho2, fxc_s, weight)
                fmat_(mol, f1oo, ao, wv, mask, shls_slice, ao_loc)
            if with_vxc:
                vxc = vxc[0]
                fmat_(mol, v1ao, ao, vxc * weight, mask, shls_slice, ao_loc)
            if with_kxc:
                # kxc in terms of the triplet coupling
                # 1/2 int (tia - tIA) kxc (tjb - tJB) = tia kxc_t tjb
                kxc = kxc[0, :, 0] - kxc[0, :, 1]
                kxc = kxc[:, :, 0] - kxc[:, :, 1]
                wv = np.einsum('yg,zg,xyzg,g->xg', rho1, rho1, kxc, weight)
                fmat_(mol, k1ao, ao, wv, mask, shls_slice, ao_loc)

    f1vo[1:] *= -1
    if f1oo is not None:
        f1oo[1:] *= -1
    if v1ao is not None:
        v1ao[1:] *= -1
    if k1ao is not None:
        k1ao[1:] *= -1

    return f1vo, f1oo, v1ao, k1ao


class Gradients(rhf_grad.Gradients):
    cphf_max_cycle = 20
    cphf_conv_tol = 1e-8

    _keys = {
        'cphf_max_cycle',
        'cphf_conv_tol',
        'mol',
        'base',
        'chkfile',
        'state',
        'atmlst',
        'de',
    }

    def __init__(self, pprpa, mult='t', state=0):
        assert isinstance(pprpa._scf, scf.hf.RHF)
        self.base = pprpa
        self.mol = pprpa._scf.mol
        self.state = state
        self.verbose = self.mol.verbose
        self.mult = mult

        self.rdm1e = None
        self.atmlst = None
        self.de = None

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** %s gradients for %s ********', self.base.__class__, self.base._scf.__class__)
        log.info('cphf_conv_tol = %g', self.cphf_conv_tol)
        log.info('cphf_max_cycle = %d', self.cphf_max_cycle)
        log.info('State ID = %d', self.state)
        log.info('max_memory %d MB (current use %d MB)', self.base._scf.max_memory, lib.current_memory()[0])
        log.info('\n')
        return self

    def grad_elec(self, xy, mult, atmlst):
        return grad_elec(self, xy, mult, atmlst)

    def _finalize(self):
        if self.verbose >= logger.NOTE:
            logger.note(
                self, '--------- %s gradients for state %d ----------', self.base.__class__.__name__, self.state
            )
            self._write(self.mol, self.de, self.atmlst)
            logger.note(self, '----------------------------------------------')

    def kernel(self, xy=None, state=None, mult=None, atmlst=None):
        if mult is None:
            mult = self.mult
        if xy is None:
            if state is None:
                state = self.state
            else:
                self.state = state
            if mult == 't':
                xy = self.base.xy_t[state]
            else:
                xy = self.base.xy_s[state]
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst

        if self.verbose >= logger.INFO:
            self.dump_flags()

        de = self.grad_elec(xy, mult, atmlst)
        self.de = de = de + self.grad_nuc(atmlst=atmlst)
        if self.mol.symmetry:
            self.de = self.symmetrize(self.de, atmlst)
        self._finalize()
        return self.de


Grad = Gradients

from pyscf.pprpa.rpprpa_direct import RppRPADirect
from pyscf.pprpa.rpprpa_davidson import RppRPADavidson

RppRPADirect.Gradients = RppRPADavidson.Gradients = lib.class_as_method(Gradients)