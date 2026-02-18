#!/usr/bin/env python
# Copyright 2014-2024 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file can be merged into pyscf.tdscf.uhf.py

import numpy as np
from pyscf import lib
from pyscf import scf, dft
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.tdscf import rhf
from pyscf import __config__
from pyscf import symm
from pyscf.data import nist
from pyscf.sftda import uks_sf
from pyscf.tdscf._lr_eig import eigh as lr_eigh

# import function
from pyscf.sftda.scf_genrep_sftd import gen_uhf_response_sf
from pyscf.sftda.numint2c_sftd import cache_xc_kernel_sf

# import class
from pyscf.tdscf.uhf import TDBase
from pyscf.scf import uhf_symm

MO_BASE = getattr(__config__, 'MO_BASE', 1)

def oscillator_strength(tdobj, ref=1, state=None):
    r'''
    Oscillator strengths between excited states for spin-flip TDDFT/TDA.
    Only applicable to length gauge.

    Args:
            tdobj : an instance of TDA_SF/TDDFT_SF
            ref : int
                Index of the reference excited state (1-based). Default is 1.
            state : int, list, or ndarray, optional
                Index/indices of the target excited state(s) (1-based).
                If None, all excited states except the 'ref' state are calculated.

        Returns:
            float or ndarray
            Oscillator strength(s) between the reference and target state(s).
    '''
    if state is None:
        states = np.arange(tdobj.nstates) + 1
    else:
        states = np.atleast_1d(state)
    states = states[states != ref]

    trans_dip = transition_dipole(tdobj, ref, states)

    ref -= 1
    states -= 1
    es = tdobj.e[states] - tdobj.e[ref]
    f = (2./3.) * lib.einsum('n,nx,nx->n', es, trans_dip.conj(), trans_dip).real
    if isinstance(state, int):
        return f[0]
    else:
        return f

def transition_dipole(tdobj, ref=1, state=None):
    '''
    Transition dipole moments between excited states for Spin-flip TDDFT/TDA.
    Only applicable to length gauge.
    '''
    if state is None:
        states = np.arange(tdobj.nstates) + 1
    else:
        states = np.atleast_1d(state)
    states = states[states != ref]
    ref -= 1
    states -= 1

    mf = tdobj._scf
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    occidxa = mo_occ[0] > 0
    occidxb = mo_occ[1] > 0
    viridxa = mo_occ[0] == 0
    viridxb = mo_occ[1] == 0
    orboa = mo_coeff[0][:, occidxa]
    orbob = mo_coeff[1][:, occidxb]
    orbva = mo_coeff[0][:, viridxa]
    orbvb = mo_coeff[1][:, viridxb]

    mx = tdobj.xy[ref][0]
    nxs = np.array([tdobj.xy[i][0] for i in states])
    if isinstance(tdobj.xy[0][1], np.ndarray):
        my = tdobj.xy[ref][1]
        nys = np.array([tdobj.xy[i][1] for i in states])
    else:
        nys = None
        my = None
    if tdobj.extype==0:
        gamma_oo_bb = - lib.einsum('ia,nja->nij', mx.conj(), nxs)
        gamma_bb = lib.einsum('uj,vi,nij->nvu', orbob.conj(), orbob, gamma_oo_bb)
        gamma_vv_aa = lib.einsum('ib,nia->nab', mx.conj(), nxs)
        gamma_aa = lib.einsum('ub,va,nab->nvu', orbva.conj(), orbva, gamma_vv_aa)
        if my is not None:
            gamma_oo_aa = - lib.einsum('ja,nia->nij', my.conj(), nys)
            gamma_aa += lib.einsum('uj,vi,nij->nvu', orboa.conj(), orboa, gamma_oo_aa)
            gamma_vv_bb = lib.einsum('ia,nib->nab', my.conj(), nys)
            gamma_bb += lib.einsum('ub,va,nab->nvu', orbvb.conj(), orbvb, gamma_vv_bb)
    elif tdobj.extype==1:
        gamma_oo_aa = - lib.einsum('ia,nja->nij', mx.conj(), nxs)
        gamma_aa = lib.einsum('uj,vi,nij->nvu', orboa.conj(), orboa, gamma_oo_aa)
        gamma_vv_bb = lib.einsum('ib,nia->nab', mx.conj(), nxs)
        gamma_bb = lib.einsum('ub,va,nab->nvu', orbvb.conj(), orbvb, gamma_vv_bb)
        if my is not None:
            gamma_oo_bb = - lib.einsum('ja,nia->nij', my.conj(), nys)
            gamma_bb += lib.einsum('uj,vi,nij->nvu', orbob.conj(), orbob, gamma_oo_bb)
            gamma_vv_aa = lib.einsum('ia,nib->nab', my.conj(), nys)
            gamma_aa += lib.einsum('ub,va,nab->nvu', orbva.conj(), orbva, gamma_vv_aa)

    gamma = gamma_aa + gamma_bb
    dip_int = mf.mol.intor_symmetric('int1e_r', comp=3)
    pol = lib.einsum('nvu,xuv->nx', gamma, dip_int)
    return pol.real

def spin_square(tdobj, state=None):
    r'''
    <S^2> of excited states for Spin-flip TDDFT/TDA.
    Ref: J. Chem. Phys. 2011, 134, 134101.

    Args:
            tdobj : an instance of TDA_SF/TDDFT_SF
            state : int, list, or ndarray, optional
                Index/indices of the target excited state(s) (1-based).
                If None, all excited states are calculated.

    Returns:
            float or ndarray
            <S^2> of the target excited state(s).
    '''
    mf = tdobj._scf
    s20, _ = mf.spin_square()
    sz = mf.mol.spin / 2.0

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    occidxa = mo_occ[0] > 0
    occidxb = mo_occ[1] > 0
    viridxa = mo_occ[0] == 0
    viridxb = mo_occ[1] == 0
    orboa = mo_coeff[0][:, occidxa]
    orbob = mo_coeff[1][:, occidxb]
    orbva = mo_coeff[0][:, viridxa]
    orbvb = mo_coeff[1][:, viridxb]

    ovlp = mf.get_ovlp()
    sab_oo = orboa.conj().T @ ovlp @ orbob
    sba_oo = sab_oo.conj().T
    sab_vo = orbva.conj().T @ ovlp @ orbob
    sba_ov = sab_vo.conj().T
    sba_vo = orbvb.conj().T @ ovlp @ orboa
    sab_ov = sba_vo.conj().T

    if state is None:
        states = np.arange(tdobj.nstates)
    else:
        states = np.atleast_1d(state) - 1
    xs = np.array([tdobj.xy[i][0].T for i in states])
    if isinstance(tdobj.xy[0][1], np.ndarray):
        ys = np.array([tdobj.xy[i][1].T for i in states])
    else:
        ys = None

    if tdobj.extype==0:
        assert xs[0].shape==sab_vo.shape
        P_ab = lib.einsum('nai,naj,jk,ki->n', xs.conj(), xs, sba_oo, sab_oo) \
               - lib.einsum('nai,nbi,kb,ak->n', xs.conj(), xs, sba_ov, sab_vo) \
               + lib.einsum('nai,nbj,jb,ai->n', xs.conj(), xs, sba_ov, sab_vo)
        if ys is not None:
            assert ys[0].shape==sba_vo.shape
            P_ab += lib.einsum('nai,naj,ik,kj->n', ys.conj(), ys, sab_oo, sba_oo) \
                    - lib.einsum('nai,nbi,ka,bk->n', ys.conj(), ys, sab_ov, sba_vo) \
                    + lib.einsum('nai,nbj,ia,bj->n', ys.conj(), ys, sab_ov, sba_vo) \
                    - 2 * lib.einsum('nai,nbj,ai,bj->n', xs.conj(), ys, sab_vo, sba_vo).real
        ds2 = P_ab + 2 * sz + 1
    elif tdobj.extype==1:
        assert xs[0].shape==sba_vo.shape
        P_ab = lib.einsum('nai,naj,jk,ki->n', xs.conj(), xs, sab_oo, sba_oo) \
               - lib.einsum('nai,nbi,kb,ak->n', xs.conj(), xs, sab_ov, sba_vo) \
               + lib.einsum('nai,nbj,jb,ai->n', xs.conj(), xs, sab_ov, sba_vo)
        if ys is not None:
            assert ys[0].shape==sab_vo.shape
            P_ab += lib.einsum('nai,naj,ik,kj->n', ys.conj(), ys, sba_oo, sab_oo) \
                    - lib.einsum('nai,nbi,ka,bk->n', ys.conj(), ys, sba_ov, sab_vo) \
                    + lib.einsum('nai,nbj,ia,bj->n', ys.conj(), ys, sba_ov, sab_vo) \
                    - 2 * lib.einsum('nai,nbj,ai,bj->n', xs.conj(), ys, sba_vo, sab_vo).real
        ds2 = P_ab - 2 * sz + 1

    s2s = s20 + ds2.real
    if isinstance(state, int):
        return s2s[0]
    else:
        return s2s

def _analyze_wfnsym(tdobj, x_sym, x):
    '''
    Guess the excitation symmetry of TDDFT X amplitude.
    Return a label.
    x_sym and x are of the same shape.'''
    possible_sym = x_sym[(x > 0.1) | (x < -0.1)]
    wfnsym = symm.MULTI_IRREPS
    ids = possible_sym[possible_sym != symm.MULTI_IRREPS]
    if len(ids) > 0 and all(ids == ids[0]):
        wfnsym = ids[0]
    if wfnsym == symm.MULTI_IRREPS:
        wfnsym_label = '???'
    else:
        wfnsym_label = symm.irrep_id2name(tdobj.mol.groupname, wfnsym)
    return wfnsym, wfnsym_label

def analyze(tdobj, verbose=None):
    log = logger.new_logger(tdobj, verbose)
    mol = tdobj.mol
    maska, maskb = tdobj.get_frozen_mask()
    mo_coeff = (tdobj._scf.mo_coeff[0][:, maska], tdobj._scf.mo_coeff[1][:, maskb])
    mo_occ = (tdobj._scf.mo_occ[0][maska], tdobj._scf.mo_occ[1][maskb])
    nocc_a = np.count_nonzero(mo_occ[0] == 1)
    nocc_b = np.count_nonzero(mo_occ[1] == 1)

    if mol.symmetry and mol.groupname!='C1':
        orbsyma, orbsymb = uhf_symm.get_orbsym(mol, mo_coeff)
        x_symab = symm.direct_prod(orbsyma[mo_occ[0]==1], orbsymb[mo_occ[1]==0], mol.groupname)
        x_symba = symm.direct_prod(orbsymb[mo_occ[1]==1], orbsyma[mo_occ[0]==0], mol.groupname)
    else:
        x_symab = x_symba = None
    S2s = spin_square(tdobj)
    for i in range(tdobj.nstates):
        x, y = tdobj.xy[i]
        if tdobj.extype==0:
            x_sym = x_symba
        elif tdobj.extype==1:
            x_sym = x_symab
        S2 = S2s[i]
        e_ev = np.asarray(tdobj.e[i]) * nist.HARTREE2EV
        if x_symab is None:
            log.note('Excited State %3d: %12.5f eV   <S^2>: %6.4f', i+1, e_ev, S2)
        else:
            wfnsymid, wfnsymlabel = _analyze_wfnsym(tdobj, x_sym, x)
            refsym = tdobj._scf.get_wfnsym()
            statesymid = wfnsymid ^ refsym
            if refsym == symm.MULTI_IRREPS or wfnsymid == symm.MULTI_IRREPS:
                statesymlabel = '???'
            else:
                statesymlabel = symm.irrep_id2name(mol.groupname, statesymid)
            log.note('Excited State %3d: %4s (State: %4s) %12.5f eV   <S^2>: %6.4f',
                     i+1, wfnsymlabel, statesymlabel, e_ev, S2)

        if log.verbose >= logger.INFO:
            if tdobj.extype==0:
                for o, v in zip(* np.where(abs(x) > 0.1)):
                    log.info('    %4db -> %4da %12.5f', o+MO_BASE, v+MO_BASE+nocc_a, x[o,v])
            elif tdobj.extype==1:
                for o, v in zip(* np.where(abs(x) > 0.1)):
                    log.info('    %4da -> %4db %12.5f', o+MO_BASE, v+MO_BASE+nocc_b, x[o,v])

def get_ab_sf(mf, mo_energy=None, mo_coeff=None, mo_occ=None, collinear_samples=200):
    r'''A and B matrices for TDDFT response function.

    A[i,a,j,b] = \delta_{ab}\delta_{ij}(E_a - E_i) + (ia||bj)
    B[i,a,j,b] = (ia||jb)

    Spin symmetry is not considered in the returned A, B lists.
    List A has two items: (A_baba, A_abab).
    List B has two items: (B_baab, B_abba).
    '''
    if isinstance(mf, scf.rohf.ROHF) or isinstance(mf, scf.hf_symm.SymAdaptedROHF):
        if isinstance(mf, dft.roks.ROKS) or isinstance(mf, dft.rks_symm.SymAdaptedROKS):
            mf = mf.to_uks()
        else:
            mf = mf.to_uhf()
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ

    mol = mf.mol
    nao = mol.nao_nr()
    occidx_a = np.where(mo_occ[0]==1)[0]
    viridx_a = np.where(mo_occ[0]==0)[0]
    occidx_b = np.where(mo_occ[1]==1)[0]
    viridx_b = np.where(mo_occ[1]==0)[0]
    orbo_a = mo_coeff[0][:,occidx_a]
    orbv_a = mo_coeff[0][:,viridx_a]
    orbo_b = mo_coeff[1][:,occidx_b]
    orbv_b = mo_coeff[1][:,viridx_b]
    nocc_a = orbo_a.shape[1]
    nvir_a = orbv_a.shape[1]
    nocc_b = orbo_b.shape[1]
    nvir_b = orbv_b.shape[1]

    if np.allclose(mf.mo_coeff[0], mf.mo_coeff[1]):
        logger.info(mf, 'Restricted open-shell detected.')
        fock_ao_a, fock_ao_b = mf.get_fock()
        fock_oo_a = orbo_a.T @ fock_ao_a @ orbo_a
        fock_vv_a = orbv_a.T @ fock_ao_a @ orbv_a
        fock_oo_b = orbo_b.T @ fock_ao_b @ orbo_b
        fock_vv_b = orbv_b.T @ fock_ao_b @ orbv_b

        a_b2a = np.zeros((nocc_b,nvir_a,nocc_b,nvir_a))
        a_a2b = np.zeros((nocc_a,nvir_b,nocc_a,nvir_b))
        a_b2a += lib.einsum('ik,ab->iakb', np.eye(nocc_b), fock_vv_a)
        a_b2a -= lib.einsum('ac,ik->iakc', np.eye(nvir_a), fock_oo_b.T)
        a_a2b += lib.einsum('ik,ab->iakb', np.eye(nocc_a), fock_vv_b)
        a_a2b -= lib.einsum('ac,ik->iakc', np.eye(nvir_b), fock_oo_a.T)

    else:
        e_ia_b2a = (mo_energy[0][viridx_a,None] - mo_energy[1][occidx_b]).T
        e_ia_a2b = (mo_energy[1][viridx_b,None] - mo_energy[0][occidx_a]).T

        a_b2a = np.diag(e_ia_b2a.ravel()).reshape(nocc_b,nvir_a,nocc_b,nvir_a)
        a_a2b = np.diag(e_ia_a2b.ravel()).reshape(nocc_a,nvir_b,nocc_a,nvir_b)
    b_b2a = np.zeros((nocc_b,nvir_a,nocc_a,nvir_b))
    b_a2b = np.zeros((nocc_a,nvir_b,nocc_b,nvir_a))
    a = (a_b2a, a_a2b)
    b = (b_b2a, b_a2b)

    def add_hf_(a, b, hyb=1):
        # In spin flip TDA/ TDDFT, hartree potential is zero.
        # A : iabj ---> ijba; B : iajb ---> ibja
        eri_a_b2a = ao2mo.general(mol, [orbo_b,orbo_b,orbv_a,orbv_a], compact=False)
        eri_a_a2b = ao2mo.general(mol, [orbo_a,orbo_a,orbv_b,orbv_b], compact=False)
        eri_b_b2a = ao2mo.general(mol, [orbo_b,orbv_b,orbo_a,orbv_a], compact=False)
        eri_b_a2b = ao2mo.general(mol, [orbo_a,orbv_a,orbo_b,orbv_b], compact=False)

        eri_a_b2a = eri_a_b2a.reshape(nocc_b,nocc_b,nvir_a,nvir_a)
        eri_a_a2b = eri_a_a2b.reshape(nocc_a,nocc_a,nvir_b,nvir_b)
        eri_b_b2a = eri_b_b2a.reshape(nocc_b,nvir_b,nocc_a,nvir_a)
        eri_b_a2b = eri_b_a2b.reshape(nocc_a,nvir_a,nocc_b,nvir_b)

        a_b2a, a_a2b = a
        b_b2a, b_a2b = b

        a_b2a-= lib.einsum('ijba->iajb', eri_a_b2a) * hyb
        a_a2b-= lib.einsum('ijba->iajb', eri_a_a2b) * hyb
        b_b2a-= lib.einsum('ibja->iajb', eri_b_b2a) * hyb
        b_a2b-= lib.einsum('ibja->iajb', eri_b_a2b) * hyb

    if isinstance(mf, dft.KohnShamDFT):
        from pyscf.dft import xc_deriv
        from pyscf.dft import numint2c
        ni0 = mf._numint
        ni = numint2c.NumInt2C()
        ni.collinear = 'mcol'
        ni.collinear_samples = collinear_samples
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        if mf.nlc or ni.libxc.is_nlc(mf.xc):
            logger.warn(mf, 'NLC functional found in DFT object.  Its second '
                        'deriviative is not available. Its contribution is '
                        'not included in the response function.')
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)

        add_hf_(a, b, hyb)

        if omega != 0:
            with mol.with_range_coulomb(omega):
                eri_a_b2a = ao2mo.general(mol, [orbo_b,orbo_b,orbv_a,orbv_a], compact=False)
                eri_a_a2b = ao2mo.general(mol, [orbo_a,orbo_a,orbv_b,orbv_b], compact=False)
                eri_b_b2a = ao2mo.general(mol, [orbo_b,orbv_b,orbo_a,orbv_a], compact=False)
                eri_b_a2b = ao2mo.general(mol, [orbo_a,orbv_a,orbo_b,orbv_b], compact=False)

                eri_a_b2a = eri_a_b2a.reshape(nocc_b,nocc_b,nvir_a,nvir_a)
                eri_a_a2b = eri_a_a2b.reshape(nocc_a,nocc_a,nvir_b,nvir_b)
                eri_b_b2a = eri_b_b2a.reshape(nocc_b,nvir_b,nocc_a,nvir_a)
                eri_b_a2b = eri_b_a2b.reshape(nocc_a,nvir_a,nocc_b,nvir_b)

                k_fac = alpha - hyb
                a_b2a, a_a2b = a
                b_b2a, b_a2b = b
                a_b2a -= lib.einsum('ijba->iajb', eri_a_b2a) * k_fac
                a_a2b -= lib.einsum('ijba->iajb', eri_a_a2b) * k_fac
                b_b2a -= lib.einsum('ibja->iajb', eri_b_b2a) * k_fac
                b_a2b -= lib.einsum('ibja->iajb', eri_b_a2b) * k_fac

        if collinear_samples < 0:
            return a, b

        xctype = ni._xc_type(mf.xc)
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, mf.max_memory*.8-mem_now)

        # it should be optimized, which is the disadvantage of mc approach.
        fxc = cache_xc_kernel_sf(ni, mol, mf.grids, mf.xc, mo_coeff, mo_occ,deriv=2,spin=1)[2]
        p0,p1=0,0 # the two parameters are used for counts the batch of grids.

        if xctype == 'LDA':
            ao_deriv = 0
            for ao, mask, weight, coords \
                    in ni0.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
                p0 = p1
                p1+= weight.shape[0]
                wfxc= fxc[0,0][...,p0:p1] * weight

                rho_o_a = lib.einsum('rp,pi->ri', ao, orbo_a)
                rho_v_a = lib.einsum('rp,pi->ri', ao, orbv_a)
                rho_o_b = lib.einsum('rp,pi->ri', ao, orbo_b)
                rho_v_b = lib.einsum('rp,pi->ri', ao, orbv_b)
                rho_ov_b2a = lib.einsum('ri,ra->ria', rho_o_b, rho_v_a)
                rho_ov_a2b = lib.einsum('ri,ra->ria', rho_o_a, rho_v_b)

                w_ov = lib.einsum('ria,r->ria', rho_ov_b2a, wfxc*2.0)
                iajb = lib.einsum('ria,rjb->iajb', rho_ov_b2a, w_ov)
                a_b2a += iajb
                iajb = lib.einsum('ria,rjb->iajb', rho_ov_a2b, w_ov)
                b_a2b += iajb

                w_ov = lib.einsum('ria,r->ria', rho_ov_a2b, wfxc*2.0)
                iajb = lib.einsum('ria,rjb->iajb', rho_ov_a2b, w_ov)
                a_a2b += iajb
                iajb = lib.einsum('ria,rjb->iajb', rho_ov_b2a, w_ov)
                b_b2a += iajb

        elif xctype == 'GGA':
            ao_deriv = 1
            for ao, mask, weight, coords \
                    in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
                p0 = p1
                p1+= weight.shape[0]
                wfxc= fxc[...,p0:p1] * weight

                rho_o_a = lib.einsum('xrp,pi->xri', ao, orbo_a)
                rho_v_a = lib.einsum('xrp,pi->xri', ao, orbv_a)
                rho_o_b = lib.einsum('xrp,pi->xri', ao, orbo_b)
                rho_v_b = lib.einsum('xrp,pi->xri', ao, orbv_b)
                rho_ov_b2a = lib.einsum('xri,ra->xria', rho_o_b, rho_v_a[0])
                rho_ov_a2b = lib.einsum('xri,ra->xria', rho_o_a, rho_v_b[0])
                rho_ov_b2a[1:4] += lib.einsum('ri,xra->xria', rho_o_b[0], rho_v_a[1:4])
                rho_ov_a2b[1:4] += lib.einsum('ri,xra->xria', rho_o_a[0], rho_v_b[1:4])

                w_ov = lib.einsum('xyr,xria->yria', wfxc*2.0, rho_ov_b2a)
                iajb = lib.einsum('xria,xrjb->iajb', w_ov, rho_ov_b2a)
                a_b2a += iajb
                iajb = lib.einsum('xria,xrjb->iajb', w_ov, rho_ov_a2b)
                b_b2a += iajb

                w_ov = lib.einsum('xyr,xria->yria', wfxc*2.0, rho_ov_a2b)
                iajb = lib.einsum('xria,xrjb->iajb', w_ov, rho_ov_a2b)
                a_a2b += iajb
                iajb = lib.einsum('xria,xrjb->iajb', w_ov, rho_ov_b2a)
                b_a2b += iajb

        elif xctype == 'HF':
            pass

        elif xctype == 'NLC':
            raise NotImplementedError('NLC')

        elif xctype == 'MGGA':
            ao_deriv = 1
            for ao, mask, weight, coords \
                    in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
                p0 = p1
                p1+= weight.shape[0]
                wfxc = fxc[...,p0:p1] * weight

                rho_oa = lib.einsum('xrp,pi->xri', ao, orbo_a)
                rho_ob = lib.einsum('xrp,pi->xri', ao, orbo_b)
                rho_va = lib.einsum('xrp,pi->xri', ao, orbv_a)
                rho_vb = lib.einsum('xrp,pi->xri', ao, orbv_b)
                rho_ov_b2a = lib.einsum('xri,ra->xria', rho_ob, rho_va[0])
                rho_ov_a2b = lib.einsum('xri,ra->xria', rho_oa, rho_vb[0])
                rho_ov_b2a[1:4] += lib.einsum('ri,xra->xria', rho_ob[0], rho_va[1:4])
                rho_ov_a2b[1:4] += lib.einsum('ri,xra->xria', rho_oa[0], rho_vb[1:4])
                tau_ov_b2a = lib.einsum('xri,xra->ria', rho_ob[1:4], rho_va[1:4]) * .5
                tau_ov_a2b = lib.einsum('xri,xra->ria', rho_oa[1:4], rho_vb[1:4]) * .5
                rho_ov_b2a = np.vstack([rho_ov_b2a, tau_ov_b2a[np.newaxis]])
                rho_ov_a2b = np.vstack([rho_ov_a2b, tau_ov_a2b[np.newaxis]])

                w_ov = lib.einsum('xyr,xria->yria', wfxc*2.0, rho_ov_b2a)
                iajb = lib.einsum('xria,xrjb->iajb', w_ov, rho_ov_b2a)
                a_b2a += iajb
                iajb = lib.einsum('xria,xrjb->iajb', w_ov, rho_ov_a2b)
                b_b2a += iajb

                w_ov = lib.einsum('xyr,xria->yria', wfxc*2.0, rho_ov_a2b)
                iajb = lib.einsum('xria,xrjb->iajb', w_ov, rho_ov_a2b)
                a_a2b += iajb
                iajb = lib.einsum('xria,xrjb->iajb', w_ov, rho_ov_b2a)
                b_a2b += iajb
    else:
        add_hf_(a, b)

    return a, b

@lib.with_doc(rhf.TDA.__doc__)
class TDA_SF(TDBase):
    extype = getattr(__config__, 'tdscf_uhf_sf_SF-TDA_extype', 0)
    collinear_samples = getattr(__config__, 'tdscf_uhf_sf_SF-TDA_collinear_samples', 200)

    _keys = {'extype','collinear_samples'}

    def __init__(self,mf,extype=0,collinear_samples=200):
        TDBase.__init__(self,mf)
        # extype is used to determine which spin flip excitation will be calculated.
        # spin flip up: exytpe=0, spin flip down: exytpe=1.
        self.extype=extype
        # collinear_samples controls the 1d spin sample points in TDDFT/TDA.
        self.collinear_samples = collinear_samples

    def gen_vind(self):
        '''
        Generate function to compute A*x for spin-flip TDDFT case.
        '''
        mf = self._scf
        mo_energy = mf.mo_energy
        mo_coeff = mf.mo_coeff
        assert (mo_coeff[0].dtype == np.double)
        mo_occ = mf.mo_occ

        extype = self.extype
        if extype==0:
            occidxb = mo_occ[1] > 0
            viridxa = mo_occ[0] == 0
            orbo = mo_coeff[1][:, occidxb]
            orbv = mo_coeff[0][:, viridxa]
            ndim = (int(occidxb.sum()), int(viridxa.sum()))
            if np.allclose(mo_coeff[0], mo_coeff[1]):
                fock_a, fock_b = mf.get_fock()
                focko = orbo.conj().T @ fock_b @ orbo
                fockv = orbv.conj().T @ fock_a @ orbv
                hdiag = (fockv.diagonal()[None, :] - focko.diagonal()[:, None]).ravel()
            else:
                e_ia = (mo_energy[0][None, viridxa] - mo_energy[1][occidxb, None])
                hdiag = e_ia.ravel()
        elif extype==1:
            occidxa = mo_occ[0] > 0
            viridxb = mo_occ[1] == 0
            orbo = mo_coeff[0][:, occidxa]
            orbv = mo_coeff[1][:, viridxb]
            ndim = (int(occidxa.sum()), int(viridxb.sum()))
            if np.allclose(mo_coeff[0], mo_coeff[1]):
                fock_a, fock_b = mf.get_fock()
                focko = orbo.conj().T @ fock_a @ orbo
                fockv = orbv.conj().T @ fock_b @ orbv
                hdiag = (fockv.diagonal()[None, :] - focko.diagonal()[:, None]).ravel()
            else:
                e_ia = (mo_energy[1][None, viridxb] - mo_energy[0][occidxa, None])
                hdiag = e_ia.ravel()

        # TODO: change the response function
        vresp = gen_uhf_response_sf(mf, hermi=0, collinear_samples=self.collinear_samples)

        def vind(zs):
            zs = np.asarray(zs).reshape(-1, *ndim)
            dms = lib.einsum('xov,pv,qo->xpq', zs, orbv, orbo.conj())
            v1ao = vresp(dms)
            v1mo = lib.einsum('xpq,qo,pv->xov', v1ao, orbo, orbv.conj())
            if np.allclose(mo_coeff[0], mo_coeff[1]):
                v1mo += lib.einsum('ab,xib->xia', fockv, zs)
                v1mo -= lib.einsum('ji,xja->xia', focko, zs)
            else:
                v1mo += zs * e_ia
            return v1mo.reshape(len(v1mo), -1)

        return vind, hdiag

    def _init_guess(self, mf, nstates):
        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ

        if self.extype==0:
            occidxb = mo_occ[1] > 0
            viridxa = mo_occ[0] == 0
            e_ia = mo_energy[0][None, viridxa] - mo_energy[1][occidxb, None]
        elif self.extype==1:
            occidxa = mo_occ[0] > 0
            viridxb = mo_occ[1] == 0
            e_ia = mo_energy[1][None, viridxb] - mo_energy[0][occidxa, None]

        e_ia = e_ia.ravel()
        nov = e_ia.size
        nstates = min(nstates, nov)
        e_threshold = np.partition(e_ia, nstates-1)[nstates-1]
        e_threshold += self.deg_eia_thresh
        idx = np.where(e_ia <= e_threshold)[0]
        nstates = idx.size
        idx = idx[np.argsort(e_ia[idx])]
        x0 = np.zeros((nstates, nov))
        for i, j in enumerate(idx):
            x0[i, j] = 1   # Koopmans' excitations
        return x0

    def init_guess(self, mf=None, nstates=None, wfnsym=None):
        if mf is None: mf = self._scf
        if nstates is None: nstates = self.nstates
        nstates += 3
        x0 = self._init_guess(mf, nstates)
        return x0

    def kernel(self, x0=None, nstates=None, extype=None):
        '''
        Spin-Flip TDA diagonalization solver
        '''
        cpu0 = (logger.process_clock(), logger.perf_counter())

        self.check_sanity()
        self.dump_flags()

        if extype is None:
            extype = self.extype
        else:
            self.extype = extype

        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates

        log = logger.Logger(self.stdout, self.verbose)

        def all_eigs(w, v, nroots, envs):
            return w, v, np.arange(w.size)

        vind, hdiag = self.gen_vind()
        precond = self.get_precond(hdiag)

        x0sym = None
        if x0 is None:
            x0 = self.init_guess()

        self.converged, self.e, x1 = lr_eigh(
            vind, x0, precond, tol_residual=self.conv_tol, lindep=self.lindep,
            nroots=nstates, x0sym=x0sym, pick=all_eigs, max_cycle=self.max_cycle,
            max_memory=self.max_memory, verbose=log)

        nmo = self._scf.mo_occ[0].size
        nocca, noccb = self._scf.nelec
        nvira = nmo - nocca
        nvirb = nmo - noccb

        if self.extype==0:
            self.xy = [(xi.reshape(noccb, nvira), 0) for xi in x1]
        elif self.extype==1:
            self.xy = [(xi.reshape(nocca, nvirb), 0) for xi in x1]

        if self.chkfile:
            lib.chkfile.save(self.chkfile, 'tddft/e', self.e)
            lib.chkfile.save(self.chkfile, 'tddft/xy', self.xy)

        log.timer('TDA_SF', *cpu0)
        self._finalize()
        return self.e, self.xy

    def get_ab_sf(self, mf=None, collinear_samples=None):
        if mf is None: mf = self._scf
        if collinear_samples is None:
            collinear_samples = self.collinear_samples
        return get_ab_sf(mf, collinear_samples=collinear_samples)

    def nuc_grad_method(self):
        from pyscf.grad import tduks_sf
        return tduks_sf.Gradients(self)

    analyze = analyze
    transition_dipole = transition_dipole
    oscillator_strength = oscillator_strength
    spin_square = spin_square
