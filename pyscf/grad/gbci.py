#!/usr/bin/env python
#
# Copyright 2026 The PySCF Developers. All Rights Reserved.
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
#
# Author: Jiseong Park <fark4308@snu.ac.kr>
# Edited by: Seunghoon Lee <seunghoonlee@snu.ac.kr>

"""
Analytic gradients for GBCI

This module provides nuclear gradients for GBCI calculations.

"""

import numpy
from functools import reduce
from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf import scf
from pyscf.fci import cistring
from pyscf.scf import cphf
from pyscf.scf import hf
from pyscf.grad import rohf as rohf_grad
from pyscf.grad import rhf as rhf_grad
from pyscf.grad.mp2 import _shell_prange
from pyscf.fci import fci_slow
from pyscf.gbci.direct_gbci import str2occ
from pyscf.gbci.gbci import group_info_list, optimize_mo, GBCI


def mo_to_um(ncas, ncore, ref_mo_coeff, mo_list, s1e):
    nbas = ref_mo_coeff.shape[0]
    core_index = numpy.arange(0, ncore)
    vir_index = numpy.arange(ncore+ncas, nbas)
    bath_index = numpy.concatenate((core_index, vir_index))
    ref_bath = ref_mo_coeff[:,bath_index]
    num_group = mo_list.shape[0]
    um_list = numpy.zeros((num_group, nbas - ncas, nbas - ncas))
    for i in range(0,num_group):
        bath_mo = mo_list[i][:,bath_index]
        um = ref_bath.conj().T @ s1e @ bath_mo
        um_list[i] = um
    return um_list

def make_svd_list(um_list, ncore):
    num_group = um_list.shape[0]
    nbath = um_list.shape[1]
    W_list = numpy.zeros((num_group, num_group, ncore, ncore))
    M_list = numpy.zeros((num_group, num_group, nbath, nbath))
    for p1 in range(num_group):
        for p2 in range(num_group):
            M_list[p1,p2] =um_list[p1].T @ um_list[p2]
            U, s, Vt = numpy.linalg.svd(M_list[p1,p2][:ncore,:ncore])
            W_list[p1,p2] = U @ numpy.diag(1/s) @ Vt
    return W_list, M_list

def make_1rdm_list(ci, ncas, nelecas, conf_info_list, ov_list):
    stringsa = cistring.make_strings(range(ncas), nelecas[0])
    stringsb = cistring.make_strings(range(ncas), nelecas[1])
    link_indexa = cistring.gen_linkstr_index(range(ncas), nelecas[0])
    link_indexb = cistring.gen_linkstr_index(range(ncas), nelecas[1])
    na = len(stringsa)
    nb = len(stringsb)
    ci = ci.reshape(na,nb)
    num_group = ov_list.shape[0]
    ordm_list = numpy.zeros((num_group, num_group, ncas, ncas))
    for str0a , taba in enumerate(link_indexa):
        for aa, ia, str1a, signa in link_indexa[str0a]:
            for str0b, strsb in enumerate(stringsb):
                p1 = conf_info_list[str1a, str0b]
                p2 = conf_info_list[str0a, str0b]
                ordm_list[p1,p2,aa,ia] += signa * numpy.conjugate(ci[str1a,str0b]) * ci[str0a,str0b] * ov_list[p1,p2]
    for str0b, tabb in enumerate(link_indexb):
        for ab, ib, str1b, signb in link_indexb[str0b]:
            for str0a, strsa in enumerate(stringsa):
                p1 = conf_info_list[str0a, str1b]
                p2 = conf_info_list[str0a, str0b]
                ordm_list[p1,p2,ab,ib] += signb * numpy.conjugate(ci[str0a,str1b]) * ci[str0a,str0b] * ov_list[p1,p2]
    return ordm_list

def make_2rdm_list(ci, ncas, nelecas, conf_info_list, ov_list):
    stringsa = cistring.make_strings(range(ncas), nelecas[0])
    stringsb = cistring.make_strings(range(ncas), nelecas[1])
    link_indexa = cistring.gen_linkstr_index(range(ncas), nelecas[0])
    link_indexb = cistring.gen_linkstr_index(range(ncas), nelecas[1])
    na = len(stringsa)
    nb = len(stringsb)
    ci = ci.reshape(na,nb)
    num_group = ov_list.shape[0]
    trdm_list = numpy.zeros((num_group, num_group, ncas, ncas,ncas,ncas))
    t2aa = numpy.zeros((ncas,ncas,ncas,ncas,na,na))
    t2bb = numpy.zeros((ncas,ncas,ncas,ncas,nb,nb))
    t1a = numpy.zeros((ncas,ncas,na,na))
    t1b = numpy.zeros((ncas,ncas,nb,nb))

    for str0a , taba in enumerate(link_indexa):
        for a1, i1, str1a, signa1 in link_indexa[str0a]:
            t1a[a1,i1,str1a,str0a] += signa1
            for a2 , i2, str2a, signa2 in link_indexa[str1a]:
                t2aa[a2, i2, a1, i1, str2a, str0a] += signa1 * signa2
    for str0b , tabb in enumerate(link_indexb):
        for a1, i1, str1b, signb1 in link_indexb[str0b]:
            t1b[a1,i1,str1b,str0b] += signb1
            for a2 , i2, str2b, signb2 in link_indexb[str1b]:
                t2bb[a2, i2, a1, i1, str2b, str0b] += signb1 * signb2
    for str0a, strs0a in enumerate(stringsa):
        for str0b, strs0b in enumerate(stringsb):
            p2 = conf_info_list[str0a, str0b]
            for str1a, strs1a in enumerate(stringsa):
                p1 = conf_info_list[str1a, str0b]
                trdm_list[p1,p2,:,:,:,:] += numpy.conjugate(ci[str1a,str0b])*ci[str0a,str0b]\
                    *t2aa[:,:,:,:,str1a,str0a]*ov_list[p1,p2]
                for k in range(ncas):
                    trdm_list[p1,p2,:,k,k,:] -= numpy.conjugate(ci[str1a,str0b])*ci[str0a,str0b]\
                        *t1a[:,:,str1a,str0a]*ov_list[p1,p2]
            for str1b, strs1b in enumerate(stringsb):
                p1 = conf_info_list[str0a, str1b]
                trdm_list[p1,p2,:,:,:,:] += numpy.conjugate(ci[str0a,str1b])*ci[str0a,str0b]\
                    *t2bb[:,:,:,:,str1b,str0b]*ov_list[p1,p2]
                for k in range(ncas):
                    trdm_list[p1,p2,:,k,k,:] -= numpy.conjugate(ci[str0a,str1b])*ci[str0a,str0b] \
                        * t1b[:,:,str1b,str0b]*ov_list[p1,p2]
            for str1a, strs1a in enumerate(stringsa):
                for str1b, strs1b in enumerate(stringsb):
                    p1 = conf_info_list[str1a, str1b]
                    trdm_list[p1,p2,:,:,:,:] += numpy.conjugate(ci[str1a,str1b])*ci[str0a,str0b]\
                                *lib.einsum('pq,rs->pqrs',t1a[:,:,str1a,str0a],t1b[:,:,str1b,str0b])*ov_list[p1,p2]
                    trdm_list[p1,p2,:,:,:,:] += numpy.conjugate(ci[str1a,str1b])*ci[str0a,str0b]\
                                *lib.einsum('pq,rs->pqrs', t1b[:,:,str1b,str0b],t1a[:,:,str1a,str0a])*ov_list[p1,p2]
    return trdm_list

def make_contracted_H_list(ci, ncas, nelecas, ncore, conf_info_list, h1eff, eri, ecore_list,ov_list):
    stringsa = cistring.make_strings(range(ncas), nelecas[0])
    stringsb = cistring.make_strings(range(ncas), nelecas[1])
    link_indexa = cistring.gen_linkstr_index(range(ncas), nelecas[0])
    link_indexb = cistring.gen_linkstr_index(range(ncas), nelecas[1])
    na = len(stringsa)
    nb = len(stringsb)
    ci = ci.reshape(na,nb)
    num_group = h1eff.shape[0]
    H_list = numpy.zeros((num_group, num_group))

    t2aa = numpy.zeros((ncas,ncas,ncas,ncas,na,na))
    t2bb = numpy.zeros((ncas,ncas,ncas,ncas,nb,nb))
    t1a = numpy.zeros((ncas,ncas,na,na))
    t1b = numpy.zeros((ncas,ncas,nb,nb))
    for str0a , taba in enumerate(link_indexa):
        for a1, i1, str1a, signa1 in link_indexa[str0a]:
            t1a[a1,i1,str1a,str0a] += signa1
            for a2 , i2, str2a, signa2 in link_indexa[str1a]:
                t2aa[a2, i2, a1, i1, str2a, str0a] += signa1 * signa2
    for str0b , tabb in enumerate(link_indexb):
        for a1, i1, str1b, signb1 in link_indexb[str0b]:
            t1b[a1,i1,str1b,str0b] += signb1
            for a2 , i2, str2b, signb2 in link_indexb[str1b]:
                t2bb[a2, i2, a1, i1, str2b, str0b] += signb1 * signb2
    t1a_nonzero = numpy.array(numpy.nonzero(t1a)).T
    t1b_nonzero = numpy.array(numpy.nonzero(t1b)).T
    t2aa_nonzero = numpy.array(numpy.nonzero(t2aa)).T
    t2bb_nonzero = numpy.array(numpy.nonzero(t2bb)).T

    for aa, ia, str1a, str0a in t1a_nonzero:
        for str0b, stringb in enumerate(stringsb):
            p1 = conf_info_list[str1a, str0b]
            p2 = conf_info_list[str0a, str0b]
            H_list[p1,p2] += h1eff[p1,p2, ncore + aa, ia] * t1a[aa,ia,str1a,str0a] * \
                ci[str1a,str0b] * ci[str0a,str0b] * ov_list[p1,p2]


    for ab, ib, str1b, str0b in t1b_nonzero:
        for str0a, stringa in enumerate(stringsa):
            p1 = conf_info_list[str0a, str1b]
            p2 = conf_info_list[str0a, str0b]
            H_list[p1,p2] += h1eff[p1,p2,ncore+ab, ib] * t1b[ab,ib,str1b,str0b] * \
                ci[str0a, str1b] * ci[str0a, str0b] * ov_list[p1,p2]

    h2 = fci_slow.absorb_h1e(h1eff[0,0][ncore:ncore+ncas,:]*0, eri, ncas, nelecas)
    for aa, ia, str1a, str0a in t1a_nonzero:
        for ab, ib, str1b, str0b in t1b_nonzero:
            p1 = conf_info_list[str1a, str1b]
            p2 = conf_info_list[str0a, str0b]
            H_list[p1,p2] += h2[aa,ia,ab,ib] * t1a[aa,ia,str1a,str0a] * \
                t1b[ab,ib,str1b,str0b] *ci[str1a, str1b] *ci[str0a, str0b] * ov_list[p1,p2]

    for a1, i1, a2,i2, str1a, str0a in t2aa_nonzero:
        for str0b, stringb in enumerate(stringsb):
            p1 = conf_info_list[str1a, str0b]
            p2 = conf_info_list[str0a, str0b]
            H_list[p1,p2] += h2[a1,i1,a2,i2] *t2aa[a1,i1,a2,i2,str1a,str0a] * \
                ci[str1a, str0b] * ci[str0a, str0b] * ov_list[p1,p2] *.5

    for a1, i1, a2,i2, str1b, str0b in t2bb_nonzero:
        for str0a, stringa in enumerate(stringsa):
            p1 = conf_info_list[str0a, str1b]
            p2 = conf_info_list[str0a, str0b]
            H_list[p1,p2] += h2[a1,i1,a2,i2] * t2bb[a1,i1,a2,i2,str1b,str0b] * \
                ci[str0a, str1b] * ci[str0a, str0b] * ov_list[p1,p2] *.5

    for str0a, stringa in enumerate(stringsa):
        for str0b, stringb in enumerate(stringsb):
            p = conf_info_list[str0a, str0b]
            H_list[p,p] += ecore_list[p] * ci[str0a, str0b] * ci[str0a, str0b]

    return H_list

def get_X(gbci, h1eff, ov_list, ordm_list, trdm_list, um_list, H_list, group_prob):
    ncore = gbci.ncore
    ncas = gbci.ncas
    mol = gbci.mol
    num_group = ov_list.shape[0]
    ref_mo = gbci._scf.mo_coeff
    nbas = ref_mo.shape[0]
    bath = list(numpy.arange(0,ncore)) + list(numpy.arange(ncore+ncas, nbas))
    h1 = gbci._scf.get_hcore()
    mo_cas = ref_mo[:,ncore:ncore + ncas]
    aapa = ao2mo.kernel(mol, (mo_cas, mo_cas, ref_mo, mo_cas), compact=False)
    aapa = aapa.reshape(ncas,ncas,nbas,ncas)

    Xa = numpy.zeros((nbas, nbas))
    Xx = numpy.zeros(((num_group, nbas - ncas, nbas - ncas)))
    Xa[:,ncore:ncore+ncas] += lib.einsum('xwij, xwmj -> mi', ordm_list, h1eff)
    Xa[:,ncore:ncore+ncas] += lib.einsum('xwji, wxmj -> mi', ordm_list, h1eff)
    Xa[:,ncore:ncore+ncas] += lib.einsum('xwijkl, klmj -> mi ', trdm_list, aapa) *.5
    Xa[:,ncore:ncore+ncas] += lib.einsum('xwjikl, klmj-> mi ', trdm_list, aapa) *.5
    Xa[:,ncore:ncore+ncas] += lib.einsum('xwkjil, kjml-> mi ', trdm_list, aapa) *.5
    Xa[:,ncore:ncore+ncas] += lib.einsum('xwljki, ljmk-> mi ', trdm_list, aapa) *.5
    aapa = None

    W_list, M_list = make_svd_list(um_list, ncore)

    for p1 in range(num_group):
        p1_mo = ref_mo[:,bath] @ um_list[p1]
        p1_core = p1_mo[:,:ncore]
        dm_p1_core = numpy.dot(p1_core, p1_core.T) * 2
        vj, vk = gbci._scf.get_jk(mol, dm_p1_core)
        vhf_c = vj - vk *.5
        Xa[:, bath] += 4* reduce(numpy.dot, (ref_mo.T, h1 + vhf_c, p1_core)) @ um_list[p1][:,:ncore].T * group_prob[p1]
        Xx[p1, :, :ncore] += 4* reduce(numpy.dot, (p1_mo.T, h1 + vhf_c, p1_core)) * group_prob[p1]

        for p2 in range(num_group):
            p2_mo = ref_mo[:,bath] @ um_list[p2]
            p2_core = p2_mo[:,:ncore]
            dm_cas = reduce(numpy.dot, (mo_cas, ordm_list[p1,p2], mo_cas.T))

            vj, vk = gbci._scf.get_jk(mol, dm_cas.T, hermi = 0)
            vhf_a = vj - vk *.5

            Xa[:,bath] += 2 * reduce(numpy.dot, (ref_mo.T, vhf_a, p2_core)) @ W_list[p1,p2].T @ um_list[p1][:,:ncore].T
            Xa[:,bath] += 2 * numpy.array(reduce(numpy.dot, (p1_core.T, vhf_a, ref_mo))
                                          ).T @ W_list[p1,p2] @ um_list[p2][:,:ncore].T

            vhf_a_mo = reduce(numpy.dot,(p1_mo.T, vhf_a, p2_mo)) * 2
            Xx[p1][:,:ncore] += 2* vhf_a_mo[:,:ncore] @ W_list[p1,p2].T


            if p1 != p2:
                Xx[p1][:,:ncore] -= lib.einsum('kl, ns, ks, rl -> nr', vhf_a_mo[:ncore,:ncore],
                                               M_list[p1,p2][:,:ncore], W_list[p1,p2], W_list[p1,p2])
                Xx[p1][:,:ncore] -= lib.einsum('kl, ns, kr, sl -> nr', vhf_a_mo[:ncore,
                                               :ncore].T, M_list[p1,p2][:,:ncore], W_list[p2,p1], W_list[p2,p1])
                Xx[p1][:,:ncore] += H_list[p1,p2] * M_list[p1,p2][:,:ncore] @ W_list[p1,p2].T * 4

    M_list = None
    return Xa, Xx

def _solve_bath_cphf(
    fvind,
    mo_energy,
    mo_occ,
    h1,
    tol=1e-10,
    lindep=1e-20,
    max_cycle=100,
    level_shift=1e-5,
):
    """Solve the bath CPHF equation with an explicit Krylov lindep."""
    e_vir = mo_energy[mo_occ == 0]
    e_occ = mo_energy[mo_occ > 0]
    e_ai = 1.0 / (
        e_vir[:, None] + level_shift - e_occ
    )

    mo1base = -numpy.asarray(h1) * e_ai
    nvir, nocc = e_ai.shape

    def vind_vo(mo1):
        mo1 = mo1.reshape(-1, nvir, nocc)
        v = fvind(mo1).reshape(-1, nvir, nocc)
        if level_shift != 0:
            v -= mo1 * level_shift

        v *= e_ai
        return v.reshape(-1, nvir * nocc)

    mo1 = lib.krylov(
        vind_vo,
        mo1base.reshape(-1, nvir * nocc),
        tol=tol,
        max_cycle=max_cycle,
        lindep=lindep,
    )

    return mo1.reshape(h1.shape)

def _bath_rotation_zvec(mc, ref_mo_coeff, num_group, bath, mo_list, moe_list, um_list, conf_info_list, Xx):
    mol = mc.mol
    ncas = mc.ncas
    ncore = mc.ncore
    nelecas = mc.nelecas
    nao = ref_mo_coeff.shape[0]

    stringsa = cistring.make_strings(range(ncas),nelecas[0])
    stringsb = cistring.make_strings(range(ncas),nelecas[1])

    nb = len(stringsb)
    mo_occ = numpy.zeros(nao - ncas)
    mo_occ[:ncore] = 2
    xzvec = numpy.zeros((num_group, nao - ncas, nao-ncas))
    xzvec_ao = numpy.zeros((num_group, nao, nao))

    #RHF for reference orbital
    zy = numpy.zeros((nao, nao))
    as_occ = numpy.zeros((num_group, ncas))

    for p in range(num_group):
        as_dm_a = numpy.zeros((nao,nao))
        as_dm_b = numpy.zeros((nao,nao))
        target_conf = numpy.where(conf_info_list == p)[0]
        for conf in target_conf:
            stra = conf // nb
            strb = conf % nb
            mo_occa = str2occ(stringsa[stra], ncas)
            mo_occb = str2occ(stringsb[strb], ncas)
            mo_occ = (mo_occa, mo_occb)
            dm_a = hf.make_rdm1(ref_mo_coeff[:,ncore:ncore+ncas], mo_occa)
            dm_b = hf.make_rdm1(ref_mo_coeff[:,ncore:ncore+ncas], mo_occb)
            as_dm_a += dm_a
            as_dm_b += dm_b
        as_dm_a = as_dm_a / len(target_conf)
        as_dm_b = as_dm_b / len(target_conf)
        core_mo_coeff = mo_list[p][:, :ncore]
        dm0_core = (core_mo_coeff ).dot(core_mo_coeff.conj().T)
        dm = numpy.asarray((dm0_core  + as_dm_a , dm0_core + as_dm_b))
        dm = dm[0] + dm[1]
        # Bath rotations only couple doubly occupied core orbitals to empty
        # virtual orbitals.  Their response is the restricted (charge)
        # response even when the reference orbitals came from ROHF.
        fock = mc.get_hcore(mol) + hf.get_veff(mol, dm)

        fock = ref_mo_coeff.T @ fock @ mo_list[p]
        xvo = Xx[p][ncore:,:ncore]
        orbv = mo_list[p][:,ncore+ncas:]
        orbo = mo_list[p][:,:ncore]
        def fvind(x):
            x = x.reshape(xvo.shape)
            dm = reduce(numpy.dot, (orbv, x, orbo.T))
            v = hf.get_veff(mol, dm + dm.T)
            v = reduce(numpy.dot, (orbv.T, v, orbo))
            return v * 2
        mo_occ = numpy.zeros((len(bath)))
        mo_occ[:ncore] = 2
        dm1resp = _solve_bath_cphf(fvind, moe_list[p][bath], mo_occ, xvo, max_cycle=30, level_shift = 1e-5)

        xzvec[p][ncore:,:ncore] = dm1resp

        zvec_ao = reduce(numpy.dot, (mo_list[p][:,bath], xzvec[p], mo_list[p][:,bath].T))
        xzvec_ao[p] += zvec_ao
        vj, vk = hf.get_jk(mol, zvec_ao.T, hermi = 0)
        vhf_z = vj - vk * .5

        zy[:,bath] += fock[:,bath] @ (xzvec[p] + xzvec[p].T) @ um_list[p].T
        zy[:,bath] += 2 * reduce(numpy.dot, (ref_mo_coeff.T, vhf_z + vhf_z.T,
                                 mo_list[p][:,:ncore])) @ um_list[p][:,:ncore].T
        target_conf = numpy.where(conf_info_list == p)[0]
        for conf in target_conf:
            stra = conf // nb
            strb = conf % nb
            mo_occa = str2occ(stringsa[stra], ncas)
            mo_occb = str2occ(stringsb[strb], ncas)
            as_occ[p] += mo_occa + mo_occb
        as_occ[p] = as_occ[p] / len(target_conf)
        zy[:,ncore:ncore+ncas] += reduce(numpy.dot, (ref_mo_coeff.T, vhf_z + vhf_z.T,
                                         ref_mo_coeff[:,ncore:ncore+ncas])) @ numpy.diag(as_occ[p])
    return xzvec, xzvec_ao, zy, as_occ

def grad_elec(
    gbci_grad, ref_mo_coeff, ref_mo_energy, mo_list, moe_list,
    conf_info_list, dmet_core_list, ov_list, ecore_list, ci,
    atmlst=None, verbose=None,
):
    mc = gbci_grad.base
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.new_logger(gbci_grad, verbose)
    mol = gbci_grad.mol
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    nelecas = mc.nelecas
    neleca = nelecas[0] + ncore
    nao, nmo = ref_mo_coeff.shape
    nao_pair = nao * (nao+1) // 2
    bath = list(numpy.arange(0,ncore)) + list(numpy.arange(ncore+ncas, nao))
    num_group = mo_list.shape[0]

    s1e = mc._scf.get_ovlp(mol)
    um_list = mo_to_um(ncas, ncore, ref_mo_coeff, mo_list, s1e)

    mo_cas = ref_mo_coeff[:,ncore:nocc]
    ordm_list = make_1rdm_list(ci, ncas, nelecas, conf_info_list, ov_list)
    trdm_list = make_2rdm_list(ci, ncas, nelecas, conf_info_list, ov_list)
    h1eff = get_h1eff_for_grad(mc, ref_mo_coeff, mo_cas, dmet_core_list)
    eri = mc.get_h2eff(ref_mo_coeff)

    H_list = make_contracted_H_list(ci, ncas, nelecas, ncore, conf_info_list, h1eff, eri, ecore_list, ov_list)
    group_prob = numpy.zeros(num_group)
    conf_info_list = conf_info_list.reshape(-1)
    for i in range(num_group):
        ci = ci.reshape(-1)
        group_where = numpy.where(conf_info_list == i)
        group_prob[i] = (numpy.abs(ci)**2)[group_where].sum()
    Xa, Xx  = get_X(mc, h1eff, ov_list, ordm_list, trdm_list, um_list, H_list, group_prob)

    xzvec, xzvec_ao, zy, as_occ = _bath_rotation_zvec(
        mc, ref_mo_coeff, num_group, bath, mo_list, moe_list, um_list, conf_info_list, Xx)

    orbv = ref_mo_coeff[:,neleca:]
    orbo = ref_mo_coeff[:,:neleca]
    #RHF reference
    ee = ref_mo_energy[:,None] - ref_mo_energy
    Imat = Xa + zy
    azvec = numpy.zeros_like(Imat)
    azvec[:ncore,ncore:neleca] = Imat[:ncore,ncore:neleca] / -ee[:ncore,ncore:neleca]
    azvec[ncore:neleca,:ncore] = Imat[ncore:neleca,:ncore] / -ee[ncore:neleca,:ncore]
    azvec[nocc:,neleca:nocc] = Imat[nocc:,neleca:nocc] / -ee[nocc:,neleca:nocc]
    azvec[neleca:nocc,nocc:] = Imat[neleca:nocc,nocc:] / -ee[neleca:nocc,nocc:]
    active_same_pairs = []
    for space in (
        numpy.arange(ncore, neleca),
        numpy.arange(neleca, nocc),
    ):
        for p_pos, p in enumerate(space):
            for q in space[:p_pos]:
                denominator = ref_mo_energy[p] - ref_mo_energy[q]
                gradient = Imat[p, q] - Imat[q, p]
                if abs(denominator) < 1e-10:
                    if abs(gradient) > 1e-8:
                        raise RuntimeError(
                            'Degenerate active-active response is ambiguous')
                    continue
                weight = -gradient / denominator
                azvec[p, q] = weight
                active_same_pairs.append((p, q))
    zvec_ao = reduce(numpy.dot, (ref_mo_coeff, azvec+azvec.T, ref_mo_coeff.T))
    vhf = mc._scf.get_veff(mol, zvec_ao) * 2
    xvo = reduce(numpy.dot, (orbv.T, vhf, orbo))
    xvo += Imat[neleca:, :neleca] - Imat[:neleca, neleca:].T
    def fvind(x):
        x = x.reshape(xvo.shape)
        dm = reduce(numpy.dot, (orbv, x, orbo.T))
        v = mc._scf.get_veff(mol, dm + dm.T)
        v = reduce(numpy.dot, (orbv.T, v, orbo))
        return v * 2
    mo_occ = numpy.zeros((nao))
    mo_occ[:neleca] = 2
    dm1resp = _solve_bath_cphf(fvind, ref_mo_energy, mo_occ, xvo, level_shift = 1e-5, max_cycle = 30)
    azvec[neleca:, :neleca] = dm1resp

    zeta = numpy.einsum('ij,j->ij', azvec, ref_mo_energy)
    zeta = reduce(numpy.dot, (ref_mo_coeff, zeta, ref_mo_coeff.T))
    zvec_ao = reduce(numpy.dot, (ref_mo_coeff, azvec+azvec.T, ref_mo_coeff.T)) *.5
    p1 = numpy.dot(ref_mo_coeff[:,:neleca], ref_mo_coeff[:,:neleca].T)
    vhf_s1occ = reduce(numpy.dot, (p1, mc._scf.get_veff(mol, zvec_ao), p1))


    Imat[:ncore,ncore:neleca] = 0
    Imat[ncore:neleca,:ncore] = 0
    Imat[nocc:,neleca:nocc] = 0
    Imat[neleca:nocc,nocc:] = 0
    for p, q in active_same_pairs:
        Imat[p, q] = 0
        Imat[q, p] = 0
    Imat[neleca:,:neleca] = Imat[:neleca,neleca:].T
    im1 = reduce(numpy.dot, (ref_mo_coeff, Imat, ref_mo_coeff.T)) * .5

    casdm1 = ref_mo_coeff[:,ncore:ncore+ncas] @ ordm_list.sum(axis = (0,1)) @ ref_mo_coeff[:,ncore:ncore+ncas].T
    casdm2 = trdm_list.sum(axis = (0,1))
    hf_dm1 = mc._scf.make_rdm1(ref_mo_coeff, mo_occ)
    hcore_deriv = gbci_grad.hcore_generator(mol)
    s1 = gbci_grad.get_ovlp(mol)

    diag_idx = numpy.arange(nao)
    diag_idx = diag_idx * (diag_idx+1) // 2 + diag_idx
    casdm2_cc = casdm2 + casdm2.transpose(0,1,3,2)
    dm2buf = ao2mo._ao2mo.nr_e2(casdm2_cc.reshape(ncas**2,ncas**2), mo_cas.T,
                                (0, nao, 0, nao)).reshape(ncas**2,nao,nao)
    dm2buf = lib.pack_tril(dm2buf)
    dm2buf[:,diag_idx] *= .5
    dm2buf = dm2buf.reshape(ncas,ncas,nao_pair)
    casdm2 = casdm2_cc = None

    if atmlst is None:
        atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    de = numpy.zeros((len(atmlst),3))

    mo_cores = numpy.stack([mo_list[p][:,:ncore] for p in range(num_group)], axis = 0)
    dm_cores = 2 * numpy.einsum('xpc, xqc -> xpq', mo_cores, mo_cores)
    dm_acts = numpy.einsum('pa, xa, qa -> xpq', mo_cas, as_occ, mo_cas)
    dms = dm_cores + dm_acts
    xz = numpy.asarray(xzvec_ao)
    ordm_aos = numpy.einsum('pa,xwab,qb->xwpq', mo_cas, ordm_list, mo_cas, optimize=True)

    max_memory = gbci_grad.max_memory - lib.current_memory()[0]
    blksize = int(max_memory*.9e6/8 / ((aoslices[:,3]-aoslices[:,2]).max()*nao_pair))
    blksize = min(nao, max(2, blksize))
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        h1ao = hcore_deriv(ia)
        de[k] += numpy.einsum('xij, ij ->x', h1ao, casdm1)
        de[k] += numpy.einsum('xij, ij ->x', h1ao, zvec_ao)
        de[k] += numpy.einsum('xij, pij ->x', h1ao, xzvec_ao)
        for x in range(num_group):
            dm_core = dm_cores[x]
            de[k] += numpy.einsum('xij, ij -> x', h1ao, dm_core) * group_prob[x]
        q1 = 0
        for b0, b1, nf in _shell_prange(mol, 0, mol.nbas, blksize):
            q0, q1 = q1, q1 + nf
            dm2_ao = lib.einsum('ijw,pi,qj->pqw', dm2buf, mo_cas[p0:p1], mo_cas[q0:q1])
            shls_slice = (shl0,shl1,b0,b1,0,mol.nbas,0,mol.nbas)
            eri1 = mol.intor('int2e_ip1', comp=3, aosym='s2kl',
                             shls_slice=shls_slice).reshape(3,p1-p0,nf,nao_pair)
            de[k] -= numpy.einsum('xijw,ijw->x', eri1, dm2_ao) * 2

            xz_pq_sym = xz[:,p0:p1, q0:q1] + xz[:,q0:q1, p0:p1].transpose(0,2,1)
            xz_p_sym = xz[:,p0:p1, :] + xz[:, :, p0:p1].transpose(0,2,1)
            xz_q_sym = xz[:,q0:q1, :] + xz[:, :, q0:q1].transpose(0,2,1)

            ordm_pq_sym = ordm_aos[:, :, p0:p1, q0:q1]+ ordm_aos[:, :, q0:q1, p0:p1].transpose(0, 1, 3, 2)
            dmet_pq_sym = dmet_core_list[:, :, p0:p1, q0:q1]+ dmet_core_list[:, :, q0:q1, p0:p1].transpose(0, 1, 3, 2)

            for i in range(3):
                eri1tmp = lib.unpack_tril(eri1[i].reshape((p1-p0)*nf,-1))
                eri1tmp = eri1tmp.reshape(p1-p0,nf,nao,nao)
                de[k,i] -= numpy.einsum('ijkl,ij,kl', eri1tmp, hf_dm1[p0:p1,q0:q1], zvec_ao) * 2
                de[k,i] -= numpy.einsum('ijkl,kl,ij', eri1tmp, hf_dm1, zvec_ao[p0:p1,q0:q1]) * 2
                de[k,i] += numpy.einsum('ijkl,il,kj', eri1tmp, hf_dm1[p0:p1], zvec_ao[:,q0:q1])
                de[k,i] += numpy.einsum('ijkl,jk,il', eri1tmp, hf_dm1[q0:q1], zvec_ao[p0:p1])

                de[k,i] -= 2 * numpy.einsum('ijkl, xlk, xij, x ->', eri1tmp, dm_cores,
                                            dm_cores[:,p0:p1, q0:q1], group_prob, optimize = True)
                de[k,i] += numpy.einsum('ijkl, xjk, xil, x ->', eri1tmp, dm_cores[:,q0:q1,:],
                                        dm_cores[:,p0:p1, :], group_prob, optimize = True)

                de[k,i] -= numpy.einsum('ijkl, xij, xkl ->', eri1tmp, xz_pq_sym, dms, optimize = True)
                de[k,i] -= 2 * numpy.einsum('ijkl, xkl, xij ->', eri1tmp, xz, dms[:,p0:p1, q0:q1], optimize = True)
                de[k,i] += 0.5 * numpy.einsum('ijkl, xil, xjk ->', eri1tmp, xz_p_sym, dms[:,q0:q1,:], optimize = True)
                de[k,i] += 0.5 * numpy.einsum('ijkl, xjl, xik ->', eri1tmp, xz_q_sym, dms[:,p0:p1,:], optimize = True)

                de[k,i] -= 2 * numpy.einsum('ijkl, xwij, xwkl ->', eri1tmp, ordm_pq_sym,
                                            dmet_core_list, optimize = True)
                de[k,i] -= 2 * numpy.einsum('ijkl, xwkl, xwij ->', eri1tmp, ordm_aos, dmet_pq_sym, optimize = True)
                de[k,i] += numpy.einsum('ijkl, xwil, xwkj', eri1tmp, ordm_aos[:, :, p0:p1, :],
                                        dmet_core_list[:, :, :,q0:q1], optimize = True)
                de[k,i] += numpy.einsum('ijkl, xwjl, xwki', eri1tmp, ordm_aos[:, :, q0:q1, :],
                                        dmet_core_list[:, :, :,p0:p1], optimize = True)
                de[k,i] += numpy.einsum('ijkl, xwkj, xwil', eri1tmp, ordm_aos[:,:,:,q0:q1],
                                        dmet_core_list[:,:,p0:p1,:], optimize = True)
                de[k,i] += numpy.einsum('ijkl, xwli, xwjk', eri1tmp, ordm_aos[:,:,:,p0:p1],
                                        dmet_core_list[:,:,q0:q1,:], optimize = True)

            eri1 = eri1tmp = None
        de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], im1[p0:p1])
        de[k] -= numpy.einsum('xij,ji->x', s1[:,p0:p1], im1[:,p0:p1])

        de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], zeta[p0:p1])
        de[k] -= numpy.einsum('xij,ji->x', s1[:,p0:p1], zeta[:,p0:p1])

        de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], vhf_s1occ[p0:p1]) * 2
        de[k] -= numpy.einsum('xij,ji->x', s1[:,p0:p1], vhf_s1occ[:,p0:p1]) * 2

    log.timer('GBCI nuclear gradients', *time0)
    return de

# (ngroup, ngroup, nbas ,ncas)
def get_h1eff_for_grad(mc, ref_mo, mo_cas, dmet_core_list):
    hcore = mc.get_hcore()
    nbas = ref_mo.shape[0]
    ncas = mc.ncas
    p = dmet_core_list.shape[0]
    h1e = numpy.zeros((p,p,nbas,ncas))
    ha1e = lib.einsum('ai,ab,bj->ij',ref_mo,hcore, mo_cas)
    for i in range(0,p):
        for j in range(0,p):
            corevhf = mc.get_veff(dm = 2 * dmet_core_list[i,j].T, hermi = 0)
            h1e[i,j] = ha1e + lib.einsum('ai, bj ,ab -> ij', ref_mo, mo_cas , corevhf)
    return h1e

def as_scanner(gbci_grad, state = None):
    if isinstance(gbci_grad, lib.GradScanner):
        return gbci_grad
    logger.info(gbci_grad, 'Create scanner for %s', gbci_grad.__class__)
    name = gbci_grad.__class__.__name__ + GBCI_GradScanner.__name_mixin__
    return lib.set_class(GBCI_GradScanner(gbci_grad, state), (GBCI_GradScanner,
                                                                 gbci_grad.__class__), name)

class GBCI_GradScanner(lib.GradScanner):
    def __init__(self, g, state):
        lib.GradScanner.__init__(self, g)
        if state is not None:
            self.state = state

    def __call__(self, mol_or_geom, state = None, **kwargs):
        if isinstance(mol_or_geom, gto.MoleBase):
            assert mol_or_geom.__class__ == gto.Mole
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace = False)
        self.reset(mol)

        if state is None:
            state = self.state

        gbci_scanner = self.base

        e_tot = gbci_scanner(mol)
        if not isinstance(e_tot, float):
            if state >= gbci_scanner.fcisolver.nroots:
                raise ValueError('State ID greater than the number of GBCI roots')
            e_tot = e_tot[state]
        de = self.kernel(state=state, **kwargs)
        return e_tot, de


class Gradients(rhf_grad.GradientsBase):
    '''Non-relativistic restricted Hartree-Fock gradients'''

    _keys = {'state'}

    def __init__(self, mc):
        self.state = 0
        rhf_grad.GradientsBase.__init__(self,mc)

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self,verbose)
        log.info('\n')
        if not self.base.converged:
            log.warn('Ground state %s not converged', self.base.__class__)
        log.info('******** %s for %s ********',
                 self.__class__, self.base.__class__)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        return self

    def grad_elec(
        self, ref_mo_coeff, ref_mo_energy, mo_list, moe_list,
        conf_info_list, dmet_core_list, ov_list, ecore_list, ci,
        atmlst=None, verbose=None,
    ):
        if isinstance(self.base._scf, scf.rohf.ROHF):
            from pyscf.grad import rohf_gbci
            return rohf_gbci.grad_elec(
                self, ref_mo_coeff, ref_mo_energy, mo_list, moe_list,
                conf_info_list, dmet_core_list, ov_list, ecore_list, ci,
                atmlst=atmlst, verbose=verbose,
            )
        return grad_elec(
            self, ref_mo_coeff, ref_mo_energy, mo_list, moe_list,
            conf_info_list, dmet_core_list, ov_list, ecore_list, ci,
            atmlst=atmlst, verbose=verbose,
        )

    def kernel(self, ci=None, atmlst=None, state=None, verbose=None):
        intermediates = getattr(self.base, '_gbci_intermediates', None)
        if intermediates is None:
            raise RuntimeError(
                'GBCI intermediates are unavailable. Run the GBCI kernel '
                'before requesting gradients')

        ref_mo_coeff = self.base.mo_coeff
        ref_mo_energy = self.base._scf.mo_energy
        mo_list = intermediates['mo_list']
        moe_list = intermediates['mo_energy']
        conf_info_list = intermediates['conf_info_list']
        dmet_core_list = intermediates['dmet_core_list']
        ov_list = intermediates['ov_list']
        ecore_list = intermediates['ecore_list']

        if ci is None:
            ci = self.base.ci
        if ci is None:
            raise RuntimeError(
                'GBCI CI coefficients are unavailable. Run the GBCI kernel '
                'before requesting gradients')

        if state is None:
            state = self.state
        else:
            self.state = state
        nroots = getattr(self.base.fcisolver, 'nroots', 1)
        if nroots > 1:
            if state < 0 or state >= nroots:
                raise ValueError(
                    'State ID greater than the number of GBCI roots')
            ci = ci[state]

        de = self.grad_elec(ref_mo_coeff, ref_mo_energy, mo_list, moe_list,conf_info_list,
                            dmet_core_list, ov_list, ecore_list, ci, atmlst, verbose)
        self.de = de + self.grad_nuc(atmlst=atmlst)
        self._finalize()
        return self.de

        # Initialize hcore_deriv with the underlying SCF object because some
        # extensions (e.g. x2c, QM/MM, solvent) modifies the SCF object only.
    def hcore_generator(self, mol=None):
        mf_grad = self.base._scf.nuc_grad_method()
        return mf_grad.hcore_generator(mol)

        # Calling the underlying SCF nuclear gradients because it may be modified
        # by external modules (e.g. QM/MM, solvent)
    def grad_nuc(self, mol=None, atmlst=None):
        mf_grad = self.base._scf.nuc_grad_method()
        return mf_grad.grad_nuc(mol, atmlst)

    def _finalize(self):
        if self.verbose >= logger.NOTE:
            if self.state is None:
                logger.note(self, '--------- %s gradients ----------',
                            self.base.__class__.__name__)
            else:
                logger.note(self, '--------- %s gradients for state %d ----------',
                            self.base.__class__.__name__, self.state)
            self._write(self.mol, self.de, self.atmlst)
            logger.note(self, '----------------------------------------------')

    as_scanner = as_scanner

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd
    lib.num_threads(1)
    delta = 1e-5
    i = 1.5
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = None
    mol.atom = [['Li', (0,0,0)], ['Cl',(0,0,i - delta)]]
    mol.basis = 'ccpvdz'
    mol.build()
    mol.set_common_orig([0,0,0])
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    mo_coeff = mf.mo_coeff
    e_hf = mf.e_tot

    mygbci = GBCI(mf, 4, (2,2), group_a = {"atom": [0]})
    mygbci.fcisolver.conv_tol = 1e-12
    gbci_grad = Gradients(mygbci)
    mygbci.mo_coeff = mo_coeff
    mo = mo_coeff
    mo_list, moe_list, po_list, group = optimize_mo(mygbci, mo, group_a = {"atom": [0]})
    p = mo_list.shape[0]
    dmet_core_list, ov_list = mygbci.get_svd_matrices(mo_list, group)
    dmet_act_list = mygbci.get_active_dm(mo)
    h1e, ecore_list = mygbci.get_h1cas(dmet_act_list , mo_list , dmet_core_list)
    eri = mygbci.get_h2eff(mo)

    ncas = mygbci.ncas
    nelecas = mygbci.nelecas
    conf_info_list = group_info_list(ncas, nelecas, po_list, group)

    e_tot, fcivec = mygbci.fcisolver.kernel(h1e, eri, ncas, nelecas,
                                            conf_info_list, ov_list, ecore_list,
                                            ci0=None, verbose=mol.verbose)


    mol = gto.Mole()
    mol.verbose = 5
    mol.output = None
    mol.atom = [['Li', (0,0,0)], ['Cl',(0,0,i + delta)]]
    mol.basis = 'ccpvdz'
    mol.build()
    mol.set_common_orig([0,0,0])

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()

    e_hf_new = mf.e_tot
    mo_coeff = mf.mo_coeff
    mygbci = GBCI(mf, 4, (2,2), group_a = {"atom": [0]})
    mygbci.fcisolver.conv_tol = 1e-12
    mygbci.mo_coeff = mo_coeff
    mo = mo_coeff
    mo_list, moe_list, po_list, group = optimize_mo(mygbci, mo,  group_a = {"atom": [0]})
    p = mo_list.shape[0]

    dmet_core_list, ov_list = mygbci.get_svd_matrices(mo_list, group)
    dmet_act_list = mygbci.get_active_dm(mo)
    h1e, ecore_list = mygbci.get_h1cas(dmet_act_list , mo_list , dmet_core_list)
    eri = mygbci.get_h2eff(mo)
    ncas = mygbci.ncas
    nelecas = mygbci.nelecas
    conf_info_list = group_info_list(ncas, nelecas, po_list, group)
    e_new, fcivec = mygbci.fcisolver.kernel(h1e, eri, ncas, nelecas,
                                            conf_info_list, ov_list, ecore_list,
                                            ci0=None, verbose=mol.verbose)

    mol = gto.Mole()
    mol.verbose = 5
    mol.output = None
    mol.atom = [['Li', (0,0,0)], ['Cl',(0,0,i)]]
    mol.basis = 'ccpvdz'
    mol.build()
    mol.set_common_orig([0,0,0])

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()

    mo_coeff = mf.mo_coeff

    mygbci = GBCI(mf, 4, (2,2), group_a = {"atom": [0]})
    mygbci.fcisolver.conv_tol = 1e-12
    gbci_grad = Gradients(mygbci)
    mygbci.kernel(mo_coeff)
    de = gbci_grad.kernel()

    ANG2BOHR = 1.0 / lib.param.BOHR
    nu_gbci = (e_new - e_tot)/(2*delta * ANG2BOHR)

    print("Diff: %.12f " % (nu_gbci - de[1][2]))
    print("Numerical gradient: %.12f" % nu_gbci)
