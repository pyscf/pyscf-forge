#!/usr/bin/env python
#
# Copyright 2025 The PySCF Developers. All Rights Reserved.
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
# Authors: Jiseong Park <fark4308@snu.ac.kr>
#          Minseok Oh <msjeff2001@snu.ac.kr>
# Edited by: Seunghoon Lee <seunghoonlee@snu.ac.kr>

import numpy
import numpy as np
from pyscf import lib
from pyscf.fci import cistring

def precompute_rdm1s_data(mo_coeff, ncas, nelecas, ncore,
                          dmet_core_list, conf_info_list, ov_list):
    """Precompute CI-independent data for spin-separated 1-RDMs."""
    nao, nmo = mo_coeff.shape
    sl = slice(ncore, ncore + ncas)
    conf_info_list = np.asarray(conf_info_list, dtype=np.int64)
    ov_list = np.asarray(ov_list)

    link_indexa = cistring.gen_linkstr_index(range(ncas), nelecas[0])
    link_indexb = cistring.gen_linkstr_index(range(ncas), nelecas[1])
    na = cistring.num_strings(ncas, nelecas[0])
    nb = cistring.num_strings(ncas, nelecas[1])
    dtype = np.result_type(mo_coeff, dmet_core_list, ov_list, np.float64)

    t1a = np.zeros((ncas, ncas, na, na), dtype=dtype)
    t1b = np.zeros((ncas, ncas, nb, nb), dtype=dtype)
    for str0a in range(na):
        for a1, i1, str1a, signa1 in link_indexa[str0a]:
            t1a[a1, i1, str1a, str0a] += signa1
    for str0b in range(nb):
        for a1, i1, str1b, signb1 in link_indexb[str0b]:
            t1b[a1, i1, str1b, str0b] += signb1

    return {
        "mo_coeff": mo_coeff,
        "ncas": ncas,
        "nelecas": nelecas,
        "ncore": ncore,
        "nao": nao,
        "nmo": nmo,
        "na": na,
        "nb": nb,
        "sl": sl,
        "t1a": t1a,
        "t1b": t1b,
        "t1a_nz": np.array(np.nonzero(t1a)).T.astype(np.int64),
        "t1b_nz": np.array(np.nonzero(t1b)).T.astype(np.int64),
        "dmet_core_list": dmet_core_list,
        "conf_info_list": conf_info_list,
        "ov_list": ov_list,
        "plist": conf_info_list.reshape(-1),
        "dtype": dtype,
    }


def trans_rdm1s_precomputed(ci_bra, ci_ket, data, dmet_core_list=None,
                            mo_coeff=None):
    """Build AO-basis spin-separated transition 1-RDMs."""
    if mo_coeff is None:
        mo_coeff = data["mo_coeff"]
    if dmet_core_list is None:
        dmet_core_list = data["dmet_core_list"]

    ncas = data["ncas"]
    na = data["na"]
    nb = data["nb"]
    sl = data["sl"]
    t1a = data["t1a"]
    t1b = data["t1b"]
    t1a_nz = data["t1a_nz"]
    t1b_nz = data["t1b_nz"]
    conf_info_list = data["conf_info_list"]
    ov_list = data["ov_list"]
    plist = data["plist"]

    dtype = np.result_type(
        ci_bra, ci_ket, data["dtype"], dmet_core_list, mo_coeff)
    ci_bra = np.asarray(ci_bra, dtype=dtype).reshape(na, nb)
    ci_ket = np.asarray(ci_ket, dtype=dtype).reshape(na, nb)

    ngroup = dmet_core_list.shape[0]
    w_diag = np.conjugate(ci_bra.reshape(-1)) * ci_ket.reshape(-1)
    wcore = np.zeros(ngroup, dtype=dtype)
    np.add.at(wcore, plist, w_diag)
    rdm1c_ao = lib.einsum(
        "p,pij->ij", wcore,
        dmet_core_list[np.arange(ngroup), np.arange(ngroup)])

    rdm1a_act = np.zeros((ncas, ncas), dtype=dtype)
    rdm1b_act = np.zeros((ncas, ncas), dtype=dtype)

    for aa, ia, str1a, str0a in t1a_nz:
        p1 = conf_info_list[str1a, :]
        p2 = conf_info_list[str0a, :]
        fac = (
            np.conjugate(ci_bra[str1a, :]) * ci_ket[str0a, :] *
            ov_list[p1, p2])
        rdm1a_act[aa, ia] += t1a[aa, ia, str1a, str0a] * fac.sum()

    for ab, ib, str1b, str0b in t1b_nz:
        p1 = conf_info_list[:, str1b]
        p2 = conf_info_list[:, str0b]
        fac = (
            np.conjugate(ci_bra[:, str1b]) * ci_ket[:, str0b] *
            ov_list[p1, p2])
        rdm1b_act[ab, ib] += t1b[ab, ib, str1b, str0b] * fac.sum()

    mo_cas = mo_coeff[:, sl]
    rdm1a_act_ao = lib.einsum(
        "ia,ab,jb->ij", mo_cas.conj(), rdm1a_act, mo_cas)
    rdm1b_act_ao = lib.einsum(
        "ia,ab,jb->ij", mo_cas.conj(), rdm1b_act, mo_cas)

    return rdm1c_ao + rdm1a_act_ao, rdm1c_ao + rdm1b_act_ao


def make_rdm1s_precomputed(ci, data, dmet_core_list=None, mo_coeff=None):
    """Build AO-basis spin-separated 1-RDMs from precomputed GBCI data."""
    return trans_rdm1s_precomputed(
        ci, ci, data, dmet_core_list=dmet_core_list, mo_coeff=mo_coeff)


def trans_rdm1s(mo_coeff, ci_bra, ci_ket, ncas, nelecas, ncore,
                dmet_core_list, conf_info_list, ov_list):
    data = precompute_rdm1s_data(
        mo_coeff, ncas, nelecas, ncore, dmet_core_list,
        conf_info_list, ov_list)
    return trans_rdm1s_precomputed(ci_bra, ci_ket, data)


def trans_rdm1(mo_coeff, ci_bra, ci_ket, ncas, nelecas, ncore,
               dmet_core_list, conf_info_list, ov_list):
    rdm1a, rdm1b = trans_rdm1s(
        mo_coeff, ci_bra, ci_ket, ncas, nelecas, ncore,
        dmet_core_list, conf_info_list, ov_list)
    return rdm1a + rdm1b


def make_rdm1s(mo_coeff, ci, ncas, nelecas, ncore, dmet_core_list, conf_info_list, ov_list):
    data = precompute_rdm1s_data(
        mo_coeff, ncas, nelecas, ncore, dmet_core_list,
        conf_info_list, ov_list)
    return make_rdm1s_precomputed(ci, data)

def make_rdm1(mo_coeff, ci, ncas, nelecas, ncore, dmet_core_list, conf_info_list, ov_list):
    rdm1a, rdm1b = make_rdm1s(mo_coeff, ci, ncas, nelecas, ncore, dmet_core_list, conf_info_list, ov_list)
    return rdm1a + rdm1b

def make_rdm2s(mo_coeff, ci, ncas, nelecas, ncore,  dmet_core_list, conf_info_list, ov_list):
    mo_cas = mo_coeff[:,ncore:ncore+ncas]
    N = mo_coeff.shape[0]
    rdm2aa = numpy.zeros((N,N,N,N))
    rdm2ab = numpy.zeros((N,N,N,N))
    rdm2ba = numpy.zeros((N,N,N,N))
    rdm2bb = numpy.zeros((N,N,N,N))
    stringsa = cistring.make_strings(range(ncas),nelecas[0])
    stringsb = cistring.make_strings(range(ncas),nelecas[1])
    link_indexa = cistring.gen_linkstr_index(range(ncas),nelecas[0])
    link_indexb = cistring.gen_linkstr_index(range(ncas),nelecas[1])
    na = cistring.num_strings(ncas,nelecas[0])
    nb = cistring.num_strings(ncas,nelecas[1])
    ci = ci.reshape(na,nb)
    t2aa = numpy.zeros((ncas,ncas,ncas,ncas,na,na))
    t2bb = numpy.zeros((ncas,ncas,ncas,ncas,nb,nb))
    t1a = numpy.zeros((ncas,ncas,na,na))
    t1b = numpy.zeros((ncas,ncas,nb,nb))

    rdm2aaac = numpy.zeros((ncas,ncas,ncas,ncas))
    rdm2abac = numpy.zeros((ncas,ncas,ncas,ncas))
    rdm2baac = numpy.zeros((ncas,ncas,ncas,ncas))
    rdm2bbac = numpy.zeros((ncas,ncas,ncas,ncas))
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
            rdm2aa += numpy.conjugate(ci[str0a,str0b])*ci[str0a,str0b] * (
                lib.einsum('pq,rs -> pqrs', dmet_core_list[p2,p2,:,:],dmet_core_list[p2,p2,:,:])
                - lib.einsum('ps,rq -> pqrs',dmet_core_list[p2,p2,:,:],dmet_core_list[p2,p2,:,:]))
            rdm2ab += numpy.conjugate(ci[str0a,str0b])*ci[str0a,str0b] \
                        * lib.einsum('pq,rs -> pqrs',dmet_core_list[p2,p2,:,:],dmet_core_list[p2,p2,:,:])
            rdm2ba += numpy.conjugate(ci[str0a,str0b])*ci[str0a,str0b] \
                        * lib.einsum('pq,rs -> pqrs',dmet_core_list[p2,p2,:,:],dmet_core_list[p2,p2,:,:])
            rdm2bb += numpy.conjugate(ci[str0a,str0b])*ci[str0a,str0b] * (
                lib.einsum('pq,rs -> pqrs', dmet_core_list[p2,p2,:,:],dmet_core_list[p2,p2,:,:])
                - lib.einsum('ps,rq -> pqrs',dmet_core_list[p2,p2,:,:],dmet_core_list[p2,p2,:,:]))
            for str1a, strs1a in enumerate(stringsa):
                p1 = conf_info_list[str1a, str0b]
                rdm2aaac[:,:,:,:] += numpy.conjugate(ci[str1a,str0b])*ci[str0a,str0b]\
                    *t2aa[:,:,:,:,str1a,str0a]*ov_list[p1,p2]
                for k in range(ncas):
                    rdm2aaac[:,k,k,:] -= numpy.conjugate(ci[str1a,str0b])*ci[str0a,str0b]\
                        *t1a[:,:,str1a,str0a]*ov_list[p1,p2]
            for str1b, strs1b in enumerate(stringsb):
                p1 = conf_info_list[str0a, str1b]
                rdm2bbac[:,:,:,:] += numpy.conjugate(ci[str0a,str1b])*ci[str0a,str0b]\
                    *t2bb[:,:,:,:,str1b,str0b]*ov_list[p1,p2]
                for k in range(ncas):
                    rdm2bbac[:,k,k,:] -= numpy.conjugate(ci[str0a,str1b])*ci[str0a,str0b] \
                        * t1b[:,:,str1b,str0b]*ov_list[p1,p2]
            for str1a, strs1a in enumerate(stringsa):
                for str1b, strs1b in enumerate(stringsb):
                    p1 = conf_info_list[str1a, str1b]
                    rdm2abac += numpy.conjugate(ci[str1a,str1b])*ci[str0a,str0b]\
                                *lib.einsum('pq,rs-> pqrs',t1a[:,:,str1a,str0a],t1b[:,:,str1b,str0b])*ov_list[p1,p2]
                    rdm2baac += numpy.conjugate(ci[str1a,str1b])*ci[str0a,str0b]\
                                *lib.einsum('pq,rs-> pqrs',t1b[:,:,str1b,str0b],t1a[:,:,str1a,str0a])*ov_list[p1,p2]

    rdm2aa += lib.einsum('pa,qb,rc,sd,abcd -> pqrs',mo_cas,mo_cas,mo_cas,mo_cas,rdm2aaac)
    rdm2ab += lib.einsum('pa,qb,rc,sd,abcd -> pqrs',mo_cas,mo_cas,mo_cas,mo_cas,rdm2abac)
    rdm2ba += lib.einsum('pa,qb,rc,sd,abcd -> pqrs',mo_cas,mo_cas,mo_cas,mo_cas,rdm2baac)
    rdm2bb += lib.einsum('pa,qb,rc,sd,abcd -> pqrs',mo_cas,mo_cas,mo_cas,mo_cas,rdm2bbac)
    t1aao = lib.einsum('ia,jb,abcd -> ijcd', mo_cas, mo_cas, t1a)
    t1bao = lib.einsum('ia,jb,abcd -> ijcd', mo_cas, mo_cas, t1b)


    for str0a, taba in enumerate(link_indexa):
        for str1a in numpy.unique(link_indexa[str0a][:,2]):
            for str0b, strsb in enumerate(stringsb):
                p1 = conf_info_list[str1a, str0b]
                p2 = conf_info_list[str0a, str0b]
                rdm2aa += numpy.conjugate(ci[str1a,str0b])*ci[str0a,str0b] *(
                    lib.einsum('pq,rs->pqrs',t1aao[:,:,str1a,str0a],dmet_core_list[p1,p2,:,:])
                    + lib.einsum('rs,pq->pqrs',t1aao[:,:,str1a,str0a],dmet_core_list[p1,p2,:,:])
                    - lib.einsum('ps,rq->pqrs',t1aao[:,:,str1a,str0a],dmet_core_list[p1,p2,:,:])
                    - lib.einsum('rq,ps->pqrs',t1aao[:,:,str1a,str0a],dmet_core_list[p1,p2,:,:]))\
                        *ov_list[p1,p2]
                rdm2ab += numpy.conjugate(ci[str1a,str0b])*ci[str0a,str0b]\
                    *(lib.einsum('pq,rs->pqrs',t1aao[:,:,str1a,str0a],dmet_core_list[p1,p2,:,:]))*ov_list[p1,p2]
                rdm2ba += numpy.conjugate(ci[str1a,str0b])*ci[str0a,str0b]\
                    *(lib.einsum('rs,pq->pqrs',t1aao[:,:,str1a,str0a],dmet_core_list[p1,p2,:,:]))*ov_list[p1,p2]

    for str0b, tabb in enumerate(link_indexb):
        for str1b in numpy.unique(link_indexb[str0b][:,2]):
            for str0a, strsa, in enumerate(stringsa):
                p1 = conf_info_list[str0a, str1b]
                p2 = conf_info_list[str0a, str0b]
                rdm2bb += numpy.conjugate(ci[str0a,str1b])*ci[str0a,str0b] * (
                    lib.einsum('pq,rs->pqrs',t1bao[:,:,str1b,str0b],dmet_core_list[p1,p2,:,:])
                    + lib.einsum('rs,pq->pqrs',t1bao[:,:,str1b,str0b],dmet_core_list[p1,p2,:,:])
                    - lib.einsum('ps,rq->pqrs',t1bao[:,:,str1b,str0b],dmet_core_list[p1,p2,:,:])
                    - lib.einsum('rq,ps->pqrs',t1bao[:,:,str1b,str0b],dmet_core_list[p1,p2,:,:]))\
                        *ov_list[p1,p2]
                rdm2ab += numpy.conjugate(ci[str0a,str1b])*ci[str0a,str0b]\
                    * (lib.einsum('rs,pq->pqrs',t1bao[:,:,str1b,str0b],dmet_core_list[p1,p2,:,:]))*ov_list[p1,p2]
                rdm2ba += numpy.conjugate(ci[str0a,str1b])*ci[str0a,str0b]\
                    * (lib.einsum('pq,rs->pqrs',t1bao[:,:,str1b,str0b],dmet_core_list[p1,p2,:,:]))*ov_list[p1,p2]

    return rdm2aa, rdm2ab, rdm2ba, rdm2bb

def precompute_rdm2s_mo_data(S, mo_coeff, ncas, nelecas, ncore,
                             dmet_core_list, conf_info_list, ov_list):
    '''
    Precompute all CI-independent quantities needed to build spin-resolved
    2-RDMs in the MO basis defined by mo_coeff.

    Args:
        mol : Mole object
            PySCF molecule object.

        mo_coeff : ndarray of shape (nao, nmo)
            Molecular orbital coefficients defining the final MO basis.

        ncas : int
            Number of active orbitals.

        nelecas : tuple of int
            Number of active alpha and beta electrons, (neleca, nelecb).

        ncore : int
            Number of core orbitals before the active block.

        dmet_core_list : ndarray of shape (nconf, nconf, nao, nao)
            Core transition 1-RDMs in AO basis.

        conf_info_list : ndarray of shape (na, nb)
            Map from (alpha-string index, beta-string index) to configuration index.

        ov_list : ndarray of shape (nconf, nconf)
            Overlap matrix between nonorthogonal configurations.

    Returns:
        data : dict
            Dictionary containing all precomputed arrays.
    '''

    nao, nmo = mo_coeff.shape
    sl = slice(ncore, ncore + ncas)

    stringsa = cistring.make_strings(range(ncas), nelecas[0])
    stringsb = cistring.make_strings(range(ncas), nelecas[1])
    link_indexa = cistring.gen_linkstr_index(range(ncas), nelecas[0])
    link_indexb = cistring.gen_linkstr_index(range(ncas), nelecas[1])
    na = cistring.num_strings(ncas, nelecas[0])
    nb = cistring.num_strings(ncas, nelecas[1])

    dtype = np.result_type(mo_coeff, dmet_core_list, ov_list, np.float64)

    # ------------------------------------------------------------------
    # Fixed operator tables in active-string basis
    # ------------------------------------------------------------------
    t1a = np.zeros((ncas, ncas, na, na), dtype=dtype)
    t1b = np.zeros((ncas, ncas, nb, nb), dtype=dtype)
    t2aa = np.zeros((ncas, ncas, ncas, ncas, na, na), dtype=dtype)
    t2bb = np.zeros((ncas, ncas, ncas, ncas, nb, nb), dtype=dtype)

    for str0a in range(na):
        for a1, i1, str1a, signa1 in link_indexa[str0a]:
            t1a[a1, i1, str1a, str0a] += signa1
            for a2, i2, str2a, signa2 in link_indexa[str1a]:
                t2aa[a2, i2, a1, i1, str2a, str0a] += signa1 * signa2

    for str0b in range(nb):
        for a1, i1, str1b, signb1 in link_indexb[str0b]:
            t1b[a1, i1, str1b, str0b] += signb1
            for a2, i2, str2b, signb2 in link_indexb[str1b]:
                t2bb[a2, i2, a1, i1, str2b, str0b] += signb1 * signb2

    # ------------------------------------------------------------------
    # Embed active 1-body operators into full MO space once
    # ------------------------------------------------------------------
    t1a_mo = np.zeros((nmo, nmo, na, na), dtype=dtype)
    t1b_mo = np.zeros((nmo, nmo, nb, nb), dtype=dtype)
    t1a_mo[sl, sl, :, :] = t1a
    t1b_mo[sl, sl, :, :] = t1b

    # ------------------------------------------------------------------
    # AO -> MO transform for core transition densities
    # D_mo = C^\dagger S D_ao S C
    # ------------------------------------------------------------------
    nconf = ov_list.shape[0]
    dcore_mo = np.empty((nconf, nconf, nmo, nmo), dtype=dtype)

    for p1 in range(nconf):
        for p2 in range(nconf):
            dcore_mo[p1, p2] = lib.einsum(
                'ui,uv,vj->ij',
                mo_coeff.conj(),
                S @ dmet_core_list[p1, p2] @ S,
                mo_coeff
            )

    # ------------------------------------------------------------------
    # Precompute linked-string lists used in mixed terms
    # ------------------------------------------------------------------
    linked_a_list = []
    for str0a in range(na):
        if len(link_indexa[str0a]) == 0:
            linked_a_list.append(np.array([], dtype=int))
        else:
            linked_a_list.append(np.unique(link_indexa[str0a][:, 2]))

    linked_b_list = []
    for str0b in range(nb):
        if len(link_indexb[str0b]) == 0:
            linked_b_list.append(np.array([], dtype=int))
        else:
            linked_b_list.append(np.unique(link_indexb[str0b][:, 2]))

    data = {
        'S': S,
        'mo_coeff': mo_coeff,
        'ncas': ncas,
        'nelecas': nelecas,
        'ncore': ncore,
        'nao': nao,
        'nmo': nmo,
        'na': na,
        'nb': nb,
        'sl': sl,
        'stringsa': stringsa,
        'stringsb': stringsb,
        'link_indexa': link_indexa,
        'link_indexb': link_indexb,
        'linked_a_list': linked_a_list,
        'linked_b_list': linked_b_list,
        't1a': t1a,
        't1b': t1b,
        't2aa': t2aa,
        't2bb': t2bb,
        't1a_mo': t1a_mo,
        't1b_mo': t1b_mo,
        'dcore_mo': dcore_mo,
        'conf_info_list': conf_info_list,
        'ov_list': ov_list,
        'dtype': dtype,
    }
    return data

def make_rdm2s_mo(ci, data):
    ncas = data['ncas']
    na = data['na']
    nb = data['nb']
    nmo = data['nmo']
    sl = data['sl']

    t1a = data['t1a']
    t1b = data['t1b']
    t2aa = data['t2aa']
    t2bb = data['t2bb']
    t1a_mo = data['t1a_mo']
    t1b_mo = data['t1b_mo']
    dcore_mo = data['dcore_mo']
    conf_info_list = data['conf_info_list']
    ov_list = data['ov_list']
    dtype = np.result_type(ci, data['dtype'])

    ci = np.asarray(ci, dtype=dtype).reshape(na, nb)
    rdm2aa = np.zeros((nmo, nmo, nmo, nmo), dtype=dtype)
    rdm2ab = np.zeros((nmo, nmo, nmo, nmo), dtype=dtype)
    rdm2ba = np.zeros((nmo, nmo, nmo, nmo), dtype=dtype)
    rdm2bb = np.zeros((nmo, nmo, nmo, nmo), dtype=dtype)

    # ------------------------------------------------------------
    # core-core part (batched)
    # ------------------------------------------------------------
    w_diag = np.abs(ci.reshape(-1))**2

    # conf_info_list maps (str0a,str0b) -> p
    # build Ddiag in the same order as ci.reshape(-1)
    plist = conf_info_list.reshape(-1)
    Ddiag = dcore_mo[plist, plist]   # shape (nconf_used, nmo, nmo)

    core_dir = lib.einsum('p,pij,pkl->ijkl', w_diag, Ddiag, Ddiag)
    core_exc = lib.einsum('p,pil,pkj->ijkl', w_diag, Ddiag, Ddiag)
    rdm2ab += core_dir
    rdm2ba += core_dir
    rdm2aa += core_dir - core_exc
    rdm2bb += core_dir - core_exc

    # ------------------------------------------------------------
    # build weights
    # ------------------------------------------------------------
    Waa = np.zeros((na, na), dtype=dtype)
    Wbb = np.zeros((nb, nb), dtype=dtype)
    Mab = np.zeros((na, na, nb, nb), dtype=dtype)
    Deff_a = np.zeros((na, na, nmo, nmo), dtype=dtype)
    Deff_b = np.zeros((nb, nb, nmo, nmo), dtype=dtype)

    for str1a in range(na):
        for str0a in range(na):
            s = 0.0
            Dacc = np.zeros((nmo, nmo), dtype=dtype)
            for str0b in range(nb):
                p1 = conf_info_list[str1a, str0b]
                p2 = conf_info_list[str0a, str0b]
                fac = np.conjugate(ci[str1a, str0b]) * ci[str0a, str0b] * ov_list[p1, p2]
                s += fac
                Dacc += fac * dcore_mo[p1, p2]
            Waa[str1a, str0a] = s
            Deff_a[str1a, str0a] = Dacc

    for str1b in range(nb):
        for str0b in range(nb):
            s = 0.0
            Dacc = np.zeros((nmo, nmo), dtype=dtype)
            for str0a in range(na):
                p1 = conf_info_list[str0a, str1b]
                p2 = conf_info_list[str0a, str0b]
                fac = np.conjugate(ci[str0a, str1b]) * ci[str0a, str0b] * ov_list[p1, p2]
                s += fac
                Dacc += fac * dcore_mo[p1, p2]
            Wbb[str1b, str0b] = s
            Deff_b[str1b, str0b] = Dacc

    for str1a in range(na):
        for str0a in range(na):
            for str1b in range(nb):
                for str0b in range(nb):
                    p1 = conf_info_list[str1a, str1b]
                    p2 = conf_info_list[str0a, str0b]
                    Mab[str1a, str0a, str1b, str0b] = (
                        np.conjugate(ci[str1a, str1b]) * ci[str0a, str0b] * ov_list[p1, p2]
                    )

    # ------------------------------------------------------------
    # active-active contraction
    # ------------------------------------------------------------
    rdm2aa_ac = lib.einsum('xy,abcdxy->abcd', Waa, t2aa)
    rdm2bb_ac = lib.einsum('xy,abcdxy->abcd', Wbb, t2bb)

    Gaa1 = lib.einsum('xy,abxy->ab', Waa, t1a)
    Gbb1 = lib.einsum('xy,abxy->ab', Wbb, t1b)

    for k in range(ncas):
        rdm2aa_ac[:, k, k, :] -= Gaa1
        rdm2bb_ac[:, k, k, :] -= Gbb1

    rdm2ab_ac = lib.einsum('xuyv,abxu,cdyv->abcd', Mab, t1a, t1b)
    rdm2ba_ac = lib.einsum('xuyv,cdxu,abyv->abcd', Mab, t1b, t1a)

    rdm2aa[sl, sl, sl, sl] += rdm2aa_ac
    rdm2ab[sl, sl, sl, sl] += rdm2ab_ac
    rdm2ba[sl, sl, sl, sl] += rdm2ba_ac
    rdm2bb[sl, sl, sl, sl] += rdm2bb_ac

    # ------------------------------------------------------------
    # mixed terms
    # ------------------------------------------------------------
    Ta = t1a_mo.reshape(nmo, nmo, na*na)
    Da = Deff_a.reshape(na*na, nmo, nmo).transpose(1, 2, 0)

    mix_a_dir = lib.einsum('ijx,klx->ijkl', Ta, Da)
    mix_a_exc = lib.einsum('ilx,kjx->ijkl', Ta, Da)

    rdm2ab += mix_a_dir
    rdm2ba += lib.einsum('klx,ijx->ijkl', Ta, Da)
    rdm2aa += mix_a_dir + lib.einsum('klx,ijx->ijkl', Ta, Da) - mix_a_exc - lib.einsum('kjx,ilx->ijkl', Ta, Da)

    Tb = t1b_mo.reshape(nmo, nmo, nb*nb)
    Db = Deff_b.reshape(nb*nb, nmo, nmo).transpose(1, 2, 0)

    mix_b_dir = lib.einsum('ijx,klx->ijkl', Tb, Db)
    mix_b_exc = lib.einsum('ilx,kjx->ijkl', Tb, Db)

    rdm2ba += mix_b_dir
    rdm2ab += lib.einsum('klx,ijx->ijkl', Tb, Db)
    rdm2bb += mix_b_dir + lib.einsum('klx,ijx->ijkl', Tb, Db) - mix_b_exc - lib.einsum('kjx,ilx->ijkl', Tb, Db)

    return rdm2aa, rdm2ab, rdm2ba, rdm2bb

def make_rdm2s_mo_slow(S, mo_coeff, ci, ncas, nelecas, ncore,
                  dmet_core_list, conf_info_list, ov_list):
    """
    Build spin-resolved 2-RDM directly in the MO basis defined by mo_coeff.
    AO basis is non-orthogonal, so AO->MO transforms use S.

    Returns
    -------
    rdm2aa, rdm2ab, rdm2ba, rdm2bb : (nmo,nmo,nmo,nmo)
    """

    nao, nmo = mo_coeff.shape
    # final MO 2-RDMs
    dtype = numpy.result_type(ci, dmet_core_list, mo_coeff)
    rdm2aa = numpy.zeros((nmo, nmo, nmo, nmo), dtype=dtype)
    rdm2ab = numpy.zeros((nmo, nmo, nmo, nmo), dtype=dtype)
    rdm2ba = numpy.zeros((nmo, nmo, nmo, nmo), dtype=dtype)
    rdm2bb = numpy.zeros((nmo, nmo, nmo, nmo), dtype=dtype)

    # CI string info
    link_indexa = cistring.gen_linkstr_index(range(ncas), nelecas[0])
    link_indexb = cistring.gen_linkstr_index(range(ncas), nelecas[1])
    na = cistring.num_strings(ncas, nelecas[0])
    nb = cistring.num_strings(ncas, nelecas[1])
    ci = ci.reshape(na, nb)

    # operator tables in active string basis
    t2aa = numpy.zeros((ncas, ncas, ncas, ncas, na, na), dtype=ci.dtype)
    t2bb = numpy.zeros((ncas, ncas, ncas, ncas, nb, nb), dtype=ci.dtype)
    t1a  = numpy.zeros((ncas, ncas, na, na), dtype=ci.dtype)
    t1b  = numpy.zeros((ncas, ncas, nb, nb), dtype=ci.dtype)

    for str0a in range(na):
        for a1, i1, str1a, signa1 in link_indexa[str0a]:
            t1a[a1, i1, str1a, str0a] += signa1
            for a2, i2, str2a, signa2 in link_indexa[str1a]:
                t2aa[a2, i2, a1, i1, str2a, str0a] += signa1 * signa2

    for str0b in range(nb):
        for a1, i1, str1b, signb1 in link_indexb[str0b]:
            t1b[a1, i1, str1b, str0b] += signb1
            for a2, i2, str2b, signb2 in link_indexb[str1b]:
                t2bb[a2, i2, a1, i1, str2b, str0b] += signb1 * signb2

    # active-space 2-RDM blocks in CAS basis
    rdm2aa_ac = numpy.zeros((ncas, ncas, ncas, ncas), dtype=ci.dtype)
    rdm2ab_ac = numpy.zeros((ncas, ncas, ncas, ncas), dtype=ci.dtype)
    rdm2ba_ac = numpy.zeros((ncas, ncas, ncas, ncas), dtype=ci.dtype)
    rdm2bb_ac = numpy.zeros((ncas, ncas, ncas, ncas), dtype=ci.dtype)

    # AO -> MO transform for core transition density
    # D_mo = C^\dagger S D_ao S C
    nconf = ov_list.shape[0]
    dcore_mo = numpy.empty((nconf, nconf, nmo, nmo), dtype=dtype)

    for p1 in range(nconf):
        for p2 in range(nconf):
            dcore_mo[p1, p2] = lib.einsum(
                'ui,uv,vj->ij',
                mo_coeff.conj(),
                S @ dmet_core_list[p1, p2] @ S,
                mo_coeff
            )
    # 3 block (C-C, C-A, A-A)
    for str0a in range(na):
        for str0b in range(nb):
            p2 = conf_info_list[str0a, str0b]
            c00 = numpy.conjugate(ci[str0a, str0b]) * ci[str0a, str0b]
            Dc22 = dcore_mo[p2, p2]

            # core-core
            rdm2aa += c00 * (
                lib.einsum('ij,kl->ijkl', Dc22, Dc22)
                - lib.einsum('il,kj->ijkl', Dc22, Dc22)
            )
            rdm2ab += c00 * lib.einsum('ij,kl->ijkl', Dc22, Dc22)
            rdm2ba += c00 * lib.einsum('ij,kl->ijkl', Dc22, Dc22)
            rdm2bb += c00 * (
                lib.einsum('ij,kl->ijkl', Dc22, Dc22)
                - lib.einsum('il,kj->ijkl', Dc22, Dc22)
            )

            # A-A alpha
            for str1a in range(na):
                p1 = conf_info_list[str1a, str0b]
                fac = numpy.conjugate(ci[str1a, str0b]) * ci[str0a, str0b] * ov_list[p1, p2]
                rdm2aa_ac += fac * t2aa[:, :, :, :, str1a, str0a]
                for k in range(ncas):
                    rdm2aa_ac[:, k, k, :] -= fac * t1a[:, :, str1a, str0a]

            # A-A beta
            for str1b in range(nb):
                p1 = conf_info_list[str0a, str1b]
                fac = numpy.conjugate(ci[str0a, str1b]) * ci[str0a, str0b] * ov_list[p1, p2]
                rdm2bb_ac += fac * t2bb[:, :, :, :, str1b, str0b]
                for k in range(ncas):
                    rdm2bb_ac[:, k, k, :] -= fac * t1b[:, :, str1b, str0b]

            # active alpha-beta / beta-alpha
            for str1a in range(na):
                for str1b in range(nb):
                    p1 = conf_info_list[str1a, str1b]
                    fac = numpy.conjugate(ci[str1a, str1b]) * ci[str0a, str0b] * ov_list[p1, p2]
                    rdm2ab_ac += fac * lib.einsum(
                        'ab,cd->abcd',
                        t1a[:, :, str1a, str0a],
                        t1b[:, :, str1b, str0b]
                    )
                    rdm2ba_ac += fac * lib.einsum(
                        'ab,cd->abcd',
                        t1b[:, :, str1b, str0b],
                        t1a[:, :, str1a, str0a]
                    )

    # ----------------------------------------------------------
    # active-active: directly embed CAS block into full MO tensor
    # active orbitals are mo indices ncore : ncore+ncas
    # ----------------------------------------------------------
    sl = slice(ncore, ncore+ncas)
    rdm2aa[sl, sl, sl, sl] += rdm2aa_ac
    rdm2ab[sl, sl, sl, sl] += rdm2ab_ac
    rdm2ba[sl, sl, sl, sl] += rdm2ba_ac
    rdm2bb[sl, sl, sl, sl] += rdm2bb_ac

    # ----------------------------------------------------------
    # active-core mixed terms in MO basis
    # t1a/t1b live only on active-active block of full MO 1-body space
    # ----------------------------------------------------------
    t1a_mo = numpy.zeros((nmo, nmo, na, na), dtype=ci.dtype)
    t1b_mo = numpy.zeros((nmo, nmo, nb, nb), dtype=ci.dtype)
    t1a_mo[sl, sl, :, :] = t1a
    t1b_mo[sl, sl, :, :] = t1b

    # alpha-active with core
    for str0a in range(na):
        linked_a = numpy.unique(link_indexa[str0a][:, 2])
        for str1a in linked_a:
            for str0b in range(nb):
                p1 = conf_info_list[str1a, str0b]
                p2 = conf_info_list[str0a, str0b]
                fac = numpy.conjugate(ci[str1a, str0b]) * ci[str0a, str0b] * ov_list[p1, p2]
                T = t1a_mo[:, :, str1a, str0a]
                Dc = dcore_mo[p1, p2]

                rdm2aa += fac * (
                    lib.einsum('ij,kl->ijkl', T, Dc)
                    + lib.einsum('kl,ij->ijkl', T, Dc)
                    - lib.einsum('il,kj->ijkl', T, Dc)
                    - lib.einsum('kj,il->ijkl', T, Dc)
                )
                rdm2ab += fac * lib.einsum('ij,kl->ijkl', T, Dc)
                rdm2ba += fac * lib.einsum('kl,ij->ijkl', T, Dc)

    # beta-active with core
    for str0b in range(nb):
        linked_b = numpy.unique(link_indexb[str0b][:, 2])
        for str1b in linked_b:
            for str0a in range(na):
                p1 = conf_info_list[str0a, str1b]
                p2 = conf_info_list[str0a, str0b]
                fac = numpy.conjugate(ci[str0a, str1b]) * ci[str0a, str0b] * ov_list[p1, p2]
                T = t1b_mo[:, :, str1b, str0b]
                Dc = dcore_mo[p1, p2]

                rdm2bb += fac * (
                    lib.einsum('ij,kl->ijkl', T, Dc)
                    + lib.einsum('kl,ij->ijkl', T, Dc)
                    - lib.einsum('il,kj->ijkl', T, Dc)
                    - lib.einsum('kj,il->ijkl', T, Dc)
                )
                rdm2ab += fac * lib.einsum('kl,ij->ijkl', T, Dc)
                rdm2ba += fac * lib.einsum('ij,kl->ijkl', T, Dc)

    return rdm2aa, rdm2ab, rdm2ba, rdm2bb

def make_rdm2(mo_coeff, ci, ncas, nelecas, ncore, dmet_core_list, conf_info_list, ov_list):
    rdm2aa, rdm2ab, rdm2ba, rdm2bb = \
        make_rdm2s(mo_coeff, ci, ncas, nelecas, ncore,dmet_core_list, conf_info_list, ov_list)
    return rdm2aa + rdm2ab + rdm2ba + rdm2bb

def get_core_density(mo_coeff, ci, ncas, nelecas, ncore, dmet_core_list, conf_info_list):
    N = mo_coeff.shape[0]
    stringsa = cistring.make_strings(range(ncas),nelecas[0])
    stringsb = cistring.make_strings(range(ncas),nelecas[1])
    na = cistring.num_strings(ncas,nelecas[0])
    nb = cistring.num_strings(ncas,nelecas[1])
    rdm1c = numpy.zeros((N,N))
    ci = ci.reshape(na,nb)
    for str0a, strsa in enumerate(stringsa):
        for str0b, strsb in enumerate(stringsb):
            p = conf_info_list[str0a, str0b]
            coeff = ci[str0a, str0b]
            coeff_T = numpy.conjugate(coeff)
            unit = coeff_T * coeff * dmet_core_list[p,p]
            rdm1c += unit
    return rdm1c
