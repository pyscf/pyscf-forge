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
# Author: Jiseong Park <fark4308@snu.ac.kr>
# Edited by: Seunghoon Lee <seunghoonlee@snu.ac.kr>

'''
Spin Flip Non-Orthogonal Configuration Interaction (SF-NOCI)
and Grouped-Bath Ansatz for SF-NOCI (SF-GNOCI)

References:
[1] Spin-flip non-orthogonal configuration interaction: a variational and
    almost black-box method for describing strongly correlated molecules
    Nicholas J. Mayhall, Paul R. Horn, Eric J. Sundstrom and Martin Head-Gordon
    Phys. Chem. Chem. Phys. 2014, 16, 22694
[2] Efficient grouped-bath ansatz for spin-flip non-orthogonal configuration
    interaction (SF-GNOCI) in transition-metal charge-transfer complexes
    Jiseong Park and Seunghoon Lee
    J. Chem. Theory Comput. 2025
'''

import sys

import numpy
import ctypes
import scipy.linalg
import types
from pyscf import fci
from pyscf import ao2mo
from pyscf import lib
from pyscf import __config__
from pyscf.lib import logger
from pyscf.fci import spin_op
from pyscf.fci import cistring
from pyscf.fci import direct_uhf
from pyscf.fci.direct_spin1 import FCIBase, FCISolver, FCIvector

libsf = lib.load_library("libsfnoci")

PENALTY = getattr(__config__, 'sfnoci_SFNOCI_fix_spin_shift', 0.2)

def make_hdiag(h1e, eri, ncas, nelecas, conf_info_list, ecore_list, opt=None):
    if isinstance(nelecas, (int, numpy.integer)):
        nelecb = nelecas//2
        neleca = nelecas - nelecb
    else:
        neleca, nelecb = nelecas
    occslista = cistring.gen_occslst(range(ncas), neleca)
    occslistb = cistring.gen_occslst(range(ncas), nelecb)
    eri = ao2mo.restore(1, eri, ncas)
    diagj = numpy.einsum('iijj->ij', eri)
    diagk = numpy.einsum('ijji->ij', eri)
    hdiag = []
    for str0a, aocc in enumerate(occslista):
        for str0b, bocc in enumerate(occslistb):
            occ = numpy.zeros(ncas)
            for i in aocc:
                occ[i] += 1
            for i in bocc:
                occ[i] +=1
            p = conf_info_list[str0a, str0b]
            e1 = h1e[p,p,aocc,aocc].sum() + h1e[p,p,bocc,bocc].sum()
            e2 = diagj[aocc][:,aocc].sum() + diagj[aocc][:,bocc].sum() \
               + diagj[bocc][:,aocc].sum() + diagj[bocc][:,bocc].sum() \
               - diagk[aocc][:,aocc].sum() - diagk[bocc][:,bocc].sum()
            hdiag.append(e1 + e2*.5 + ecore_list[p])
    return numpy.array(hdiag)

def absorb_h1e(h1e, eri, ncas, nelecas, fac=1):
    '''Modify 2e Hamiltonian to include effective 1e Hamiltonian contribution

    input : h1e : (nbath, nbath, ncas, ncas)
            eri   : (ncas, ncas, ncas, ncas)

    return : erieff : (ngroup,ngroup,ncas,ncas,ncas,ncas)
    '''
    if not isinstance(nelecas, (int, numpy.number)):
        nelecas = sum(nelecas)
    h2e = ao2mo.restore(1, eri.copy(), ncas)
    p = h1e.shape[0]
    f1e = h1e.copy()
    f1e -= numpy.einsum('jiik->jk', h2e)[numpy.newaxis, numpy.newaxis, :, :] * .5
    f1e = f1e * (1./(nelecas+1e-100))
    erieff = numpy.zeros((p, p, ncas, ncas, ncas, ncas))
    erieff += h2e[numpy.newaxis, numpy.newaxis, :, :, :, :]
    for k in range(ncas):
        erieff[:,:,k,k,:,:] += f1e
        erieff[:,:,:,:,k,k] += f1e
    return erieff * fac

def gen_excitations(ncas, nelecas, na, nb, link_index=None):
    if isinstance(nelecas, (int, numpy.integer)):
        nelecb = nelecas//2
        neleca = nelecas - nelecb
    else:
        neleca, nelecb = nelecas
    if link_index is None:
        link_indexa = cistring.gen_linkstr_index(range(ncas), neleca)
        link_indexb = cistring.gen_linkstr_index(range(ncas), nelecb)
    else:
        link_indexa, link_indexb = link_index
    t2aa = numpy.zeros((ncas,ncas,ncas,ncas,na,na), dtype=numpy.int32)
    t2bb = numpy.zeros((ncas,ncas,ncas,ncas,nb,nb), dtype=numpy.int32)
    t1a = numpy.zeros((ncas,ncas,na,na), dtype=numpy.int32)
    t1b = numpy.zeros((ncas,ncas,nb,nb), dtype=numpy.int32)
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
    return t1a, t1b, t2aa, t2bb

def gen_nonzero_excitations(t1a, t1b, t2aa, t2bb):
    t1a_nonzero = numpy.array(numpy.array(numpy.nonzero(t1a)).T, order = 'C', dtype = numpy.int32)
    t1b_nonzero = numpy.array(numpy.array(numpy.nonzero(t1b)).T, order = 'C', dtype = numpy.int32)
    t2aa_nonzero = numpy.array(numpy.array(numpy.nonzero(t2aa)).T, order = 'C', dtype = numpy.int32)
    t2bb_nonzero = numpy.array(numpy.array(numpy.nonzero(t2bb)).T, order = 'C', dtype = numpy.int32)
    return t1a_nonzero, t1b_nonzero, t2aa_nonzero, t2bb_nonzero

def contract_H(erieff, civec, ncas, nelecas, conf_info_list, ov_list, ecore_list,
               link_index=None, ts=None, t_nonzero=None):
    '''Compute H|CI>
    '''
    if isinstance(nelecas, (int, numpy.integer)):
        nelecb = nelecas//2
        neleca = nelecas - nelecb
    else:
        neleca, nelecb = nelecas

    na = cistring.num_strings(ncas,neleca)
    nb = cistring.num_strings(ncas,nelecb)

    if ts is None:
        if link_index is None:
            link_indexa = cistring.gen_linkstr_index(range(ncas), neleca)
            link_indexb = cistring.gen_linkstr_index(range(ncas), nelecb)
            link_index = (link_indexa, link_indexb)
        else:
            link_indexa, link_indexb = link_index
        t1a, t1b, t2aa, t2bb= gen_excitations(ncas, nelecas,na,nb,link_index)
    else:
        t1a, t1b, t2aa, t2bb = ts
    if t_nonzero is None:
        t1a_nonzero, t1b_nonzero, t2aa_nonzero, t2bb_nonzero = \
            gen_nonzero_excitations(t1a, t1b, t2aa, t2bb)
    else:
        t1a_nonzero, t1b_nonzero, t2aa_nonzero, t2bb_nonzero = t_nonzero
    civec = numpy.asarray(civec, order = 'C')
    cinew = numpy.zeros_like(civec)
    erieff = numpy.asarray(erieff, order = 'C', dtype= numpy.float64)
    conf_info_list = numpy.asarray(conf_info_list, order = 'C', dtype = numpy.int32)
    stringsa = cistring.make_strings(range(ncas),neleca)
    stringsb = cistring.make_strings(range(ncas),nelecb)
    t1ann = t1a_nonzero.shape[0]
    t1bnn = t1b_nonzero.shape[0]
    t2aann = t2aa_nonzero.shape[0]
    t2bbnn = t2bb_nonzero.shape[0]
    ov_list = numpy.asarray(ov_list, order = 'C', dtype=numpy.float64)
    ecore_list = numpy.asarray(ecore_list, order = 'C', dtype=numpy.float64)
    mo_num = erieff.shape[0]
    libsf.SFNOCIcontract_H_spin1(erieff.ctypes.data_as(ctypes.c_void_p),
         civec.ctypes.data_as(ctypes.c_void_p),
         cinew.ctypes.data_as(ctypes.c_void_p),
         ctypes.c_int(ncas),
         ctypes.c_int(neleca), ctypes.c_int(nelecb),
         conf_info_list.ctypes.data_as(ctypes.c_void_p),
         ctypes.c_int(na), stringsa.ctypes.data_as(ctypes.c_void_p),
         ctypes.c_int(nb), stringsb.ctypes.data_as(ctypes.c_void_p),
         ctypes.c_int(mo_num),
         t1a.ctypes.data_as(ctypes.c_void_p),
         t1a_nonzero.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(t1ann),
         t1b.ctypes.data_as(ctypes.c_void_p),
         t1b_nonzero.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(t1bnn),
         t2aa.ctypes.data_as(ctypes.c_void_p),
         t2aa_nonzero.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(t2aann),
         t2bb.ctypes.data_as(ctypes.c_void_p),
         t2bb_nonzero.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(t2bbnn),
         ov_list.ctypes.data_as(ctypes.c_void_p), ecore_list.ctypes.data_as(ctypes.c_void_p))
    return cinew

def contract_H_slow(erieff, civec, ncas, nelecas, conf_info_list, ov_list, ecore_list, link_index=None):
    '''Compute H|CI>
    '''
    if isinstance(nelecas, (int, numpy.integer)):
        nelecb = nelecas//2
        neleca = nelecas - nelecb
    else:
        neleca, nelecb = nelecas
    if link_index is None:
        link_indexa = cistring.gen_linkstr_index(range(ncas), neleca)
        link_indexb = cistring.gen_linkstr_index(range(ncas), nelecb)
    else:
        link_indexa, link_indexb = link_index
    na = cistring.num_strings(ncas,neleca)
    nb = cistring.num_strings(ncas,nelecb)
    civec = civec.reshape(na,nb)
    cinew = numpy.zeros((na,nb))
    stringsa = cistring.make_strings(range(ncas),neleca)
    stringsb = cistring.make_strings(range(ncas),nelecb)
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
        for ab, ib, str1b, str0b in t1b_nonzero:
            p1 = conf_info_list[str1a, str1b]
            p2 = conf_info_list[str0a, str0b]
            cinew[str1a,str1b] += civec[str0a,str0b] * erieff[p1,p2,aa,ia,ab,ib] \
                                    * t1a[aa,ia,str1a,str0a]* t1b[ab,ib,str1b,str0b] \
                                    * ov_list[p1,p2] *2
    for a1, i1, a2,i2, str1a, str0a in t2aa_nonzero:
        for str0b, stringb in enumerate(stringsb):
            p1 = conf_info_list[str1a, str0b]
            p2 = conf_info_list[str0a, str0b]
            cinew[str1a,str0b] += civec[str0a,str0b] * erieff[p1,p2,a1,i1,a2,i2] \
                                    *t2aa[a1,i1,a2,i2,str1a,str0a] * ov_list[p1,p2]
    for a1, i1, a2,i2, str1b, str0b in t2bb_nonzero:
        for str0a, stringa in enumerate(stringsa):
            p1 = conf_info_list[str0a, str1b]
            p2 = conf_info_list[str0a, str0b]
            cinew[str0a,str1b] += civec[str0a,str0b] * erieff[p1,p2,a1,i1,a2,i2] \
                                    * t2bb[a1,i1,a2,i2,str1b,str0b] * ov_list[p1,p2]
    for str0a, stringa in enumerate(stringsa):
        for str0b, stringb in enumerate(stringsb):
            p = conf_info_list[str0a, str0b]
            cinew[str0a,str0b] += ecore_list[p] * civec[str0a,str0b]
    cinew.reshape(-1)
    return cinew

def kernel_sfnoci(sfnoci, h1e, eri, ncas, nelecas, conf_info_list, ov_list, ecore_list,
                  ci0=None, link_index=None, tol=None, lindep=None,
                  max_cycle=None, max_space=None, nroots=None,
                  davidson_only=None, pspace_size=None, hop=None,
                  max_memory=None, verbose=None, **kwargs):
    '''
    Args:
        h1e: ndarray
            effective 1-electron Hamiltonian defined in SF-NOCI space : (nbath, nbath, N, N)
        eri: ndarray
            2-electron integrals in chemist's notation
        ncas: int
            Number of active orbitals
        nelecas: (int, int)
            Number of active electrons of the system
        conf_info_list : ndarray, (nstringsa, nstringsb)
            The optimized bath orbitals indices for each configuration.
        ov_list : ndarray (nbath, nbath)
            overlap matrix between different baths.
        ecore_list : ndarray (nbath)
            1D numpy array of core energies for each bath

    Kwargs:
        ci0: ndarray
            Initial guess
        link_index: ndarray
            A lookup table to cache the addresses of CI determinants in
            wave-function vector
        tol: float
            Convergence tolerance
        lindep: float
            Linear dependence threshold
        max_cycle: int
            Max. iterations for diagonalization
        max_space: int
            Max. trial vectors to store for sub-space diagonalization method
        nroots: int
            Number of states to solve
        davidson_only: bool
            Whether to call subspace diagonalization (davidson solver) or do a
            full diagonalization (lapack eigh) for small systems
        pspace_size: int
            Number of determinants as the threshold of "small systems",
        hop: function(c) => array_like_c
            Function to use for the Hamiltonian multiplication with trial vector

    Note: davidson solver requires more arguments. For the parameters not
    dispatched, they can be passed to davidson solver via the extra keyword
    arguments **kwargs
    '''
    if nroots is None: nroots = sfnoci.nroots
    if davidson_only is None: davidson_only = sfnoci.davidson_only
    if pspace_size is None: pspace_size = sfnoci.pspace_size
    if max_memory is None:
        max_memory = sfnoci.max_memory - lib.current_memory()[0]
    log = logger.new_logger(sfnoci, verbose)
    nelec = nelecas
    assert (0 <= nelec[0] <= ncas and 0 <= nelec[1] <= ncas)
    hdiag = sfnoci.make_hdiag(h1e, eri, ncas, nelec, conf_info_list, ecore_list).ravel()
    num_dets = hdiag.size
    civec_size = num_dets
    precond = sfnoci.make_precond(hdiag)
    addr = [0]
    erieff = sfnoci.absorb_h1e(h1e, eri, ncas, nelec, .5)
    na = cistring.num_strings(ncas, nelec[0])
    nb = cistring.num_strings(ncas, nelec[1])
    if link_index is None:
        link_indexa = cistring.gen_linkstr_index(range(ncas), nelec[0])
        link_indexb = cistring.gen_linkstr_index(range(ncas), nelec[1])
        link_index = (link_indexa, link_indexb)
    else:
        link_indexa, link_indexb = link_index

    ts = gen_excitations(ncas, nelecas, na, nb, link_index)
    t_nonzero = gen_nonzero_excitations(ts[0], ts[1], ts[2], ts[3])
    if hop is None:
        cpu0 = [logger.process_clock(), logger.perf_counter()]
        def hop(c):
            hc = sfnoci.contract_H(erieff, c, ncas, nelecas, conf_info_list,
                                   ov_list, ecore_list,link_index, ts, t_nonzero)
            cpu0[:] = log.timer_debug1('contract_H', *cpu0)
            return hc.ravel()
    def init_guess():
        if callable(getattr(sfnoci, 'get_init_guess', None)):
            return sfnoci.get_init_guess(ncas, nelecas, nroots, hdiag)
        else:
            x0 = []
            for i in range(min(len(addr), nroots)):
                x = numpy.zeros(civec_size)
                x[addr[i]] = 1
                x0.append(x)
            return x0
    if ci0 is None:
        ci0 = init_guess
    if tol is None: tol = sfnoci.conv_tol
    if lindep is None: lindep = sfnoci.lindep
    if max_cycle is None: max_cycle = sfnoci.max_cycle
    if max_space is None: max_space = sfnoci.max_space
    with lib.with_omp_threads(None):
        e, c = sfnoci.eig(hop, ci0, precond, tol=tol, lindep=lindep,
                       max_cycle=max_cycle, max_space=max_space, nroots=nroots,
                       max_memory=max_memory, verbose=log, follow_state=True,
                       tol_residual=None, **kwargs)
    return e, c

def make_rdm1s(mo_coeff, ci, ncas, nelecas, ncore, dmet_core_list, conf_info_list, ov_list):
    N = mo_coeff.shape[0]
    mo_cas = mo_coeff[:,ncore:ncore+ncas]
    stringsa = cistring.make_strings(range(ncas),nelecas[0])
    stringsb = cistring.make_strings(range(ncas),nelecas[1])
    link_indexa = cistring.gen_linkstr_index(range(ncas),nelecas[0])
    link_indexb = cistring.gen_linkstr_index(range(ncas),nelecas[1])
    na = cistring.num_strings(ncas,nelecas[0])
    nb = cistring.num_strings(ncas,nelecas[1])
    rdm1c = numpy.zeros((N,N))
    ci = ci.reshape(na,nb)
    for str0a, strsa in enumerate(stringsa):
        for str0b, strsb in enumerate(stringsb):
            p = conf_info_list[str0a, str0b]
            rdm1c += numpy.conjugate(ci[str0a,str0b])*ci[str0a,str0b]*dmet_core_list[p,p]

    rdm1asmoa = numpy.zeros((ncas,ncas))
    rdm1asmob = numpy.zeros((ncas,ncas))
    for str0a , taba in enumerate(link_indexa):
        for aa, ia, str1a, signa in link_indexa[str0a]:
            for str0b, strsb in enumerate(stringsb):
                p1 = conf_info_list[str1a, str0b]
                p2 = conf_info_list[str0a, str0b]
                rdm1asmoa[aa,ia] += signa * numpy.conjugate(ci[str1a,str0b]) * ci[str0a,str0b] * ov_list[p1,p2]
    for str0b, tabb in enumerate(link_indexb):
        for ab, ib, str1b, signb in link_indexb[str0b]:
            for str0a, strsa in enumerate(stringsa):
                p1 = conf_info_list[str0a, str1b]
                p2 = conf_info_list[str0a, str0b]
                rdm1asmob[ab,ib] += signb * numpy.conjugate(ci[str0a,str1b]) * ci[str0a,str0b] * ov_list[p1,p2]
    rdm1a = lib.einsum('ia,ab,jb -> ij', numpy.conjugate(mo_cas),rdm1asmoa,mo_cas)
    rdm1b = lib.einsum('ia,ab,jb-> ij', numpy.conjugate(mo_cas),rdm1asmob,mo_cas)
    rdm1a += rdm1c
    rdm1b += rdm1c

    return rdm1a, rdm1b

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
                p1 = conf_info_list[str0a, str0b]
                p2 = conf_info_list[str0a, str1b]
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

def make_rdm2(mo_coeff, ci, ncas, nelecas, ncore, dmet_core_list, conf_info_list, ov_list):
    rdm2aa, rdm2ab, rdm2ba, rdm2bb = \
        make_rdm2s(mo_coeff, ci, ncas, nelecas, ncore,dmet_core_list, conf_info_list, ov_list)
    return rdm2aa + rdm2ab + rdm2ba + rdm2bb

def fix_spin(fciobj, shift=PENALTY, ss=None, **kwargs):
    r'''If FCI solver cannot stay on spin eigenfunction, this function can
    add a shift to the states which have wrong spin.

    .. math::

        (H + shift*S^2) |\Psi\rangle = E |\Psi\rangle

    Args:
        fciobj : An instance of :class:`FCISolver`

    Kwargs:
        shift : float
            Level shift for states which have different spin
        ss : number
            S^2 expection value == s*(s+1)

    Returns
            A modified FCI object based on fciobj.
    '''
    if isinstance(fciobj, direct_uhf.FCISolver):
        raise NotImplementedError

    if isinstance (fciobj, types.ModuleType):
        raise DeprecationWarning('fix_spin should be applied on FCI object only')

    if 'ss_value' in kwargs:
        sys.stderr.write('fix_spin_: kwarg "ss_value" will be removed in future release. '
                         'It was replaced by "ss"\n')
        ss_value = kwargs['ss_value']
    else:
        ss_value = ss

    if isinstance (fciobj, SpinPenaltySFNOCISolver):
        # recursion avoidance
        fciobj.ss_penalty = shift
        fciobj.ss_value = ss_value
        return fciobj

    return lib.set_class(SpinPenaltySFNOCISolver(fciobj, shift, ss_value),
                         (SpinPenaltySFNOCISolver, fciobj.__class__))

def fix_spin_(fciobj, shift=.1, ss=None):
    sp_fci = fix_spin(fciobj, shift, ss)
    fciobj.__class__ = sp_fci.__class__
    fciobj.__dict__ = sp_fci.__dict__
    return fciobj


class SFNOCISolver(FCISolver):
    '''SF-NOCI
    '''
    def make_hdiag(self, h1e, eri, ncas, nelecas, conf_info_list, ecore_list, opt=None):
        return make_hdiag(h1e, eri, ncas, nelecas, conf_info_list, ecore_list, opt)

    def make_precond(self, hdiag, level_shift=0):
        return lib.make_diag_precond(hdiag, level_shift)

    def absorb_h1e(self, h1e, eri, ncas, nelecas, fac=1):
        return absorb_h1e(h1e, eri, ncas, nelecas, fac)

    def contract_H(self, erieff, civec, ncas, nelecas, conf_info_list, ov_list,
                 ecore_list, link_index=None, ts=None, t_nonzero=None):
        return contract_H(erieff, civec, ncas, nelecas, conf_info_list, ov_list,
                        ecore_list ,link_index, ts, t_nonzero)

    def get_init_guess(self, ncas, nelecas, nroots, hdiag):
        return fci.direct_spin1.get_init_guess(ncas, nelecas, nroots, hdiag)

    def eig(self, op, x0=None, precond=None, **kwargs):
        if isinstance(op, numpy.ndarray):
            self.converged = True
            return scipy.linalg.eigh(op)

        self.converged, e, ci = \
                lib.davidson1(lambda xs: [op(x) for x in xs],
                              x0, precond, lessio=False, **kwargs)
        if kwargs['nroots'] == 1:
            self.converged = self.converged[0]
            e = e[0]
            ci = ci[0]
        return e, ci

    def kernel(self, h1e, eri, norb, nelec, conf_info_list, ov_list, ecore_list, ci0=None,
             tol=None, lindep=None, max_cycle=None, max_space=None,
             nroots=None, davidson_only=None, pspace_size=None,
             orbsym=None, wfnsym=None, **kwargs):
        if nroots is None: nroots = self.nroots
        if self.verbose >= logger.WARN:
            self.check_sanity()
        assert self.spin is None or self.spin == 0
        self.norb = norb
        self.nelec = nelec
        link_indexa = cistring.gen_linkstr_index(range(norb), nelec[0])
        link_indexb = cistring.gen_linkstr_index(range(norb), nelec[1])
        link_index = (link_indexa, link_indexb)

        e, c = kernel_sfnoci(self, h1e, eri, norb, nelec, conf_info_list, ov_list, ecore_list, ci0,
                           link_index, tol, lindep, max_cycle, max_space, nroots,
                           davidson_only, pspace_size, **kwargs)
        self.eci = e

        na = link_index[0].shape[0]
        nb = link_index[1].shape[0]
        if nroots > 1:
            self.ci = [x.reshape(na,nb).view(FCIvector) for x in c]
        else:
            self.ci = c.reshape(na,nb).view(FCIvector)

        return self.eci, self.ci

    def make_rdm1s(self, mo_coeff, ci, ncas, nelecas, ncore, dmet_core_list, conf_info_list, ov_list):
        return make_rdm1s(mo_coeff, ci, ncas, nelecas, ncore, dmet_core_list, conf_info_list, ov_list)

    def make_rdm1(self, mo_coeff, ci, ncas, nelecas, ncore, dmet_core_list, conf_info_list, ov_list):
        return make_rdm1(mo_coeff, ci, ncas, nelecas, ncore, dmet_core_list,conf_info_list, ov_list)

    def make_rdm2s(self, mo_coeff, ci, ncas, nelecas, ncore,dmet_core_list, conf_info_list, ov_list):
        return make_rdm2s(mo_coeff, ci, ncas, nelecas, ncore, dmet_core_list, conf_info_list, ov_list)

    def make_rdm2(self, mo_coeff, ci, ncas, nelecas, ncore, dmet_core_list, conf_info_list, ov_list):
        return make_rdm2(mo_coeff, ci, ncas, nelecas, ncore, dmet_core_list, conf_info_list, ov_list)

    def contract_ss(self, civec, ncas=None, nelecas=None):
        if ncas is None : ncas = self.ncas
        if nelecas is None : nelecas = self.nelecas
        return spin_op.contract_ss(civec,ncas,nelecas)

    def fix_spin_(self, shift=PENALTY, ss = None):
        r'''Use level shift to control FCI solver spin.

        .. math::

            (H + shift*S^2) |\Psi\rangle = E |\Psi\rangle

        Kwargs:
            shift : float
                Energy penalty for states which have wrong spin
            ss : number
                S^2 expection value == s*(s+1)
        '''
        fix_spin_(self, shift, ss)
        return self
    fix_spin = fix_spin_

    def spin_square(self, civec, ncas = None, nelecas = None):
        if ncas is None : ncas = self.ncas
        if nelecas is None : nelecas = self.nelecas
        return spin_op.spin_square0(civec, ncas, nelecas)

class SpinPenaltySFNOCISolver:
    __name_mixin__ = 'SpinPenalty'
    _keys = {'ss_value', 'ss_penalty', 'base'}

    def __init__(self, sfnocibase, shift, ss_value):
        self.base = sfnocibase.copy()
        self.__dict__.update (sfnocibase.__dict__)
        self.ss_value = ss_value
        self.ss_penalty = shift
        self.davidson_only = self.base.davidson_only = True

    def undo_fix_spin(self):
        obj = lib.view(self, lib.drop_class(self.__class__, SpinPenaltySFNOCISolver))
        del obj.base
        del obj.ss_value
        del obj.ss_penalty
        return obj

    def base_contract_H(self, *args, **kwargs):
        return super().contract_H(*args, **kwargs)

    def contract_H(self, erieff, civec, ncas, nelecas, conf_info_list, ov_list,
                   ecore_list, link_index=None, ts=None, t_nonzero=None, **kwargs):
        if isinstance(nelecas, (int, numpy.number)):
            sz = (nelecas % 2) * .5
        else:
            sz = abs(nelecas[0]-nelecas[1]) * .5
        if self.ss_value is None:
            ss = sz*(sz+1)
        else:
            ss = self.ss_value
        if ss < sz*(sz+1)+.1:
            # (S^2-ss)|Psi> to shift state other than the lowest state
            ci1 = self.contract_ss(civec, ncas, nelecas).reshape(civec.shape)
            ci1 -= ss * civec
        else:
            # (S^2-ss)^2|Psi> to shift states except the given spin.
            # It still relies on the quality of initial guess
            tmp = self.contract_ss(civec, ncas, nelecas).reshape(civec.shape)
            tmp -= ss * civec
            ci1 = -ss * tmp
            ci1 += self.contract_ss(tmp, ncas, nelecas).reshape(civec.shape)
            tmp = None
        ci1 *= self.ss_penalty
        ci0 = super().contract_H(erieff, civec, ncas, nelecas, conf_info_list, ov_list,
                                 ecore_list, link_index, ts, t_nonzero, **kwargs)
        ci1 += ci0.reshape(civec.shape)
        return ci1
