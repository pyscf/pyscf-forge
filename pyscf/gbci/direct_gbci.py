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

import sys

import numpy
import numpy as np
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

libgbci = lib.load_library("libgbci")

PENALTY = getattr(__config__, 'gbci_GBCI_fix_spin_shift', 0.2)

def str2occ(str0,norb):
    occ=numpy.zeros(norb)
    for i in range(norb):
        if str0 & ( 1 << i ):
            occ[i]=1

    return occ

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

def contract_h(erieff, civec, ncas, nelecas, conf_info_list, ov_list, ecore_list,
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
    libgbci.gbci_contract_h_spin1(erieff.ctypes.data_as(ctypes.c_void_p),
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

def contract_h_slow(erieff, civec, ncas, nelecas, conf_info_list, ov_list, ecore_list, link_index=None):
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

def kernel(gbci, h1e, eri, ncas, nelecas, conf_info_list, ov_list, ecore_list,
           ci0=None, link_index=None, tol=None, lindep=None,
           max_cycle=None, max_space=None, nroots=None,
           davidson_only=None, pspace_size=None, hop=None,
           max_memory=None, verbose=None, **kwargs):
    '''
    Args:
        h1e: ndarray
            effective 1-electron Hamiltonian defined in GBCI space : (nbath, nbath, N, N)
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
    if nroots is None: nroots = gbci.nroots
    if davidson_only is None: davidson_only = gbci.davidson_only
    if pspace_size is None: pspace_size = gbci.pspace_size
    if max_memory is None:
        max_memory = gbci.max_memory - lib.current_memory()[0]
    log = logger.new_logger(gbci, verbose)
    nelec = nelecas
    assert (0 <= nelec[0] <= ncas and 0 <= nelec[1] <= ncas)
    hdiag = gbci.make_hdiag(h1e, eri, ncas, nelec, conf_info_list, ecore_list).ravel()
    num_dets = hdiag.size
    civec_size = num_dets
    precond = gbci.make_precond(hdiag)
    addr = [0]
    erieff = gbci.absorb_h1e(h1e, eri, ncas, nelec, .5)
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
            hc = gbci.contract_h(erieff, c, ncas, nelecas, conf_info_list,
                                   ov_list, ecore_list,link_index, ts, t_nonzero)
            cpu0[:] = log.timer_debug1('contract_h', *cpu0)
            return hc.ravel()
    def init_guess():
        if callable(getattr(gbci, 'get_init_guess', None)):
            return gbci.get_init_guess(ncas, nelecas, nroots, hdiag)
        else:
            x0 = []
            for i in range(min(len(addr), nroots)):
                x = numpy.zeros(civec_size)
                x[addr[i]] = 1
                x0.append(x)
            return x0
    if ci0 is None:
        ci0 = init_guess
    if tol is None: tol = gbci.conv_tol
    if lindep is None: lindep = gbci.lindep
    if max_cycle is None: max_cycle = gbci.max_cycle
    if max_space is None: max_space = gbci.max_space
    with lib.with_omp_threads(None):
        e, c = gbci.eig(hop, ci0, precond, tol=tol, lindep=lindep,
                       max_cycle=max_cycle, max_space=max_space, nroots=nroots,
                       max_memory=max_memory, verbose=log, follow_state=True,
                       tol_residual=None, **kwargs)
    return e, c

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

    if isinstance (fciobj, SpinPenaltyGBCISolver):
        # recursion avoidance
        fciobj.ss_penalty = shift
        fciobj.ss_value = ss_value
        return fciobj
    return lib.set_class(SpinPenaltyGBCISolver(fciobj, shift, ss_value),
                         (SpinPenaltyGBCISolver, fciobj.__class__))

def fix_spin_(fciobj, shift=.1, ss=None):
    sp_fci = fix_spin(fciobj, shift, ss)
    fciobj.__class__ = sp_fci.__class__
    fciobj.__dict__ = sp_fci.__dict__
    return fciobj


class GBCISolver(FCISolver):
    '''GBCI FCI solver.
    '''
    def make_hdiag(self, h1e, eri, ncas, nelecas, conf_info_list, ecore_list, opt=None):
        return make_hdiag(h1e, eri, ncas, nelecas, conf_info_list, ecore_list, opt)

    def make_precond(self, hdiag, level_shift=0):
        return lib.make_diag_precond(hdiag, level_shift)

    def absorb_h1e(self, h1e, eri, ncas, nelecas, fac=1):
        return absorb_h1e(h1e, eri, ncas, nelecas, fac)

    def contract_h(self, erieff, civec, ncas, nelecas, conf_info_list, ov_list,
                 ecore_list, link_index=None, ts=None, t_nonzero=None):
        return contract_h(erieff, civec, ncas, nelecas, conf_info_list, ov_list,
                        ecore_list ,link_index, ts, t_nonzero)

    def contract_h_slow(self, erieff, civec, ncas, nelecas, conf_info_list, ov_list,
                 ecore_list, link_index=None):
        return contract_h_slow(erieff, civec, ncas, nelecas, conf_info_list, ov_list,
                        ecore_list ,link_index)

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
        self.norb = norb
        self.nelec = nelec
        link_indexa = cistring.gen_linkstr_index(range(norb), nelec[0])
        link_indexb = cistring.gen_linkstr_index(range(norb), nelec[1])
        link_index = (link_indexa, link_indexb)

        e, c = kernel(self, h1e, eri, norb, nelec, conf_info_list, ov_list, ecore_list, ci0,
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
    def fix_spin(self, shift=PENALTY, ss=None):
        return self.fix_spin_(shift=shift, ss=ss)

    def spin_square(self, civec, ncas = None, nelecas = None):
        if ncas is None : ncas = self.ncas
        if nelecas is None : nelecas = self.nelecas
        return spin_op.spin_square0(civec, ncas, nelecas)

class SpinPenaltyGBCISolver:
    __name_mixin__ = 'SpinPenalty'
    _keys = {'ss_value', 'ss_penalty', 'base'}

    def __init__(self, gbcibase, shift, ss_value):
        object.__setattr__(self, 'base', gbcibase.copy())
        object.__setattr__(self, 'ss_value', ss_value)
        object.__setattr__(self, 'ss_penalty', float(shift))

    # Delegate everything else to base
    def __getattr__(self, name):
        base = object.__getattribute__(self, 'base')
        return getattr(base, name)

    def __setattr__(self, name, value):
        if name in ('__dict__', '__class__', '__weakref__'):
            object.__setattr__(self, name, value)
            return

        d = object.__getattribute__(self, '__dict__')
        if 'base' not in d or name in ('base', 'ss_value', 'ss_penalty',
                                       '_contract_h_base'):
            object.__setattr__(self, name, value)
            return

        base = object.__getattribute__(self, 'base')
        setattr(base, name, value)

    def undo_fix_spin(self):
        return self.base

    def contract_h(self, erieff, civec, ncas, nelecas, conf_info_list, ov_list,
                   ecore_list, link_index=None, ts=None, t_nonzero=None, **kwargs):
        # --- spin penalty part (uses base.contract_ss, which is safe) ---
        if isinstance(nelecas, (int, np.number)):
            sz = (nelecas % 2) * 0.5
        else:
            sz = abs(nelecas[0]-nelecas[1]) * 0.5
        ss_tgt = self.ss_value if self.ss_value is not None else sz*(sz+1)

        if ss_tgt < sz*(sz+1) + 0.1:
            ci1 = self.base.contract_ss(civec, ncas, nelecas).reshape(civec.shape)
            ci1 -= ss_tgt * civec
        else:
            tmp = self.base.contract_ss(civec, ncas, nelecas).reshape(civec.shape)
            tmp -= ss_tgt * civec
            ci1 = -ss_tgt * tmp
            ci1 += self.base.contract_ss(tmp, ncas, nelecas).reshape(civec.shape)
        ci1 *= self.ss_penalty

        ci0 = self._contract_h_base(erieff, civec, ncas, nelecas,
                                    conf_info_list, ov_list, ecore_list,
                                    link_index, ts, t_nonzero, **kwargs)
        return ci0.reshape(civec.shape) + ci1

    def kernel(self, *args, **kwargs):
        self._contract_h_base = self.base.contract_h

        def _proxy_contract_h(erieff, civec, ncas, nelecas, conf_info_list, ov_list,
                              ecore_list, link_index=None, ts=None, t_nonzero=None, **kw):
            return self.contract_h(erieff, civec, ncas, nelecas, conf_info_list,
                                   ov_list, ecore_list, link_index, ts, t_nonzero, **kw)

        self.base.contract_h = _proxy_contract_h
        try:
            return self.base.kernel(*args, **kwargs)
        finally:
            self.base.contract_h = self._contract_h_base
            del self._contract_h_base
    def select_spin_roots(self, civecs, ncas, nelecas, tol=0.20,
                          use_base=True, also_return_s2=False,
                          nroots=None, dbg=False):

        s2_op  = self.base.contract_ss if use_base else self.contract_ss
        target = self.ss_value
        if target is None:
            if isinstance(nelecas, (int, np.number)):
                sz = (nelecas % 2) * 0.5
            else:
                sz = abs(nelecas[0] - nelecas[1]) * 0.5
            target = sz * (sz + 1.0)
        target = float(target)

        # decide how many roots we expect
        if nroots is None:
            nroots = getattr(self, "nroots", None)
            if nroots is None:
                # try to infer from last axis length
                arr = np.asarray(civecs)
                nroots = arr.shape[-1] if arr.ndim >= 2 else 1

        # robust iterator over roots
        def _iter_roots(ci):
            arr = np.asarray(ci)
            if isinstance(ci, (list, tuple)):
                for v in ci:
                    yield np.asarray(v).ravel()
                return
            if arr.ndim == 1:
                yield arr.ravel()
                return
            if arr.ndim == 2 and nroots in arr.shape:
                # pick axis that equals nroots
                axis = 0 if arr.shape[0] == nroots and arr.shape[1] != nroots else 1
                if axis == 0:
                    for i in range(arr.shape[0]):
                        yield arr[i].ravel()
                else:
                    for i in range(arr.shape[1]):
                        yield arr[:, i].ravel()
                return
            # generic: treat last axis as roots
            arr2 = arr.reshape((-1, arr.shape[-1]))
            for i in range(arr2.shape[1]):
                yield arr2[:, i].ravel()

        s2_vals = []
        for c in _iter_roots(civecs):
            c = np.asarray(c).ravel()
            if c.size == 0:
                s2_vals.append(np.nan)
                continue
            Sc  = np.asarray(s2_op(c, ncas, nelecas)).ravel()
            den = (np.vdot(c, c)).real  # assumes orthonormal working space
            s2  = (np.vdot(c, Sc) / den).real
            s2_vals.append(s2)

        s2_vals = np.asarray(s2_vals, dtype=float)
        dists   = np.abs(s2_vals - target)
        idx     = [int(i) for i, d in enumerate(dists) if np.isfinite(d) and d <= tol]

        if dbg:
            logger.debug(self, "[select_spin_roots] target ss=%.3f, tol=%.2f",
                         target, tol)
            logger.debug(self, "[select_spin_roots] <S^2> per root: %s", s2_vals)
            logger.debug(self, "[select_spin_roots] |S^2 - target|: %s", dists)
            logger.debug(self, "[select_spin_roots] chosen idx: %s", idx)

        return (idx, s2_vals) if also_return_s2 else idx
