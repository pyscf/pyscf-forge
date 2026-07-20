#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#         Yu Jin <yjin@flatironinstitute.org>
#

'''
ULNO-UCCSD(T)
'''


import ctypes
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc.uccsd_t import _sort_eri


libcc = lib.load_library('liblno')
CCulnoccsd_t_aaa  = libcc.CCulnoccsd_t_aaa
CCulnoccsd_t_zaaa = libcc.CCulnoccsd_t_zaaa
CCulnoccsd_t_baa  = libcc.CCulnoccsd_t_baa
CCulnoccsd_t_zbaa = libcc.CCulnoccsd_t_zbaa


def kernel(mycc, eris, prjlo, t1=None, t2=None, verbose=logger.NOTE):
    '''
    adapted from pyscf.cc.uccsd_t

    Args:
       prjlo[mu,i] = <mu|i> is the overlap between the mu-th LO and the i-th occ MO.
    '''
    cpu1 = cpu0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mycc, verbose)
    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    prjloa, prjlob = prjlo

    nocca, noccb = mycc.nocc
    nmoa = eris.focka.shape[0]
    nmob = eris.fockb.shape[0]
    nvira = nmoa - nocca
    nvirb = nmob - noccb

    if prjloa.shape[1] != nocca:
        raise ValueError(f'Alpha projector shape {prjloa.shape} does not match nocca={nocca}')
    if prjlob.shape[1] != noccb:
        raise ValueError(f'Beta projector shape {prjlob.shape} does not match noccb={noccb}')

    dtype = numpy.result_type(t1a, t1b, t2aa, t2ab, t2bb, eris.ovoo.dtype)
    prjloa = numpy.asarray(prjloa, dtype=dtype, order='C')
    prjlob = numpy.asarray(prjlob, dtype=dtype, order='C')

    et_sum = numpy.zeros(1, dtype=dtype)
    if mycc.incore_complete:
        ftmp = None
    else:
        ftmp = lib.H5TmpFile()
    t1aT = t1a.T.copy()
    t1bT = t1b.T.copy()
    t2aaT = t2aa.transpose(2,3,0,1).copy()
    t2bbT = t2bb.transpose(2,3,0,1).copy()

    eris_vooo = numpy.asarray(eris.ovoo).transpose(1,3,0,2).conj().copy()
    eris_VOOO = numpy.asarray(eris.OVOO).transpose(1,3,0,2).conj().copy()
    eris_vOoO = numpy.asarray(eris.ovOO).transpose(1,3,0,2).conj().copy()
    eris_VoOo = numpy.asarray(eris.OVoo).transpose(1,3,0,2).conj().copy()

    eris_vvop, eris_VVOP, eris_vVoP, eris_VvOp = _sort_eri(mycc, eris, ftmp, log)
    cpu1 = log.timer_debug1('UCCSD(T) sort_eri', *cpu1)

    mem_now = lib.current_memory()[0]
    max_memory = max(0, mycc.max_memory - mem_now)
    # aaa
    bufsize = max(8, int((max_memory*.5e6/8-nocca**3*3*lib.num_threads())*.4/max(1,nocca*nmoa)))
    log.debug('max_memory %d MB (%d MB in use)', max_memory, mem_now)
    orbsym = numpy.zeros(nocca, dtype=int)
    contract = _gen_contract_aaa(t1aT, t2aaT, eris_vooo, eris.focka,
                                 eris.mo_energy[0], prjloa, orbsym, log)
    with lib.call_in_background(contract, sync=not mycc.async_io) as ctr:
        for a0, a1 in reversed(list(lib.prange_tril(0, nvira, bufsize))):
            cache_row_a = numpy.asarray(eris_vvop[a0:a1,:a1], order='C')
            if a0 == 0:
                cache_col_a = cache_row_a
            else:
                cache_col_a = numpy.asarray(eris_vvop[:a0,a0:a1], order='C')
            ctr(et_sum, a0, a1, a0, a1, (cache_row_a,cache_col_a,
                                         cache_row_a,cache_col_a))

            for b0, b1 in lib.prange_tril(0, a0, max(1, bufsize//8)):
                cache_row_b = numpy.asarray(eris_vvop[b0:b1,:b1], order='C')
                if b0 == 0:
                    cache_col_b = cache_row_b
                else:
                    cache_col_b = numpy.asarray(eris_vvop[:b0,b0:b1], order='C')
                ctr(et_sum, a0, a1, b0, b1, (cache_row_a,cache_col_a,
                                             cache_row_b,cache_col_b))
    cpu1 = log.timer_debug1('contract_aaa', *cpu1)

    # bbb
    bufsize = max(8, int((max_memory*.5e6/8-noccb**3*3*lib.num_threads())*.4/max(1,noccb*nmob)))
    log.debug('max_memory %d MB (%d MB in use)', max_memory, mem_now)
    orbsym = numpy.zeros(noccb, dtype=int)
    contract = _gen_contract_aaa(t1bT, t2bbT, eris_VOOO, eris.fockb,
                                 eris.mo_energy[1], prjlob, orbsym, log)
    with lib.call_in_background(contract, sync=not mycc.async_io) as ctr:
        for a0, a1 in reversed(list(lib.prange_tril(0, nvirb, bufsize))):
            cache_row_a = numpy.asarray(eris_VVOP[a0:a1,:a1], order='C')
            if a0 == 0:
                cache_col_a = cache_row_a
            else:
                cache_col_a = numpy.asarray(eris_VVOP[:a0,a0:a1], order='C')
            ctr(et_sum, a0, a1, a0, a1, (cache_row_a,cache_col_a,
                                         cache_row_a,cache_col_a))

            for b0, b1 in lib.prange_tril(0, a0, max(1, bufsize//8)):
                cache_row_b = numpy.asarray(eris_VVOP[b0:b1,:b1], order='C')
                if b0 == 0:
                    cache_col_b = cache_row_b
                else:
                    cache_col_b = numpy.asarray(eris_VVOP[:b0,b0:b1], order='C')
                ctr(et_sum, a0, a1, b0, b1, (cache_row_a,cache_col_a,
                                             cache_row_b,cache_col_b))
    cpu1 = log.timer_debug1('contract_bbb', *cpu1)

    # Premature termination for fully spin-polarized systems
    if nocca*noccb == 0:
        et_sum *= .25
        if abs(et_sum[0].imag) > 1e-4:
            logger.warn(mycc, 'Non-zero imaginary part of UCCSD(T) energy was found %s',
                        et_sum[0])
        et = et_sum[0].real
        log.timer('UCCSD(T)', *cpu0)
        log.note('UCCSD(T) correction = %.15g', et)
        return et

    # Cache t2abT in t2ab to reduce memory footprint
    assert (t2ab.flags.c_contiguous)
    t2abT = lib.transpose(t2ab.copy().reshape(nocca*noccb,nvira*nvirb), out=t2ab)
    t2abT = t2abT.reshape(nvira,nvirb,nocca,noccb)
    # baa
    bufsize = int(max(12, (max_memory*.5e6/8-noccb*nocca**2*5)*.7/max(1,nocca*nmob)))
    ts = t1aT, t1bT, t2aaT, t2abT
    fock = (eris.focka, eris.fockb)
    vooo = (eris_vooo, eris_vOoO, eris_VoOo)
    contract = _gen_contract_baa(ts, vooo, fock, eris.mo_energy, (prjloa, prjlob), log)
    with lib.call_in_background(contract, sync=not mycc.async_io) as ctr:
        for a0, a1 in lib.prange(0, nvirb, int(bufsize/nvira+1)):
            cache_row_a = numpy.asarray(eris_VvOp[a0:a1,:], order='C')
            cache_col_a = numpy.asarray(eris_vVoP[:,a0:a1], order='C')
            for b0, b1 in lib.prange_tril(0, nvira, max(1, bufsize//12)):
                cache_row_b = numpy.asarray(eris_vvop[b0:b1,:b1], order='C')
                cache_col_b = numpy.asarray(eris_vvop[:b0,b0:b1], order='C')
                ctr(et_sum, a0, a1, b0, b1, (cache_row_a,cache_col_a,
                                             cache_row_b,cache_col_b))
    cpu1 = log.timer_debug1('contract_baa', *cpu1)

    t2baT = numpy.ndarray((nvirb,nvira,noccb,nocca), buffer=t2abT,
                          dtype=t2abT.dtype)
    t2baT[:] = t2abT.copy().transpose(1,0,3,2)
    # abb
    ts = t1bT, t1aT, t2bbT, t2baT
    fock = (eris.fockb, eris.focka)
    mo_energy = (eris.mo_energy[1], eris.mo_energy[0])
    vooo = (eris_VOOO, eris_VoOo, eris_vOoO)
    contract = _gen_contract_baa(ts, vooo, fock, mo_energy, (prjlob, prjloa), log)
    for a0, a1 in lib.prange(0, nvira, int(bufsize/nvirb+1)):
        with lib.call_in_background(contract, sync=not mycc.async_io) as ctr:
            cache_row_a = numpy.asarray(eris_vVoP[a0:a1,:], order='C')
            cache_col_a = numpy.asarray(eris_VvOp[:,a0:a1], order='C')
            for b0, b1 in lib.prange_tril(0, nvirb, max(1, bufsize//12)):
                cache_row_b = numpy.asarray(eris_VVOP[b0:b1,:b1], order='C')
                cache_col_b = numpy.asarray(eris_VVOP[:b0,b0:b1], order='C')
                ctr(et_sum, a0, a1, b0, b1, (cache_row_a,cache_col_a,
                                             cache_row_b,cache_col_b))
    cpu1 = log.timer_debug1('contract_abb', *cpu1)

    # Restore t2ab
    lib.transpose(t2baT.transpose(1,0,3,2).copy().reshape(nvira*nvirb,nocca*noccb),
                  out=t2ab)
    et_sum *= .25
    if abs(et_sum[0].imag) > 1e-4:
        logger.warn(mycc, 'Non-zero imaginary part of UCCSD(T) energy was found %s',
                    et_sum[0])
    et = et_sum[0].real
    log.timer('UCCSD(T)', *cpu0)
    log.note('UCCSD(T) correction = %.15g', et)
    return et


def _gen_contract_aaa(t1T, t2T, vooo, fock, mo_energy, prjlo, orbsym, log):
    nvir, nocc = t1T.shape
    nlo = prjlo.shape[0]
    mo_energy = numpy.asarray(mo_energy, order='C')
    fvo = numpy.asarray(fock[nocc:, :nocc], dtype=t1T.dtype, order='C')
    prjlo = numpy.asarray(prjlo, dtype=t1T.dtype, order='C')

    cpu2 = [logger.process_clock(), logger.perf_counter()]
    orbsym = numpy.hstack((numpy.sort(orbsym[:nocc]), numpy.sort(orbsym[nocc:])))
    o_ir_loc = numpy.append(0, numpy.cumsum(numpy.bincount(orbsym[:nocc], minlength=8)))
    v_ir_loc = numpy.append(0, numpy.cumsum(numpy.bincount(orbsym[nocc:], minlength=8)))
    o_sym = orbsym[:nocc]
    oo_sym = (o_sym[:, None] ^ o_sym).ravel()
    oo_ir_loc = numpy.append(0, numpy.cumsum(numpy.bincount(oo_sym, minlength=8)))
    if len(oo_sym) == 0:
        nirrep = 0
    else:
        nirrep = max(oo_sym) + 1

    orbsym = orbsym.astype(numpy.int32)
    o_ir_loc = o_ir_loc.astype(numpy.int32)
    v_ir_loc = v_ir_loc.astype(numpy.int32)
    oo_ir_loc = oo_ir_loc.astype(numpy.int32)
    dtype = numpy.result_type(t2T.dtype, vooo.dtype, fock.dtype)
    if dtype == numpy.complex128:
        drv = CCulnoccsd_t_zaaa
    else:
        drv = CCulnoccsd_t_aaa

    def contract(et_sum, a0, a1, b0, b1, cache):
        cache_row_a, cache_col_a, cache_row_b, cache_col_b = cache
        drv(et_sum.ctypes.data_as(ctypes.c_void_p),
            mo_energy.ctypes.data_as(ctypes.c_void_p),
            t1T.ctypes.data_as(ctypes.c_void_p),
            t2T.ctypes.data_as(ctypes.c_void_p),
            vooo.ctypes.data_as(ctypes.c_void_p),
            fvo.ctypes.data_as(ctypes.c_void_p),
            prjlo.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nlo),
            ctypes.c_int(nocc), ctypes.c_int(nvir),
            ctypes.c_int(a0), ctypes.c_int(a1),
            ctypes.c_int(b0), ctypes.c_int(b1),
            ctypes.c_int(nirrep),
            o_ir_loc.ctypes.data_as(ctypes.c_void_p),
            v_ir_loc.ctypes.data_as(ctypes.c_void_p),
            oo_ir_loc.ctypes.data_as(ctypes.c_void_p),
            orbsym.ctypes.data_as(ctypes.c_void_p),
            cache_row_a.ctypes.data_as(ctypes.c_void_p),
            cache_col_a.ctypes.data_as(ctypes.c_void_p),
            cache_row_b.ctypes.data_as(ctypes.c_void_p),
            cache_col_b.ctypes.data_as(ctypes.c_void_p))
        cpu2[:] = log.timer_debug1('contract %d:%d,%d:%d' % (a0, a1, b0, b1), *cpu2)
    return contract

def _gen_contract_baa(ts, vooo, fock, mo_energy, prjlo, log):
    t1aT, t1bT, t2aaT, t2abT = ts
    focka, fockb = fock
    vooo, vOoO, VoOo = vooo
    prjloa, prjlob = prjlo
    nvira, nocca = t1aT.shape
    nvirb, noccb = t1bT.shape
    nloa = prjloa.shape[0]
    nlob = prjlob.shape[0]
    mo_ea = numpy.asarray(mo_energy[0], order='C')
    mo_eb = numpy.asarray(mo_energy[1], order='C')
    fvo = numpy.asarray(focka[nocca:, :nocca], dtype=t1aT.dtype, order='C')
    fVO = numpy.asarray(fockb[noccb:, :noccb], dtype=t1bT.dtype, order='C')
    prjloa = numpy.asarray(prjloa, dtype=t1aT.dtype, order='C')
    prjlob = numpy.asarray(prjlob, dtype=t1aT.dtype, order='C')

    cpu2 = [logger.process_clock(), logger.perf_counter()]
    dtype = numpy.result_type(t2aaT.dtype, vooo.dtype, prjloa.dtype, prjlob.dtype)
    if dtype == numpy.complex128:
        drv = CCulnoccsd_t_zbaa
    else:
        drv = CCulnoccsd_t_baa
    def contract(et_sum, a0, a1, b0, b1, cache):
        cache_row_a, cache_col_a, cache_row_b, cache_col_b = cache
        drv(et_sum.ctypes.data_as(ctypes.c_void_p),
            mo_ea.ctypes.data_as(ctypes.c_void_p),
            mo_eb.ctypes.data_as(ctypes.c_void_p),
            t1aT.ctypes.data_as(ctypes.c_void_p),
            t1bT.ctypes.data_as(ctypes.c_void_p),
            t2aaT.ctypes.data_as(ctypes.c_void_p),
            t2abT.ctypes.data_as(ctypes.c_void_p),
            vooo.ctypes.data_as(ctypes.c_void_p),
            vOoO.ctypes.data_as(ctypes.c_void_p),
            VoOo.ctypes.data_as(ctypes.c_void_p),
            fvo.ctypes.data_as(ctypes.c_void_p),
            fVO.ctypes.data_as(ctypes.c_void_p),
            prjloa.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nloa),
            prjlob.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nlob),
            ctypes.c_int(nocca), ctypes.c_int(noccb),
            ctypes.c_int(nvira), ctypes.c_int(nvirb),
            ctypes.c_int(a0), ctypes.c_int(a1),
            ctypes.c_int(b0), ctypes.c_int(b1),
            cache_row_a.ctypes.data_as(ctypes.c_void_p),
            cache_col_a.ctypes.data_as(ctypes.c_void_p),
            cache_row_b.ctypes.data_as(ctypes.c_void_p),
            cache_col_b.ctypes.data_as(ctypes.c_void_p))
        cpu2[:] = log.timer_debug1('contract %d:%d,%d:%d' % (a0, a1, b0, b1), *cpu2)
    return contract
