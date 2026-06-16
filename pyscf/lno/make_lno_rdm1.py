#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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
# Author: Hong-Zhou Ye <hzyechem@gmail.com>
#

''' Make MP2 rdm1 of different flavors

    Args:
        eris :
            Provide access to the MO basis DF integral `ovL` through the following methods:
                get_occ_blk(i0,i1) -> ([i0:i1]v|L)
                get_vir_blk(a0,a1) -> (o[a0:a1]|L)
                xform_occ(u)       -> einsum('iaL,iI->IaL', ovL, u)
                xform_vir(u)       -> einsum('iaL,aA->iAL', ovL, u)
        orbact_data : tuple
            moeocc, moevir, uoccact, uviract = orbact_data
            where
                - `moeocc` and `moevir` are the MO energy for the occupied and virtual MOs
                  used to obtain `ovL`
                - `uoccact` and `uviract` are the overlap matrix between canonical and active
                  orbitals, i.e., uoccact[i,I] = <i|I> and uviract[a,A] = <a|A>.
        dm_type : str
            '1h'/'1p'/'2p' for occ and '1p'/'1h'/'2h' for vir.
'''
import sys
import numpy as np
from functools import reduce

from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__

DEBUG_BLKSIZE = getattr(__config__, 'lno_base_make_rdm1_DEBUG_BLKSIZE', False)


def make_lo_rdm1_occ(eris, moeocc, moevir, uocc, uvir, dm_type):
    if dm_type == '1h':
        dm = make_lo_rdm1_occ_1h(eris, moeocc, moevir, uocc)
    elif dm_type == '1p':
        dm = make_lo_rdm1_occ_1p(eris, moeocc, moevir, uvir)
    elif dm_type == '2p':
        dm = make_lo_rdm1_occ_2p(eris, moeocc, moevir, uvir)
    else:
        raise RuntimeError('Requested occ LNO type "%s" is unknown.' % dm_type)
    return dm

def make_lo_rdm1_vir(eris, moeocc, moevir, uocc, uvir, dm_type):
    if dm_type == '1p':
        dm = make_lo_rdm1_vir_1p(eris, moeocc, moevir, uvir)
    elif dm_type == '1h':
        dm = make_lo_rdm1_vir_1h(eris, moeocc, moevir, uocc)
    elif dm_type == '2h':
        dm = make_lo_rdm1_vir_2h(eris, moeocc, moevir, uocc)
    else:
        raise RuntimeError('Requested vir LNO type "%s" is unknown.' % dm_type)
    return dm

def _mp2_rdm1_occblksize(nocc, nvir, naux, n1, n2, M, dsize):
    r''' Estimate block size for the occupied index in MP2 rdm1 evaluation.

        Model:
            Assuming storing n1 copies of ([O]V|L) and n2 copies of ([O]V|[O]V), [O] is
            determined by solving a quadratic equation:
                (n2*V^2) * [O]^2 + (n1*V*X) * [O] - M = 0

        Args:
            nocc/nvir/naux : int
                Number of occ/vir/aux orbitals.
            n1/n2 : int
                Number of copies of tensors of size nvir*naux*occblksize and nvir^2*occblksize^2.
            M: float or int
                Available memory in terms how many numbers to store, i.e., mem_in_MB * 1e6/dsize.

        Return:
            occblksize (int)
            mem_peak (float) : peak memory (in MB)
    '''
    occblksize = max(1, min(nocc, int(np.floor((((n1*naux)**2 + 4*n2*M)**0.5 - n1*naux) /
                                                (2*n2*nvir)))))
    mem_peak = (occblksize * nvir*naux * n1 + occblksize**2 * nvir**2 * n2) * dsize/1e6
    return occblksize, mem_peak

def _mp2_rdm1_virblksize(nocc, nvir, naux, n1, n2, M, dsize):
    r''' Estimate block size for the virtual index in MP2 rdm1 evaluation.

        See `_mp2_rdm1_occblksize`.
    '''
    return _mp2_rdm1_occblksize(nvir, nocc, naux, n1, n2, M, dsize)

def make_full_rdm1(eris, moeocc, moevir, with_occ=True, with_vir=True):
    r''' Occ-occ and vir-vir blocks of MP2 density matrix

        Math:
            dm(i,j)
                = 2 * \sum_{kab} t2(ikab).conj() * ( 2*t2(jkab) - t2(jkba) )
            dm(a,b)
                = 2 * \sum_{ijc} t2(ijac) * ( 2*t2(ijbc) - t2(ijcb) ).conj()
    '''
    assert( with_occ or with_vir )

    nocc, nvir, naux = eris.nocc, eris.nvir, eris.naux
    dtype = eris.dtype
    dsize = eris.dsize

    # determine occblksize
    mem_avail = eris.max_memory - lib.current_memory()[0]
    M = mem_avail * 0.7 * 1e6/dsize
    occblksize, mem_peak = _mp2_rdm1_occblksize(nocc,nvir,naux, 2, 3, M, dsize)
    if DEBUG_BLKSIZE: occblksize = max(1,nocc//2)
    logger.debug1(eris, 'make_full_rdm1  :  nocc = %d  nvir = %d  naux = %d  occblksize = %d  '
                  'peak mem = %.2f MB', nocc, nvir, naux, occblksize, mem_peak)

    eov = moeocc[:,None] - moevir

    dmoo = np.zeros((nocc,nocc), dtype=dtype) if with_occ else None
    dmvv = np.zeros((nvir,nvir), dtype=dtype) if with_vir else None
    for ibatch,(i0,i1) in enumerate(lib.prange(0,nocc,occblksize)):
        ivL = eris.get_occ_blk(i0,i1)
        eiv = eov[i0:i1]
        for jbatch,(j0,j1) in enumerate(lib.prange(0,nocc,occblksize)):
            if jbatch == ibatch:
                jvL = ivL
                ejv = eiv
            else:
                jvL = eris.get_occ_blk(j0,j1)
                ejv = eov[j0:j1]
            denom = lib.direct_sum('ia+jb->ijab', eiv, ejv)
            t2ijvv = np.conj(lib.einsum('iax,jbx->ijab', ivL, jvL)) / denom
            jvL = None
            denom = None
            if with_occ:
                dmoo[i0:i1,j0:j1]  = 4*lib.einsum('ikab,jkab->ij', np.conj(t2ijvv), t2ijvv)
                dmoo[i0:i1,j0:j1] -= 2*lib.einsum('ikab,jkba->ij', np.conj(t2ijvv), t2ijvv)
            if with_vir:
                dmvv  = 4*lib.einsum('ijac,ijbc->ab', t2ijvv, np.conj(t2ijvv))
                dmvv -= 2*lib.einsum('ijac,ijcb->ab', t2ijvv, np.conj(t2ijvv))
            t2ijvv = None
        ivL = None

    return dmoo, dmvv

def make_full_rdm1_occ(eris, moeocc, moevir):
    r''' Occupied MP2 density matrix

        Math:
            dm(i,j)
                = 2 * \sum_{kab} t2(ikab) ( 2*t2(jkab) - t2(jkba) )
    '''
    nocc, nvir, naux = eris.nocc, eris.nvir, eris.naux
    dtype = eris.dtype
    dsize = eris.dsize

    # determine occblksize
    mem_avail = eris.max_memory - lib.current_memory()[0]
    M = mem_avail * 0.7 * 1e6/dsize
    occblksize, mem_peak = _mp2_rdm1_occblksize(nocc,nvir,naux, 2, 3, M, dsize)
    if DEBUG_BLKSIZE: occblksize = max(1,nocc//2)
    logger.debug1(eris, 'make_full_rdm1_occ  :  nocc = %d  nvir = %d  naux = %d  occblksize = %d  '
                  'peak mem = %.2f MB', nocc, nvir, naux, occblksize, mem_peak)

    eov = moeocc[:,None] - moevir

    dm = np.zeros((nocc,nocc), dtype=dtype)
    for ibatch,(i0,i1) in enumerate(lib.prange(0,nocc,occblksize)):
        ivL = eris.get_occ_blk(i0,i1)
        eiv = eov[i0:i1]
        for jbatch,(j0,j1) in enumerate(lib.prange(0,nocc,occblksize)):
            if jbatch == ibatch:
                jvL = ivL
                ejv = eiv
            else:
                jvL = eris.get_occ_blk(j0,j1)
                ejv = eov[j0:j1]
            denom = lib.direct_sum('ia+jb->ijab', eiv, ejv)
            t2ijvv = np.conj(lib.einsum('iax,jbx->ijab', ivL, jvL)) / denom
            jvL = None
            denom = None
            dm[i0:i1,j0:j1]  = 4*lib.einsum('ikab,jkab->ij', np.conj(t2ijvv), t2ijvv)
            dm[i0:i1,j0:j1] -= 2*lib.einsum('ikab,jkba->ij', np.conj(t2ijvv), t2ijvv)
            t2ijvv = None
        ivL = None

    return dm

def make_full_rdm1_vir(eris, moeocc, moevir):
    r''' Virtual MP2 density matrix

        Math:
            dm(a,b)
                = 2 * \sum_{ijc} t2(ijac) ( 2*t2(ijbc) - t2(ijcb) )
    '''
    nocc, nvir, naux = eris.nocc, eris.nvir, eris.naux
    dtype = eris.dtype
    dsize = eris.dsize

    # determine occblksize
    mem_avail = eris.max_memory - lib.current_memory()[0]
    M = mem_avail * 0.7 * 1e6/dsize
    occblksize, mem_peak = _mp2_rdm1_occblksize(nocc,nvir,naux, 2, 3, M, dsize)
    if DEBUG_BLKSIZE: occblksize = max(1,nocc//2)
    logger.debug1(eris, 'make_full_rdm1_vir  :  nocc = %d  nvir = %d  naux = %d  occblksize = %d  '
                  'peak mem = %.2f MB', nocc, nvir, naux, occblksize, mem_peak)

    eov = moeocc[:,None] - moevir

    dm = np.zeros((nvir,nvir), dtype=dtype)
    for ibatch,(i0,i1) in enumerate(lib.prange(0,nocc,occblksize)):
        ivL = eris.get_occ_blk(i0,i1)
        eiv = eov[i0:i1]
        for jbatch,(j0,j1) in enumerate(lib.prange(0,nocc,occblksize)):
            if jbatch == ibatch:
                jvL = ivL
                ejv = eiv
            else:
                jvL = eris.get_occ_blk(j0,j1)
                ejv = eov[j0:j1]
            eijvv = lib.direct_sum('ia+jb->ijab', eiv, ejv)
            t2ijvv = np.conj(lib.einsum('iax,jbx->ijab', ivL, jvL)) / eijvv
            jvL = None
            eijvv = None

            dm  = 4*lib.einsum('ijac,ijbc->ab', t2ijvv, np.conj(t2ijvv))
            dm -= 2*lib.einsum('ijac,ijcb->ab', t2ijvv, np.conj(t2ijvv))

            t2ijvv = None
        ivL = None

    return dm

def make_lo_rdm1_occ_1h(eris, moeocc, moevir, u):
    r''' Occupied MP2 density matrix with one localized hole

        Math:
            dm(i,j)
                = 2 * \sum_{k'ab} t2(ik'ab) ( 2*t2(jk'ab) - t2(jk'ba) )

        Args:
            eris : ERI object
                Provides `ovL` in the canonical MOs.
            u : np.ndarray
                Overlap between the canonical and localized occupied orbitals, i.e.,
                    u(i,i') = <i|i'>
    '''
    nocc, nvir, naux = eris.nocc, eris.nvir, eris.naux
    dtype = eris.dtype
    dsize = eris.dsize
    nOcc = u.shape[1]

    # determine occblksize
    mem_avail = eris.max_memory - lib.current_memory()[0]
    M = mem_avail * 0.7 * 1e6/dsize
    occblksize, mem_peak = _mp2_rdm1_occblksize(nocc,nvir,naux, 3, 3, M, dsize)
    if DEBUG_BLKSIZE: occblksize = max(1,nocc//2)
    logger.debug1(eris, 'make_lo_rdm1_occ_1h :  nocc = %d  nvir = %d  nOcc = %d  naux = %d  '
                  'occblksize = %d  peak mem = %.2f MB',
                  nocc, nvir, nOcc, naux, occblksize, mem_peak)

    moeOcc, u = subspace_eigh(np.diag(moeocc), u)
    eov = moeocc[:,None] - moevir
    eOv = moeOcc[:,None] - moevir

    dm = np.zeros((nocc,nocc), dtype=dtype)
    for Kbatch,(K0,K1) in enumerate(lib.prange(0,nOcc,occblksize)):
        KvL = eris.xform_occ(u[:,K0:K1])
        eKv = eOv[K0:K1]
        for ibatch,(i0,i1) in enumerate(lib.prange(0,nocc,occblksize)):
            ivL = eris.get_occ_blk(i0,i1)
            eiv = eov[i0:i1]
            eiKvv = lib.direct_sum('ia+Kb->iKab', eiv, eKv)
            t2iKvv = np.conj(lib.einsum('iax,Kbx->iKab', ivL, KvL)) / eiKvv
            ivL = None
            eiKvv = None
            for jbatch,(j0,j1) in enumerate(lib.prange(0,nocc,occblksize)):
                if jbatch == ibatch:
                    t2jKvv = t2iKvv
                else:
                    jvL = eris.get_occ_blk(j0,j1)
                    ejv = eov[j0:j1]
                    ejKvv = lib.direct_sum('ia+Kb->iKab', ejv, eKv)
                    t2jKvv = np.conj(lib.einsum('iax,Kbx->iKab', jvL, KvL)) / ejKvv
                    jvL = None
                    ejKvv = None

                dm[i0:i1,j0:j1] -= 4 * lib.einsum('iKab,jKab->ij', np.conj(t2iKvv), t2jKvv)
                dm[i0:i1,j0:j1] += 2 * lib.einsum('iKab,jKba->ij', np.conj(t2iKvv), t2jKvv)

                t2jKvv = None
            t2iKvv = None
        KvL = None

    return dm

def make_lo_rdm1_occ_1p(eris, moeocc, moevir, u):
    r''' Occupied MP2 density matrix with one localized particle

        Math:
            dm(i,j)
                = 2 * \sum_{k'ab} t2(ik'ab) ( 2*t2(jk'ab) - t2(jk'ba) )

        Args:
            eris : ERI object
                Provides `ovL` in the canonical MOs.
            u : np.ndarray
                Overlap between the canonical and localized virtual orbitals, i.e.,
                    u(a,a') = <a|a'>
    '''
    nocc, nvir, naux = eris.nocc, eris.nvir, eris.naux
    dtype = eris.dtype
    dsize = eris.dsize
    nVir = u.shape[1]

    # determine occblksize
    mem_avail = eris.max_memory - lib.current_memory()[0]
    M = mem_avail * 0.7 * 1e6/dsize
    virblksize, mem_peak = _mp2_rdm1_virblksize(nocc,nvir,naux, 2, 3, M, dsize)
    if DEBUG_BLKSIZE: virblksize = max(1,nvir//2)
    logger.debug1(eris, 'make_lo_rdm1_occ_1p :  nocc = %d  nvir = %d  nVir = %d  naux = %d  '
                  'virblksize = %d  peak mem = %.2f MB',
                  nocc, nvir, nVir, naux, virblksize, mem_peak)

    moeVir, u = subspace_eigh(np.diag(moevir), u)
    eov = moeocc[:,None] - moevir
    eoV = moeocc[:,None] - moeVir

    dm = np.zeros((nocc,nocc), dtype=dtype)
    for Abatch,(A0,A1) in enumerate(lib.prange(0,nVir,virblksize)):
        oAL = eris.xform_vir(u[:,A0:A1])
        eoA = eoV[:,A0:A1]
        for bbatch,(b0,b1) in enumerate(lib.prange(0,nvir,virblksize)):
            obL = eris.get_vir_blk(b0,b1)
            eob = eov[:,b0:b1]

            eooAb = lib.direct_sum('iA+jb->ijAb', eoA, eob)
            t2ooAb = np.conj(lib.einsum('iAx,jbx->ijAb', oAL, obL)) / eooAb
            obL = None
            eooAb = None

            dm -= 2 * lib.einsum('ikAb,jkAb->ij', np.conj(t2ooAb), t2ooAb)
            dm +=     lib.einsum('ikAb,kjAb->ij', np.conj(t2ooAb), t2ooAb)
            dm +=     lib.einsum('kiAb,jkAb->ij', np.conj(t2ooAb), t2ooAb)
            dm -= 2 * lib.einsum('kiAb,kjAb->ij', np.conj(t2ooAb), t2ooAb)

            t2ooAb = None
        oAL = None

    return dm

def make_lo_rdm1_occ_2p(eris, moeocc, moevir, u):
    r''' Occupied MP2 density matrix with two localized particles

        Math:
            dm(i,j)
                = 2 * \sum_{ka'b'} t2(ika'b') ( 2*t2(jka'b') - t2(jkb'a') )

        Args:
            eris : ERI object
                Provides `ovL` in the canonical MOs.
            u : np.ndarray
                Overlap between the canonical and localized virtual orbitals, i.e.,
                    u(a,a') = <a|a'>
    '''
    nocc, nvir, naux = eris.nocc, eris.nvir, eris.naux
    dtype = eris.dtype
    dsize = eris.dsize
    nVir = u.shape[1]

    # determine Occblksize
    mem_avail = eris.max_memory - lib.current_memory()[0]
    M = mem_avail * 0.7 * 1e6/dsize
    Virblksize, mem_peak = _mp2_rdm1_virblksize(nocc,nVir,naux, 2, 3, M, dsize)
    if DEBUG_BLKSIZE: Virblksize = max(1,nVir//2)
    logger.debug1(eris, 'make_lo_rdm1_occ_2p:  nocc = %d  nvir = %d  nVir = %d  naux = %d  '
                  'Virblksize = %d  peak mem = %.2f MB',
                  nocc, nvir, nVir, naux, Virblksize, mem_peak)

    moeVir, u = subspace_eigh(np.diag(moevir), u)
    eoV = moeocc[:,None] - moeVir

    dm = np.zeros((nocc,nocc), dtype=dtype)
    for Abatch,(A0,A1) in enumerate(lib.prange(0,nVir,Virblksize)):
        oAL = eris.xform_vir(u[:,A0:A1])
        eoA = eoV[:,A0:A1]
        for Bbatch,(B0,B1) in enumerate(lib.prange(0,nVir,Virblksize)):
            if Bbatch == Abatch:
                oBL = oAL
                eoB = eoA
            else:
                oBL = eris.xform_vir(u[:,B0:B1])
                eoB = eoV[:,B0:B1]
            eooAB = lib.direct_sum('iA+jB->ijAB', eoA, eoB)
            t2ooAB = np.conj(lib.einsum('iAx,jBx->ijAB', oAL, oBL)) / eooAB
            oBL = None
            eooAB = None

            dm -= 4 * lib.einsum('ikAB,jkAB->ij', np.conj(t2ooAB), t2ooAB)
            dm += 2 * lib.einsum('ikAB,kjAB->ij', np.conj(t2ooAB), t2ooAB)

            t2ooAB = None
        oAL = None

    return dm

def make_lo_rdm1_vir_1p(eris, moeocc, moevir, u):
    r''' Virtual MP2 density matrix with one localized particle

        Math:
            dm(a,b)
                = \sum_{ijc'} 2 * t2(ijac') * ( 2 * t2(ijbc') - t2(jibc') )

        Args:
            eris : ERI object
                Provides `ovL` in the canonical MOs.
            u : np.ndarray
                Overlap between the canonical and localized occupied orbitals, i.e.,
                    u(a,a') = <a|a'>
    '''
    nocc, nvir, naux = eris.nocc, eris.nvir, eris.naux
    dtype = eris.dtype
    dsize = eris.dsize
    nVir = u.shape[1]

    # determine occblksize
    mem_avail = eris.max_memory - lib.current_memory()[0]
    M = mem_avail * 0.7 * 1e6/dsize
    virblksize, mem_peak = _mp2_rdm1_virblksize(nocc,nvir,naux, 3, 3, M, dsize)
    if DEBUG_BLKSIZE: virblksize = max(1,nvir//2)
    logger.debug1(eris, 'make_lo_rdm1_vir_1p :  nocc = %d  nvir = %d  nVir = %d  naux = %d  '
                  'virblksize = %d  peak mem = %.2f MB',
                  nocc, nvir, nVir, naux, virblksize, mem_peak)

    moeVir, u = subspace_eigh(np.diag(moevir), u)
    eov = moeocc[:,None] - moevir
    eoV = moeocc[:,None] - moeVir

    # TODO: can we batch over occ index?
    dm = np.zeros((nvir,nvir), dtype=dtype)
    for Abatch,(A0,A1) in enumerate(lib.prange(0,nVir,virblksize)):
        oAL = eris.xform_vir(u[:,A0:A1])
        eoA = eoV[:,A0:A1]
        for abatch,(a0,a1) in enumerate(lib.prange(0,nvir,virblksize)):
            oaL = eris.get_vir_blk(a0,a1)
            eoa = eov[:,a0:a1]
            eooAa = lib.direct_sum('iA+jb->ijAb', eoA, eoa)
            t2ooAa = np.conj(lib.einsum('iAx,jbx->ijAb', oAL, oaL)) / eooAa
            oaL = None
            eooAa = None
            for bbatch,(b0,b1) in enumerate(lib.prange(0,nvir,virblksize)):
                if abatch == bbatch:
                    t2ooAb = t2ooAa
                else:
                    obL = eris.get_vir_blk(b0,b1)
                    eob = eov[:,b0:b1]
                    eooAb = lib.direct_sum('iA+jb->ijAb', eoA, eob)
                    t2ooAb = np.conj(lib.einsum('iAx,jbx->ijAb', oAL, obL)) / eooAb
                    obL = None
                    eooAb = None

                dm[a0:a1,b0:b1] += 4 * lib.einsum('ijAa,ijAb->ab', t2ooAa, np.conj(t2ooAb))
                dm[a0:a1,b0:b1] -= 2 * lib.einsum('ijAa,jiAb->ab', t2ooAa, np.conj(t2ooAb))

                t2ooAb = None
            t2ooAa = None
        oAL = None

    return dm

def make_lo_rdm1_vir_1h(eris, moeocc, moevir, u):
    r''' Occupied MP2 density matrix with one localized hole

        Math:
            dm(a,b)
                = \sum_{i'jc} 2 * t2(i'jac) * t2(i'jbc) + 2 * t2(i'jca) * t2(i'jcb)
                            - t2(i'jac) * t2(i'jcb) - t2(i'jca) * t2(i'jbc)

        Args:
            eris : ERI object
                Provides `ovL` in the canonical MOs.
            u : np.ndarray
                Overlap between the canonical and localized occupied orbitals, i.e.,
                    u(i,i') = <i|i'>
    '''
    nocc, nvir, naux = eris.nocc, eris.nvir, eris.naux
    dtype = eris.dtype
    dsize = eris.dsize
    nOcc = u.shape[1]

    # determine Occblksize
    mem_avail = eris.max_memory - lib.current_memory()[0]
    M = mem_avail * 0.7 * 1e6/dsize
    occblksize, mem_peak = _mp2_rdm1_occblksize(nocc,nvir,naux, 2, 3, M, dsize)
    if DEBUG_BLKSIZE: occblksize = max(1,nocc//2)
    logger.debug1(eris, 'make_lo_rdm1_vir_1h :  nocc = %d  nvir = %d  nOcc = %d  naux = %d  '
                  'occblksize = %d  peak mem = %.2f MB',
                  nocc, nvir, nOcc, naux, occblksize, mem_peak)

    moeOcc, u = subspace_eigh(np.diag(moeocc), u)
    eOv = moeOcc[:,None] - moevir
    eov = moeocc[:,None] - moevir

    dm = np.zeros((nvir,nvir), dtype=dtype)
    for Ibatch,(I0,I1) in enumerate(lib.prange(0,nOcc,occblksize)):
        IvL = eris.xform_occ(u[:,I0:I1])
        eIv = eOv[I0:I1]
        for jbatch,(j0,j1) in enumerate(lib.prange(0,nocc,occblksize)):
            jvL = eris.get_occ_blk(j0,j1)
            ejv = eov[j0:j1]

            eIjvv = lib.direct_sum('Ia+jb->Ijab', eIv, ejv)
            t2Ijvv = np.conj(lib.einsum('Iax,jbx->Ijab', IvL, jvL)) / eIjvv
            jvL = None
            eIjvv = None

            dm += 2 * lib.einsum('Ijac,Ijbc->ab', t2Ijvv, np.conj(t2Ijvv))
            dm -=     lib.einsum('Ijac,Ijcb->ab', t2Ijvv, np.conj(t2Ijvv))
            dm -=     lib.einsum('Ijca,Ijbc->ab', t2Ijvv, np.conj(t2Ijvv))
            dm += 2 * lib.einsum('Ijca,Ijcb->ab', t2Ijvv, np.conj(t2Ijvv))

            t2Ijvv = None
        IvL = None

    return dm

def make_lo_rdm1_vir_2h(eris, moeocc, moevir, u):
    r''' Occupied MP2 density matrix with two localized holes

        Math:
            dm(a,b)
                = 2 * \sum_{i'j'c} t2(i'j'ac) ( 2*t2(i'j'bc) - t2(i'j'cb) )

        Args:
            eris : ERI object
                Provides `ovL` in the canonical MOs.
            u : np.ndarray
                Overlap between the canonical and localized occupied orbitals, i.e.,
                    u(i,i') = <i|i'>
    '''
    nocc, nvir, naux = eris.nocc, eris.nvir, eris.naux
    dtype = eris.dtype
    dsize = eris.dsize
    nOcc = u.shape[1]

    # determine Occblksize
    mem_avail = eris.max_memory - lib.current_memory()[0]
    M = mem_avail * 0.7 * 1e6/dsize
    Occblksize, mem_peak = _mp2_rdm1_occblksize(nOcc,nvir,naux, 2, 3, M, dsize)
    if DEBUG_BLKSIZE: Occblksize = max(1,nOcc//2)
    logger.debug1(eris, 'make_lo_rdm1_vir_2h:  nocc = %d  nvir = %d  nOcc = %d  naux = %d  '
                  'Occblksize = %d  peak mem = %.2f MB',
                  nocc, nvir, nOcc, naux, Occblksize, mem_peak)

    moeOcc, u = subspace_eigh(np.diag(moeocc), u)
    eOv = moeOcc[:,None] - moevir

    dm = np.zeros((nvir,nvir), dtype=dtype)
    for Ibatch,(I0,I1) in enumerate(lib.prange(0,nOcc,Occblksize)):
        IvL = eris.xform_occ(u[:,I0:I1])
        eIv = eOv[I0:I1]
        for Jbatch,(J0,J1) in enumerate(lib.prange(0,nOcc,Occblksize)):
            if Jbatch == Ibatch:
                JvL = IvL
                eJv = eIv
            else:
                JvL = eris.xform_occ(u[:,J0:J1])
                eJv = eOv[J0:J1]
            eIJvv = lib.direct_sum('Ia+Jb->IJab', eIv, eJv)
            t2IJvv = np.conj(lib.einsum('Iax,Jbx->IJab', IvL, JvL)) / eIJvv
            JvL = None
            eIJvv = None

            dm += 4 * lib.einsum('IJac,IJbc->ab', t2IJvv, np.conj(t2IJvv))
            dm -= 2 * lib.einsum('IJac,IJcb->ab', t2IJvv, np.conj(t2IJvv))

            t2IJvv = None
        IvL = None

    return dm

def subspace_eigh(fock, orb):
    f = reduce(np.dot, (orb.T.conj(), fock, orb))
    if orb.shape[1] == 1:
        moe = np.array([f[0,0]])
    else:
        moe, u = np.linalg.eigh(f)
        orb = np.dot(orb, u)
    return moe, orb
