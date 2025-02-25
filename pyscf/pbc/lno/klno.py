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


''' Generic framework for k-point local natural orbital (KLNO)-based methods. This code
    can be used to implement LNO-based local correlation approximation to many correlated
    wavefunction methods with periodic boundary condition. See `klnoccsd.py` for the
    implementation of KLNO-CCSD as an example.

    - Original publication of molecular LNO by Kállay and co-workers:
        Rolik and Kállay, J. Chem. Phys. 135, 104111 (2011)

    - Publication for periodic KLNO by Ye and Berkelbach:
        Ye and Berkelbach, J. Chem. Theory Comput. 2024, 20, 20, 8948–8959
'''


import sys
import numpy as np
import h5py

from pyscf.lib import logger
from pyscf import lib
from pyscf.pbc.df.df import _load3c
from pyscf.pbc.lib.kpts_helper import gamma_point
from pyscf import __config__

from pyscf.lno import LNO
from pyscf.pbc.lno.tools import K2SDF, k2s_scf
from pyscf.pbc.lno.make_lno_rdm1 import make_lo_rdm1_occ, make_lo_rdm1_vir

einsum = lib.einsum

DEBUG_BLKSIZE = getattr(__config__, 'lno_base_klno_base_DEBUG_BLKSIZE', False)


class KLNO(LNO):
    r''' Base class for LNO-based methods with k-point mean-field reference.

    This base class provides common functions for constructing KLNO subspace.
    Specific LNO-based methods (e.g., KLNO-CCSD, KLNO-CCSD(T)) can be implemented as
    derived classes from this base class with appropriately defined method
    `impurity_solve`.

    Input:
        kmf (PySCF KSCF object):
            KSCF mean-field object.
        lo_coeff (np.ndarray):
            Supercell AO coefficient matrix of LOs. Must span occupied space.
        frag_lolist (nested list):
            Fragment assignment in terms of LO index. E.g., [[0,2], [1], ...] means
            frag 1 consists of LO 0 and 2, frag 2 consists of LO 1, etc.
        lno_type (len-2 list):
            lno_type = [occ_lno_type, vir_lno_type], where 'occ_lno_type' can be
            '1h', '1p', or '2p' and 'vir_lno_type' can be '1p', '1h', '2h'.
            Default is ['1h','1h'].
        lno_thresh (float of len-2 list):
            Thresholds for LNO truncation. Use a len-2 list to specify thresh for
            occ and vir separately. Default is [1e-5,1e-6].
        frozen (int or list):
            Same as the `frozen` attr in MP2/CCSD etc. modules.
    '''

    def __init__(self, kmf, lo_coeff, frag_lolist, lno_type=None, lno_thresh=None, frozen=None,
                 mf=None):
        if mf is None: mf = k2s_scf(kmf)
        LNO.__init__(self, mf, lo_coeff, frag_lolist, lno_type=lno_type,
                     lno_thresh=lno_thresh, frozen=frozen)
        self._kscf = kmf
        self.with_df = kmf.with_df
        self.unit_cell = kmf.cell
        self.kpts = kmf.kpts

    def ao2mo(self):
        log = logger.new_logger(self)

        if self.with_df is None:
            log.error('DF is not found. Rerun KSCF with DF.')
            raise NotImplementedError
        else:
            cput0 = (logger.process_clock(), logger.perf_counter())
            orbocc, orbvir = self.split_mo_coeff()[1:3]
            dsize = 16  # Lov is always complex (but the eri from contracting Lov may be real)
            nocc = orbocc.shape[1]
            nvir = orbvir.shape[1]
            # FIXME: more accurate mem estimate
            mem_now = self.max_memory - lib.current_memory()[0]
            nkpts = len(self.kpts)
            naux = self.with_df.get_naoaux()
            nk = nkpts//2+nkpts%2 if gamma_point(self.kpts[0]) and np.isrealobj(orbocc) else nkpts
            mem_df = nk*nocc*nvir*naux*dsize/1024**2.
            log.debug('ao2mo est mem= %.2f MB  avail mem= %.2f MB', mem_df, mem_now)
            if ( (self._ovL_to_save is not None) or (self._ovL is not None) or
                 self.force_outcore_ao2mo or (mem_df > mem_now*0.5) ):
                eris = _KLNODFOUTCOREERIS(self.with_df, orbocc, orbvir, self.max_memory,
                                          ovL=self._ovL, ovL_to_save=self._ovL_to_save,
                                          verbose=self.verbose, stdout=self.stdout)
            else:
                eris = _KLNODFINCOREERIS(self.with_df, orbocc, orbvir, self.max_memory,
                                         verbose=self.verbose, stdout=self.stdout)
            eris.build()
            log.timer('Integral xform   ', *cput0)

            return eris

    def make_lo_rdm1_occ(self, eris, moeocc, moevir, uocc_loc, uvir_loc, occ_lno_type):
        return make_lo_rdm1_occ(eris, moeocc, moevir, uocc_loc, uvir_loc, occ_lno_type)

    def make_lo_rdm1_vir(self, eris, moeocc, moevir, uocc_loc, uvir_loc, vir_lno_type):
        return make_lo_rdm1_vir(eris, moeocc, moevir, uocc_loc, uvir_loc, vir_lno_type)


def _KLNODFINCOREERIS(with_df, orbocc, orbvir, max_memory, verbose=None, stdout=None):
    if gamma_point(with_df.kpts[0]) and np.isrealobj(orbocc.dtype):
        _ERIS = _KLNODFINCOREERIS_REAL
    else:
        _ERIS = _KLNODFINCOREERIS_COMPLEX
    return _ERIS(with_df, orbocc, orbvir, max_memory, verbose, stdout)

def _KLNODFOUTCOREERIS(with_df, orbocc, orbvir, max_memory, ovL=None, ovL_to_save=None,
                       verbose=None, stdout=None):
    if gamma_point(with_df.kpts[0]) and np.isrealobj(orbocc.dtype):
        _ERIS = _KLNODFOUTCOREERIS_REAL
    else:
        _ERIS = _KLNODFOUTCOREERIS_COMPLEX
    return _ERIS(with_df, orbocc, orbvir, max_memory, ovL, ovL_to_save, verbose, stdout)


''' DF ERI for real orbitals
'''
class _KLNODFINCOREERIS_REAL(K2SDF):
    def __init__(self, with_df, orbocc, orbvir, max_memory, verbose=None, stdout=None):
        K2SDF.__init__(self, with_df)
        self.orbocc = orbocc
        self.orbvir = orbvir

        self.max_memory = max_memory
        self.verbose = verbose
        self.stdout = stdout

        self.dtype = np.float64
        self.dsize = 8
        self.dtype_eri, self.dsize_eri = self.get_eri_dtype_dsize(orbocc, orbvir)

        self.ovLR = None
        self.ovLI = None

    @property
    def nocc(self):
        return self.orbocc.shape[1]
    @property
    def nvir(self):
        return self.orbvir.shape[1]

    def build(self):
        log = logger.new_logger(self)
        self.ovLR, self.ovLI = _init_mp_df_eris_real(self, self.orbocc, self.orbvir,
                                                     self.max_memory, ovLR=self.ovLR, ovLI=self.ovLI, log=log)

    def get_occ_blk(self, i0,i1):
        return np.asarray(self.ovLR[i0:i1]), np.asarray(self.ovLI[i0:i1])

    def get_vir_blk(self, a0,a1, real_and_imag=False):
        return np.asarray(self.ovLR[:,a0:a1], order='C'), np.asarray(self.ovLI[:,a0:a1], order='C')

    def xform_occ(self, u):
        assert( u.dtype == np.float64 )
        nocc, nvir, Naux = self.nocc, self.nvir, self.Naux_ibz
        nOcc = u.shape[1]
        M = (self.max_memory - lib.current_memory()[0])*1e6 / self.dsize
        occblksize = min(nocc, max(1, int(np.floor(M*0.5/(nvir*Naux) - nOcc))))
        if DEBUG_BLKSIZE: occblksize = max(1,nocc//2)
        OvLR = np.empty((nOcc,nvir,Naux), dtype=np.float64)
        OvLI = np.empty((nOcc,nvir,Naux), dtype=np.float64)
        for iblk,(i0,i1) in enumerate(lib.prange(0,nocc,occblksize)):
            ovLR, ovLI = self.get_occ_blk(i0,i1)
            if iblk == 0:
                OvLR[:]  = lib.einsum('iax,iI->Iax', ovLR, u[i0:i1])
                OvLI[:]  = lib.einsum('iax,iI->Iax', ovLI, u[i0:i1])
            else:
                OvLR[:] += lib.einsum('iax,iI->Iax', ovLR, u[i0:i1])
                OvLI[:] += lib.einsum('iax,iI->Iax', ovLI, u[i0:i1])
            ovLR = ovLI = None
        return OvLR, OvLI

    def xform_vir(self, u):
        assert( u.dtype == np.float64 )
        nocc, nvir, Naux = self.nocc, self.nvir, self.Naux_ibz
        nVir = u.shape[1]
        M = (self.max_memory - lib.current_memory()[0])*1e6 / self.dsize
        occblksize = min(nocc, max(1, int(np.floor(M*0.5/(nvir*Naux) - nocc*nVir/float(nvir)))))
        if DEBUG_BLKSIZE: occblksize = max(1,nocc//2)
        oVLR = np.empty((nocc,nVir,Naux), dtype=np.float64)
        oVLI = np.empty((nocc,nVir,Naux), dtype=np.float64)
        for i0,i1 in lib.prange(0,nocc,occblksize):
            ovLR, ovLI = self.get_occ_blk(i0,i1)
            oVLR[i0:i1] = lib.einsum('iax,aA->iAx', ovLR, u)
            oVLI[i0:i1] = lib.einsum('iax,aA->iAx', ovLI, u)
            ovLR = ovLI = None
        return oVLR, oVLI


class _KLNODFOUTCOREERIS_REAL(_KLNODFINCOREERIS_REAL):
    def __init__(self, with_df, orbocc, orbvir, max_memory, ovL=None, ovL_to_save=None,
                 verbose=None, stdout=None):
        _KLNODFINCOREERIS_REAL.__init__(self, with_df, orbocc, orbvir, max_memory, verbose, stdout)

        self._ovL = ovL
        self._ovL_to_save = ovL_to_save

    def build(self):
        log = logger.new_logger(self)
        ovL_shape = (self.nocc,self.nvir,self.Naux_ibz)
        ovL_dtype = self.dtype
        if self._ovL is None:
            if isinstance(self._ovL_to_save, str):
                self.feri = h5py.File(self._ovL_to_save, 'w')
            else:
                self.feri = lib.H5TmpFile()
            log.info('ovL is saved to %s', self.feri.filename)
            # TODO: determine a chunks size
            self.ovLR = self.feri.create_dataset('ovLR', ovL_shape, ovL_dtype)
            self.ovLI = self.feri.create_dataset('ovLI', ovL_shape, ovL_dtype)
            _init_mp_df_eris_real(self, self.orbocc, self.orbvir, self.max_memory,
                                  ovLR=self.ovLR, ovLI=self.ovLI, log=log)
        elif isinstance(self._ovL, str):
            self.feri = h5py.File(self._ovL, 'r')
            log.info('ovL is read from %s', self.feri.filename)
            for key in ['ovLR', 'ovLI']:
                assert( key in self.feri )
                assert( self.feri[key].shape == ovL_shape )
                assert( self.feri[key].dtype == ovL_dtype )
                setattr(self, key, self.feri[key])
        else:
            raise RuntimeError

def _init_mp_df_eris_real(k2sdf, orbocc, orbvir, max_memory, ovLR=None, ovLI=None, log=None):
    r''' ovL[q,I,A,L] := (I A | L,q)
                      = \sum_{k} \sum_{mu,nu} (mu,k nu,k-q | L,q) C(mu,k I).conj() C(nu,k-q A)
    '''
    from pyscf.ao2mo import _ao2mo

    if log is None: log = logger.Logger(sys.stdout, 3)

    korbocc = k2sdf.s2k_mo_coeff(orbocc)
    korbvir = k2sdf.s2k_mo_coeff(orbvir)
    naux_by_q = k2sdf.naux_by_q
    naux = k2sdf.naux
    Naux = k2sdf.Naux_ibz
    nqpts = len(k2sdf.qpts_ibz)

    REAL = np.float64
    COMPLEX = np.complex128
    dsize = 8

    nao, nocc = korbocc[0].shape
    nvir = korbvir[0].shape[1]

    if ovLR is None:
        ovLR = np.empty((nocc,nvir,Naux), dtype=REAL)
        ovLI = np.empty((nocc,nvir,Naux), dtype=REAL)

    cput1 = (logger.process_clock(), logger.perf_counter())

    mem_avail  = max_memory - lib.current_memory()[0]

    if isinstance(ovLR, np.ndarray):
        mem_avail -= Naux*nocc*nvir*2 * dsize/1e6   # subtract mem for holding ovLR/I incore
        mode = 'incore'
    else:
        mode = 'outcore'

    # batching aux (OV*3 + Nao_pair) * [X] = M
    mem_auxblk = (nao**2+nocc*nvir*3) * dsize/1e6
    aux_blksize = min(naux, max(1, int(np.floor(mem_avail*0.7 / mem_auxblk))))
    if DEBUG_BLKSIZE: aux_blksize = max(1,naux//2)
    log.debug('aux blksize for %s ao2mo: %d/%d', mode, aux_blksize, naux)
    buf = np.empty(aux_blksize*nocc*nvir, dtype=COMPLEX)
    bufR = np.empty(aux_blksize*nocc*nvir, dtype=REAL)
    bufI = np.empty(aux_blksize*nocc*nvir, dtype=REAL)

    for qi,q in enumerate(k2sdf.ibz2bz):
        nauxq = naux_by_q[q]
        if nauxq < naux:
            ovLR[:,:,naux*qi+nauxq:naux*(qi+1)] = 0
            ovLI[:,:,naux*qi+nauxq:naux*(qi+1)] = 0
        for p0,p1 in lib.prange(0, nauxq, aux_blksize):
            auxslice = (p0,p1)
            dp = p1 - p0
            LovR = np.ndarray((dp,nocc,nvir), dtype=REAL, buffer=bufR)
            LovI = np.ndarray((dp,nocc,nvir), dtype=REAL, buffer=bufI)
            LovR.fill(0)
            LovI.fill(0)
            for (ki,kj),LpqR,LpqI in k2sdf.loop_ao2mo(q, orbocc, orbvir, buf=buf,
                                                      real_and_imag=True, auxslice=auxslice):
                LovR += LpqR.reshape(dp,nocc,nvir)
                LovI += LpqI.reshape(dp,nocc,nvir)
                LpqR = LpqI = None
            w = k2sdf.qpts_ibz_weights[qi]
            LovR *= w
            LovI *= w
            b0 = naux*qi + p0
            b1 = b0 + dp
            ovLR[:,:,b0:b1] = LovR.transpose(1,2,0)
            ovLI[:,:,b0:b1] = LovI.transpose(1,2,0)
            LovR = LovI = None
        cput1 = log.timer('ao2mo for qidx %d/%d'%(qi+1,nqpts), *cput1)

    buf = bufR = bufI = None

    return ovLR, ovLI


''' DF ERI for complex orbitals
'''
class _KLNODFINCOREERIS_COMPLEX(K2SDF):
    def __init__(self, with_df, orbocc, orbvir, max_memory, verbose=None, stdout=None):
        K2SDF.__init__(self, with_df)
        self.orbocc = orbocc
        self.orbvir = orbvir

        self.max_memory = max_memory
        self.verbose = verbose
        self.stdout = stdout

        self.dtype = np.complex128
        self.dsize = 16
        self.dtype_eri, self.dsize_eri = self.get_eri_dtype_dsize(orbocc, orbvir)

        self.ovL = None

    @property
    def nocc(self):
        return self.orbocc.shape[1]
    @property
    def nvir(self):
        return self.orbvir.shape[1]

    def build(self):
        log = logger.new_logger(self)
        self.ovL = _init_mp_df_eris_complex(self, self.orbocc, self.orbvir,
                                            self.max_memory, ovL=self.ovL, log=log)

    def get_occ_blk(self, q, i0,i1, real_and_imag=False):
        if real_and_imag:
            out = self.ovL[q,i0:i1]
            return np.asarray(out.real, order='C'), np.asarray(out.imag, order='C')
        else:
            return np.asarray(self.ovL[q,i0:i1], order='C')

    def get_vir_blk(self, q, a0,a1, real_and_imag=False):
        if real_and_imag:
            out = self.ovL[q,:,a0:a1]
            return np.asarray(out.real, order='C'), np.asarray(out.imag, order='C')
        else:
            return np.asarray(self.ovL[q,:,a0:a1], order='C')

    def xform_occ(self, q, u, real_and_imag=False):
        nocc, nvir, naux = self.nocc, self.nvir, self.naux
        nOcc = u.shape[1]
        M = (self.max_memory - lib.current_memory()[0])*1e6 / self.dsize
        occblksize = min(nocc, max(1, int(np.floor(M*0.5/(nvir*naux) - nOcc))))
        if DEBUG_BLKSIZE: occblksize = max(1,nocc//2)
        OvL = np.empty((nOcc,nvir,naux), dtype=self.dtype)
        for iblk,(i0,i1) in enumerate(lib.prange(0,nocc,occblksize)):
            if iblk == 0:
                OvL[:]  = lib.einsum('iax,iI->Iax', self.get_occ_blk(q,i0,i1), u[i0:i1].conj())
            else:
                OvL[:] += lib.einsum('iax,iI->Iax', self.get_occ_blk(q,i0,i1), u[i0:i1].conj())
        if real_and_imag:
            return np.asarray(OvL.real, order='C'), np.asarray(OvL.imag, order='C')
        else:
            return OvL

    def xform_vir(self, q, u, real_and_imag=False):
        nocc, nvir, naux = self.nocc, self.nvir, self.naux
        nVir = u.shape[1]
        M = (self.max_memory - lib.current_memory()[0])*1e6 / self.dsize
        occblksize = min(nocc, max(1, int(np.floor(M*0.5/(nvir*naux) - nocc*nVir/float(nvir)))))
        if DEBUG_BLKSIZE: occblksize = max(1,nocc//2)
        oVL = np.empty((nocc,nVir,naux), dtype=self.dtype)
        for i0,i1 in lib.prange(0,nocc,occblksize):
            oVL[i0:i1] = lib.einsum('iax,aA->iAx', self.get_occ_blk(q,i0,i1), u)
        if real_and_imag:
            return np.asarray(oVL.real, order='C'), np.asarray(oVL.imag, order='C')
        else:
            return oVL


class _KLNODFOUTCOREERIS_COMPLEX(_KLNODFINCOREERIS_COMPLEX):
    def __init__(self, with_df, orbocc, orbvir, max_memory, ovL=None, ovL_to_save=None,
                 verbose=None, stdout=None):
        raise NotImplementedError
        _KLNODFINCOREERIS_COMPLEX.__init__(self, with_df, orbocc, orbvir, max_memory,
                                           verbose, stdout)

        self._ovL = ovL
        self._ovL_to_save = ovL_to_save

    def build(self):
        log = logger.new_logger(self)
        if self._ovL is None:
            if isinstance(self._ovL_to_save, str):
                self.feri = h5py.File(self._ovL_to_save, 'w')
            else:
                self.feri = lib.H5TmpFile()
            log.info('ovL is saved to %s', self.feri.filename)
            shape = (len(self.qpts),self.nocc,self.nvir,self.naux)
            self.ovL = self.feri.create_dataset('ovL', shape, self.dtype, chunks=(1,*shape[1:]))
            _init_mp_df_eris_complex(self, self.orbocc, self.orbvir, self.max_memory,
                                     ovL=self.ovL, log=log)
        elif isinstance(self._ovL, str):
            self.feri = h5py.File(self._ovL, 'r')
            log.info('ovL is read from %s', self.feri.filename)
            assert( 'ovL' in self.feri )
            ovL_shape = (self.nocc,self.nvir,self.naux)
            assert( self.feri['ovL/0'].shape == ovL_shape )
            self.ovL = self.feri['ovL']
        else:
            raise RuntimeError

def _init_mp_df_eris_complex(k2sdf, orbocc, orbvir, max_memory, ovL=None, log=None):
    r''' ovL[q,I,A,L] := (I A | L,q)
                      = \sum_{k} \sum_{mu,nu} (mu,k nu,k-q | L,q) C(mu,k I).conj() C(nu,k-q A)
    '''
    from pyscf.ao2mo import _ao2mo

    if log is None: log = logger.Logger(sys.stdout, 3)

    korbocc = k2sdf.s2k_mo_coeff(orbocc)
    korbvir = k2sdf.s2k_mo_coeff(orbvir)
    with_df = k2sdf.with_df
    kpts = k2sdf.kpts
    qpts = k2sdf.qpts
    kikj_by_q = k2sdf.kikj_by_q
    nqpts = len(qpts)
    naux_by_q = k2sdf.naux_by_q
    naux = k2sdf.naux

    dtype = k2sdf.dtype
    dsize = k2sdf.dsize

    nao, nocc = korbocc[0].shape
    nvir = korbvir[0].shape[1]
    nmo = nocc + nvir

    def fao2mo(j3c, mo, i0, i1, p0, p1, buf):
        tao = []
        ao_loc = None
        ijslice = (i0,i1,nocc,nmo)

        if dtype == np.float64:
            Lpq_ao = np.asarray(j3c[p0:p1].real)
            return _ao2mo.nr_e2(Lpq_ao, mo, ijslice, aosym='s2', out=buf)
        else:
            Lpq_ao = np.asarray(j3c[p0:p1])
            if Lpq_ao[0].size != nao**2:  # aosym = 's2'
                Lpq_ao = lib.unpack_tril(Lpq_ao).astype(np.complex128)
            return _ao2mo.r_e2(Lpq_ao, mo, ijslice, tao, ao_loc, out=buf)

    if ovL is None:
        ovL = np.empty((nqpts,nocc,nvir,naux), dtype=dtype)

    cput1 = (logger.process_clock(), logger.perf_counter())

    mem_avail  = max_memory - lib.current_memory()[0]
    if isinstance(ovL, np.ndarray):
        # subtract mem for holding ovL incore
        mem_avail -= nqpts*naux*nocc*nvir * dsize/1e6
        # incore: batching aux (OV + Nao_pair) * [X] = M
        mem_auxblk = (nao**2+nocc*nvir) * dsize/1e6
        aux_blksize = min(naux, max(1, int(np.floor(mem_avail*0.7 / mem_auxblk))))
        if DEBUG_BLKSIZE: aux_blksize = max(1,naux//2)
        log.debug('aux blksize for incore ao2mo: %d/%d', aux_blksize, naux)
        buf = np.empty(aux_blksize*nocc*nvir, dtype=dtype)

        ovL.fill(0)
        for q in range(nqpts):
            nauxq = naux_by_q[q]
            ovLq = ovL[q]
            if nauxq < naux:
                ovLq[:,:,nauxq:naux] = 0
            for ki,kj in kikj_by_q[q]:
                kpti_kptj = np.asarray((kpts[ki],kpts[kj]))
                mo = np.asarray(np.hstack((korbocc[ki], korbvir[kj])), order='F')
                with _load3c(with_df._cderi, with_df._dataname, kpti_kptj=kpti_kptj) as j3c:
                    for p0,p1 in lib.prange(0,nauxq,aux_blksize):
                        out = fao2mo(j3c, mo, 0, nocc, p0, p1, buf)
                        ovLq[:,:,p0:p1] += out.reshape(p1-p0,nocc,nvir).transpose(1,2,0)
                        out = None
            ovLq /= nqpts**0.5
            ovLq = None
            cput1 = log.timer('ao2mo for qidx %d/%d'%(q+1,nqpts), *cput1)

        buf = None
    else:
        # outcore: batching occ [O]XV and aux ([O]V + Nao_pair)*[X]
        mem_occblk = naux*nvir * dsize/1e6
        occ_blksize = min(nocc, max(1, int(np.floor(mem_avail*0.6 / mem_occblk))))
        if DEBUG_BLKSIZE: occ_blksize = max(1,nocc//2)
        mem_auxblk = (occ_blksize*nvir+nao**2) * dsize/1e6
        aux_blksize = min(naux, max(1, int(np.floor(mem_avail*0.3 / mem_auxblk))))
        if DEBUG_BLKSIZE: aux_blksize = max(1,naux//2)
        log.debug('occ blksize for outcore ao2mo: %d/%d', occ_blksize, nocc)
        log.debug('aux blksize for outcore ao2mo: %d/%d', aux_blksize, naux)
        buf = np.empty(naux*occ_blksize*nvir, dtype=dtype)
        buf2 = np.empty(aux_blksize*occ_blksize*nvir, dtype=dtype)

        for q in range(nqpts):
            nauxq = naux_by_q[q]
            if nauxq < naux:
                ovL[q,:,:,nauxq:naux] = 0
            for i0,i1 in lib.prange(0, nocc, occ_blksize):
                OvL = np.ndarray((i1-i0,nvir,nauxq), buffer=buf, dtype=dtype)
                OvL.fill(0)
                for ki,kj in kikj_by_q[q]:
                    kpti_kptj = np.asarray((kpts[ki],kpts[kj]))
                    mo = np.asarray(np.hstack((korbocc[ki], korbvir[kj])), order='F')
                    with _load3c(with_df._cderi, with_df._dataname, kpti_kptj=kpti_kptj) as j3c:
                        for p0,p1 in lib.prange(0,nauxq,aux_blksize):
                            out = fao2mo(j3c, mo, i0, i1, p0, p1, buf2)
                            OvL[:,:,p0:p1] += out.reshape(p1-p0,i1-i0,nvir).transpose(1,2,0)
                            out = None
                ovL[q,i0:i1] = OvL/nqpts**0.5
                OvL = None
            cput1 = log.timer('ao2mo for qidx %d/%d'%(q+1,nqpts), *cput1)

        buf = buf2 = None
    return ovL
