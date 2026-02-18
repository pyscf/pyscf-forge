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


''' KLNO-RCCSD and KLNO-CCSD(T):

    - Original publication of molecular LNO by Kállay and co-workers:
        Rolik and Kállay, J. Chem. Phys. 135, 104111 (2011)

    - Publication for periodic KLNO by Ye and Berkelbach:
        Ye and Berkelbach, J. Chem. Theory Comput. 2024, 20, 20, 8948–8959
'''


import numpy as np

from pyscf.pbc.lib.kpts_helper import gamma_point
from pyscf import lib
logger = lib.logger

# for _contract_vvvv_t2
from pyscf import __config__
from pyscf.cc import _ccsd
import ctypes

from pyscf.lno.lnoccsd import (LNOCCSD, MODIFIED_CCSD, MODIFIED_DFCCSD, _ChemistsERIs,
                               impurity_solve, get_maskact, _cp)
from pyscf.pbc.lno.klno import KLNO
from pyscf.pbc.lno.tools import K2SDF, zdotCNtoR, zdotNNtoR, k2s_aoint

FORCE_DFKCC = getattr(__config__, 'lno_cc_kccsd_FORCE_DFKCC', False)    # force using DF CC
DEBUG_BLKSIZE = getattr(__config__, 'lno_cc_kccsd_DEBUG_BLKSIZE', False)


r''' Beginning of modification of PySCF's (DF)CCSD class for K2S (DF)CCSD

    These functions are modified from pyscf.cc and parallel those in pyscf.cc.lnoccsd.
    The major change compared to the latter is using KSCF DF integrals.

    For KSCF that includes the Gamma-point, the two classes
        - MODIFIED_K2SCCSD
        - MODIFIED_DFK2SCCSD
    assume time-reversal symmetry is conserved. As a result, the restored supercell
    ERIs are real-valued and can be calculated by
        (pq|rs) = \sum_{P}^{naux} \sum_{q}^{Nk} (pq|P,q) (rs|P,q).conj()

    For the general case where time-reversal symmetry is not conserved or a twisted
    k-point mesh is used, the two classes
        - MODIFIED_K2SCCSD_complex
        - MODIFIED_DFK2SCCSD_complex
    will be used. The restored supercell ERIs are in general complex-valued and
    calculated by
        (pq|rs) = \sum_{P}^{naux} \sum_{q}^{Nk} (pq|P,q) (rs|P,-q)
'''
from pyscf.cc import ccsd, dfccsd, rccsd
def K2SCCSD(mf, with_df, frozen, mo_coeff, mo_occ):
    ''' with_df is KSCF DF object
    '''
    import numpy
    from pyscf import lib
    from pyscf.soscf import newton_ah
    from pyscf import scf

    log = logger.new_logger(mf)

    if isinstance(mf, newton_ah._CIAH_SOSCF) or not isinstance(mf, scf.hf.RHF):
        mf = scf.addons.convert_to_rhf(mf)

    ''' auto-choose if using DFCCSD (storing Lvv) or CCSD (storing vvvv) by memory
    '''
    k2sdf = K2SDF(with_df)
    naux = k2sdf.Naux_ibz
    maskocc = mo_occ > 1e-10
    frozen, maskact = get_maskact(frozen, len(mo_occ))
    nvir = np.count_nonzero(~maskocc & maskact)
    nvir_pair = nvir*(nvir+1)//2
    mem_avail = mf.max_memory - lib.current_memory()[0]
    mem_need = (nvir_pair**2 + nvir_pair*naux) * 8/1024**2.
    log.debug1('naux= %d  nvir_pair= %d  mem_avail= %.1f  mem_vvvv= %.1f',
               naux, nvir_pair, mem_avail, mem_need)

    if gamma_point(with_df.kpts[0]) and np.isrealobj(mo_coeff):
        ''' Gamma-inclusive k-point mesh and time-reversal symmetry conserved
        '''
        if not FORCE_DFKCC and (naux > nvir_pair or mem_need < mem_avail * 0.7):
            log.debug1('Using CCSD')
            return MODIFIED_K2SCCSD(mf, with_df, frozen, mo_coeff, mo_occ)
        else:
            log.debug1('Using DFCCSD')
            return MODIFIED_DFK2SCCSD(mf, with_df, frozen, mo_coeff, mo_occ)
    else:
        raise NotImplementedError
        if not FORCE_DFKCC and (naux > nvir_pair or mem_need < mem_avail * 0.7):
            log.debug1('Using complex CCSD')
            return MODIFIED_K2SCCSD_complex(mf, with_df, frozen, mo_coeff, mo_occ)
        else:
            log.debug1('Using complex DFCCSD')
            raise NotImplementedError('LNO-DFCCSD not implemented for complex orbitals.')
            return MODIFIED_DFK2SCCSD_complex(mf, with_df, frozen, mo_coeff, mo_occ)

class MODIFIED_K2SCCSD(MODIFIED_CCSD):
    _keys = {"k2sdf"}
    def __init__(self, mf, with_df, frozen, mo_coeff, mo_occ):
        MODIFIED_CCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.k2sdf = K2SDF(with_df)

    def ao2mo(self, mo_coeff=None):
        return _make_df_eris_outcore(self, mo_coeff)

def _make_df_eris_outcore(mycc, mo_coeff=None):
    from pyscf.ao2mo import _ao2mo

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(mycc.stdout, mycc.verbose)
    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)

    mo_coeff = eris.mo_coeff
    nocc = eris.nocc
    nmo = mo_coeff.shape[1]
    nvir = nmo - nocc
    nvir_pair = nvir*(nvir+1)//2

    k2sdf = mycc.k2sdf
    Naux = k2sdf.Naux_ibz
    naux = k2sdf.naux
    naux_by_q = k2sdf.naux_by_q
    REAL = np.float64
    COMPLEX = np.complex128
    LooR = np.zeros((Naux,nocc,nocc), dtype=REAL)
    LooI = np.zeros((Naux,nocc,nocc), dtype=REAL)
    LovR = np.zeros((Naux,nocc,nvir), dtype=REAL)
    LovI = np.zeros((Naux,nocc,nvir), dtype=REAL)
    LvoR = np.zeros((Naux,nvir,nocc), dtype=REAL)
    LvoI = np.zeros((Naux,nvir,nocc), dtype=REAL)
    LvvR = np.zeros((Naux,nvir_pair), dtype=REAL)
    LvvI = np.zeros((Naux,nvir_pair), dtype=REAL)
    buf = np.empty((naux,nmo,nmo), dtype=COMPLEX)

    for qi,q in enumerate(k2sdf.ibz2bz):
        nauxq = naux_by_q[q]
        p0 = naux * qi
        p1 = p0 + nauxq
        for (ki,kj),LpqR,LpqI in k2sdf.loop_ao2mo(q,mo_coeff,mo_coeff,buf=buf,real_and_imag=True):
            LpqR = LpqR.reshape(nauxq,nmo,nmo)
            LpqI = LpqI.reshape(nauxq,nmo,nmo)
            LooR[p0:p1] += LpqR[:,:nocc,:nocc]
            LooI[p0:p1] += LpqI[:,:nocc,:nocc]
            LovR[p0:p1] += LpqR[:,:nocc,nocc:]
            LovI[p0:p1] += LpqI[:,:nocc,nocc:]
            LvoR[p0:p1] += LpqR[:,nocc:,:nocc]
            LvoI[p0:p1] += LpqI[:,nocc:,:nocc]
            LvvR[p0:p1] += lib.pack_tril(LpqR[:,nocc:,nocc:])
            LvvI[p0:p1] += lib.pack_tril(LpqI[:,nocc:,nocc:])
            LpqR = LpqI = None
        w = k2sdf.qpts_ibz_weights[qi]
        LooR[p0:p1] *= w
        LooI[p0:p1] *= w
        LovR[p0:p1] *= w
        LovI[p0:p1] *= w
        LvoR[p0:p1] *= w
        LvoI[p0:p1] *= w
        LvvR[p0:p1] *= w
        LvvI[p0:p1] *= w

    LooR = LooR.reshape(Naux,nocc*nocc)
    LooI = LooI.reshape(Naux,nocc*nocc)
    LovR = LovR.reshape(Naux,nocc*nvir)
    LovI = LovI.reshape(Naux,nocc*nvir)
    LvoR = LvoR.reshape(Naux,nocc*nvir)
    LvoI = LvoI.reshape(Naux,nocc*nvir)

    eris.feri1 = lib.H5TmpFile()
    eris.oooo = eris.feri1.create_dataset('oooo', (nocc,nocc,nocc,nocc), 'f8')
    eris.oovv = eris.feri1.create_dataset('oovv', (nocc,nocc,nvir,nvir), 'f8',
                                          chunks=(nocc,nocc,1,nvir))
    eris.ovoo = eris.feri1.create_dataset('ovoo', (nocc,nvir,nocc,nocc), 'f8',
                                          chunks=(nocc,1,nocc,nocc))
    eris.ovvo = eris.feri1.create_dataset('ovvo', (nocc,nvir,nvir,nocc), 'f8',
                                          chunks=(nocc,1,nvir,nocc))
    eris.ovov = eris.feri1.create_dataset('ovov', (nocc,nvir,nocc,nvir), 'f8',
                                           chunks=(nocc,1,nocc,nvir))
    eris.ovvv = eris.feri1.create_dataset('ovvv', (nocc,nvir,nvir_pair), 'f8')
    eris.vvvv = eris.feri1.create_dataset('vvvv', (nvir_pair,nvir_pair), 'f8')
    eris.oooo[:] = zdotCNtoR(LooR.T, LooI.T, LooR, LooI).reshape(nocc,nocc,nocc,nocc)
    eris.ovoo[:] = zdotCNtoR(LovR.T, LovI.T, LooR, LooI).reshape(nocc,nvir,nocc,nocc)
    eris.oovv[:] = lib.unpack_tril(zdotCNtoR(LooR.T, LooI.T,
                                             LvvR  , LvvI)).reshape(nocc,nocc,nvir,nvir)
    eris.ovvo[:] = zdotCNtoR(LovR.T, LovI.T, LvoR, LvoI).reshape(nocc,nvir,nvir,nocc)
    eris.ovov[:] = zdotCNtoR(LovR.T, LovI.T, LovR, LovI).reshape(nocc,nvir,nocc,nvir)
    eris.ovvv[:] = zdotCNtoR(LovR.T, LovI.T, LvvR, LvvI).reshape(nocc,nvir,nvir_pair)
    eris.vvvv[:] = zdotCNtoR(LvvR.T, LvvI.T, LvvR, LvvI)

    log.timer('CCSD integral transformation', *cput0)
    return eris


class MODIFIED_DFK2SCCSD(MODIFIED_DFCCSD):
    def __init__(self, mf, with_df, frozen, mo_coeff, mo_occ):
        MODIFIED_DFCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.k2sdf = K2SDF(with_df)

    def ao2mo(self, mo_coeff=None):
        return _make_df_eris(self, mo_coeff)

class _K2SDFChemistsERIs(_ChemistsERIs):
    def _contract_vvvv_t2(self, mycc, t2, direct=False, out=None, verbose=None):
        assert(not direct)
        return _contract_vvvv_t2(mycc, self.mol, self.vvLR, self.vvLI, t2, out, verbose)
def _contract_vvvv_t2(mycc, mol, vvLR, vvLI, t2, out=None, verbose=None):
    '''Ht2 = np.einsum('ijcd,acdb->ijab', t2, vvvv)

    Args:
        vvvv : None or integral object
            if vvvv is None, contract t2 to AO-integrals using AO-direct algorithm
    '''
    MEMORYMIN = getattr(__config__, 'cc_ccsd_memorymin', 2000)
    _dgemm = lib.numpy_helper._dgemm
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.new_logger(mol, verbose)

    naux = vvLR.shape[-1]
    nvira, nvirb = t2.shape[-2:]
    x2 = t2.reshape(-1,nvira,nvirb)
    nocc2 = x2.shape[0]
    nvir2 = nvira * nvirb
    Ht2 = np.ndarray(x2.shape, buffer=out)
    Ht2[:] = 0

    max_memory = max(MEMORYMIN, mycc.max_memory - lib.current_memory()[0])
    def contract_blk_(eri, i0, i1, j0, j1):
        ic = i1 - i0
        jc = j1 - j0
        #:Ht2[:,j0:j1] += np.einsum('xef,efab->xab', x2[:,i0:i1], eri)
        _dgemm('N', 'N', nocc2, jc*nvirb, ic*nvirb,
               x2.reshape(-1,nvir2), eri.reshape(-1,jc*nvirb),
               Ht2.reshape(-1,nvir2), 1, 1, i0*nvirb, 0, j0*nvirb)

        if i0 > j0:
            #:Ht2[:,i0:i1] += np.einsum('xef,abef->xab', x2[:,j0:j1], eri)
            _dgemm('N', 'T', nocc2, ic*nvirb, jc*nvirb,
                   x2.reshape(-1,nvir2), eri.reshape(-1,jc*nvirb),
                   Ht2.reshape(-1,nvir2), 1, 1, j0*nvirb, 0, i0*nvirb)

#TODO: check if vvL can be entirely loaded into memory
    nvir_pair = nvirb * (nvirb+1) // 2
    dmax = np.sqrt(max_memory*.7e6/8/nvirb**2/2)
    dmax = int(min((nvira+3)//4, max(ccsd.BLKMIN, dmax)))
    vvblk = (max_memory*1e6/8 - dmax**2*(nvirb**2*1.5+naux*2))/(naux*2)
    vvblk = int(min((nvira+3)//4, max(ccsd.BLKMIN, vvblk/(naux*2))))
    eribuf = np.empty((dmax,dmax,nvir_pair))
    loadbuf = np.empty((dmax,dmax,nvirb,nvirb))
    tril2sq = lib.square_mat_in_trilu_indices(nvira)

    for i0, i1 in lib.prange(0, nvira, dmax):
        off0 = i0*(i0+1)//2
        off1 = i1*(i1+1)//2
        vvLR0 = _cp(vvLR[off0:off1])
        vvLI0 = _cp(vvLI[off0:off1])
        for j0, j1 in lib.prange(0, i1, dmax):
            ijLR = vvLR0[tril2sq[i0:i1,j0:j1] - off0].reshape(-1,naux)
            ijLI = vvLI0[tril2sq[i0:i1,j0:j1] - off0].reshape(-1,naux)
            eri = np.ndarray(((i1-i0)*(j1-j0),nvir_pair), buffer=eribuf)
            for p0, p1 in lib.prange(0, nvir_pair, vvblk):
                vvLR1 = _cp(vvLR[p0:p1])
                vvLI1 = _cp(vvLI[p0:p1])
                eri[:,p0:p1] = zdotCNtoR(ijLR, ijLI, vvLR1.T, vvLI1.T)
                vvLR1 = vvLI1 = None
            ijLR = ijLI = None

            tmp = np.ndarray((i1-i0,nvirb,j1-j0,nvirb), buffer=loadbuf)
            _ccsd.libcc.CCload_eri(tmp.ctypes.data_as(ctypes.c_void_p),
                                   eri.ctypes.data_as(ctypes.c_void_p),
                                   (ctypes.c_int*4)(i0, i1, j0, j1),
                                   ctypes.c_int(nvirb))
            contract_blk_(tmp, i0, i1, j0, j1)
            time0 = log.timer_debug1('vvvv [%d:%d,%d:%d]'%(i0,i1,j0,j1), *time0)
        vvLR0 = vvLI0 = None
    return Ht2.reshape(t2.shape)
def _make_df_eris(mycc, mo_coeff=None):
    from pyscf.ao2mo import _ao2mo

    eris = _K2SDFChemistsERIs()
    eris._common_init_(mycc, mo_coeff)
    nocc = eris.nocc
    nmo = eris.fock.shape[0]
    nvir = nmo - nocc
    nvir_pair = nvir*(nvir+1)//2
    mo_coeff = eris.mo_coeff

    k2sdf = mycc.k2sdf
    naux_by_q = k2sdf.naux_by_q
    naux = k2sdf.naux
    Naux = k2sdf.Naux_ibz
    REAL = np.float64
    COMPLEX = np.complex128

    eris.feri = lib.H5TmpFile()
    eris.oooo = eris.feri.create_dataset('oooo', (nocc,nocc,nocc,nocc), 'f8')
    eris.ovoo = eris.feri.create_dataset('ovoo', (nocc,nvir,nocc,nocc), 'f8',
                                         chunks=(nocc,1,nocc,nocc))
    eris.ovov = eris.feri.create_dataset('ovov', (nocc,nvir,nocc,nvir), 'f8',
                                         chunks=(nocc,1,nocc,nvir))
    eris.ovvo = eris.feri.create_dataset('ovvo', (nocc,nvir,nvir,nocc), 'f8',
                                         chunks=(nocc,1,nvir,nocc))
    eris.oovv = eris.feri.create_dataset('oovv', (nocc,nocc,nvir,nvir), 'f8',
                                         chunks=(nocc,nocc,1,nvir))
    # nrow ~ 4e9/8/blockdim to ensure hdf5 chunk < 4GB
    chunks = (min(nvir_pair,int(4e8/k2sdf.blockdim)), min(Naux,k2sdf.blockdim))
    eris.vvLR = eris.feri.create_dataset('vvLR', (nvir_pair,Naux), 'f8', chunks=chunks)
    eris.vvLI = eris.feri.create_dataset('vvLI', (nvir_pair,Naux), 'f8', chunks=chunks)

    # estimate aux blksize
    mem_avail  = mycc.max_memory - lib.current_memory()[0]
    mem_avail -= Naux*nocc*nmo * 16/1e6
    mem_block  = (nmo**2 + nvir_pair) * 16/1e6
    aux_blksize = max(1, min(naux, int(np.round(np.floor(mem_avail*0.7/mem_block)))))
    if DEBUG_BLKSIZE: aux_blksize = max(1, naux//2)

    LooR = np.zeros((Naux,nocc,nocc), dtype=REAL)
    LooI = np.zeros((Naux,nocc,nocc), dtype=REAL)
    LovR = np.zeros((Naux,nocc,nvir), dtype=REAL)
    LovI = np.zeros((Naux,nocc,nvir), dtype=REAL)

    buf = np.empty((aux_blksize,nmo,nmo), dtype=COMPLEX)
    bufR = np.empty((aux_blksize,nvir_pair), dtype=REAL)
    bufI = np.empty((aux_blksize,nvir_pair), dtype=REAL)

    for qi,q in enumerate(k2sdf.ibz2bz):
        nauxq = naux_by_q[q]
        if nauxq < naux:
            eris.vvLR[:,naux*qi+nauxq:naux*(qi+1)] = 0
            eris.vvLI[:,naux*qi+nauxq:naux*(qi+1)] = 0
        for r0,r1 in lib.prange(0,nauxq,aux_blksize):
            auxslice = (r0,r1)
            dr = r1-r0
            p0 = naux*qi + r0
            p1 = p0 + dr
            LvvR = np.ndarray((dr,nvir_pair), dtype=REAL, buffer=bufR)
            LvvI = np.ndarray((dr,nvir_pair), dtype=REAL, buffer=bufI)
            LvvR.fill(0)
            LvvI.fill(0)
            for (ki,kj),LpqR,LpqI in k2sdf.loop_ao2mo(q,mo_coeff,mo_coeff,buf=buf,
                                                      real_and_imag=True,auxslice=auxslice):
                LpqR = LpqR.reshape(dr,nmo,nmo)
                LpqI = LpqI.reshape(dr,nmo,nmo)
                LooR[p0:p1] += LpqR[:,:nocc,:nocc]
                LooI[p0:p1] += LpqI[:,:nocc,:nocc]
                LovR[p0:p1] += LpqR[:,:nocc,nocc:]
                LovI[p0:p1] += LpqI[:,:nocc,nocc:]
                LvvR[:dr] += lib.pack_tril(LpqR[:,nocc:,nocc:])
                LvvI[:dr] += lib.pack_tril(LpqI[:,nocc:,nocc:])
                LpqR = LpqI = None
            w = k2sdf.qpts_ibz_weights[qi]
            LooR[p0:p1] *= w
            LooI[p0:p1] *= w
            LovR[p0:p1] *= w
            LovI[p0:p1] *= w
            LvvR *= w
            LvvI *= w
            eris.vvLR[:,p0:p1] = LvvR.T
            eris.vvLI[:,p0:p1] = LvvI.T
            LvvR = LvvI = None
    buf = bufR = bufI = None

    LooR = LooR.reshape(Naux,nocc*nocc)
    LooI = LooI.reshape(Naux,nocc*nocc)
    LovR = LovR.reshape(Naux,nocc*nvir)
    LovI = LovI.reshape(Naux,nocc*nvir)

    eris.oooo[:] = zdotCNtoR(LooR.T, LooI.T, LooR, LooI).reshape(nocc,nocc,nocc,nocc)
    eris.ovoo[:] = zdotCNtoR(LovR.T, LovI.T, LooR, LooI).reshape(nocc,nvir,nocc,nocc)
    ovov = zdotCNtoR(LovR.T, LovI.T, LovR, LovI).reshape(nocc,nvir,nocc,nvir)
    eris.ovov[:] = ovov
    eris.ovvo[:] = ovov.transpose(0,1,3,2)
    ovov = None

    mem_now = lib.current_memory()[0]
    max_memory = max(0, mycc.max_memory - mem_now)
    blksize = max(ccsd.BLKMIN, int((max_memory*.9e6/8-nocc**2*nvir_pair)/(nocc**2+Naux)))
    oovv_tril = np.empty((nocc*nocc,nvir_pair))
    for p0, p1 in lib.prange(0, nvir_pair, blksize):
        oovv_tril[:,p0:p1] = zdotCNtoR(LooR.T, LooI.T, _cp(eris.vvLR[p0:p1]).T,
                                       _cp(eris.vvLI[p0:p1]).T)
    eris.oovv[:] = lib.unpack_tril(oovv_tril).reshape(nocc,nocc,nvir,nvir)
    oovv_tril = LooR = LooI = None

    LovR = LovR.reshape(Naux,nocc,nvir)
    LovI = LovI.reshape(Naux,nocc,nvir)
    vblk = max(nocc, int((max_memory*.15e6/8)/(nocc*nvir_pair)))
    vvblk = int(min(nvir_pair, 4e8/nocc, max(4, (max_memory*.8e6/8)/(vblk*nocc+Naux))))
    eris.ovvv = eris.feri.create_dataset('ovvv', (nocc,nvir,nvir_pair), 'f8',
                                         chunks=(nocc,1,vvblk))
    for q0, q1 in lib.prange(0, nvir_pair, vvblk):
        vvLR = _cp(eris.vvLR[q0:q1])
        vvLI = _cp(eris.vvLI[q0:q1])
        for p0, p1 in lib.prange(0, nvir, vblk):
            tmpLovR = _cp(LovR[:,:,p0:p1]).reshape(Naux,-1)
            tmpLovI = _cp(LovI[:,:,p0:p1]).reshape(Naux,-1)
            eris.ovvv[:,p0:p1,q0:q1] = zdotCNtoR(tmpLovR.T, tmpLovI.T, vvLR.T,
                                                 vvLI.T).reshape(nocc,p1-p0,q1-q0)
        vvLR = vvLI = None
    return eris

class MODIFIED_K2SCCSD_complex:
    pass
class MODIFIED_DFK2SCCSD_complex:
    pass

class KLNOCCSD(KLNO,LNOCCSD):
    def __init__(self, kmf, lo_coeff, frag_lolist, lno_type=None, lno_thresh=None, frozen=None,
                 mf=None):
        KLNO.__init__(self, kmf, lo_coeff, frag_lolist, lno_type, lno_thresh, frozen, mf)

        self.efrag_cc = None
        self.efrag_pt2 = None
        self.efrag_cc_t = None
        self.efrag_cc_spin_comp = None
        self.efrag_pt2_spin_comp = None
        self.ccsd_t = False

        # args for impurity solver
        self.kwargs_imp = None
        self.verbose_imp = 2    # ERROR and WARNING

        # args for precompute
        self._s1e = None
        self._h1e = None
        self._vhf = None

    def impurity_solve(self, mf, mo_coeff, uocc_loc, eris, frozen=None, log=None):
        if log is None: log = logger.new_logger(self)
        mo_occ = self.mo_occ
        frozen, maskact = get_maskact(frozen, mo_occ.size)
        mcc = K2SCCSD(mf, self.with_df, frozen, mo_coeff, mo_occ).set(verbose=self.verbose_imp)
        mcc._s1e = self._s1e
        mcc._h1e = self._h1e
        mcc._vhf = self._vhf

        if self.kwargs_imp is not None:
            mcc = mcc.set(**self.kwargs_imp)

        return impurity_solve(mcc, mo_coeff, uocc_loc, mo_occ, maskact, eris, log=log,
                              ccsd_t=self.ccsd_t, verbose_imp=self.verbose_imp,
                              max_las_size_ccsd=self._max_las_size_ccsd,
                              max_las_size_ccsd_t=self._max_las_size_ccsd_t)

class KLNOCCSD_T(KLNOCCSD):
    def __init__(self, *args, **kwargs):
        KLNOCCSD.__init__(self, *args, **kwargs)
        self.ccsd_t = True


if __name__ == '__main__':
    from pyscf.pbc import gto, scf, mp
    from pyscf import lo
    from pyscf.pbc.lno.tools import k2s_scf
    from pyscf.pbc.lno.tools import sort_orb_by_cell
    from pyscf.lno import LNOCCSD

    atom = '''
    O          0.00000        0.00000        0.11779
    H          0.00000        0.75545       -0.47116
    H          0.00000       -0.75545       -0.47116
    '''
    a = np.eye(3) * 4
    basis = 'cc-pvdz'
    kmesh = [3,1,1]

    scaled_center = None

    cell = gto.M(atom=atom, basis=basis, a=a).set(verbose=4)
    kpts = cell.make_kpts(kmesh, scaled_center=scaled_center)

    kmf = scf.KRHF(cell, kpts=kpts).density_fit()
    kmf.kernel()

    mf = k2s_scf(kmf)

    # KLNO with PM localized orbitals
    # PM localization within the BvK supercell
    orbocc = mf.mo_coeff[:,mf.mo_occ>1e-6]
    mlo = lo.PipekMezey(mf.cell, orbocc)
    lo_coeff = mlo.kernel()
    while True: # always performing jacobi sweep to avoid trapping in local minimum/saddle point
        lo_coeff1 = mlo.stability_jacobi()[1]
        if lo_coeff1 is lo_coeff:
            break
        mlo = lo.PipekMezey(mf.mol, lo_coeff1).set(verbose=4)
        mlo.init_guess = None
        lo_coeff = mlo.kernel()

    # sort LOs by unit cell
    s1e = mf.get_ovlp()
    Nk = len(kpts)
    nlo = lo_coeff.shape[1]//Nk
    lo_coeff = sort_orb_by_cell(mf.cell, lo_coeff, Nk, s=s1e)

    frag_lolist = [[i] for i in range(nlo)]

    # Optional: precompute h1e within supercell from K2S transform
    h1e = k2s_aoint(cell, kpts, kmf.get_hcore())

    # KLNOCCSD(T) calculations
    # kmlno = KLNOCCSD_T(kmf, lo_coeff, frag_lolist, mf=mf).set(verbose=5)
    kmlno = KLNOCCSD(kmf, lo_coeff, frag_lolist, mf=mf).set(verbose=5)
    kmlno._h1e = h1e
    kmlno.lno_thresh = [1e-4, 1e-5]
    kmlno.kernel()

    # Supercell LNOCCSD(T) calculation (the two should match!)
    frag_lolist = [[i] for i in range(nlo*Nk)]
    # mlno = LNOCCSD_T(mf, lo_coeff, frag_lolist)
    mlno = LNOCCSD(mf, lo_coeff, frag_lolist)
    mlno._h1e = h1e
    mlno.lno_thresh = [1e-4, 1e-5]
    mlno.kernel()

    def print_compare(name, ek, es):
        print(f'{name:9s} Ecorr:  {ek: 14.9f}  {es: 14.9f}  diff: {es-ek: 14.9f}')

    print()
    print('Comparing KLNO with supercell LNO (normalized to per cell):')
    print_compare('LNOMP2', kmlno.e_corr_pt2, mlno.e_corr_pt2/Nk)
    print_compare('LNOCCSD', kmlno.e_corr_ccsd, mlno.e_corr_ccsd/Nk)
    # print_compare('LNOCCSD_T', kmlno.e_corr_ccsd_t, mlno.e_corr_ccsd_t/Nk)
