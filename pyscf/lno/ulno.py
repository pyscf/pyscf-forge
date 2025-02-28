# Copyright 2014-2025 The PySCF Developers. All Rights Reserved.
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
# Authors: Chenghan Li
#          Xing Zhang
#

from functools import reduce
import numpy as np

from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf import mp

from pyscf.lno import lno

einsum = lib.einsum

def make_las(mlno, eris, orbloc, lno_type, lno_param):
    log = logger.new_logger(mlno)
    cput1 = (logger.process_clock(), logger.perf_counter())

    s1e = mlno.s1e

    orboccfrz_core = [None,] * 2
    orbocc = [None,] * 2
    orbvir = [None,] * 2
    orbvirfrz_core = [None,] * 2
    moeocc = [None,] * 2
    moevir = [None,] * 2    

    uocc_loc = [None,] * 2
    uocc_std = [None,] * 2
    uocc_orth = [None,] * 2
    uvir_loc = [None,] * 2
    uvir_std = [None,] * 2
    uvir_orth = [None,] * 2

    mo_splits = mlno.split_mo_coeff()
    moe_splits = mlno.split_mo_energy()
    for s in range(2):
        orboccfrz_core[s], orbocc[s], orbvir[s], orbvirfrz_core[s] = mo_splits[s]
        moeocc[s], moevir[s] = moe_splits[s][1:3]

        #####################################
        # Projection of LO onto occ and vir #
        #####################################
        ovlp = reduce(np.dot, (orbloc[s].T.conj(), s1e, orbocc[s]))
        uocc_loc[s], uocc_std[s], uocc_orth[s] = \
            lno.projection_construction(ovlp, mlno.lo_proj_thresh, mlno.lo_proj_thresh_active)
        # NOTE we allow empty fragments
        #if uocc_loc[s].shape[1] == 0:
        #    log.error('LOs do not overlap with occupied space. This could be caused '
        #              'by either a bad fragment choice or too high of `lo_proj_thresh_active` '
        #              '(current value: %s).', mlno.lo_proj_thresh_active)
        #    raise RuntimeError
        log.info('LO occ proj: %d active | %d standby | %d orthogonal',
                 *[u.shape[1] for u in [uocc_loc[s], uocc_std[s], uocc_orth[s]]])

    ####################
    # LNO construction #
    ####################
    if lno_type[0] == lno_type[1] == '1h':
        dmoo, dmvv = make_lo_rdm1_1h(eris, moeocc, moevir, uocc_loc)
    else:
        raise NotImplementedError

    #if mlno._match_oldcode: dmoo *= 0.5 # TO MATCH OLD LNO CODE

    orbfrag = [None,] * 2
    frzfrag = [None,] * 2
    uoccact_loc = [None,] * 2
    frag_msg = ""

    for s in range(2):
        dmoo[s] = reduce(np.dot, (uocc_orth[s].T.conj(), dmoo[s], uocc_orth[s]))

        _param = lno_param[s][0]
        if _param['norb'] is not None:
            _param['norb'] -= uocc_loc[s].shape[1] + uocc_std[s].shape[1]

        uoccact_orth, uoccfrz_orth = lno.natorb_select(dmoo[s], uocc_orth[s], **_param)
        orboccfrz = np.hstack((orboccfrz_core[s], np.dot(orbocc[s], uoccfrz_orth)))
        uoccact = lno.subspace_eigh(np.diag(moeocc[s]), np.hstack((uoccact_orth, uocc_std[s], uocc_loc[s])))[1]
        orboccact = np.dot(orbocc[s], uoccact)
        uoccact_loc[s] = np.linalg.multi_dot((orboccact.T.conj(), s1e, orbloc[s]))

        orbviract, orbvirfrz = lno.natorb_select(dmvv[s], orbvir[s], **(lno_param[s][1]))
        orbvirfrz = np.hstack((orbvirfrz, orbvirfrz_core[s]))
        uviract = reduce(np.dot, (orbvir[s].T.conj(), s1e, orbviract))
        uviract = lno.subspace_eigh(np.diag(moevir[s]), uviract)[1]
        orbviract = np.dot(orbvir[s], uviract)

        ####################
        # LAS construction #
        ####################
        orbfragall = [orboccfrz, orboccact, orbviract, orbvirfrz]
        orbfrag[s] = np.hstack(orbfragall)
        norbfragall = np.asarray([x.shape[1] for x in orbfragall])
        locfragall = np.cumsum([0] + norbfragall.tolist()).astype(int)
        frzfrag[s] = np.concatenate((
            np.arange(locfragall[0], locfragall[1]),
            np.arange(locfragall[3], locfragall[4]))).astype(int)
        frag_msg += '\nSpin channel %d: %d/%d Occ | %d/%d Vir | %d/%d MOs\n' % (
                        s,
                        norbfragall[1], sum(norbfragall[:2]),
                        norbfragall[2], sum(norbfragall[2:4]),
                        sum(norbfragall[1:3]), sum(norbfragall)
                    )
        if len(frzfrag[s]) == 0:
            frzfrag[s] = 0

    return orbfrag, frzfrag, uoccact_loc, frag_msg

def make_lo_rdm1_1h(eris, moeocc, moevir, uocc):
    u = [None,] * 2
    eov = [None,] * 2
    eiv = [None,] * 2
    for s in range(2):
        moei, u[s] = lno.subspace_eigh(np.diag(moeocc[s]), uocc[s])
        eov[s] = moeocc[s][:,None] - moevir[s]
        eiv[s] = moei[:,None] - moevir[s]

    g = eris.get_ivov(u[0])
    t2aa = g.conj() / lib.direct_sum('ia+jb->iajb', eiv[0], eov[0])
    t2aa = t2aa - t2aa.transpose(0,3,2,1)

    g = eris.get_ivOV(u[0])
    t2ab = g.conj() / lib.direct_sum('ia+jb->iajb', eiv[0], eov[1])

    g = eris.get_IVov(u[1])
    t2ba = g.conj() / lib.direct_sum('ia+jb->iajb', eiv[1], eov[0])

    g = eris.get_IVOV(u[1])
    t2bb = g.conj() / lib.direct_sum('ia+jb->iajb', eiv[1], eov[1])
    t2bb = t2bb - t2bb.transpose(0,3,2,1)

    dmoo  = einsum('iajb,iakb->jk', t2aa, t2aa.conj()) * .5
    dmoo += einsum('iajb,iakb->jk', t2ba, t2ba.conj())
    dmOO  = einsum('iajb,iakb->jk', t2bb, t2bb.conj()) * .5
    dmOO += einsum('iajb,iakb->jk', t2ab, t2ab.conj())

    dmvv  = einsum('iajc,ibjc->ba', t2aa.conj(), t2aa) * .5
    dmvv += einsum('iajc,ibjc->ba', t2ab.conj(), t2ab) * .5
    dmvv += einsum('icja,icjb->ba', t2ba.conj(), t2ba) * .5
    dmVV  = einsum('iajc,ibjc->ba', t2bb.conj(), t2bb) * .5
    dmVV += einsum('iajc,ibjc->ba', t2ba.conj(), t2ba) * .5
    dmVV += einsum('icja,icjb->ba', t2ab.conj(), t2ab) * .5
    return [dmoo, dmOO], [dmvv, dmVV]


class ULNO(lno.LNO):
    def ao2mo(self, mo_coeff=None):
        log = logger.new_logger(self)
        cput0 = (logger.process_clock(), logger.perf_counter())

        nmoa, nmob = self.get_nmo()
        mem_now = self.max_memory - lib.current_memory()[0]
        if getattr(self, 'with_df', None):
            naux = self.with_df.get_naoaux()
            mem_df = nmoa**2 * naux * 8 / 1e6 * 4
            log.debug('ao2mo est mem= %.2f MB  avail mem= %.2f MB', mem_df, mem_now)
            if ((mem_df < mem_now or
                 self.mol.incore_anyway) and
                not self.force_outcore_ao2mo):
                eris = _make_df_eris_incore(self, mo_coeff)
            else:
                eris = _make_df_eris_outcore(self, mo_coeff)

        else:
            mem_incore = nmoa**4 * 8 / 1e6 * 4.
            log.debug('ao2mo est mem= %.2f MB  avail mem= %.2f MB', mem_incore, mem_now)
            if ((self._scf._eri is not None or
                 mem_incore < mem_now or
                 self.mol.incore_anyway) and
                not self.force_outcore_ao2mo):
                eris = _make_eris_incore(self, mo_coeff)
            else:
                eris = _make_eris_outcore(self, mo_coeff)

        log.timer('Integral xform   ', *cput0)
        return eris

    get_nocc = mp.ump2.get_nocc
    get_nmo = mp.ump2.get_nmo
    split_mo_coeff  = mp.dfump2.DFUMP2.split_mo_coeff
    split_mo_energy = mp.dfump2.DFUMP2.split_mo_energy
    split_mo_occ    = mp.dfump2.DFUMP2.split_mo_occ
    make_las = make_las


class _ULNO_ERIs:
    def __init__(self):
        self.orbo = None
        self.orbO = None
        self.orbv = None
        self.orbV = None

    def _common_init_(self, mlno, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = mlno.mo_coeff

        mos = mlno.split_mo_coeff(mo_coeff)
        self.orbo, self.orbv = mos[0][1:3]
        self.orbO, self.orbV = mos[1][1:3]

class _ULNO_Incore_ERIRs(_ULNO_ERIs):
    def _common_init_(self, mlno, mo_coeff=None):
        _ULNO_ERIs._common_init_(self, mlno, mo_coeff)
        if mlno._scf._eri is None:
            self._eri = mlno.mol.intor('int2e', aosym='s8')
        else:
            self._eri = mlno._scf._eri

    def get_ivov(self, u):
        orbi = np.dot(self.orbo, u)
        return ao2mo.general(self._eri, [orbi, self.orbv, self.orbo, self.orbv], compact=False)

    def get_ivOV(self, u):
        orbi = np.dot(self.orbo, u)
        return ao2mo.general(self._eri, [orbi, self.orbv, self.orbO, self.orbV], compact=False)

    def get_IVov(self, u):
        orbI = np.dot(self.orbO, u)
        return ao2mo.general(self._eri, [orbI, self.orbV, self.orbo, self.orbv], compact=False)

    def get_IVOV(self, u):
        orbI = np.dot(self.orbO, u)
        return ao2mo.general(self._eri, [orbI, self.orbV, self.orbO, self.orbV], compact=False)

def _make_eris_incore(mlno, mo_coeff=None):
    eris = _ULNO_Incore_ERIRs()
    eris._common_init_(mlno, mo_coeff)
    return eris

def _make_eris_outcore(mlno, mo_coeff=None):
    raise NotImplementedError

class _ULNO_DF_Incore_ERIRs(_ULNO_ERIs):
    def _common_init_(self, mlno, mo_coeff=None):
        _ULNO_ERIs._common_init_(self, mlno, mo_coeff)

        with_df = mlno.with_df
        naux = with_df.get_naoaux()
        nao, nocca = self.orbo.shape
        noccb = self.orbO.shape[-1]
        nvira = self.orbv.shape[-1]
        nvirb = self.orbV.shape[-1]
        self.Lov = np.empty((naux, nocca, nvira))
        self.LOV = np.empty((naux, noccb, nvirb))

        p1 = 0
        for eri1 in with_df.loop():
            eri1 = lib.unpack_tril(eri1).reshape(-1, nao, nao)
            p0, p1 = p1, p1 + eri1.shape[0]
            self.Lov[p0:p1] = einsum('Lpq,pi,qa->Lia', eri1, self.orbo, self.orbv)
            self.LOV[p0:p1] = einsum('Lpq,pi,qa->Lia', eri1, self.orbO, self.orbV)

    def get_iv(self, u):
        return einsum('iI,Lia->LIa', u, self.Lov)

    def get_IV(self, u):
        return einsum('iI,Lia->LIa', u, self.LOV)

    @staticmethod
    def _get_eris(Lia, Ljb):
        return einsum('Lia,Ljb->iajb', Lia, Ljb)

    def get_ivov(self, u):
        Liv = self.get_iv(u)
        return self._get_eris(Liv, self.Lov)

    def get_ivOV(self, u):
        Liv = self.get_iv(u)
        return self._get_eris(Liv, self.LOV)

    def get_IVov(self, u):
        LIV = self.get_IV(u)
        return self._get_eris(LIV, self.Lov)

    def get_IVOV(self, u):
        LIV = self.get_IV(u)
        return self._get_eris(LIV, self.LOV)

def _make_df_eris_incore(mlno, mo_coeff=None):
    eris = _ULNO_DF_Incore_ERIRs()
    eris._common_init_(mlno, mo_coeff)
    return eris

def _make_df_eris_outcore(mlno, mo_coeff=None):
    raise NotImplementedError

