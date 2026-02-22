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
#          Ardavan Farahvash

import sys
from functools import reduce
import numpy as np
import h5py

from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf import mp

from pyscf.lno import lno
from pyscf.lno.make_lno_rdm1 import _mp2_rdm1_occblksize, DEBUG_BLKSIZE

einsum = lib.einsum


def make_las(mlno, eris, orbloc, lno_type, lno_param):
    """
    Create localized active space for a given set of localized orbitals
    given in orbloc
    """
    log = logger.new_logger(mlno)
    # cput1 = (logger.process_clock(), logger.perf_counter())

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
            lno.projection_construction(
                ovlp, mlno.lo_proj_thresh, mlno.lo_proj_thresh_active)
        # NOTE we allow empty fragments
        # if uocc_loc[s].shape[1] == 0:
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
        # NOTE: uvir_loc is not used in 1h/1h, so we pass None
        if getattr(mlno, 'with_df', None):
            dmoo, dmvv = make_lo_rdm1_1h_df(eris, moeocc, moevir, uocc_loc)
        else:
            dmoo, dmvv = make_lo_rdm1_1h(eris, moeocc, moevir, uocc_loc)
    else:
        raise NotImplementedError('Unsupported LNO type')

    orbfrag = [None,] * 2
    frzfrag = [None,] * 2
    uoccact_loc = [None,] * 2
    frag_msg = ""

    for s in range(2):
        dmoo[s] = reduce(
            np.dot, (uocc_orth[s].T.conj(), dmoo[s], uocc_orth[s]))

        _param = lno_param[s][0]
        if _param['norb'] is not None:
            _param['norb'] -= uocc_loc[s].shape[1] + uocc_std[s].shape[1]

        uoccact_orth, uoccfrz_orth = lno.natorb_select(
            dmoo[s], uocc_orth[s], **_param)
        orboccfrz = np.hstack(
            (orboccfrz_core[s], np.dot(orbocc[s], uoccfrz_orth)))
        uoccact = lno.subspace_eigh(np.diag(moeocc[s]), np.hstack(
            (uoccact_orth, uocc_std[s], uocc_loc[s])))[1]
        orboccact = np.dot(orbocc[s], uoccact)
        uoccact_loc[s] = np.linalg.multi_dot(
            (orboccact.T.conj(), s1e, orbloc[s]))

        orbviract, orbvirfrz = lno.natorb_select(
            dmvv[s], orbvir[s], **(lno_param[s][1]))
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
    r'''
    Create unrestricted MP2 density matrix with one localized hole

    Math:
        dmoo_a(i,j) = sum_{Kab} [ T(Kaib) * T(Kajb)]
                      + 2* sum_{K'ab'} [ t(K'aib') * t(K'ajb') ]

        dmoo_b(i,j) - reverse alpha/beta spin indices from dmoo_a

        dmvv_a(a,b) = sum_{Kic} [ T(Kaic) * T(Kbjc) ]
                      + sum_{K'ic'} [ t(K'c'ib) * t(K'c'ia) ]
                      + sum_{Ki'c'} [ t(Kai'c) * t(Kbi'c') ]

        dmvv_b(a,b) - reverse alpha/beta spin indices from dmvv_a

    Notation:
    i,j - canonical occupied orbitals
    a,b,c - canonical virtual orbitals
    K - local occupied orbitals
    indices with ' are beta spin, others are alpha.

    are canonical alpha; I,J,K are local alpha;
    A,B,C are canonical beta;

    t(ijab) = (ia|jb) / (e_i+e_j-e_a-e_b)
    T(ijab) = t(ijab) - t(ijba)

    #
    Args:
        eris : ERI object
            Provides `ovL` in the canonical MOs.
        moeocc : [moeocc,_a moeocc_b]
            Occ MO energies
        moeocc : [moevir,_a moevir_b]
            Vir MO energies
        uocc : [u_a, u_b]
            Overlap between the canonical and localized occupied orbitals.
            u_a(i,I) = <i|I> (alpha), u_b(i',I') = <i'|I'> (beta)
    '''
    log = logger.new_logger(eris)

    # Unpack spins
    moeocc_a, moeocc_b = moeocc
    moevir_a, moevir_b = moevir
    u_a, u_b = uocc

    nocca, nvira = eris.nocc[0], eris.nvir[0]
    noccb, nvirb = eris.nocc[1], eris.nvir[1]
    nOcca = u_a.shape[1]
    nOccb = u_b.shape[1]
    dtype = eris.dtype
    dsize = eris.dsize

    # Determine block sizes
    mem_avail = eris.max_memory - lib.current_memory()[0]
    M = mem_avail * 0.1 * 1e6 / dsize

    # Alpha block size (based on alpha canonical occ)
    occblksize_a, mem_peak_a = _mp2_rdm1_occblksize(
        nocca, nvira, 0, 0, 3, M/2, dsize)
    if DEBUG_BLKSIZE:
        occblksize_a = max(1, nocca // 2)

    # Beta block size (based on beta canonical occ)
    occblksize_b, mem_peak_b = _mp2_rdm1_occblksize(
        noccb, nvirb, 0, 0, 3, M/2, dsize)
    if DEBUG_BLKSIZE:
        occblksize_b = max(1, noccb // 2)

    #
    log.debug1('make_lo_rdm1_1h (alpha): nocc=%d nvir=%d nOcc=%d blksize=%d peak_mem=%.2f MB',
               nocca, nvira, nOcca, occblksize_a, mem_peak_a)

    log.debug1('make_lo_rdm1_1h (beta):  nocc=%d nvir=%d nOcc=%dblksize=%d peak_mem=%.2f MB',
               noccb, nvirb, nOccb, occblksize_b, mem_peak_b)

    # Localized MO energies
    moeI_a, u_a = lno.subspace_eigh(np.diag(moeocc_a), u_a)
    moeI_b, u_b = lno.subspace_eigh(np.diag(moeocc_b), u_b)

    # Energy denominators
    eov_a = moeocc_a[:, None] - moevir_a
    eIv_a = moeI_a[:, None] - moevir_a
    eov_b = moeocc_b[:, None] - moevir_b
    eIv_b = moeI_b[:, None] - moevir_b

    # Initialize RDMs
    dmoo_a = np.zeros((nocca, nocca), dtype=dtype)
    dmoo_b = np.zeros((noccb, noccb), dtype=dtype)
    dmvv_a = np.zeros((nvira, nvira), dtype=dtype)
    dmvv_b = np.zeros((nvirb, nvirb), dtype=dtype)

    # -- construct t2aa and contract
    for Kbatch, (K0, K1) in enumerate(lib.prange(0, nOcca, occblksize_a)):
        # fragment-occ DF energies
        eKv_a = eIv_a[K0:K1]

        for ibatch, (i0, i1) in enumerate(lib.prange(0, nocca, occblksize_a)):
            # full-occ DF energies
            eiv_a = eov_a[i0:i1]

            # form t2_ovov-block
            denom_aa = lib.direct_sum('Ka+ib->Kaib', eKv_a, eiv_a)
            t2aa_i = eris.get_ivov(u_a, K0, K1, i0, i1) / denom_aa
            t2aa_i = t2aa_i - t2aa_i.transpose(0, 3, 2, 1)
            denom_aa = None
            eiv_a = None

            for jbatch, (j0, j1) in enumerate(lib.prange(0, nocca, occblksize_a)):
                if ibatch == jbatch:
                    t2aa_j = t2aa_i
                else:
                    # full-occ DF energies
                    ejv_a = eov_a[j0:j1]

                    # form t2_ovov-block
                    denom_aa = lib.direct_sum('Ka+jb->Kajb', eKv_a, ejv_a)
                    t2aa_j = eris.get_ivov(u_a, K0, K1, j0, j1) / denom_aa
                    t2aa_j = t2aa_j - t2aa_j.transpose(0, 3, 2, 1)
                    denom_aa = None
                    ejv_a = None

                # contract block to make occupied MP2-DM
                dmoo_a[i0:i1,
                       j0:j1] += lib.einsum('Kaib,Kajb->ij', t2aa_i, t2aa_j.conj())
                t2aa_j = None

            # contract block to make virtual MP2-DM
            dmvv_a += lib.einsum('Kaic,Kbic->ba', np.conj(t2aa_i), t2aa_i)
            t2aa_i = None

        eKv_a = None

    # -- construct t2bb and contract
    for Kbatch, (K0, K1) in enumerate(lib.prange(0, nOccb, occblksize_b)):
        # fragment-occ DF energies
        eKv_b = eIv_b[K0:K1]

        for ibatch, (i0, i1) in enumerate(lib.prange(0, noccb, occblksize_b)):
            # full-occ DF integrals/energies
            eiv_b = eov_b[i0:i1]

            # form t2_ovov-block
            denom_bb = lib.direct_sum('Ka+ib->Kaib', eKv_b, eiv_b)
            t2bb_i = eris.get_IVOV(u_b, K0, K1, i0, i1) / denom_bb
            t2bb_i = t2bb_i - t2bb_i.transpose(0, 3, 2, 1)
            denom_bb = None
            eiv_b = None

            for jbatch, (j0, j1) in enumerate(lib.prange(0, noccb, occblksize_b)):
                if ibatch == jbatch:
                    t2bb_j = t2bb_i
                else:
                    # full-occ DF integrals/energies
                    ejv_b = eov_b[j0:j1]

                    # form t2_ovov-block
                    denom_bb = lib.direct_sum('Ka+jb->Kajb', eKv_b, ejv_b)
                    t2bb_j = eris.get_IVOV(u_b, K0, K1, j0, j1) / denom_bb
                    t2bb_j = t2bb_j - t2bb_j.transpose(0, 3, 2, 1)
                    denom_bb = None
                    ejv_b = None

                # contract block to make occupied MP2/DM
                dmoo_b[i0:i1,
                       j0:j1] += lib.einsum('Kaib,Kajb->ij', t2bb_i, t2bb_j.conj())
                t2bb_j = None

            # contract block to make virtual MP2/DM
            dmvv_b += lib.einsum('Kaic,Kbic->ba', t2bb_i.conj(), t2bb_i)
            t2bb_i = None
        eKv_b = None

    # -- construct t2ba and contract
    for Kbatch, (K0, K1) in enumerate(lib.prange(0, nOccb, occblksize_b)):
        # fragment-occ DF energies
        eKv_b = eIv_b[K0:K1]

        for ibatch, (i0, i1) in enumerate(lib.prange(0, nocca, occblksize_a)):
            # full-occ DF energies
            eiv_a = eov_a[i0:i1]

            # form t2_ovov-block
            denom_ba = lib.direct_sum('Ka+ib->Kaib', eKv_b, eiv_a)
            t2ba_i = eris.get_IVov(u_b, K0, K1, i0, i1) / denom_ba
            denom_ba = None
            eiv_a = None

            for jbatch, (j0, j1) in enumerate(lib.prange(0, nocca, occblksize_a)):
                if ibatch == jbatch:
                    t2ba_j = t2ba_i
                else:
                    # full-occ DF energies
                    ejv_a = eov_a[j0:j1]

                    # form t2_ovov-block
                    denom_ba = lib.direct_sum('Ka+jb->Kajb', eKv_b, ejv_a)
                    t2ba_j = eris.get_IVov(u_b, K0, K1, j0, j1) / denom_ba
                    denom_ba = None
                    ejv_a = None

                # contract block to make occupied MP2/DM
                dmoo_a[i0:i1, j0:j1] += 2 * \
                    lib.einsum('Kaib,Kajb->ij', t2ba_i, t2ba_j.conj())
                t2ba_j = None

            # contract block to make virtual MP2/DM
            dmvv_a += lib.einsum('Kcia,Kcib->ba', t2ba_i.conj(), t2ba_i)
            dmvv_b += lib.einsum('Kaic,Kbic->ba', t2ba_i.conj(), t2ba_i)
            t2ba_i = None
        eKv_b = None

    # -- construct t2ab and contract
    for Kbatch, (K0, K1) in enumerate(lib.prange(0, nOcca, occblksize_a)):
        # fragment-occ DF energies
        eKv_a = eIv_a[K0:K1]

        for ibatch, (i0, i1) in enumerate(lib.prange(0, noccb, occblksize_b)):
            # full-occ DF energies
            eiv_b = eov_b[i0:i1]

            # form t2_ovov-block
            denom_ab = lib.direct_sum('Ka+ib->Kaib', eKv_a, eiv_b)
            t2ab_i = eris.get_ivOV(u_a, K0, K1, i0, i1) / denom_ab
            denom_ba = None
            eiv_b = None

            for jbatch, (j0, j1) in enumerate(lib.prange(0, noccb, occblksize_b)):
                if ibatch == jbatch:
                    t2ab_j = t2ab_i
                else:
                    # full-occ DF energies
                    ejv_b = eov_b[j0:j1]

                    # form t2_ovov-block
                    denom_ab = lib.direct_sum('Ka+jb->Kajb', eKv_a, ejv_b)
                    t2ab_j = eris.get_ivOV(u_a, K0, K1, j0, j1) / denom_ab
                    denom_ba = None
                    ejv_b = None

                # contract block to make occupied MP2/DM
                dmoo_b[i0:i1, j0:j1] += 2 * \
                    lib.einsum('Kaib,Kajb->ij', t2ab_i, t2ab_j.conj())
                t2ab_j = None

            dmvv_a += lib.einsum('Kaic,Kbic->ba', t2ab_i.conj(), t2ab_i)
            dmvv_b += lib.einsum('Kcia,Kcib->ba', t2ab_i.conj(), t2ab_i)
            t2ab_i = None
        eKv_a = None

    return [dmoo_a, dmoo_b], [dmvv_a, dmvv_b]


def make_lo_rdm1_1h_df(eris, moeocc, moevir, uocc):
    r'''
    Create unrestricted MP2 density matrix with one localized hole
    Density-fitted version

    Math:
        dmoo_a(i,j) = sum_{Kab} [ T(Kaib) * T(Kajb)]
                      + 2* sum_{K'ab'} [ t(K'aib') * t(K'ajb') ]

        dmoo_b(i,j) - reverse alpha/beta spin indices from dmoo_a

        dmvv_a(a,b) = sum_{Kic} [ T(Kaic) * T(Kbjc) ]
                      + sum_{K'ic'} [ t(K'c'ib) * t(K'c'ia) ]
                      + sum_{Ki'c'} [ t(Kai'c) * t(Kbi'c') ]


        dmvv_b(a,b) - reverse alpha/beta spin indices from dmvv_a

    Notation:
    i,j - canonical occupied orbitals
    a,b,c - canonical virtual orbitals
    K - local occupied orbitals
    indices with ' are beta spin, others are alpha.

    are canonical alpha; I,J,K are local alpha;
    A,B,C are canonical beta;

    t(ijab) = (ia|jb) / (e_i+e_j-e_a-e_b)
    T(ijab) = t(ijab) - t(ijba)

    #
    Args:
        eris : ERI object
            Provides `ovL` in the canonical MOs.
        moeocc : [moeocc,_a moeocc_b]
            Occ MO energies
        moeocc : [moevir,_a moevir_b]
            Vir MO energies
        uocc : [u_a, u_b]
            Overlap between the canonical and localized occupied orbitals.
            u_a(i,I) = <i|I> (alpha), u_b(i',I') = <i'|I'> (beta)
    '''
    log = logger.new_logger(eris)

    # Unpack spins
    moeocc_a, moeocc_b = moeocc
    moevir_a, moevir_b = moevir
    u_a, u_b = uocc

    nocca, nvira, naux = eris.nocc[0], eris.nvir[0], eris.naux
    noccb, nvirb = eris.nocc[1], eris.nvir[1]
    nOcca = u_a.shape[1]
    nOccb = u_b.shape[1]
    dtype = eris.dtype
    dsize = eris.dsize

    # Determine block sizes
    mem_avail = eris.max_memory - lib.current_memory()[0]
    M = mem_avail * 0.7 * 1e6 / dsize

    # Alpha block size (based on alpha canonical occ)
    occblksize_a, mem_peak_a = _mp2_rdm1_occblksize(
        nocca, nvira, naux, 3, 3, M/2, dsize)
    if DEBUG_BLKSIZE:
        occblksize_a = max(1, nocca // 2)

    # Beta block size (based on beta canonical occ)
    occblksize_b, mem_peak_b = _mp2_rdm1_occblksize(
        noccb, nvirb, naux, 3, 3, M/2, dsize)
    if DEBUG_BLKSIZE:
        occblksize_b = max(1, noccb // 2)

    #
    log.debug1('make_lo_rdm1_1h (alpha): nocc=%d nvir=%d nOcc=%d naux=%d blksize=%d peak_mem=%.2f MB',
               nocca, nvira, nOcca, naux, occblksize_a, mem_peak_a)

    log.debug1('make_lo_rdm1_1h (beta):  nocc=%d nvir=%d nOcc=%d naux=%d blksize=%d peak_mem=%.2f MB',
               noccb, nvirb, nOccb, naux, occblksize_b, mem_peak_b)

    # Localized MO energies
    moeI_a, u_a = lno.subspace_eigh(np.diag(moeocc_a), u_a)
    moeI_b, u_b = lno.subspace_eigh(np.diag(moeocc_b), u_b)

    # Energy denominators
    eov_a = moeocc_a[:, None] - moevir_a
    eIv_a = moeI_a[:, None] - moevir_a
    eov_b = moeocc_b[:, None] - moevir_b
    eIv_b = moeI_b[:, None] - moevir_b

    # Initialize RDMs
    dmoo_a = np.zeros((nocca, nocca), dtype=dtype)
    dmoo_b = np.zeros((noccb, noccb), dtype=dtype)
    dmvv_a = np.zeros((nvira, nvira), dtype=dtype)
    dmvv_b = np.zeros((nvirb, nvirb), dtype=dtype)

    # -- construct t2aa and contract
    for Kbatch, (K0, K1) in enumerate(lib.prange(0, nOcca, occblksize_a)):

        # fragment-occ DF integrals/energies
        KvL_a = eris.xform_occ(u_a[:, K0:K1], spin='a')
        eKv_a = eIv_a[K0:K1]

        for ibatch, (i0, i1) in enumerate(lib.prange(0, nocca, occblksize_a)):

            # full-occ DF integrals/energies
            ivL_a = eris.get_occ_blk(i0, i1, spin='a')
            eiv_a = eov_a[i0:i1]

            # form t2-block
            denom_aa = lib.direct_sum('Ka+ib->Kaib', eKv_a, eiv_a)
            t2aa_i = lib.einsum('Kax,ibx->Kaib', KvL_a, ivL_a) / denom_aa
            t2aa_i = t2aa_i - t2aa_i.transpose(0, 3, 2, 1)
            denom_aa = None
            ivL_a = None
            eiv_a = None

            for jbatch, (j0, j1) in enumerate(lib.prange(0, nocca, occblksize_a)):
                if jbatch == ibatch:
                    t2aa_j = t2aa_i
                else:
                    # full-occ DF integrals/energies
                    jvL_a = eris.get_occ_blk(j0, j1, spin='a')
                    ejv_a = eov_a[j0:j1]

                    # form t2-block
                    denom_aa = lib.direct_sum('Ka+jb->Kajb', eKv_a, ejv_a)
                    t2aa_j = lib.einsum(
                        'Kax,jbx->Kajb', KvL_a, jvL_a) / denom_aa
                    t2aa_j = t2aa_j - t2aa_j.transpose(0, 3, 2, 1)
                    denom_aa = None
                    jvL_a = None
                    ejv_a = None

                # contract block to make occupied MP2-DM
                dmoo_a[i0:i1,
                       j0:j1] += lib.einsum('Kaib,Kajb->ij', t2aa_i, t2aa_j.conj())
                t2aa_j = None

            # contract block to make virtual MP2-DM
            dmvv_a += lib.einsum('Kaic,Kbic->ba', np.conj(t2aa_i), t2aa_i)
            t2aa_i = None

        KvL_a = None
        eKv_a = None

    # -- construct t2bb and contract
    for Kbatch, (K0, K1) in enumerate(lib.prange(0, nOccb, occblksize_b)):
        # fragment-occ DF integrals/energies
        KvL_b = eris.xform_occ(u_b[:, K0:K1], spin='b')
        eKv_b = eIv_b[K0:K1]

        for ibatch, (i0, i1) in enumerate(lib.prange(0, noccb, occblksize_b)):
            # full-occ DF integrals/energies
            ivL_b = eris.get_occ_blk(i0, i1, spin='b')
            eiv_b = eov_b[i0:i1]

            # form t2-block
            denom_bb = lib.direct_sum('Ka+ib->Kaib', eKv_b, eiv_b)
            t2bb_i = lib.einsum('Kax,ibx->Kaib', KvL_b, ivL_b) / denom_bb
            t2bb_i = t2bb_i - t2bb_i.transpose(0, 3, 2, 1)
            denom_bb = None
            ivL_b = None
            eiv_b = None

            for jbatch, (j0, j1) in enumerate(lib.prange(0, noccb, occblksize_b)):
                if jbatch == ibatch:
                    t2bb_j = t2bb_i
                else:
                    # full-occ DF integrals/energies
                    jvL_b = eris.get_occ_blk(j0, j1, spin='b')
                    ejv_b = eov_b[j0:j1]

                    # form t2-block
                    denom_bb = lib.direct_sum('Ka+jb->Kajb', eKv_b, ejv_b)
                    t2bb_j = lib.einsum(
                        'Kax,jbx->Kajb', KvL_b, jvL_b) / denom_bb
                    t2bb_j = t2bb_j - t2bb_j.transpose(0, 3, 2, 1)
                    denom_bb = None
                    jvL_b = None
                    ejv_b = None

                # contract block to make occupied MP2-DM
                dmoo_b[i0:i1,
                       j0:j1] += lib.einsum('Kaib,Kajb->ij', t2bb_i, t2bb_j.conj())
                t2bb_j = None

            # contract block to make virtual MP2-DM
            dmvv_b += lib.einsum('Kaic,Kbic->ba', t2bb_i.conj(), t2bb_i)
            t2bb_i = None

        KvL_b = None
        eKv_b = None

    # -- construct t2ba and contract
    for Kbatch, (K0, K1) in enumerate(lib.prange(0, nOccb, occblksize_b)):
        # fragment-occ DF integrals/energies
        KvL_b = eris.xform_occ(u_b[:, K0:K1], spin='b')
        eKv_b = eIv_b[K0:K1]

        for ibatch, (i0, i1) in enumerate(lib.prange(0, nocca, occblksize_a)):
            # full-occ DF integrals/energies
            ivL_a = eris.get_occ_blk(i0, i1, spin='a')
            eiv_a = eov_a[i0:i1]

            # form t2-block
            denom_ba = lib.direct_sum('Ka+ib->Kaib', eKv_b, eiv_a)
            t2ba_i = lib.einsum('Kax,ibx->Kaib', KvL_b, ivL_a) / denom_ba
            ivL_a = None
            eiv_a = None
            denom_ba = None

            for jbatch, (j0, j1) in enumerate(lib.prange(0, nocca, occblksize_a)):
                if jbatch == ibatch:
                    t2ba_j = t2ba_i

                else:
                    # full-occ DF integrals/energies
                    jvL_a = eris.get_occ_blk(j0, j1, spin='a')
                    ejv_a = eov_a[j0:j1]

                    # form t2-block
                    denom_ba = lib.direct_sum('Ka+jb->Kajb', eKv_b, ejv_a)
                    t2ba_j = lib.einsum(
                        'Kax,jbx->Kajb', KvL_b, jvL_a) / denom_ba
                    jvL_a = None
                    ejv_a = None
                    denom_ba = None

                # contract block to make occupied MP2-DM
                dmoo_a[i0:i1, j0:j1] += 2 * \
                    lib.einsum('Kaib,Kajb->ij', t2ba_i, t2ba_j.conj())
                t2ba_j = None

            # contract block to make virtual MP2-DM
            dmvv_a += lib.einsum('Kcia,Kcib->ba', t2ba_i.conj(), t2ba_i)
            dmvv_b += lib.einsum('Kaic,Kbic->ba', t2ba_i.conj(), t2ba_i)
            t2ba_i = None

        KvL_b = None
        eKv_b = None

    # -- construct t2ab and contract
    for Kbatch, (K0, K1) in enumerate(lib.prange(0, nOcca, occblksize_a)):
        # fragment-occ DF integrals/energies
        KvL_a = eris.xform_occ(u_a[:, K0:K1], spin='a')
        eKv_a = eIv_a[K0:K1]

        for ibatch, (i0, i1) in enumerate(lib.prange(0, noccb, occblksize_b)):
            # full-occ DF integrals/energies
            ivL_b = eris.get_occ_blk(i0, i1, spin='b')
            eiv_b = eov_b[i0:i1]

            # form t2-block
            denom_ab = lib.direct_sum('Ka+ib->Kaib', eKv_a, eiv_b)
            t2ab_i = lib.einsum('Kax,ibx->Kaib', KvL_a, ivL_b) / denom_ab
            ivL_b = None
            eiv_b = None
            denom_ba = None

            for jbatch, (j0, j1) in enumerate(lib.prange(0, noccb, occblksize_b)):
                if jbatch == ibatch:
                    t2ab_j = t2ab_i
                else:
                    # full-occ DF integrals/energies
                    jvL_b = eris.get_occ_blk(j0, j1, spin='b')
                    ejv_b = eov_b[j0:j1]

                    # form t2-block
                    denom_ab = lib.direct_sum('Ka+jb->Kajb', eKv_a, ejv_b)
                    t2ab_j = lib.einsum(
                        'Kax,jbx->Kajb', KvL_a, jvL_b) / denom_ab
                    jvL_b = None
                    ejv_b = None
                    denom_ba = None

                # contract block to make occupied MP2-DM
                dmoo_b[i0:i1, j0:j1] += 2 * \
                    lib.einsum('Kaib,Kajb->ij', t2ab_i, t2ab_j.conj())
                t2ba_j = None

            # contract block to make virtual MP2-DM
            dmvv_a += lib.einsum('Kaic,Kbic->ba', t2ab_i.conj(), t2ab_i)
            dmvv_b += lib.einsum('Kcia,Kcib->ba', t2ab_i.conj(), t2ab_i)
            t2ab_i = None

        KvL_a = None
        eKv_a = None

    return [dmoo_a, dmoo_b], [dmvv_a, dmvv_b]


# ------ Density Fitted ERIS code
class _ULNO_DF_ERIs:
    # This class is now more of a holder for common propert0es,
    # matching the structure of _LNODFINCOREERIS
    def __init__(self, with_df, orbocc, orbvir, max_memory, verbose=None, stdout=None):
        self.with_df = with_df
        self.orbocc = orbocc  # [orb_a, orb_b]
        self.orbvir = orbvir  # [orb_v, orb_V]

        self.max_memory = max_memory
        self.verbose = verbose
        self.stdout = stdout

        self.dtype = self.orbocc[0].dtype
        self.dsize = self.orbocc[0].itemsize

        self.ovL = None  # Alpha (o,v,L)
        self.OVL = None  # Beta (O,V,L)

    @property
    def nocc(self):
        return [self.orbocc[0].shape[1], self.orbocc[1].shape[1]]

    @property
    def nvir(self):
        return [self.orbvir[0].shape[1], self.orbvir[1].shape[1]]

    @property
    def naux(self):
        return self.with_df.get_naoaux()

    def get_occ_blk(self, i0, i1, spin='a'):
        if spin == 'a':
            return np.asarray(self.ovL[i0:i1], order='C')
        else:
            return np.asarray(self.OVL[i0:i1], order='C')

    def get_vir_blk(self, a0, a1, spin='a'):
        if spin == 'a':
            return np.asarray(self.ovL[:, a0:a1], order='C')
        else:
            return np.asarray(self.OVL[:, a0:a1], order='C')

    def xform_occ(self, u, spin='a'):
        if spin == 'a':
            ovL = self.ovL
            nocc, nvir, naux = self.nocc[0], self.nvir[0], self.naux
        else:
            ovL = self.OVL
            nocc, nvir, naux = self.nocc[1], self.nvir[1], self.naux

        nOcc = u.shape[1]
        M = (self.max_memory - lib.current_memory()[0])*1e6 / self.dsize
        occblksize = min(nocc, max(1, int(np.floor(M*0.5/(nvir*naux) - nOcc))))
        if DEBUG_BLKSIZE:
            occblksize = max(1, nocc // 2)

        ovL = np.empty((nOcc, nvir, naux), dtype=self.dtype)
        for iblk, (i0, i1) in enumerate(lib.prange(0, nocc, occblksize)):
            if iblk == 0:
                ovL[:] = lib.einsum(
                    'iax,iI->Iax', self.get_occ_blk(i0, i1, spin=spin), u[i0:i1].conj())
            else:
                ovL[:] += lib.einsum('iax,iI->Iax', self.get_occ_blk(i0,
                                     i1, spin=spin), u[i0:i1].conj())
        return ovL

    def xform_vir(self, u, spin='a'):
        if spin == 'a':
            ovL = self.ovL
            nocc, nvir, naux = self.nocc[0], self.nvir[0], self.naux
        else:
            ovL = self.OVL
            nocc, nvir, naux = self.nocc[1], self.nvir[1], self.naux

        nVir = u.shape[1]
        M = (self.max_memory - lib.current_memory()[0])*1e6 / self.dsize
        occblksize = min(
            nocc, max(1, int(np.floor(M*0.5/(nvir*naux) - nocc*nVir/float(nvir)))))
        if DEBUG_BLKSIZE:
            occblksize = max(1, nocc // 2)

        ovL = np.empty((nocc, nVir, naux), dtype=self.dtype)
        for i0, i1 in lib.prange(0, nocc, occblksize):
            ovL[i0:i1] = lib.einsum(
                'iax,aA->iAx', self.get_occ_blk(i0, i1, spin=spin), u)
        return ovL


class _ULNO_DF_Incore_ERIs(_ULNO_DF_ERIs):
    def __init__(self, with_df, orbocc, orbvir, max_memory, verbose=None, stdout=None):
        super().__init__(with_df, orbocc, orbvir, max_memory, verbose, stdout)

    def build(self):
        log = logger.new_logger(self)
        self.ovL, self.OVL = _init_ump_df_eris(self.with_df,
                                               self.orbocc[0], self.orbvir[0],
                                               self.orbocc[1], self.orbvir[1],
                                               self.max_memory, ovL_a=self.ovL, ovL_b=self.OVL, log=log)


class _ULNO_DF_Outcore_ERIs(_ULNO_DF_ERIs):
    def __init__(self, with_df, orbocc, orbvir, max_memory, ovL=None, ovL_to_save=None,
                 verbose=None, stdout=None):
        super().__init__(with_df, orbocc, orbvir, max_memory, verbose, stdout)
        self._ovL = ovL  # Can be a path or list of paths
        self._ovL_to_save = ovL_to_save  # Can be a path

    def build(self):
        log = logger.new_logger(self)
        nocca, nvira, naux = self.nocc[0], self.nvir[0], self.naux
        noccb, nvirb = self.nocc[1], self.nvir[1]
        ovL_shape_a = (nocca, nvira, naux)
        ovL_shape_b = (noccb, nvirb, naux)

        ovL_a_dataset_name = 'ovL_a'
        ovL_b_dataset_name = 'ovL_b'

        if self._ovL is None:
            if isinstance(self._ovL_to_save, str):
                self.feri = h5py.File(self._ovL_to_save, 'w')
                log.info('ovL (alpha/beta) is saved to %s', self.feri.filename)
            else:
                self.feri = lib.H5TmpFile()
                log.info('ovL (alpha/beta) is saved to tmpfile %s',
                         self.feri.filename)

            self.ovL = self.feri.create_dataset(ovL_a_dataset_name, ovL_shape_a, dtype=self.dtype,
                                                chunks=(1, *ovL_shape_a[1:]))
            self.OVL = self.feri.create_dataset(ovL_b_dataset_name, ovL_shape_b, dtype=self.dtype,
                                                chunks=(1, *ovL_shape_b[1:]))

            _init_ump_df_eris(self.with_df, self.orbocc[0], self.orbvir[0],
                              self.orbocc[1], self.orbvir[1], self.max_memory,
                              ovL_a=self.ovL, ovL_b=self.OVL, log=log)

        elif isinstance(self._ovL, str):
            self.feri = h5py.File(self._ovL, 'r')
            log.info('ovL (alpha/beta) is read from %s', self.feri.filename)
            assert (ovL_a_dataset_name in self.feri)
            assert (ovL_b_dataset_name in self.feri)
            assert (self.feri[ovL_a_dataset_name].shape == ovL_shape_a)
            assert (self.feri[ovL_b_dataset_name].shape == ovL_shape_b)
            self.ovL = self.feri[ovL_a_dataset_name]
            self.OVL = self.feri[ovL_b_dataset_name]
        else:
            # Handle case where self._ovL is [ovL_a_obj, ovL_b_obj] (not paths)
            # This path is less common but supported in restricted code.
            # For simplicity, we assume string path or None.
            raise RuntimeError(
                "Invalid _ovL input. Expecting None or HDF5 file path.")

# Helper function for DF ERI generation


def _init_ump_df_eris(with_df, occ_coeff_a, vir_coeff_a, occ_coeff_b, vir_coeff_b,
                      max_memory, ovL_a=None, ovL_b=None, log=None):
    from pyscf.ao2mo import _ao2mo

    if log is None:
        log = logger.Logger(sys.stdout, 3)

    # array shapes
    nao, nocca = occ_coeff_a.shape
    nvira = vir_coeff_a.shape[1]
    nmoa = nocca + nvira

    nao, noccb = occ_coeff_b.shape
    nvirb = vir_coeff_b.shape[1]
    nmob = noccb + nvirb

    nao_pair = nao**2
    naux = with_df.get_naoaux()

    dtype = occ_coeff_a.dtype
    dsize = occ_coeff_a.itemsize

    mo_a = np.asarray(np.hstack((occ_coeff_a, vir_coeff_a)), order='F')
    mo_b = np.asarray(np.hstack((occ_coeff_b, vir_coeff_b)), order='F')
    ijslice_a = (0, nocca, nocca, nmoa)
    ijslice_b = (0, noccb, noccb, nmob)

    if ovL_a is None:
        ovL_a = np.empty((nocca, nvira, naux), dtype=dtype)

    if ovL_b is None:
        ovL_b = np.empty((noccb, nvirb, naux), dtype=dtype)

    mem_avail = max_memory - lib.current_memory()[0]

    # --- Define DF loop and ao2mo functions ---
    if dtype == np.float64:
        def loop_df(blksize):
            for Lpq in with_df.loop(blksize=blksize):
                yield Lpq
                Lpq = None

        def ao2mo_df(Lpq, mo, ijslice, out):
            return _ao2mo.nr_e2(Lpq, mo, ijslice, aosym='s2', out=out)

    else:
        def loop_df(blksize):
            kpti_kptj = [with_df.kpts[0]]*2
            for LpqR, LpqI, sign in with_df.sr_loop(blksize=blksize,
                                                    kpti_kptj=kpti_kptj):
                Lpq = LpqR + LpqI*1j
                LpqR = LpqI = None
                if Lpq.shape[1] != nao_pair:
                    Lpq = lib.unpack_tril(Lpq).astype(dtype)
                yield Lpq
                Lpq = None

        def ao2mo_df(Lpq, mo, ijslice, out):
            return _ao2mo.r_e2(Lpq, mo, ijslice, [], None, aosym='s1', out=out)

    # --- In-core ERI generation ---
    if isinstance(ovL_a, np.ndarray) and isinstance(ovL_b, np.ndarray):
        mem_auxblk = (nao_pair + nocca*nvira + noccb*nvirb) * dsize / 1e6
        aux_blksize = min(
            naux, max(1, int(np.floor(mem_avail * 0.5 / mem_auxblk))))
        if DEBUG_BLKSIZE:
            aux_blksize = max(1, naux // 2)
        log.debug('aux blksize for incore ao2mo (unrestricted): %d/%d',
                  aux_blksize, naux)

        buf_a = np.empty(aux_blksize * nocca * nvira, dtype=dtype)
        buf_b = np.empty(aux_blksize * noccb * nvirb, dtype=dtype)

        p1 = 0
        for Lpq in loop_df(aux_blksize):
            p0, p1 = p1, p1 + Lpq.shape[0]
            out_a = ao2mo_df(Lpq, mo_a, ijslice_a, buf_a)
            out_b = ao2mo_df(Lpq, mo_b, ijslice_b, buf_b)
            ovL_a[:, :,
                  p0:p1] = out_a.reshape(-1, nocca, nvira).transpose(1, 2, 0)
            ovL_b[:, :,
                  p0:p1] = out_b.reshape(-1, noccb, nvirb).transpose(1, 2, 0)
            Lpq = out_a = out_b = None
        buf_a = buf_b = None

    # --- Out-of-core (HDF5) ERI generation ---
    else:

        # batching occ [O]XV and aux ([O]V + Nao_pair)*[X]
        # We process alpha and beta spins sequentially to save memory

        # Process Alpha
        mem_occblk_a = naux * nvira * dsize / 1e6
        occ_blksize_a = min(nocca, max(
            1, int(np.floor(mem_avail * 0.6 / mem_occblk_a))))
        if DEBUG_BLKSIZE:
            occ_blksize_a = max(1, nocca // 2)
        mem_auxblk_a = (occ_blksize_a * nvira + nao_pair) * dsize / 1e6
        aux_blksize_a = min(
            naux, max(1, int(np.floor(mem_avail * 0.3 / mem_auxblk_a))))
        if DEBUG_BLKSIZE:
            aux_blksize_a = max(1, naux // 2)
        log.debug('occ blksize (alpha) for outcore ao2mo: %d/%d',
                  occ_blksize_a, nocca)
        log.debug('aux blksize (alpha) for outcore ao2mo: %d/%d',
                  aux_blksize_a, naux)

        buf_a = np.empty(naux * occ_blksize_a * nvira, dtype=dtype)
        buf2_a = np.empty(aux_blksize_a * occ_blksize_a * nvira, dtype=dtype)

        for i0, i1 in lib.prange(0, nocca, occ_blksize_a):
            nocci = i1 - i0
            ijslice = (i0, i1, nocca, nmoa)
            p1 = 0
            ovL_block = np.ndarray((nocci, nvira, naux),
                                   dtype=dtype, buffer=buf_a)
            for Lpq in loop_df(aux_blksize_a):
                p0, p1 = p1, p1 + Lpq.shape[0]
                out = ao2mo_df(Lpq, mo_a, ijslice, buf2_a)
                ovL_block[:, :,
                          p0:p1] = out.reshape(-1, nocci, nvira).transpose(1, 2, 0)
                Lpq = out = None
            ovL_a[i0:i1] = ovL_block
            ovL_block = None
        buf_a = buf2_a = None

        # Process Beta
        mem_occblk_b = naux * nvirb * dsize / 1e6
        occ_blksize_b = min(noccb, max(
            1, int(np.floor(mem_avail * 0.6 / mem_occblk_b))))
        if DEBUG_BLKSIZE:
            occ_blksize_b = max(1, noccb // 2)
        mem_auxblk_b = (occ_blksize_b * nvirb + nao_pair) * dsize / 1e6
        aux_blksize_b = min(
            naux, max(1, int(np.floor(mem_avail * 0.3 / mem_auxblk_b))))
        if DEBUG_BLKSIZE:
            aux_blksize_b = max(1, naux // 2)
        log.debug('occ blksize (beta) for outcore ao2mo: %d/%d',
                  occ_blksize_b, noccb)
        log.debug('aux blksize (beta) for outcore ao2mo: %d/%d',
                  aux_blksize_b, naux)

        buf_b = np.empty(naux * occ_blksize_b * nvirb, dtype=dtype)
        buf2_b = np.empty(aux_blksize_b * occ_blksize_b * nvirb, dtype=dtype)

        for i0, i1 in lib.prange(0, noccb, occ_blksize_b):
            nocci = i1 - i0
            ijslice = (i0, i1, noccb, nmob)
            p1 = 0
            OVL_block = np.ndarray((nocci, nvirb, naux),
                                   dtype=dtype, buffer=buf_b)
            for Lpq in loop_df(aux_blksize_b):
                p0, p1 = p1, p1 + Lpq.shape[0]
                out = ao2mo_df(Lpq, mo_b, ijslice, buf2_b)
                OVL_block[:, :,
                          p0:p1] = out.reshape(-1, nocci, nvirb).transpose(1, 2, 0)
                Lpq = out = None
            ovL_b[i0:i1] = OVL_block
            OVL_block = None
        buf_b = buf2_b = None

    return ovL_a, ovL_b


# ------ Non-Density Fitted ERIS code
class _ULNO_ERIs:
    def __init__(self, mlno, orbocc, orbvir, max_memory, verbose=None, stdout=None):
        self.orbo = orbocc[0]
        self.orbO = orbocc[1]
        self.orbv = orbvir[0]
        self.orbV = orbvir[1]

        self.max_memory = max_memory
        self.verbose = verbose
        self.stdout = stdout

        self.dtype = self.orbo.dtype
        self.dsize = self.orbo.itemsize

        if mlno._scf._eri is None:
            self._eri = mlno.mol.intor('int2e', aosym='s8')
        else:
            self._eri = mlno._scf._eri

    @property
    def nocc(self):
        return [self.orbo.shape[1], self.orbO.shape[1]]

    @property
    def nvir(self):
        return [self.orbv.shape[1], self.orbV.shape[1]]


class _ULNO_Incore_ERIs(_ULNO_ERIs):
    def _common_init_(self, mlno, mo_coeff=None):
        super()._common_init_(mlno, mo_coeff)
        if mlno._scf._eri is None:
            self._eri = mlno.mol.intor('int2e', aosym='s8')
        else:
            self._eri = mlno._scf._eri

    def get_ivov(self, u, i0, i1, j0, j1):
        orbi = np.dot(self.orbo, u)
        g = ao2mo.general(self._eri, [
                          orbi[:, i0:i1], self.orbv, self.orbo[:, j0:j1], self.orbv], compact=False)
        return g.reshape(orbi[:, i0:i1].shape[1], self.orbv.shape[1], self.orbo[:, j0:j1].shape[1], self.orbv.shape[1])

    def get_ivOV(self, u, i0, i1, j0, j1):
        orbi = np.dot(self.orbo, u)
        g = ao2mo.general(self._eri, [
                          orbi[:, i0:i1], self.orbv, self.orbO[:, j0:j1], self.orbV], compact=False)
        return g.reshape(orbi[:, i0:i1].shape[1], self.orbv.shape[1], self.orbO[:, j0:j1].shape[1], self.orbV.shape[1])

    def get_IVov(self, u, i0, i1, j0, j1):
        orbI = np.dot(self.orbO, u)
        g = ao2mo.general(self._eri, [
                          orbI[:, i0:i1], self.orbV, self.orbo[:, j0:j1], self.orbv], compact=False)
        return g.reshape(orbI[:, i0:i1].shape[1], self.orbV.shape[1], self.orbo[:, j0:j1].shape[1], self.orbv.shape[1])

    def get_IVOV(self, u, i0, i1, j0, j1):
        orbI = np.dot(self.orbO, u)
        g = ao2mo.general(self._eri, [
                          orbI[:, i0:i1], self.orbV, self.orbO[:, j0:j1], self.orbV], compact=False)
        return g.reshape(orbI[:, i0:i1].shape[1], self.orbV.shape[1], self.orbO[:, j0:j1].shape[1], self.orbV.shape[1])

# unrestricted LNO class


class ULNO(lno.LNO):
    def ao2mo(self, mo_coeff=None):
        log = logger.new_logger(self)
        cput0 = (logger.process_clock(), logger.perf_counter())

        if mo_coeff is None:
            mo_coeff = self.mo_coeff

        mos = self.split_mo_coeff(mo_coeff)
        orbocc = [mos[0][1], mos[1][1]]
        orbvir = [mos[0][2], mos[1][2]]

        nmoa, nmob = self.get_nmo()
        mem_now = self.max_memory - lib.current_memory()[0]

        if getattr(self, 'with_df', None):
            naux = self.with_df.get_naoaux()
            nocca, nvira = orbocc[0].shape[1], orbvir[0].shape[1]
            noccb, nvirb = orbocc[1].shape[1], orbvir[1].shape[1]

            dsize = orbocc[0].itemsize
            mem_df = (nocca * nvira + noccb * nvirb) * naux * dsize / 1024**2.
            log.debug('ao2mo est mem= %.2f MB  avail mem= %.2f MB',
                      mem_df, mem_now)

            if ((self._ovL_to_save is not None) or (self._ovL is not None) or
                    self.force_outcore_ao2mo or (mem_df > mem_now * 0.5)):
                eris = _ULNO_DF_Outcore_ERIs(self.with_df, orbocc, orbvir, self.max_memory,
                                             ovL=self._ovL, ovL_to_save=self._ovL_to_save,
                                             verbose=self.verbose, stdout=self.stdout)
            else:
                eris = _ULNO_DF_Incore_ERIs(self.with_df, orbocc, orbvir, self.max_memory,
                                            verbose=self.verbose, stdout=self.stdout)
            eris.build()

        else:
            mem_incore = nmoa**4 * 8 / 1e6 * 4.  # Rough estimate
            log.debug(
                'ao2mo (non-DF) est mem= %.2f MB  avail mem= %.2f MB', mem_incore, mem_now)
            if ((self._scf._eri is not None or
                 mem_incore < mem_now or
                 self.mol.incore_anyway) and
                    not self.force_outcore_ao2mo):
                eris = _ULNO_Incore_ERIs(self, orbocc, orbvir, self.max_memory,
                                         verbose=self.verbose, stdout=self.stdout)
            else:
                raise NotImplementedError(
                    "Unrestricted non-DF out-of-core ERIs not implemented.")

        log.timer('Integral xform   ', *cput0)
        return eris

    get_nocc = mp.ump2.get_nocc
    get_nmo = mp.ump2.get_nmo

    # bug-fix core orbital fixing
    get_frozen_mask = mp.ump2.get_frozen_mask
    #

    split_mo_coeff = mp.dfump2.DFUMP2.split_mo_coeff
    split_mo_energy = mp.dfump2.DFUMP2.split_mo_energy
    split_mo_occ = mp.dfump2.DFUMP2.split_mo_occ
    make_las = make_las
