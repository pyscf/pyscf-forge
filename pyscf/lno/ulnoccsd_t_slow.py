# Copyright 2014-2024 The PySCF Developers. All Rights Reserved.
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
#

import numpy as np
from pyscf import lib

try:
    from cotengra import einsum
except ImportError:
    einsum = lib.einsum

def kernel(mcc, eris, prjlo, t1=None, t2=None):
    '''
    adapted from pyscf.cc.gccsd_t_slow

    Args:
       prjlo[mu,i] = <mu|i> is the overlap between the mu-th LO and the i-th occ MO.
    '''
    if t1 is None or t2 is None:
        t1, t2 = mcc.t1, mcc.t2

    prjloa, prjlob = prjlo
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb = t2ab.shape[:2]
    nvira, nvirb = t2ab.shape[2:]
    mo_ea, mo_eb = eris.mo_energy
    eia = mo_ea[:nocca, None] - mo_ea[nocca:]
    eIA = mo_eb[:noccb, None] - mo_eb[noccb:]
    fvo = eris.focka[nocca:, :nocca]
    fVO = eris.fockb[noccb:, :noccb]

    et = 0
    log = lib.logger.Logger(mcc.stdout, mcc.verbose)
    from multiprocessing import pool
    p = pool.ThreadPool()

    # aaa
    cput0 = (lib.logger.process_clock(), lib.logger.perf_counter())
    bcei = np.array(eris.get_ovvv()).transpose(2, 1, 3, 0)
    bcei -= bcei.transpose(1, 0, 2, 3)
    majk = np.array(eris.ovoo).conj().transpose(2, 1, 3, 0)
    majk -= majk.transpose(0, 1, 3, 2)
    bcjk = np.array(eris.ovov).conj().transpose(1, 3, 0, 2)
    bcjk -= bcjk.transpose(1, 0, 2, 3)

    def t3c_aaa1(a, b, d3):
        t3c = (einsum('jke,cei->ijkc', t2aa[:, :, a, :], bcei[b, :, :, :]) -
               einsum('imc,mjk->ijkc', t2aa[:, :, b], majk[:, a, :, :]))
        t3c = t3c - t3c.transpose(1, 0, 2, 3) - t3c.transpose(2, 1, 0, 3)
        return t3c / d3

    def t3c_aaa2(a, b, d3):
        t3c = (einsum('jkce,ei->ijkc', t2aa, bcei[b, a]) -
               einsum('im,mcjk->ijkc', t2aa[:, :, b, a], majk))
        t3c = t3c - t3c.transpose(1, 0, 2, 3) - t3c.transpose(2, 1, 0, 3)
        return t3c / d3

    def t3d_aaa1(a, b, d3):
        t3d = einsum('i,cjk->ijkc', t1a[:, a], bcjk[b])
        t3d += einsum('i,jkc->ijkc', fvo[a], t2aa[:, :, b])
        t3d = t3d - t3d.transpose(1, 0, 2, 3) - t3d.transpose(2, 1, 0, 3)
        return t3d / d3

    def t3d_aaa2(a, b, d3):
        t3d = einsum('ic,jk->ijkc', t1a, bcjk[b, a])
        t3d += einsum('ci,jk->ijkc', fvo, t2aa[:, :, b, a])
        t3d = t3d - t3d.transpose(1, 0, 2, 3) - t3d.transpose(2, 1, 0, 3)
        return t3d / d3
    #for a in range(nvira):
    def task_a(a):
        et = 0
        for b in range(a+1, nvira):
            d3 = lib.direct_sum(
                'i+j+kc->ijkc', eia[:, a], eia[:, b], eia)
            t3c = 0
            t3c += t3c_aaa1(a, b, d3)
            t3c -= t3c_aaa1(b, a, d3)
            t3c -= t3c_aaa2(a, b, d3)
            t3d = 0
            t3d += t3d_aaa1(a, b, d3)
            t3d -= t3d_aaa1(b, a, d3)
            t3d -= t3d_aaa2(a, b, d3)
            et_ij = einsum('mjkc,njkc,njkc->mn',
                           (t3c+t3d).conj(), d3, t3c) / 9
            et += 2 * einsum('ij,li,lj->', et_ij, prjloa, prjloa)
        return et
    et += sum(p.map(task_a, range(nvira)))
    cput0 = log.timer_debug1('(T) aaa', *cput0)

    # aab
    bCEi = -np.array(eris.get_ovVV()).transpose(1, 2, 3, 0)
    MajK = -np.array(eris.ovOO).transpose(2, 1, 0, 3)
    baei = bcei
    jKmC = np.array(eris.OVoo).transpose(3, 0, 2, 1)
    bCeK = np.array(eris.get_OVvv()).transpose(2, 1, 3, 0)
    maji = majk
    bCjK = np.array(eris.ovOV).transpose(1, 3, 0, 2)
    baji = np.array(eris.ovov).transpose(1, 3, 0, 2)
    baji -= baji.transpose(0, 1, 3, 2)
    # t3c:

    def t3c_aab1(a, b, d3):
        #   Pij Pab [ <bC||Ei> t_jKaE - <jK||Ma> t_MibC ]
        t3c = (einsum('jKE,CEi->ijKC', t2ab[:, :, a, :], bCEi[b]) -
               einsum('iMC,MjK->ijKC', t2ab[:, :, b], MajK[:, a, :, :]))
        t3c = t3c - t3c.transpose(1, 0, 2, 3)
        return t3c / d3

    def t3c_aab2(a, b, d3):
        #   -Pij [ <ba||ei> t_jKCe - <jK||mC> t_miba ] (a <-> C)
        t3c = (einsum('jKeC,ei->ijKC', t2ab, baei[b, a, :, :]) -
               einsum('mi,jKmC->ijKC', t2aa[:, :, b, a], jKmC))
        t3c = t3c - t3c.transpose(1, 0, 2, 3)
        return t3c / d3

    def t3c_aab3(a, b, d3):
        #   -Pab [ <bC||eK> t_jiae - <ji||ma> t_mKbC ] (i <-> K)
        t3c = (einsum('jie,CeK->ijKC',  t2aa[:, :, a, :], bCeK[b]) -
               einsum('mKC,mji->ijKC', -t2ab[:, :, b], maji[:, a, :, :]))
        return -t3c / d3
    # t3d:

    def t3d_aab1(a, b, d3):
        #   Pij Pab ( tia <bC||jK> )
        t3d = einsum('i,CjK->ijKC', t1a[:, a], bCjK[b])
        t3d += einsum('i,jKC->ijKC', fvo[a], t2ab[:, :, b])
        t3d = t3d - t3d.transpose(1, 0, 2, 3)
        return t3d / d3

    def t3d_aab2(a, b, d3):
        #   tKC <ba||ji> (i <-> K, a <-> C)
        t3d = einsum('KC,ji->ijKC', t1b, baji[b, a])
        t3d += einsum('CK,ij->ijKC', fVO, t2aa[:, :, a, b])
        return t3d / d3
    #for a in range(nvira):
    def task_a(a):
        et = 0
        for b in range(a+1, nvira):
            d3 = lib.direct_sum(
                'i+j+kC->ijkC', eia[:, a], eia[:, b], eIA)
            t3c = 0
            t3c += t3c_aab1(a, b, d3)
            t3c -= t3c_aab1(b, a, d3)
            t3c += t3c_aab2(a, b, d3)
            t3c += t3c_aab3(a, b, d3)
            t3c -= t3c_aab3(b, a, d3)
            t3d = 0
            t3d += t3d_aab1(a, b, d3)
            t3d -= t3d_aab1(b, a, d3)
            t3d += t3d_aab2(a, b, d3)
            et_ij = einsum('ijmC,ijnC,ijnC->mn', (t3c+t3d).conj(), d3, t3c)
            et += 2 * einsum('ij,li,lj->', et_ij, prjlob, prjlob) * 3 / 36 * 4
            et_ij = einsum('mjkC,njkC,njkC->mn', (t3c+t3d).conj(), d3, t3c)
            et += 2 * einsum('ij,li,lj->', et_ij, prjloa, prjloa) * 6 / 36 * 4
        return et
    et += sum(p.map(task_a, range(nvira)))
    cput0 = log.timer_debug1('(T) aab', *cput0)

    # bbb
    bcei = np.array(eris.get_OVVV()).transpose(2, 1, 3, 0)
    bcei -= bcei.transpose(1, 0, 2, 3)
    majk = np.array(eris.OVOO).conj().transpose(2, 1, 3, 0)
    majk -= majk.transpose(0, 1, 3, 2)
    bcjk = np.array(eris.OVOV).conj().transpose(1, 3, 0, 2)
    bcjk -= bcjk.transpose(1, 0, 2, 3)

    def t3c_bbb1(a, b, d3):
        t3c = (einsum('jke,cei->ijkc', t2bb[:, :, a], bcei[b]) -
               einsum('imc,mjk->ijkc', t2bb[:, :, b], majk[:, a]))
        t3c = t3c - t3c.transpose(1, 0, 2, 3) - t3c.transpose(2, 1, 0, 3)
        return t3c / d3

    def t3c_bbb2(a, b, d3):
        t3c = (einsum('jkce,ei->ijkc', t2bb, bcei[b, a]) -
               einsum('im,mcjk->ijkc', t2bb[:, :, b, a], majk))
        t3c = t3c - t3c.transpose(1, 0, 2, 3) - t3c.transpose(2, 1, 0, 3)
        return t3c / d3

    def t3d_bbb1(a, b, d3):
        t3d = einsum('i,cjk->ijkc', t1b[:, a], bcjk[b])
        t3d += einsum('i,jkc->ijkc', fVO[a], t2bb[:, :, b])
        t3d = t3d - t3d.transpose(1, 0, 2, 3) - t3d.transpose(2, 1, 0, 3)
        return t3d / d3

    def t3d_bbb2(a, b, d3):
        t3d = einsum('ic,jk->ijkc', t1b, bcjk[b, a])
        t3d += einsum('ci,jk->ijkc', fVO, t2bb[:, :, b, a])
        t3d = t3d - t3d.transpose(1, 0, 2, 3) - t3d.transpose(2, 1, 0, 3)
        return t3d / d3
    #for a in range(nvirb):
    def task_a(a):
        et = 0
        for b in range(a+1, nvirb):
            d3 = lib.direct_sum(
                'i+j+kc->ijkc', eIA[:, a], eIA[:, b], eIA)
            t3c = 0
            t3c += t3c_bbb1(a, b, d3)
            t3c -= t3c_bbb1(b, a, d3)
            t3c -= t3c_bbb2(a, b, d3)
            t3d = 0
            t3d += t3d_bbb1(a, b, d3)
            t3d -= t3d_bbb1(b, a, d3)
            t3d -= t3d_bbb2(a, b, d3)
            et_ij = einsum('mjkc,njkc,njkc->mn',
                           (t3c+t3d).conj(), d3, t3c) / 9
            et += 2 * einsum('ij,li,lj->', et_ij, prjlob, prjlob)
        return et
    et += sum(p.map(task_a, range(nvirb)))
    cput0 = log.timer_debug1('(T) bbb', *cput0)

    # bba
    bCEi = -np.array(eris.get_OVvv()).transpose(1, 2, 3, 0)
    MajK = -np.array(eris.OVoo).transpose(2, 1, 0, 3)
    baei = bcei
    jKmC = np.array(eris.ovOO).transpose(3, 0, 2, 1)
    bCeK = np.array(eris.get_ovVV()).transpose(2, 1, 3, 0)
    maji = majk
    bCjK = np.array(eris.ovOV).transpose(3, 1, 2, 0)
    baji = np.array(eris.OVOV).transpose(1, 3, 0, 2)
    baji -= baji.transpose(0, 1, 3, 2)
    # t3c:

    def t3c_bba1(a, b, d3):
        t3c = (einsum('KjE,CEi->ijKC', t2ab[:, :, :, a], bCEi[b]) -
               einsum('MiC,MjK->ijKC', t2ab[:, :, :, b], MajK[:, a]))
        t3c = t3c - t3c.transpose(1, 0, 2, 3)
        return t3c / d3

    def t3c_bba2(a, b, d3):
        t3c = (einsum('KjCe,ei->ijKC', t2ab, baei[b, a]) -
               einsum('mi,jKmC->ijKC', t2bb[:, :, b, a], jKmC))
        t3c = t3c - t3c.transpose(1, 0, 2, 3)
        return t3c / d3

    def t3c_bba3(a, b, d3):
        #   -Pab [ <bC||eK> t_jiae - <ji||ma> t_mKbC ] (i <-> K)
        t3c = (einsum('jie,CeK->ijKC',  t2bb[:, :, a], bCeK[b]) -
               einsum('KmC,mji->ijKC', -t2ab[:, :, :, b], maji[:, a]))
        return -t3c / d3
    # t3d:

    def t3d_bba1(a, b, d3):
        #   Pij Pab ( tia <bC||jK> )
        t3d = einsum('i,CjK->ijKC', t1b[:, a], bCjK[b])
        t3d += einsum('i,KjC->ijKC', fVO[a], t2ab[:, :, :, b])
        t3d = t3d - t3d.transpose(1, 0, 2, 3)
        return t3d / d3

    def t3d_bba2(a, b, d3):
        #   tKC <ba||ji> (i <-> K, a <-> C)
        t3d = einsum('KC,ji->ijKC', t1a, baji[b, a])
        t3d += einsum('CK,ij->ijKC', fvo, t2bb[:, :, a, b])
        return t3d / d3
    #for a in range(nvirb):
    def task_a(a):
        et = 0
        for b in range(a+1, nvirb):
            d3 = lib.direct_sum(
                'i+j+kc->ijkc', eIA[:, a], eIA[:, b], eia)
            t3c = 0
            t3c += t3c_bba1(a, b, d3)
            t3c -= t3c_bba1(b, a, d3)
            t3c += t3c_bba2(a, b, d3)
            t3c += t3c_bba3(a, b, d3)
            t3c -= t3c_bba3(b, a, d3)
            t3d = 0
            t3d += t3d_bba1(a, b, d3)
            t3d -= t3d_bba1(b, a, d3)
            t3d += t3d_bba2(a, b, d3)
            et_ij = einsum('ijmc,ijnc,ijnc->mn', (t3c+t3d).conj(), d3, t3c)
            et += 2 * einsum('ij,li,lj->', et_ij, prjloa, prjloa) * 3 / 36 * 4
            et_ij = einsum('mjkc,njkc,njkc->mn', (t3c+t3d).conj(), d3, t3c)
            et += 2 * einsum('ij,li,lj->', et_ij, prjlob, prjlob) * 6 / 36 * 4
        return et
    et += sum(p.map(task_a, range(nvirb)))
    cput0 = log.timer_debug1('(T) bba', *cput0)

    et *= .25
    return et
