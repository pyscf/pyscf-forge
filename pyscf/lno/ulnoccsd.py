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

import numpy as np
from functools import reduce

from pyscf.lib import logger
from pyscf import lib
from pyscf.mp.ump2 import get_frozen_mask

from pyscf.lno.ulno import ULNO
from pyscf.lno import lnoccsd
from pyscf.lno.lnoccsd import LNOCCSD, LNOCCSD_T

einsum = lib.einsum

from pyscf.cc import uccsd
def UCCSD(mf, frozen=None, mo_coeff=None, mo_occ=None):
    import numpy
    from pyscf import lib
    from pyscf.soscf import newton_ah
    from pyscf import scf

    #log = logger.new_logger(mf)

    if not mf.istype('UHF'):
        mf = scf.addons.convert_to_uhf(mf)

    if getattr(mf, 'with_df', None):
        mf.with_df.get_naoaux()

    if mo_occ is None:
        mo_occ = mf.mo_occ
    return MODIFIED_UCCSD(mf, frozen, mo_coeff, mo_occ)


class MODIFIED_UCCSD(uccsd.UCCSD):
    def ao2mo(self, mo_coeff=None):
        if self._scf._eri is not None: #and
            #(mem_incore+mem_now < self.max_memory or self.incore_complete)):
            return _make_eris_incore(self, mo_coeff)

        elif getattr(self._scf, 'with_df', None):
            logger.warn(self, 'CCSD detected DF being used in the HF object. '
                        'MO integrals are computed based on the DF 3-index tensors.\n'
                        'It\'s recommended to use dfccsd.CCSD for the '
                        'DF-CCSD calculations')
            return _make_df_eris_outcore(self, mo_coeff)

        else:
            raise NotImplementedError   # should never happen

class _ChemistsERIs(uccsd._ChemistsERIs):
    def _common_init_(self, mycc, mo_coeff=None):
        mymf = mycc._scf

        if mo_coeff is None:
            mo_coeff = mycc.mo_coeff

        moidxa, moidxb = get_frozen_mask(mycc)
        self.mo_coeff = mo_coeff = mo_coeff[0][:,moidxa], mo_coeff[1][:,moidxb]

        # Note: Recomputed fock matrix and HF energy since SCF may not be fully converged.
        # This block is modified to take precomputed 1e integrals
        s1e = getattr(mycc, '_s1e', None)
        h1e = getattr(mycc, '_h1e', None)
        vhf = getattr(mycc, '_vhf', None)
        dm = mymf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
        if vhf is None: vhf = self.get_vhf(mymf, dm, h1e=h1e, s1e=s1e)
        fockao = mymf.get_fock(vhf=vhf, dm=dm, h1e=h1e, s1e=s1e)
        self.focka = reduce(np.dot, (mo_coeff[0].conj().T, fockao[0], mo_coeff[0]))
        self.fockb = reduce(np.dot, (mo_coeff[1].conj().T, fockao[1], mo_coeff[1]))
        self.fock = (self.focka, self.fockb)
        self.e_hf = mymf.energy_tot(dm=dm, vhf=vhf, h1e=h1e)

        nocca, noccb = self.nocc = mycc.nocc
        self.mol = mycc.mol

        # Note self.mo_energy can be different to fock.diagonal().
        # self.mo_energy is used in the initial guess function (to generate
        # MP2 amplitudes) and CCSD update_amps preconditioner.
        # fock.diagonal() should only be used to compute the expectation value
        # of Slater determinants.
        mo_ea = self.focka.diagonal().real
        mo_eb = self.fockb.diagonal().real
        self.mo_energy = (mo_ea, mo_eb)
        gap_a = abs(mo_ea[:nocca,None] - mo_ea[None,nocca:])
        gap_b = abs(mo_eb[:noccb,None] - mo_eb[None,noccb:])
        if gap_a.size > 0:
            gap_a = gap_a.min()
        else:
            gap_a = 1e9
        if gap_b.size > 0:
            gap_b = gap_b.min()
        else:
            gap_b = 1e9
        if gap_a < 1e-5 or gap_b < 1e-5:
            logger.warn(mycc, 'HOMO-LUMO gap (%s,%s) too small for UCCSD',
                        gap_a, gap_b)
        return self

    def get_vhf(self, mymf, dm, h1e=None, s1e=None):
        ''' Build vhf from input dm.

        NOTE:
            If the input dm is the same as the SCF dm, vhf is built directly from the SCF
            MO and MO energy; otherwise, scf.get_vhf is called.
        '''
        dm0 = mymf.make_rdm1()
        errdm = abs(dm0-dm).max()
        if errdm < 1e-6:
            if h1e is None: h1e = mymf.get_hcore()
            if s1e is None: s1e = mymf.get_ovlp()
            moa = np.dot(s1e, mymf.mo_coeff[0])
            mob = np.dot(s1e, mymf.mo_coeff[1])
            moea, moeb = mymf.mo_energy
            vhf = np.asarray([np.dot(moa*moea, moa.T)-h1e, np.dot(mob*moeb, mob.T)-h1e])
        else:
            vhf = mymf.get_veff(mymf.mol, dm)
        return vhf

def _make_eris_incore(mycc, mo_coeff=None):
    from pyscf import ao2mo
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(mycc.stdout, mycc.verbose)
    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)

    moa, mob = eris.mo_coeff
    nocca, noccb = eris.nocc
    #nao = moa.shape[0]
    nmoa = moa.shape[1]
    nmob = mob.shape[1]

    eri_aa = ao2mo.restore(1, ao2mo.full(mycc._scf._eri, moa), nmoa)
    eri_bb = ao2mo.restore(1, ao2mo.full(mycc._scf._eri, mob), nmob)
    eri_ab = ao2mo.general(mycc._scf._eri, (moa,moa,mob,mob), compact=False)
    eri_ba = eri_ab.reshape(nmoa,nmoa,nmob,nmob).transpose(2,3,0,1)

    eri_aa = eri_aa.reshape(nmoa,nmoa,nmoa,nmoa)
    eri_ab = eri_ab.reshape(nmoa,nmoa,nmob,nmob)
    eri_ba = eri_ba.reshape(nmob,nmob,nmoa,nmoa)
    eri_bb = eri_bb.reshape(nmob,nmob,nmob,nmob)
    eris.oooo = eri_aa[:nocca,:nocca,:nocca,:nocca].copy()
    eris.ovoo = eri_aa[:nocca,nocca:,:nocca,:nocca].copy()
    eris.ovov = eri_aa[:nocca,nocca:,:nocca,nocca:].copy()
    eris.oovv = eri_aa[:nocca,:nocca,nocca:,nocca:].copy()
    eris.ovvo = eri_aa[:nocca,nocca:,nocca:,:nocca].copy()
    eris.ovvv = eri_aa[:nocca,nocca:,nocca:,nocca:].copy()
    eris.vvvv = eri_aa[nocca:,nocca:,nocca:,nocca:].copy()

    eris.OOOO = eri_bb[:noccb,:noccb,:noccb,:noccb].copy()
    eris.OVOO = eri_bb[:noccb,noccb:,:noccb,:noccb].copy()
    eris.OVOV = eri_bb[:noccb,noccb:,:noccb,noccb:].copy()
    eris.OOVV = eri_bb[:noccb,:noccb,noccb:,noccb:].copy()
    eris.OVVO = eri_bb[:noccb,noccb:,noccb:,:noccb].copy()
    eris.OVVV = eri_bb[:noccb,noccb:,noccb:,noccb:].copy()
    eris.VVVV = eri_bb[noccb:,noccb:,noccb:,noccb:].copy()

    eris.ooOO = eri_ab[:nocca,:nocca,:noccb,:noccb].copy()
    eris.ovOO = eri_ab[:nocca,nocca:,:noccb,:noccb].copy()
    eris.ovOV = eri_ab[:nocca,nocca:,:noccb,noccb:].copy()
    eris.ooVV = eri_ab[:nocca,:nocca,noccb:,noccb:].copy()
    eris.ovVO = eri_ab[:nocca,nocca:,noccb:,:noccb].copy()
    eris.ovVV = eri_ab[:nocca,nocca:,noccb:,noccb:].copy()
    eris.vvVV = eri_ab[nocca:,nocca:,noccb:,noccb:].copy()

    #eris.OOoo = eri_ba[:noccb,:noccb,:nocca,:nocca].copy()
    eris.OVoo = eri_ba[:noccb,noccb:,:nocca,:nocca].copy()
    #eris.OVov = eri_ba[:noccb,noccb:,:nocca,nocca:].copy()
    eris.OOvv = eri_ba[:noccb,:noccb,nocca:,nocca:].copy()
    eris.OVvo = eri_ba[:noccb,noccb:,nocca:,:nocca].copy()
    eris.OVvv = eri_ba[:noccb,noccb:,nocca:,nocca:].copy()
    #eris.VVvv = eri_ba[noccb:,noccb:,nocca:,nocca:].copy()

    log.timer('CCSD integral transformation', *cput0)
    return eris

def _make_df_eris_outcore(mycc, mo_coeff=None):
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(mycc.stdout, mycc.verbose)
    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)

    moa, mob = eris.mo_coeff
    nocca, noccb = eris.nocc
    nao = moa.shape[0]
    nmoa = moa.shape[1]
    nmob = mob.shape[1]
    nvira = nmoa - nocca
    nvirb = nmob - noccb
    nvira_pair = nvira*(nvira+1)//2
    nvirb_pair = nvirb*(nvirb+1)//2
    naux = mycc._scf.with_df.get_naoaux()

    # --- Three-center integrals
    # (L|aa)
    Loo = np.empty((naux,nocca,nocca))
    Lov = np.empty((naux,nocca,nvira))
    Lvo = np.empty((naux,nvira,nocca))
    Lvv = np.empty((naux,nvira_pair))
    # (L|bb)
    LOO = np.empty((naux,noccb,noccb))
    LOV = np.empty((naux,noccb,nvirb))
    LVO = np.empty((naux,nvirb,noccb))
    LVV = np.empty((naux,nvirb_pair))
    p1 = 0
    oa, va = np.s_[:nocca], np.s_[nocca:]
    ob, vb = np.s_[:noccb], np.s_[noccb:]
    # Transform three-center integrals to MO basis
    for eri1 in mycc._scf.with_df.loop():
        eri1 = lib.unpack_tril(eri1).reshape(-1,nao,nao)
        # (L|aa)
        Lpq = einsum('Lab,ap,bq->Lpq', eri1, moa, moa)
        p0, p1 = p1, p1 + Lpq.shape[0]
        blk = np.s_[p0:p1]
        Loo[blk] = Lpq[:,oa,oa]
        Lov[blk] = Lpq[:,oa,va]
        Lvo[blk] = Lpq[:,va,oa]
        # Lvv[blk] = lib.pack_tril(Lpq[:,va,va].reshape(-1,nvira,nvira))
        # Bugfix (Ardavan) for case where nvirb is 0
        if nvira > 0:
            Lvv[blk] = lib.pack_tril(Lpq[:,va,va].reshape(-1,nvira,nvira))
        else:
            np.empty((Lpq.shape[0], 0), dtype=Lpq.dtype)

        # (L|bb)
        Lpq = einsum('Lab,ap,bq->Lpq', eri1, mob, mob)
        LOO[blk] = Lpq[:,ob,ob]
        LOV[blk] = Lpq[:,ob,vb]
        LVO[blk] = Lpq[:,vb,ob]
        # LVV[blk] = lib.pack_tril(Lpq[:,vb,vb].reshape(-1,nvirb,nvirb))
        # Bugfix (Ardavan) for case where nvirb is 0
        if nvirb > 0:
            LVV[blk] = lib.pack_tril(Lpq[:,vb,vb].reshape(-1,nvirb,nvirb))
        else:
            np.empty((Lpq.shape[0], 0), dtype=Lpq.dtype)

    Loo = Loo.reshape(naux,nocca*nocca)
    Lov = Lov.reshape(naux,nocca*nvira)
    Lvo = Lvo.reshape(naux,nocca*nvira)
    LOO = LOO.reshape(naux,noccb*noccb)
    LOV = LOV.reshape(naux,noccb*nvirb)
    LVO = LVO.reshape(naux,noccb*nvirb)

    # --- Four-center integrals
    dot = lib.ddot
    eris.feri1 = lib.H5TmpFile()
    # (aa|aa)
    eris.oooo = eris.feri1.create_dataset('oooo', (nocca,nocca,nocca,nocca), 'f8')
    eris.oovv = eris.feri1.create_dataset('oovv', (nocca,nocca,nvira,nvira), 'f8', chunks=(nocca,nocca,1,nvira))
    eris.ovoo = eris.feri1.create_dataset('ovoo', (nocca,nvira,nocca,nocca), 'f8', chunks=(nocca,1,nocca,nocca))
    eris.ovvo = eris.feri1.create_dataset('ovvo', (nocca,nvira,nvira,nocca), 'f8', chunks=(nocca,1,nvira,nocca))
    eris.ovov = eris.feri1.create_dataset('ovov', (nocca,nvira,nocca,nvira), 'f8', chunks=(nocca,1,nocca,nvira))
    eris.ovvv = eris.feri1.create_dataset('ovvv', (nocca,nvira,nvira_pair), 'f8')
    eris.vvvv = eris.feri1.create_dataset('vvvv', (nvira_pair,nvira_pair), 'f8')
    eris.oooo[:] = dot(Loo.T, Loo).reshape(nocca,nocca,nocca,nocca)
    eris.ovoo[:] = dot(Lov.T, Loo).reshape(nocca,nvira,nocca,nocca)
    eris.oovv[:] = lib.unpack_tril(dot(Loo.T, Lvv)).reshape(nocca,nocca,nvira,nvira)
    eris.ovvo[:] = dot(Lov.T, Lvo).reshape(nocca,nvira,nvira,nocca)
    eris.ovov[:] = dot(Lov.T, Lov).reshape(nocca,nvira,nocca,nvira)
    eris.ovvv[:] = dot(Lov.T, Lvv).reshape(nocca,nvira,nvira_pair)
    eris.vvvv[:] = dot(Lvv.T, Lvv)
    # (bb|bb)
    eris.OOOO = eris.feri1.create_dataset('OOOO', (noccb,noccb,noccb,noccb), 'f8')
    eris.OOVV = eris.feri1.create_dataset('OOVV', (noccb,noccb,nvirb,nvirb), 'f8', chunks=(noccb,noccb,1,nvirb))
    eris.OVOO = eris.feri1.create_dataset('OVOO', (noccb,nvirb,noccb,noccb), 'f8', chunks=(noccb,1,noccb,noccb))
    eris.OVVO = eris.feri1.create_dataset('OVVO', (noccb,nvirb,nvirb,noccb), 'f8', chunks=(noccb,1,nvirb,noccb))
    eris.OVOV = eris.feri1.create_dataset('OVOV', (noccb,nvirb,noccb,nvirb), 'f8', chunks=(noccb,1,noccb,nvirb))
    eris.OVVV = eris.feri1.create_dataset('OVVV', (noccb,nvirb,nvirb_pair), 'f8')
    eris.VVVV = eris.feri1.create_dataset('VVVV', (nvirb_pair,nvirb_pair), 'f8')
    eris.OOOO[:] = dot(LOO.T, LOO).reshape(noccb,noccb,noccb,noccb)
    eris.OVOO[:] = dot(LOV.T, LOO).reshape(noccb,nvirb,noccb,noccb)
    eris.OOVV[:] = lib.unpack_tril(dot(LOO.T, LVV)).reshape(noccb,noccb,nvirb,nvirb)
    eris.OVVO[:] = dot(LOV.T, LVO).reshape(noccb,nvirb,nvirb,noccb)
    eris.OVOV[:] = dot(LOV.T, LOV).reshape(noccb,nvirb,noccb,nvirb)
    eris.OVVV[:] = dot(LOV.T, LVV).reshape(noccb,nvirb,nvirb_pair)
    eris.VVVV[:] = dot(LVV.T, LVV)
    # (aa|bb)
    eris.ooOO = eris.feri1.create_dataset('ooOO', (nocca,nocca,noccb,noccb), 'f8')
    eris.ooVV = eris.feri1.create_dataset('ooVV', (nocca,nocca,nvirb,nvirb), 'f8', chunks=(nocca,nocca,1,nvirb))
    eris.ovOO = eris.feri1.create_dataset('ovOO', (nocca,nvira,noccb,noccb), 'f8', chunks=(nocca,1,noccb,noccb))
    eris.ovVO = eris.feri1.create_dataset('ovVO', (nocca,nvira,nvirb,noccb), 'f8', chunks=(nocca,1,nvirb,noccb))
    eris.ovOV = eris.feri1.create_dataset('ovOV', (nocca,nvira,noccb,nvirb), 'f8', chunks=(nocca,1,noccb,nvirb))
    eris.ovVV = eris.feri1.create_dataset('ovVV', (nocca,nvira,nvirb_pair), 'f8')
    eris.vvVV = eris.feri1.create_dataset('vvVV', (nvira_pair,nvirb_pair), 'f8')
    eris.ooOO[:] = dot(Loo.T, LOO).reshape(nocca,nocca,noccb,noccb)
    eris.ovOO[:] = dot(Lov.T, LOO).reshape(nocca,nvira,noccb,noccb)
    eris.ooVV[:] = lib.unpack_tril(dot(Loo.T, LVV)).reshape(nocca,nocca,nvirb,nvirb)
    eris.ovVO[:] = dot(Lov.T, LVO).reshape(nocca,nvira,nvirb,noccb)
    eris.ovOV[:] = dot(Lov.T, LOV).reshape(nocca,nvira,noccb,nvirb)
    eris.ovVV[:] = dot(Lov.T, LVV).reshape(nocca,nvira,nvirb_pair)
    eris.vvVV[:] = dot(Lvv.T, LVV)
    # (bb|aa)
    eris.OOvv = eris.feri1.create_dataset('OOvv', (noccb,noccb,nvira,nvira), 'f8', chunks=(noccb,noccb,1,nvira))
    eris.OVoo = eris.feri1.create_dataset('OVoo', (noccb,nvirb,nocca,nocca), 'f8', chunks=(noccb,1,nocca,nocca))
    eris.OVvo = eris.feri1.create_dataset('OVvo', (noccb,nvirb,nvira,nocca), 'f8', chunks=(noccb,1,nvira,nocca))
    eris.OVvv = eris.feri1.create_dataset('OVvv', (noccb,nvirb,nvira_pair), 'f8')
    eris.OVoo[:] = dot(LOV.T, Loo).reshape(noccb,nvirb,nocca,nocca)
    eris.OOvv[:] = lib.unpack_tril(dot(LOO.T, Lvv)).reshape(noccb,noccb,nvira,nvira)
    eris.OVvo[:] = dot(LOV.T, Lvo).reshape(noccb,nvirb,nvira,nocca)
    eris.OVvv[:] = dot(LOV.T, Lvv).reshape(noccb,nvirb,nvira_pair)

    log.timer('CCSD integral transformation', *cput0)
    return eris


def impurity_solve(mcc, mo_coeff, uocc_loc, mo_occ, maskact, eris,
                   ccsd_t=False, log=None, verbose_imp=None,
                   max_las_size_ccsd=1000, max_las_size_ccsd_t=1000):

    log = logger.new_logger(mcc if log is None else log)
    cput1 = (logger.process_clock(), logger.perf_counter())

    occidxa = mo_occ[0]>1e-10
    occidxb = mo_occ[1]>1e-10
    nmo = mo_occ[0].size, mo_occ[1].size
    moidxa, moidxb = maskact

    orbfrzocca = mo_coeff[0][:, ~moidxa &  occidxa]
    orbactocca = mo_coeff[0][:,  moidxa &  occidxa]
    orbactvira = mo_coeff[0][:,  moidxa & ~occidxa]
    orbfrzvira = mo_coeff[0][:, ~moidxa & ~occidxa]
    nfrzocca, nactocca, nactvira, nfrzvira = [orb.shape[1]
                                              for orb in [orbfrzocca,orbactocca,
                                                          orbactvira,orbfrzvira]]
    orbfrzoccb = mo_coeff[1][:, ~moidxb &  occidxb]
    orbactoccb = mo_coeff[1][:,  moidxb &  occidxb]
    orbactvirb = mo_coeff[1][:,  moidxb & ~occidxb]
    orbfrzvirb = mo_coeff[1][:, ~moidxb & ~occidxb]
    nfrzoccb, nactoccb, nactvirb, nfrzvirb = [orb.shape[1]
                                              for orb in [orbfrzoccb,orbactoccb,
                                                          orbactvirb,orbfrzvirb]]
    nlo = [uocc_loc[0].shape[1], uocc_loc[1].shape[1]]
    prjlo = [uocc_loc[0].T.conj(), uocc_loc[1].T.conj()]

    log.debug('    impsol:  alpha  %d LOs  %d/%d MOs  %d occ  %d vir',
              nlo[0], nactocca+nactvira, nmo[0], nactocca, nactvira)
    log.debug('    impsol:  beta   %d LOs  %d/%d MOs  %d occ  %d vir',
              nlo[1], nactoccb+nactvirb, nmo[1], nactoccb, nactvirb)

    if nactocca * nactvira == 0 and nactoccb * nactvirb == 0:
        elcorr_pt2 = lib.tag_array(0., spin_comp=np.array((0., 0.)))
        elcorr_cc = lib.tag_array(0., spin_comp=np.array((0., 0.)))
        elcorr_cc_t = 0.
    else:
        # solve impurity problem
        imp_eris = mcc.ao2mo()
        cput1 = log.timer_debug1('imp sol - eri    ', *cput1)
        # MP2 fragment energy
        t1, t2 = mcc.init_amps(eris=imp_eris)[1:]
        cput1 = log.timer_debug1('imp sol - mp2 amp', *cput1)
        elcorr_pt2 = get_fragment_energy(imp_eris, t1, t2, prjlo)
        cput1 = log.timer_debug1('imp sol - mp2 ene', *cput1)
        # CCSD fragment energy
        t1, t2 = mcc.kernel(eris=imp_eris, t1=t1, t2=t2)[1:]
        cput1 = log.timer_debug1('imp sol - cc  amp', *cput1)
        elcorr_cc = get_fragment_energy(imp_eris, t1, t2, prjlo)
        cput1 = log.timer_debug1('imp sol - cc  ene', *cput1)
        if ccsd_t:
            from pyscf.lno.ulnoccsd_t_slow import kernel as UCCSD_T
            elcorr_cc_t = UCCSD_T(mcc, imp_eris, prjlo, t1=t1, t2=t2)
            cput1 = log.timer_debug1('imp sol - cc  (T)', *cput1)
        else:
            elcorr_cc_t = 0.

    frag_msg = '  '.join([f'E_corr(MP2) = {elcorr_pt2:.15g}',
                          f'E_corr(CCSD) = {elcorr_cc:.15g}',
                          f'E_corr(CCSD(T)) = {elcorr_cc_t:.15g}'])

    t1 = t2 = imp_eris = None

    return (elcorr_pt2, elcorr_cc, elcorr_cc_t), frag_msg

def get_fragment_energy(eris, t1, t2, prj):
    prja, prjb = prj
    eris_ovov = np.asarray(eris.ovov)
    eris_OVOV = np.asarray(eris.OVOV)
    eris_ovOV = np.asarray(eris.ovOV)
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb = t2ab.shape[:2]
    fov = eris.focka[:nocca,nocca:]
    fOV = eris.fockb[:noccb,noccb:]
    ea = einsum('ia,ka->ik', fov, t1a)
    eb = einsum('ia,ka->ik', fOV, t1b)
    ea += 0.25 * einsum('ijab,kajb->ik', t2aa, eris_ovov)
    ea -= 0.25 * einsum('ijab,kbja->ik', t2aa, eris_ovov)
    eb += 0.25 * einsum('ijab,kajb->ik', t2bb, eris_OVOV)
    eb -= 0.25 * einsum('ijab,kbja->ik', t2bb, eris_OVOV)
    ea +=  0.5 * einsum('iJaB,kaJB->ik', t2ab, eris_ovOV)
    eb +=  0.5 * einsum('iJaB,iaKB->JK', t2ab, eris_ovOV)
    ea +=  0.5 * einsum('ia,jb,kajb->ik', t1a, t1a, eris_ovov)
    ea -=  0.5 * einsum('ia,jb,kbja->ik', t1a, t1a, eris_ovov)
    eb +=  0.5 * einsum('ia,jb,kajb->ik', t1b, t1b, eris_OVOV)
    eb -=  0.5 * einsum('ia,jb,kbja->ik', t1b, t1b, eris_OVOV)
    ea +=  0.5 * einsum('ia,jb,kajb->ik', t1a, t1b, eris_ovOV)
    eb +=  0.5 * einsum('ia,jb,iakb->jk', t1a, t1b, eris_ovOV)

    e  = einsum('ik,li,lk->', ea, prja, prja)
    e += einsum('ik,li,lk->', eb, prjb, prjb)
    return lib.tag_array(e, spin_comp=np.array((0., 0.)))

def get_maskact(frozen, nmo):
    maskact = [None,] * 2
    for s in range(2):
        if len(frozen[s])>0:
            frozen[s], maskact[s] = lnoccsd.get_maskact(frozen[s], nmo[s])
        else:
            #update for domain lno-ccsd
            _, maskact[s] = lnoccsd.get_maskact(frozen[s], nmo[s])
    return frozen, maskact

def fock_from_mo(mymf, s1e=None, force_exxdiv_none=True):
    if s1e is None:
        s1e = mymf.get_ovlp()
    fock = []
    for s in range(2):
        mo0 = np.dot(s1e, mymf.mo_coeff[s])
        moe0 = mymf.mo_energy[s]
        fock.append(np.dot(mo0 * moe0, mo0.T.conj()))
    return fock

class ULNOCCSD(ULNO, LNOCCSD):

    get_frozen_mask = get_frozen_mask

    def _precompute(self):
        log = logger.new_logger(self)
        mf = self._scf
        s1e = self.s1e
        h1e = self.h1e
        if self._vhf is None:
            log.warn('Input vhf is not found. Building vhf from SCF MO.')
            self._vhf = fock_from_mo(mf, s1e=s1e, force_exxdiv_none=True) - h1e

    def impurity_solve(self, mf, mo_coeff, uocc_loc, eris, frozen=None, log=None):
        if log is None:
            log = logger.new_logger(self)
        mo_occ = self.mo_occ
        frozen, maskact = get_maskact(frozen, [mo_occ[0].size, mo_occ[1].size])
        mcc = UCCSD(mf, mo_coeff=mo_coeff, frozen=frozen).set(verbose=self.verbose_imp)
        mcc._s1e = self._s1e
        mcc._h1e = self._h1e
        mcc._vhf = self._vhf
        if self.kwargs_imp is not None:
            mcc = mcc.set(**self.kwargs_imp)

        return impurity_solve(mcc, mo_coeff, uocc_loc, mo_occ, maskact, eris, log=log,
                              ccsd_t=self.ccsd_t, verbose_imp=self.verbose_imp,
                              max_las_size_ccsd=self._max_las_size_ccsd,
                              max_las_size_ccsd_t=self._max_las_size_ccsd_t)


class ULNOCCSD_T(ULNOCCSD, LNOCCSD_T):
    pass


if __name__ == '__main__':
    from pyscf import gto, lo, scf, mp, cc

    # S22-2: water dimer
    atom = '''
        O   -1.485163346097   -0.114724564047    0.000000000000
        H   -1.868415346097    0.762298435953    0.000000000000
        H   -0.533833346097    0.040507435953    0.000000000000
        O    1.416468653903    0.111264435953    0.000000000000
        H    1.746241653903   -0.373945564047   -0.758561000000
        H    1.746241653903   -0.373945564047    0.758561000000
    '''
    basis = 'cc-pvdz'

    mol = gto.M(atom=atom, basis=basis, spin=0, verbose=5, max_memory=16000)
    mf = scf.UHF(mol).density_fit()
    mf.kernel()

    frozen = 2
    # canonical
    mmp = mp.UMP2(mf, frozen=frozen)
    mmp.kernel()

    mcc = cc.UCCSD(mf, frozen=frozen)
    eris = mcc.ao2mo()
    mcc.kernel(eris=eris)
    eccsd_t = mcc.ccsd_t(eris=eris)

    # PM
    orbocca = mf.mo_coeff[0][:,frozen:np.count_nonzero(mf.mo_occ[0])]
    orbloca = lo.PipekMezey(mol, orbocca).kernel()
    orboccb = mf.mo_coeff[1][:,frozen:np.count_nonzero(mf.mo_occ[1])]
    orblocb = lo.PipekMezey(mol, orboccb).kernel()
    orbloc = [orbloca, orblocb]

    # LNO
    lno_type = ['1h'] * 2
    lno_thresh = [1e-4] * 2
    oa = [[[i],[]] for i in range(orbloca.shape[1])]
    ob = [[[],[i]] for i in range(orblocb.shape[1])]
    frag_lolist = oa + ob

    mlno = ULNOCCSD_T(mf, orbloc, frag_lolist, lno_type=lno_type, lno_thresh=lno_thresh, frozen=frozen)
    mlno.lo_proj_thresh_active = None
    mlno.verbose_imp = 4
    mlno.kernel()
    ecc = mlno.e_corr_ccsd
    ecc_t = mlno.e_corr_ccsd_t
    ecc_pt2corrected = mlno.e_corr_ccsd_pt2corrected(mmp.e_corr)
    ecc_t_pt2corrected = mlno.e_corr_ccsd_t_pt2corrected(mmp.e_corr)
    log = logger.new_logger(mol)
    log.info('lno_thresh = %s\n'
             '    E_corr(CCSD)     = %.10f  rel = %6.2f%%  '
             'diff = % .10f',
             lno_thresh, ecc, ecc/mcc.e_corr*100, ecc-mcc.e_corr)
    log.info('    E_corr(CCSD_T)   = %.10f  rel = %6.2f%%  '
             'diff = % .10f',
             ecc_t, ecc_t/(mcc.e_corr+eccsd_t)*100,
             ecc_t-(mcc.e_corr+eccsd_t))
    log.info('    E_corr(CCSD+PT2) = %.10f  rel = %6.2f%%  '
             'diff = % .10f',
             ecc_pt2corrected, ecc_pt2corrected/mcc.e_corr*100,
             ecc_pt2corrected - mcc.e_corr)
    log.info('    E_corr(CCSD_T+PT2)   = %.10f  rel = %6.2f%%  '
             'diff = % .10f',
             ecc_t_pt2corrected, ecc_t_pt2corrected/(mcc.e_corr+eccsd_t)*100,
             ecc_t_pt2corrected-(mcc.e_corr+eccsd_t))
