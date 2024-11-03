#!/usr/bin/env python
#
# Copyright 2024 The PySCF Developers. All Rights Reserved.
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
# Author: Peng Bao <baopeng@iccas.ac.cn>
# Edited by: Qiming Sun <osirpt.sun@gmail.com>

'''
Multistate Density Functional Theory (MSDFT)

References:
[1] Block-Localized Excitation for Excimer Complex and Diabatic Coupling
    Peng Bao, Christian P. Hettich, Qiang Shi, and Jiali Gao
    J. Chem. Theory Comput. 2021, 17, 240-254
[2] Block-Localized Density Functional Theory (BLDFT), Diabatic Coupling, and
    Their Use in Valence Bond Theory for Representing Reactive Potential Energy
    Surfaces
    Alessandro Cembran, Lingchun Song, Yirong Mo and Jiali Gao
    J. Chem. Theory Comput. 2009, 5, 2702-2716
[3] Beyond Kohn-Sham Approximation: Hybrid Multistate Wave Function and
    Density Functional Theory
    Jiali Gao, Adam Grofe, Haisheng Ren, Peng Bao
    J. Phys. Chem. Lett. 2016, 7, 5143-5149
[4] Spin-Multiplet Components and Energy Splittings by Multistate Density
    Functional Theory
    Adam Grofe, Xin Chen, Wenjian Liu, Jiali Gao
    J. Phys. Chem. Lett. 2017, 8, 4838-4845
'''

import numpy as np
import scipy.linalg
from pyscf import lib
from pyscf import scf
from pyscf import dft
from pyscf.lib import logger
from pyscf.data.nist import HARTREE2EV as au2ev

__all__ = ['NOCI']

SVD_THRESHOLD = 1e-11

def hf_det_ovlp(msks, mfs):
    '''Compute the standard interaction <I|H|J> between two non-orthogonal
    determinants I and J
    '''
    mol = msks.mol
    _mf = mfs[0].copy()
    ovlp = _mf.get_ovlp()
    neleca, nelecb = _mf.nelec
    occ_mos = []
    for mf in mfs:
        mo_coeff_a = mf.mo_coeff[0]
        mo_coeff_b = mf.mo_coeff[1]
        mo_occ_a = mf.mo_occ[0]
        mo_occ_b = mf.mo_occ[1]
        occ_mo_a = mo_coeff_a[:,mo_occ_a>0]
        occ_mo_b = mo_coeff_b[:,mo_occ_b>0]
        occ_mos.append([occ_mo_a, occ_mo_b])
        if occ_mo_a.shape[1] != neleca or occ_mo_b.shape[1] != nelecb:
            raise RuntimeError('Electron numbers must be equal')

    dms = []
    det_ovlp = []
    for i, mf_bra in enumerate(mfs):
        c1_a, c1_b = occ_mos[i]
        for j, mf_ket in enumerate(mfs[:i]):
            c2_a, c2_b = occ_mos[j]
            o_a = c1_a.conj().T.dot(ovlp).dot(c2_a)
            o_b = c1_b.conj().T.dot(ovlp).dot(c2_b)

            u_a, s_a, vt_a = scipy.linalg.svd(o_a)
            u_b, s_b, vt_b = scipy.linalg.svd(o_b)
            s_a = np.where(abs(s_a) > SVD_THRESHOLD, s_a, SVD_THRESHOLD)
            s_b = np.where(abs(s_b) > SVD_THRESHOLD, s_b, SVD_THRESHOLD)
            x_a = (u_a/s_a).dot(vt_a)
            x_b = (u_b/s_b).dot(vt_b)
            phase = (np.linalg.det(u_a) * np.linalg.det(u_b) *
                     np.linalg.det(vt_a) * np.linalg.det(vt_b))
            det_ovlp.append(phase * np.prod(s_a)*np.prod(s_b))

            # One-particle asymmetric density matrix. See also pyscf.scf.uhf.make_asym_dm
            dm_01a = c1_a.dot(x_a).dot(c2_a.conj().T)
            dm_01b = c1_b.dot(x_b).dot(c2_b.conj().T)
            dms.append((dm_01a, dm_01b))

        det_ovlp.append(1.)
        dm_a = c1_a.dot(c1_a.conj().T)
        dm_b = c1_b.dot(c1_b.conj().T)
        dms.append((dm_a, dm_b))

    dms = np.stack(dms)
    hcore = _mf.get_hcore()
    #FIXME: dms might be very huge, integral screening does not work well in get_jk
    vjs, vks = scf.uhf.UHF.get_jk(_mf, mol, dms, hermi=0)
    h = []
    for dm_01, vj, vk in zip(dms, vjs, vks):
        vhf_01 = vj[0] + vj[1] - vk
        e_tot = scf.uhf.energy_elec(_mf, dm_01, hcore, vhf_01)[0]
        h.append(e_tot)
    det_ovlp = np.asarray(det_ovlp)
    h = lib.unpack_tril(det_ovlp * np.asarray(h))
    s = lib.unpack_tril(det_ovlp)
    return h, s

def multi_states_scf(msks, ground_ks=None):
    '''Construct multiple Kohn-Sham instances for states specified in MSDFT'''
    log = logger.new_logger(msks)
    if ground_ks is None:
        ground_ks = dft.UKS(msks.mol, xc=msks.xc).run()
    else:
        assert isinstance(ground_ks, dft.uks.UKS)

    neleca, nelecb = ground_ks.nelec
    assert neleca == nelecb

    mfs_s = []
    mfs_t = []
    for n, (i, a) in enumerate(msks.s):
        log.debug('KS for single excitation %s->%s', i, a)
        occ = ground_ks.mo_occ.copy()
        occ[0][i] = 0
        occ[0][a] = 1
        mf = ground_ks.copy()
        mf = scf.addons.mom_occ(mf, ground_ks.mo_coeff, occ)
        dm_init = mf.make_rdm1(ground_ks.mo_coeff, occ)
        mf.kernel(dm0=dm_init)
        mfs_s.append(mf)
        # single excitation for beta electrons
        mf = mf.copy()
        mf.mo_coeff = mf.mo_coeff[::-1]
        mf.mo_occ = mf.mo_occ[::-1]
        mf.mo_energy = mf.mo_energy[::-1]
        mfs_s.append(mf)

        # spin-flip excitation
        log.debug('KS for spin-flip single excitation %s->%s', i, a)
        occ = ground_ks.mo_occ.copy()
        occ[1][i] = 0
        occ[0][a] = 1
        mf = ground_ks.copy()
        mf.nelec = neleca+1, nelecb-1
        mf = scf.addons.mom_occ(mf, ground_ks.mo_coeff, occ)
        dm_init = mf.make_rdm1(ground_ks.mo_coeff, occ)
        mf.kernel(dm0=dm_init)
        mfs_t.append(mf)

    mfs_d = []
    for n, (i, a) in enumerate(msks.d):
        log.debug('KS for double excitation (%s,%s)->(%s,%s)', i, i, a, a)
        occ = ground_ks.mo_occ.copy()
        occ[0][i] = 0
        occ[0][a] = 1
        occ[1][i] = 0
        occ[1][a] = 1
        mf = ground_ks.copy()
        mf = scf.addons.mom_occ(mf, ground_ks.mo_coeff, occ)
        dm_init = mf.make_rdm1(ground_ks.mo_coeff, occ)
        mf.kernel(dm0=dm_init)
        mfs_d.append(mf)

    e_g = ground_ks.e_tot
    log.info('Ground state KS energy = %g', e_g)
    log.info('Doubly excited energy:')
    for i, mf in enumerate(mfs_d):
        e_d = mf.e_tot
        log.info('%-2d %18.15g AU %12.6g eV', i+1, e_d, (e_d-e_g)*au2ev)

    log.info('Single and triple excitation:')
    log.info('       E(S)                  E(T)                   dEt               dEs')
    for i, (mf_s, mf_t) in enumerate(zip(mfs_s[::2], mfs_t)):
        dEt = (mf_t.e_tot - e_g) * au2ev
        e_split = (mf_s.e_tot - mf_t.e_tot) * au2ev
        log.info('%-2d %18.15g AU %18.15g AU %15.9g eV %15.9g eV',
                 i+1, mf_s.e_tot, mf_t.e_tot, dEt, e_split)
    return mfs_s + mfs_d + [ground_ks]

class NOCI(lib.StreamObject):
    '''
    Nonorthogonal Configuration Interaction (NOCI) of Multistate Density Functional Theory (MSDFT)

    Attributes:
        xc : str
            Name of exchange-correlation functional
        coup : int
            How to compute the electronic coupling between diabatic states (Bao, JCTC, 17, 240).
            * 0: geometric average over diagonal terms.
            * 1: determinant-weighted average of correlation.
            * 2 (default): overlap-scaled average of correlation.
        ci_g : bool
            Whether to compute the adiabatic ground-state energy. True by default.
        s :
            A list of singly excited orbital pairs. Each pair [i, a] means an
            excitation from occupied orbital i to a.
        d :
            A list of doubly excited orbital pairs. Each pair [i, a] means both
            alpha and beta electrons at orbital i are excited to orbital a.

    Saved results:
        e_tot : float
            Total HF energy (electronic energy plus nuclear repulsion)
        csfvec : array
            CI coefficients
        mfs :
            KS instances of the underlying diabatic states)
    '''
    _keys = {
        'mol', 'verbose', 'stdout', 'xc', 'coup', 'ci_g', 's', 'd',
        'e_tot', 'csfvec', 'mfs',
    }

    def __init__(self, mol, xc=None, coup=2):
        self.mol = mol
        self.verbose = mol.verbose
        self.stdout = mol.stdout
        self.xc = xc
        self.coup = 2
        self.ci_g = True
        self.s = []
        self.d = []
##################################################
# don't modify the following attributes, they are not input options
        self.e_tot = 0
        self.csfvec = None
        self.mfs = None

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        if log.verbose < logger.INFO:
            return self

        log.info('\n')
        log.info('******** %s ********', self.__class__)
        log.info('xc = %s', self.xc)
        log.info('coup = %s', self.coup)
        log.info('ci_g = %s', self.ci_g)
        log.info('single excitation = %s', self.s)
        log.info('double excitation = %s', self.d)
        return self

    def kernel(self, ground_ks=None):
        log = logger.new_logger(self)
        cpu0 = (logger.process_clock(), logger.perf_counter())
        self.dump_flags(log)
        self.check_sanity()

        mfs = self.mfs = multi_states_scf(self, ground_ks)
        e_hf, s_csf = hf_det_ovlp(self, mfs)
        Enuc = self.mol.energy_nuc()
        e_ks = np.array([mf.e_tot for mf in mfs])
        e_ks -= Enuc

        # Compute transition density functional energy
        if self.coup == 0:
            # geometric average over diagonal terms.
            d = e_ks / e_hf.diagonal()
            h_tdf = e_hf * (d[:,None] * d)**.5 + s_csf * Enuc
        elif self.coup == 1:
            d = e_hf.diagonal()
            # determinant-weighted average of correlation.
            h_tdf = e_hf * (e_ks[:,None]+e_ks) / (d[:,None]+d) + s_csf * Enuc
        elif self.coup == 2:
            # overlap-scaled average of correlation.
            d = e_ks - e_hf.diagonal()
            h_tdf = e_hf + s_csf * ((d[:,None] + d) / 2 + Enuc)

        self.e_tot, self.csfvec = scipy.linalg.eigh(h_tdf, s_csf)
        log.note('MSDFT eigs %s', self.e_tot)
        log.timer('MSDFT', *cpu0)
        return self.e_tot

    @property
    def converged(self):
        return all(mf.converged for mf in self.mfs)
