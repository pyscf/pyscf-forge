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

def hf_det_ovlp(msks, mfs):
    '''Compute the standard interaction <I|H|J> between two non-orthogonal
    determinants I and J
    '''
    log = logger.new_logger(msks)
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
        assert mo_coeff_a.dtype == np.float64

    # <I|H|J> can be evaluated using
    # * the generalized Slater-Condon rule (J. Chem. Phys. 131, 124113, 2009) or
    # * the density matrix limitation (dml) method (J. Am. Chem. SOC. 1990, 112, 4214).
    #   E = Tr(dm_12, H[dm_12]) where dm_12 = T_I S_{IJ}^{-1} T_J.
    #def dml(mo1_a, mo1_b, mo2_a, mo2_b, f):
    #    '''See also the det_ovlp function in test_noci.py'''
    #    o_a = mo1_a.conj().T.dot(s).dot(mo2_a)
    #    o_b = mo1_b.conj().T.dot(s).dot(mo2_b)
    #    u_a, s_a, vt_a = scipy.linalg.svd(o_a)
    #    u_b, s_b, vt_b = scipy.linalg.svd(o_b)
    #    s_a = np.where(abs(s_a) > 1e-11, s_a, 1e-11)
    #    s_b = np.where(abs(s_b) > 1e-11, s_b, 1e-11)
    #    x_a = (u_a/s_a).dot(vt_a)
    #    x_b = (u_b/s_b).dot(vt_b)
    #    phase = (np.linalg.det(u_a) * np.linalg.det(u_b) *
    #             np.linalg.det(vt_a) * np.linalg.det(vt_b))
    #    det_ovlp = phase * np.prod(s_a)*np.prod(s_b)
    #    # One-particle asymmetric density matrix. See also pyscf.scf.uhf.make_asym_dm
    #    dm_a = mo1_a.dot(x_a).dot(mo2_a.conj().T)
    #    dm_b = mo1_b.dot(x_b).dot(mo2_b.conj().T)
    #    dm_01 = (dm_a, dm_b)
    #    return scf.uhf.UHF.energy_elec(mol, dm_01) * det_ovlp
    #
    # when I and J differ by symmetry and <I|H|J> is strictly zero, the density
    # matrix limitation method may encounter numerical issues since |S_{IJ}| is
    # strictly zero.
    # Here, citing the discussions in https://github.com/pyscf/pyscf-forge/pull/77:
    # * If there is only one zero singular value due to symmetry, then the symmetry of
    #   the two-electron integrals will ultimately cancel the crazy (AO-basis) JK
    #   matrix. (ii'|jj') and (ij'|ji') are only nonzero if j\to j' corresponds to the
    #   same symmetry element as i\to i', but this would imply that the jth singular
    #   value must also be zero, and the i==j case is canceled by exchange. So the
    #   two-electron part of the energy can't contribute, and neither can the
    #   one-electron part because it will be zero by symmetry.
    # * If there are two singular values corresponding to the same symmetry change,
    #   then the two states do not in fact have different symmetries. The artificial
    #   1e-11 floored singular values cancel between the two factors of the density
    #   matrix and the final multiplication by det_ovlp, and all is well.
    # * If there are more than two zero singular values, then neither of the two terms
    #   of the Hamiltonian can cancel all of the 1e-11 factors in det_ovlp, and the
    #   whole thing is at most ~1e-11, which is close enough to zero in most cases.
    #
    # Below, <I|H|J> is evaluated using dml when zero singular values in the
    # orbital overlap. Otherwise, generalized Slater-Condon rule is applied.

    # Five elements are cached in det_ovlp_cache:
    # * Determinants overlap
    # * Asymmetric density matrix for evaluating JK
    # * The second density matrix for computing HF energy: Tr(dm, hcore+VHF/2)
    # * A factor for generalized Slater-Condon integral
    # * Number of different spin-orbitals
    det_ovlp_cache = {}
    svd_threshold = mfks.svd_threshold
    for i, mf_bra in enumerate(mfs):
        mo1_a, mo1_b = occ_mos[i]
        for j, mf_ket in enumerate(mfs[:i]):
            mo2_a, mo2_b = occ_mos[j]
            o_a = mo1_a.conj().T.dot(ovlp).dot(mo2_a)
            o_b = mo1_b.conj().T.dot(ovlp).dot(mo2_b)

            u_a, s_a, vt_a = scipy.linalg.svd(o_a)
            u_b, s_b, vt_b = scipy.linalg.svd(o_b)
            s_a_overlapped = s_a[s_a > svd_threshold]
            s_b_overlapped = s_b[s_b > svd_threshold]
            differs_a = s_a.size - s_a_overlapped.size
            differs_b = s_b.size - s_b_overlapped.size
            differs = differs_a + differs_b
            log.debug1('states = (%d %d), GSC differs = %d', i, j, differs)

            if differs == 0:
                # Evaluate <I|H|J> using the density matrix limitation method
                det_ovlp = np.linalg.det(o_a) * np.linalg.det(o_b)
                # One-particle asymmetric density matrix. See also pyscf.scf.uhf.make_asym_dm
                dm_a = mo1_a.dot(np.linalg.solve(o_a.T, mo2_a.conj().T))
                dm_b = mo1_b.dot(np.linalg.solve(o_b.T, mo2_b.conj().T))
                dm_01 = (dm_a, dm_b)
                det_ovlp_cache[i, j] = (det_ovlp, dm_01, dm_01, det_ovlp, differs)

            elif differs == 1: # Generalized Slater-Condon rule
                det_ovlp = np.linalg.det(o_a) * np.linalg.det(o_b)
                c1_a = mo1_a.dot(u_a[:,s_a<svd_threshold])
                c1_b = mo1_b.dot(u_b[:,s_b<svd_threshold])
                c2_a = mo2_a.dot(vt_a[s_a<svd_threshold].conj().T)
                c2_b = mo2_b.dot(vt_b[s_b<svd_threshold].conj().T)
                dm_a = c1_a.dot(c2_a.conj().T)
                dm_b = c1_b.dot(c2_b.conj().T)
                dm_01 = (dm_a, dm_b)

                x_a = u_a[:,s_a>svd_threshold].dot(vt_a[s_a>svd_threshold])
                x_b = u_b[:,s_b>svd_threshold].dot(vt_b[s_b>svd_threshold])
                dm_a = mo1_a.dot(x_a).dot(mo2_a.conj().T)
                dm_b = mo1_b.dot(x_b).dot(mo2_b.conj().T)
                dm_r = (dm_a, dm_b)

                phase = (np.linalg.det(u_a) * np.linalg.det(u_b) *
                         np.linalg.det(vt_a) * np.linalg.det(vt_b))
                fac = phase * np.prod(s_a_overlapped)*np.prod(s_b_overlapped)

                det_ovlp_cache[i, j] = (det_ovlp, dm_r, dm_01, fac, differs)

            elif differs == 2: # Generalized Slater-Condon rule
                det_ovlp = np.linalg.det(o_a) * np.linalg.det(o_b)
                c1_a = mo1_a.dot(u_a[:,s_a<svd_threshold])
                c1_b = mo1_b.dot(u_b[:,s_b<svd_threshold])
                c2_a = mo2_a.dot(vt_a[s_a<svd_threshold].conj().T)
                c2_b = mo2_b.dot(vt_b[s_b<svd_threshold].conj().T)
                dm_a = c1_a.dot(c2_a.conj().T)
                dm_b = c1_b.dot(c2_b.conj().T)
                dm_01 = (dm_a, dm_b)

                phase = (np.linalg.det(u_a) * np.linalg.det(u_b) *
                         np.linalg.det(vt_a) * np.linalg.det(vt_b))
                fac = phase * np.prod(s_a_overlapped)*np.prod(s_b_overlapped)

                det_ovlp_cache[i, j] = (det_ovlp, dm_01, dm_01, fac, differs)

        dm_a = mo1_a.dot(mo1_a.conj().T)
        dm_b = mo1_b.dot(mo1_b.conj().T)
        dm = (dm_a, dm_b)
        det_ovlp_cache[i, i] = (1., dm, dm, 1., 0)

    dms = np.stack([v[1] for v in det_ovlp_cache.values()])
    vjs, vks = _mf.get_jk(mol, dms, hermi=0)

    hcore = _mf.get_hcore()
    n_csf = len(mfs)
    s = np.eye(n_csf)
    h = np.zeros_like(s)
    for idx, vj, vk in zip(det_ovlp_cache, vjs, vks):
        det_ovlp, _, dm_01, fac, differs = det_ovlp_cache[idx]
        s[idx] = det_ovlp
        vhf_01 = vj[0] + vj[1] - vk
        e1  = np.einsum('ij,ji->', hcore, dm_01[0]).real
        e1 += np.einsum('ij,ji->', hcore, dm_01[1]).real
        e2  = np.einsum('ij,ji->', vhf_01[0], dm_01[0]).real
        e2 += np.einsum('ij,ji->', vhf_01[1], dm_01[1]).real
        e2 *= .5
        if differs == 2:
            # When differed by 2 spin-orbitals, only the two-electron part
            # contributes to Slater-Condon integrals
            h[idx] = e2 * fac
        else:
            h[idx] = (e1 + e2) * fac

    h = lib.hermi_triu(h, inplace=True)
    s = lib.hermi_triu(s, inplace=True)
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
    return [ground_ks], mfs_s, mfs_d, mfs_t

class NOCI(lib.StreamObject):
    '''
    Nonorthogonal Configuration Interaction (NOCI) of Multistate Density Functional Theory (MSDFT)

    Attributes:
        xc : str
            Name of exchange-correlation functional
        s :
            A list of singly excited orbital pairs. Each pair [i, a] means an
            excitation from occupied orbital i to a.
        d :
            A list of doubly excited orbital pairs. Each pair [i, a] means both
            alpha and beta electrons at orbital i are excited to orbital a.
        coup : int
            How to compute the electronic coupling between diabatic states (Bao, JCTC, 17, 240).
            * 0: geometric average over diagonal terms.
            * 1: determinant-weighted average of correlation.
            * 2 (default): overlap-scaled average of correlation.
        ci_g : bool
            Whether to compute the adiabatic ground-state energy. True by default.
        sm_t: bool
            Use the energy difference between mix state and Ms=1 triplet state
            as the coupling between two symmetry-adapted mix state. This can be
            more accurate than the approximate HF coupling. True by default.

    Saved results:
        e_tot : float
            Total HF energy (electronic energy plus nuclear repulsion)
        csfvec : array
            CI coefficients
        mfs :
            KS instances of the underlying diabatic states)
    '''
    _keys = {
        'mol', 'verbose', 'stdout', 'xc', 'coup', 'ci_g', 's', 'd', 'sm_t',
        'e_tot', 'csfvec', 'mfs',
    }

    coup = 2
    ci_g = True
    sm_t = True
    svd_threshold = 1e-7

    def __init__(self, mol, xc=None):
        self.mol = mol
        self.verbose = mol.verbose
        self.stdout = mol.stdout
        self.xc = xc
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
        log.info('sm_t = %s', self.sm_t)
        log.info('single excitation = %s', self.s)
        log.info('double excitation = %s', self.d)
        log.info('Overlap svd threshold = %s', self.svd_threshold)
        return self

    def kernel(self, ground_ks=None):
        log = logger.new_logger(self)
        self.dump_flags(log)
        self.check_sanity()

        mf_gs, mfs_s, mfs_d, mfs_t = multi_states_scf(self, ground_ks)
        mfs = mfs_s + mfs_d
        if self.ci_g:
            mfs = mfs + mf_gs
        self.mfs = mfs

        e_hf, s_csf = hf_det_ovlp(self, mfs)
        Enuc = self.mol.energy_nuc()
        e_ks = np.array([mf.e_tot for mf in mfs])
        e_ks -= Enuc
        log.debug1('KS energies %s', e_ks)

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

        if self.sm_t:
            n_triplets = len(mfs_t)
            assert n_triplets * 2 == len(mfs_s)
            e_t = np.array([mf.e_tot for mf in mfs_t])
            e_t -= Enuc
            e_s = e_ks[:n_triplets*2]
            log.debug1('KS singlet energies %s', e_s)
            log.debug1('KS triplet energies %s', e_t)

            for i in range(n_triplets):
                j = 2*i
                s_t_coupling = e_s[i] + (s_csf[j,j+1] - 1.) * e_t[i]
                h_tdf[j,j+1] = h_tdf[j+1,j] = s_t_coupling
        self.e_tot, self.csfvec = scipy.linalg.eigh(h_tdf, s_csf)
        log.note('MSDFT eigs %s', self.e_tot)
        return self.e_tot

    @property
    def converged(self):
        return all(mf.converged for mf in self.mfs)

    to_gpu = lib.to_gpu
