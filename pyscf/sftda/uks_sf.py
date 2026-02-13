#!/usr/bin/env python
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

import sys
import numpy as np
import scipy
from pyscf import lib
from pyscf.lib import logger
from pyscf.sftda import uhf_sf
from pyscf import __config__

# import function
from pyscf.sftda.scf_genrep_sftd import gen_uhf_response_sf

from pyscf.lib.parameters import MAX_MEMORY
from pyscf.lib.linalg_helper import _sort_elast
from pyscf.tdscf._lr_eig import MAX_SPACE_INC


def davidson_nosym1(aop, x0, precond, tol=1e-12, max_cycle=50, lindep=1e-12, callback=None,
                    max_space=20,
                    max_memory=MAX_MEMORY, nroots=1, pick=None, verbose=logger.WARN):
    '''
    slightly modified from pyscf.lib.linalg.davidson_nosym1 with vectorization support
    '''
    def _qr(xs, lindep=1e-14):
        q, r = np.linalg.qr(xs.T, mode='reduced')
        r_diag = np.abs(np.diag(r))
        mask = r_diag > lindep
        qs = q.T[mask]
        return qs, np.where(mask)[0]

    assert callable(pick)
    assert callable(precond)

    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(sys.stdout, verbose)

    toloose = tol ** 0.5
    log.debug1('tol %g  toloose %g', tol, toloose)

    if isinstance(x0, np.ndarray) and x0.ndim == 1:
        x0 = x0[None, :]
    x0 = np.asarray(x0)
    x0_size = x0.shape[1]

    if MAX_SPACE_INC is None:
        space_inc = nroots
    else:
        space_inc = max(nroots, min(MAX_SPACE_INC, x0_size//2))
    max_space = int(max_memory*1e6/8/x0_size / 2 - nroots - space_inc)
    if max_space < nroots * 4 < x0_size:
        log.warn('Not enough memory to store trial space in _lr_eig.eigh')
    max_space = max(max_space, nroots * 4)
    max_space = min(max_space, x0_size)
    log.debug(f'Set max_space {max_space}, space_inc {space_inc}')

    xs = np.empty((max_space, x0_size), dtype=x0.dtype)
    ax = np.empty((max_space, x0_size), dtype=x0.dtype)
    heff = np.empty((max_space, max_space), dtype=x0.dtype)
    fresh_start = True
    space = 0
    e = None
    v = None
    conv = np.zeros(nroots, dtype=bool)
    conv_last = np.zeros(nroots, dtype=bool)

    for icyc in range(max_cycle):
        if fresh_start:
            xs = np.empty_like(xs)
            ax = np.empty_like(ax)
            space = 0
            x0len = len(x0)
            xt, _ = _qr(x0, lindep)
            if len(xt) != x0len:
                log.warn('QR decomposition removed %d vectors. '
                            'Check to see if `pick` function :%s: is providing linear dependent '
                            'vectors' % (x0len - len(xt), pick.__name__))
        elif len(xt) > 1:
            xt, _ = _qr(xt, lindep)
        xt = xt[:space_inc]

        add = xt.shape[0]
        axt = aop(xt)
        xs[space: space+add] = xt
        ax[space: space+add] = axt
        space_old = space
        space += add
        if fresh_start:
            heff.fill(0)
        heff[:space_old, space_old:space] = xs[:space_old].conj().dot(ax[space_old:space].T)
        heff[space_old:space, :space_old] = xs[space_old:space].conj().dot(ax[:space_old].T)
        heff[space_old:space, space_old:space] = xs[space_old:space].conj().dot(ax[space_old:space].T)
        xt = axt = None

        elast = e
        vlast = v
        conv_last = conv

        w_npu, v_npu = scipy.linalg.eig(heff[:space,:space])
        w, v = np.asarray(w_npu), np.asarray(v_npu)
        w, v, idx = pick(w, v, nroots, locals())
        if len(w) == 0:
            raise RuntimeError('Not enough eigenvalues')

        e = w[:nroots]
        v = v[:,:nroots]
        if not fresh_start:
            elast, conv_last = _sort_elast(elast, conv_last, vlast, v, log)

        if elast is None:
            de = e
        elif elast.size != e.size:
            log.debug('Number of roots different from the previous step (%d,%d)',
                      e.size, elast.size)
            de = e
        else:
            de = e - elast

        x0 = v.T.dot(xs[:space])
        ax0 = v.T.dot(ax[:space])

        xt = ax0[:nroots] - e[:, None] * x0[:nroots]
        ax0 = None

        dx_norm = np.linalg.norm(xt, axis=1)
        conv =  (abs(de) < tol) & (dx_norm < toloose)
        for k, ek in enumerate(e):
            if conv[k] and not conv_last[k]:
                log.debug('root %d converged  |r|= %4.3g  e= %s  max|de|= %4.3g',
                          k, dx_norm[k], ek, de[k])
        ax0 = None
        max_dx_norm = max(dx_norm)
        max_de = max(abs(de))
        if all(conv):
            log.debug('converged %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g',
                      icyc, len(xs), max_dx_norm, e, max_de)
            break

        mask = (~conv) & (dx_norm**2 > lindep)
        xt = precond(xt[mask], e[0], x0[mask])
        valid_xs = xs[:space]
        for _ in range(2):
            xt -= np.dot(np.dot(xt, valid_xs.T.conj()), valid_xs)
        xt_norm = np.linalg.norm(xt, axis=1)
        keep_mask = (xt_norm**2 > lindep)
        xt = xt[keep_mask]
        xt_norm = xt_norm[keep_mask]

        if len(xt)==0:
            log.debug('Linear dependency in trial subspace. |r| for each state %s',
                      dx_norm)
            conv = dx_norm < toloose
            break
        log.debug('davidson %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g  lindep= %4.3g',
                  icyc, space, max_dx_norm, e, max_de, np.linalg.norm(xt, axis=1).min())

        xt /= xt_norm[:, None]

        fresh_start = space+len(xt) > max_space
        if callable(callback):
            callback(locals())

    return conv, e, x0

class TDA_SF(uhf_sf.TDA_SF):
    def nuc_grad_method(self):
        from pyscf.grad import tduks_sf
        return tduks_sf.Gradients(self)

class CasidaTDDFT(TDA_SF):
    '''Solve the Casida TDDFT formula
       [ A  B][X]
       [-B -A][Y]
    '''

    def gen_vind(self):
        mf = self._scf
        mo_energy = mf.mo_energy
        mo_coeff = mf.mo_coeff
        assert (mo_coeff[0].dtype == np.double)
        mo_occ = mf.mo_occ

        occidxa = mo_occ[0] > 0
        occidxb = mo_occ[1] > 0
        viridxa = mo_occ[0] == 0
        viridxb = mo_occ[1] == 0
        orboa = mo_coeff[0][:, occidxa]
        orbob = mo_coeff[1][:, occidxb]
        orbva = mo_coeff[0][:, viridxa]
        orbvb = mo_coeff[1][:,viridxb]
        nocca = int(occidxa.sum())
        noccb = int(occidxb.sum())
        nvira = int(viridxa.sum())
        nvirb = int(viridxb.sum())

        if np.allclose(mo_coeff[0], mo_coeff[1]):
            fock_a, fock_b = mf.get_fock()
            fockoa = orboa.conj().T @ fock_a @ orboa
            fockva = orbva.conj().T @ fock_a @ orbva
            fockob = orbob.conj().T @ fock_b @ orbob
            fockvb = orbvb.conj().T @ fock_b @ orbvb
            if self.extype==0:
                hdiag1 = (fockva.diagonal()[None, :] - fockob.diagonal()[:, None]).ravel()
                hdiag2 = - (fockvb.diagonal()[None, :] - fockoa.diagonal()[:, None]).ravel()
                hdiag = np.hstack((hdiag1, hdiag2))
            elif self.extype==1:
                hdiag1 = (fockvb.diagonal()[None, :] - fockoa.diagonal()[:, None]).ravel()
                hdiag2 = - (fockva.diagonal()[None, :] - fockob.diagonal()[:, None]).ravel()
                hdiag = np.hstack((hdiag1, hdiag2))
        else:
            e_ia_b2a = (mo_energy[0][viridxa] - mo_energy[1][occidxb,None])
            e_ia_a2b = (mo_energy[1][viridxb] - mo_energy[0][occidxa,None])

            if self.extype==0:
                hdiag = np.hstack((e_ia_b2a.ravel(), -e_ia_a2b.ravel()))
            elif self.extype==1:
                hdiag = np.hstack((e_ia_a2b.ravel(), -e_ia_b2a.ravel()))

        vresp = gen_uhf_response_sf(mf, hermi=0, collinear_samples=self.collinear_samples)

        def vind(zs):
            nz = len(zs)
            zs = np.asarray(zs).reshape(nz,-1)
            if self.extype==0:
                zs_b2a = zs[:, :noccb*nvira].reshape(nz, noccb, nvira)
                zs_a2b = zs[:, noccb*nvira:].reshape(nz, nocca, nvirb)
                dm_b2a = lib.einsum('xov,pv,qo->xpq', zs_b2a, orbva, orbob.conj())
                dm_a2b = lib.einsum('xov,qv,po->xpq', zs_a2b, orbvb.conj(), orboa)
            elif self.extype==1:
                zs_a2b = zs[:, :nocca*nvirb].reshape(nz, nocca, nvirb)
                zs_b2a = zs[:, nocca*nvirb:].reshape(nz, noccb, nvira)
                dm_a2b = lib.einsum('xov,pv,qo->xpq', zs_a2b, orbvb, orboa.conj())
                dm_b2a = lib.einsum('xov,qv,po->xpq', zs_b2a, orbva.conj(), orbob)
            dms = dm_b2a + dm_a2b

            v1ao = vresp(dms)
            if self.extype==0:
                v1_top = lib.einsum('xpq,qo,pv->xov', v1ao, orbob, orbva.conj())
                v1_bot = lib.einsum('xpq,po,qv->xov', v1ao, orboa.conj(), orbvb)
                if np.allclose(mo_coeff[0], mo_coeff[1]):
                    v1_top += lib.einsum('ab,xib->xia', fockva, zs_b2a)
                    v1_top -= lib.einsum('ji,xja->xia', fockob, zs_b2a)
                    v1_bot += lib.einsum('ab,xib->xia', fockvb, zs_a2b)
                    v1_bot -= lib.einsum('ji,xja->xia', fockoa, zs_a2b)
                else:
                    v1_top += zs_b2a * e_ia_b2a
                    v1_bot += zs_a2b * e_ia_a2b
            elif self.extype==1:
                v1_top = lib.einsum('xpq,qo,pv->xov', v1ao, orboa, orbvb.conj())
                v1_bot = lib.einsum('xpq,po,qv->xov', v1ao, orbob.conj(), orbva)
                if np.allclose(mo_coeff[0], mo_coeff[1]):
                    v1_top += lib.einsum('ab,xib->xia', fockvb, zs_a2b)
                    v1_top -= lib.einsum('ji,xja->xia', fockoa, zs_a2b)
                    v1_bot += lib.einsum('ab,xib->xia', fockva, zs_b2a)
                    v1_bot -= lib.einsum('ji,xja->xia', fockob, zs_b2a)
                else:
                    v1_top += zs_a2b * e_ia_a2b
                    v1_bot += zs_b2a * e_ia_b2a
            hx = np.hstack((v1_top.reshape(nz, -1), -v1_bot.reshape(nz, -1)))
            return hx

        return vind, hdiag

    def init_guess(self, mf=None, nstates=None):
        if mf is None: mf = self._scf
        if nstates is None: nstates = self.nstates
        nstates += 3
        x0 = self._init_guess(mf, nstates)
        nx = len(x0)
        nmo = mf.mo_occ[0].size
        nocca, noccb = mf.nelec
        nvira = nmo - nocca
        nvirb = nmo - noccb
        if self.extype == 0:
            y0 = np.zeros((nx, nocca*nvirb))
        else:
            y0 = np.zeros((nx, noccb*nvira))
        return np.hstack([x0.reshape(nx, -1), y0])


    def gen_pickeig(self, extype=1):
        '''
        Selects physical roots based on the norm condition ||X|| > ||Y||.
        This replaces the previous empirical energy threshold (`positive_eig_threshold`).
        '''
        mo_occ = self._scf.mo_occ
        occidxa = np.where(mo_occ[0]>0)[0]
        occidxb = np.where(mo_occ[1]>0)[0]
        viridxa = np.where(mo_occ[0]==0)[0]
        viridxb = np.where(mo_occ[1]==0)[0]
        nocca = len(occidxa)
        noccb = len(occidxb)
        nvira = len(viridxa)
        nvirb = len(viridxb)
        if extype==0:
            ov = noccb * nvira
        elif extype==1:
            ov = nocca * nvirb
        def pickeig(w, v, nroots, envs):
            xs = np.vstack(envs.get('xs'))
            ritz_vectors = v.T.dot(xs[:envs.get('space')])
            x_part = ritz_vectors[:, :ov]
            y_part = ritz_vectors[:, ov:]
            norm_x2 = np.linalg.norm(x_part, axis=1)**2
            norm_y2 = np.linalg.norm(y_part, axis=1)**2
            is_physical = (norm_x2 > norm_y2 - 1e-4) & (abs(w.imag) < 1e-4)
            realidx = np.where(is_physical)[0]
            if len(realidx) < nroots:
                remaining_idx = np.setdiff1d(np.arange(len(w)), realidx)
                sorted_rem = remaining_idx[np.argsort(w[remaining_idx].real)]
                needed = nroots - len(realidx)
                realidx = np.concatenate([realidx, sorted_rem[:needed]])
            return lib.linalg_helper._eigs_cmplx2real(w, v, realidx, real_eigenvectors=True)
        return pickeig

    def kernel(self, x0=None, nstates=None, extype=None):
        '''
        Spin-flip TDDFT diagonalization solver
        '''
        cpu0 = (logger.process_clock(), logger.perf_counter())

        self.check_sanity()
        self.dump_flags()

        if extype is None:
            extype = self.extype
        else:
            self.extype = extype

        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates

        log = logger.Logger(self.stdout, self.verbose)

        vind, hdiag = self.gen_vind()
        precond = self.get_precond(hdiag)

        if x0 is None:
            x0 = self.init_guess()

        pickeig = self.gen_pickeig(extype=extype)

        self.converged, self.e, x1 = davidson_nosym1(vind, x0, precond,
                                            tol=self.conv_tol,
                                            nroots=nstates,
                                            max_cycle=self.max_cycle,
                                            pick=pickeig,
                                            verbose=log)

        nmo = self._scf.mo_occ[0].size
        nocca, noccb = self._scf.nelec
        nvira = nmo - nocca
        nvirb = nmo - noccb

        if self.extype==0:
            def norm_xy(z):
                x = z[:noccb*nvira].reshape(noccb, nvira)
                y = z[noccb*nvira:].reshape(nocca, nvirb)
                norm = lib.norm(x)**2 - lib.norm(y)**2
                norm = np.sqrt(1./norm)
                return x*norm, y*norm
        elif self.extype==1:
            def norm_xy(z):
                x = z[:nocca*nvirb].reshape(nocca, nvirb)
                y = z[nocca*nvirb:].reshape(noccb, nvira)
                norm = lib.norm(x)**2 - lib.norm(y)**2
                norm = np.sqrt(1./norm)
                return x*norm, y*norm

        self.xy = [norm_xy(z) for z in x1]

        if self.chkfile:
            lib.chkfile.save(self.chkfile, 'tddft/e', self.e)
            lib.chkfile.save(self.chkfile, 'tddft/xy', self.xy)

        log.timer('TDDFT_SF', *cpu0)

        self._finalize()
        return self.e, self.xy

    def nuc_grad_method(self):
        from pyscf.grad import tduks
        return tduks.Gradients(self)

class TDDFT_SF(CasidaTDDFT):
    pass

TDUKS_SF = TDDFT_SF

def tddft(mf):
    '''Driver to create TDDFT_SF or CasidaTDDFT_SF object'''
    return CasidaTDDFT(mf)

from pyscf import scf, dft
dft.uks.UKS.TDA_SF   = lib.class_as_method(TDA_SF)
dft.uks.UKS.TDDFT_SF = lib.class_as_method(TDDFT_SF)
scf.uhf.UHF.TDA_SF   = lib.class_as_method(TDA_SF)
scf.uhf.UHF.TDDFT_SF = lib.class_as_method(TDDFT_SF)
