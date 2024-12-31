#/usr/bin/env python
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

# This file can be merged into pyscf.tdscf.uhf.py

import numpy
from pyscf import lib
from pyscf import scf
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.tdscf import rhf
from pyscf import __config__

# import function
from pyscf.sftda.scf_genrep_sftd import _gen_uhf_tda_response_sf
from pyscf.sftda.numint2c_sftd import cache_xc_kernel_sf

# import class
from pyscf.tdscf.uhf import TDBase

def gen_tda_operation_sf(mf, fock_ao=None, wfnsym=None,extype=0,collinear_samples=200):
    '''A x for spin flip TDDFT case.

    Kwargs:
        wfnsym : int or str
            Point group symmetry irrep symbol or ID for excited CIS wavefunction.
        extype : int (0 or 1)
            Determine which spin flip excitation will be calculated.
            Spin flip up: exytpe=0. Spin flip down: exytpe=1.
    '''
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    assert (mo_coeff[0].dtype == numpy.double)
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ

    if wfnsym is not None and mol.symmetry:
        raise NotImplementedError("UKS Spin Flip TDA/ TDDFT haven't taken symmetry\
                                      into account.")

    if extype==0:
        occidxb = numpy.where(mo_occ[1]>0)[0]
        viridxa = numpy.where(mo_occ[0]==0)[0]
        noccb = len(occidxb)
        nvira = len(viridxa)
        orbob = mo_coeff[1][:,occidxb]
        orbva = mo_coeff[0][:,viridxa]
        orbov = (orbob,orbva)
        ndim = (noccb,nvira)

        e_ia = (mo_energy[0][viridxa,None] - mo_energy[1][occidxb]).T
        hdiag = e_ia.ravel()

    elif extype==1:
        occidxa = numpy.where(mo_occ[0]>0)[0]
        viridxb = numpy.where(mo_occ[1]==0)[0]
        nocca = len(occidxa)
        nvirb = len(viridxb)
        orboa = mo_coeff[0][:,occidxa]
        orbvb = mo_coeff[1][:,viridxb]
        orbov = (orboa,orbvb)
        ndim = (nocca,nvirb)

        e_ia = (mo_energy[1][viridxb,None] - mo_energy[0][occidxa]).T
        hdiag = e_ia.ravel()

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory*.8-mem_now)

    # _gen_uhf_tda_response_sf() should be used by : mf.gen_response
    vresp = _gen_uhf_tda_response_sf(mf, hermi=0, max_memory=max_memory,
                                     collinear_samples=collinear_samples)

    def vind(zs):
        zs = numpy.asarray(zs)

        ndim0,ndim1 = ndim
        orbo,orbv = orbov
        zs = zs[:,:ndim0*ndim1].reshape(-1,ndim0,ndim1)
        dmov = lib.einsum('xov,qv,po->xpq', zs, orbv.conj(), orbo)
        v1ao = vresp(dmov)
        v1 = lib.einsum('xpq,po,qv->xov', v1ao, orbo.conj(), orbv)
        # add the orbital energy difference in A matrix.
        v1 += lib.einsum('ov,xov->xov', e_ia, zs)
        nz = zs.shape[0]
        hx = v1.reshape(nz,-1)

        return hx

    return vind, hdiag

gen_tda_hop_sf = gen_tda_operation_sf

def get_ab_sf(mf, mo_energy=None, mo_coeff=None, mo_occ=None,collinear_samples=200):
    r'''A and B matrices for TDDFT response function.

    A[i,a,j,b] = \delta_{ab}\delta_{ij}(E_a - E_i) + (ia||bj)
    B[i,a,j,b] = (ia||jb)

    Spin symmetry is not considered in the returned A, B lists.
    List A has two items: (A_abab, A_baba).
    List B has two items: (B_abba, B_baab).
    '''
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ

    mol = mf.mol
    nao = mol.nao_nr()
    occidx_a = numpy.where(mo_occ[0]==1)[0]
    viridx_a = numpy.where(mo_occ[0]==0)[0]
    occidx_b = numpy.where(mo_occ[1]==1)[0]
    viridx_b = numpy.where(mo_occ[1]==0)[0]
    orbo_a = mo_coeff[0][:,occidx_a]
    orbv_a = mo_coeff[0][:,viridx_a]
    orbo_b = mo_coeff[1][:,occidx_b]
    orbv_b = mo_coeff[1][:,viridx_b]
    nocc_a = orbo_a.shape[1]
    nvir_a = orbv_a.shape[1]
    nocc_b = orbo_b.shape[1]
    nvir_b = orbv_b.shape[1]

    e_ia_b2a = (mo_energy[0][viridx_a,None] - mo_energy[1][occidx_b]).T
    e_ia_a2b = (mo_energy[1][viridx_b,None] - mo_energy[0][occidx_a]).T

    a_b2a = numpy.diag(e_ia_b2a.ravel()).reshape(nocc_b,nvir_a,nocc_b,nvir_a)
    a_a2b = numpy.diag(e_ia_a2b.ravel()).reshape(nocc_a,nvir_b,nocc_a,nvir_b)
    b_b2a = numpy.zeros((nocc_b,nvir_a,nocc_a,nvir_b))
    b_a2b = numpy.zeros((nocc_a,nvir_b,nocc_b,nvir_a))
    a = (a_b2a, a_a2b)
    b = (b_b2a, b_a2b)

    def add_hf_(a, b, hyb=1):
        # In spin flip TDA/ TDDFT, hartree potential is zero.
        # A : iabj ---> ijba; B : iajb ---> ibja
        eri_a_b2a = ao2mo.general(mol, [orbo_b,orbo_b,orbv_a,orbv_a], compact=False)
        eri_a_a2b = ao2mo.general(mol, [orbo_a,orbo_a,orbv_b,orbv_b], compact=False)
        eri_b_b2a = ao2mo.general(mol, [orbo_b,orbv_b,orbo_a,orbv_a], compact=False)
        eri_b_a2b = ao2mo.general(mol, [orbo_a,orbv_a,orbo_b,orbv_b], compact=False)

        eri_a_b2a = eri_a_b2a.reshape(nocc_b,nocc_b,nvir_a,nvir_a)
        eri_a_a2b = eri_a_a2b.reshape(nocc_a,nocc_a,nvir_b,nvir_b)
        eri_b_b2a = eri_b_b2a.reshape(nocc_b,nvir_b,nocc_a,nvir_a)
        eri_b_a2b = eri_b_a2b.reshape(nocc_a,nvir_a,nocc_b,nvir_b)

        a_b2a, a_a2b = a
        b_b2a, b_a2b = b

        a_b2a-= numpy.einsum('ijba->iajb', eri_a_b2a) * hyb
        a_a2b-= numpy.einsum('ijba->iajb', eri_a_a2b) * hyb
        b_b2a-= numpy.einsum('ibja->iajb', eri_b_b2a) * hyb
        b_a2b-= numpy.einsum('ibja->iajb', eri_b_a2b) * hyb

    if isinstance(mf, scf.hf.KohnShamDFT):
        from pyscf.dft import xc_deriv
        from pyscf.dft import numint2c
        ni0 = mf._numint
        ni = numint2c.NumInt2C()
        ni.collinear = 'mcol'
        ni.collinear_samples = collinear_samples
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        if mf.nlc or ni.libxc.is_nlc(mf.xc):
            logger.warn(mf, 'NLC functional found in DFT object.  Its second '
                        'deriviative is not available. Its contribution is '
                        'not included in the response function.')
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)

        add_hf_(a, b, hyb)

        xctype = ni._xc_type(mf.xc)
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, mf.max_memory*.8-mem_now)

        if xctype == 'LDA':
            ao_deriv = 0
            for ao, mask, weight, coords \
                    in ni0.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
                fxc = cache_xc_kernel_sf(ni, mol, mf.grids, mf.xc, mo_coeff, mo_occ, 1)[2]
                wfxc = fxc[0,0] * weight

                rho_o_a = lib.einsum('rp,pi->ri', ao, orbo_a)
                rho_v_a = lib.einsum('rp,pi->ri', ao, orbv_a)
                rho_o_b = lib.einsum('rp,pi->ri', ao, orbo_b)
                rho_v_b = lib.einsum('rp,pi->ri', ao, orbv_b)
                rho_ov_b2a = numpy.einsum('ri,ra->ria', rho_o_b, rho_v_a)
                rho_ov_a2b = numpy.einsum('ri,ra->ria', rho_o_a, rho_v_b)

                w_ov = numpy.einsum('ria,r->ria', rho_ov_b2a, wfxc*2.0)
                iajb = lib.einsum('ria,rjb->iajb', rho_ov_b2a, w_ov)
                a_b2a += iajb
                iajb = lib.einsum('ria,rjb->iajb', rho_ov_a2b, w_ov)
                b_a2b += iajb

                w_ov = numpy.einsum('ria,r->ria', rho_ov_a2b, wfxc*2.0)
                iajb = lib.einsum('ria,rjb->iajb', rho_ov_a2b, w_ov)
                a_a2b += iajb
                iajb = lib.einsum('ria,rjb->iajb', rho_ov_b2a, w_ov)
                b_b2a += iajb

        elif xctype == 'GGA':
            ao_deriv = 1
            for ao, mask, weight, coords \
                    in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
                fxc = cache_xc_kernel_sf(ni, mol, mf.grids, mf.xc, mo_coeff, mo_occ, 1)[2]
                wfxc = fxc * weight

                rho_o_a = lib.einsum('xrp,pi->xri', ao, orbo_a)
                rho_v_a = lib.einsum('xrp,pi->xri', ao, orbv_a)
                rho_o_b = lib.einsum('xrp,pi->xri', ao, orbo_b)
                rho_v_b = lib.einsum('xrp,pi->xri', ao, orbv_b)
                rho_ov_b2a = numpy.einsum('xri,ra->xria', rho_o_b, rho_v_a[0])
                rho_ov_a2b = numpy.einsum('xri,ra->xria', rho_o_a, rho_v_b[0])
                rho_ov_b2a[1:4] += numpy.einsum('ri,xra->xria', rho_o_b[0], rho_v_a[1:4])
                rho_ov_a2b[1:4] += numpy.einsum('ri,xra->xria', rho_o_a[0], rho_v_b[1:4])

                w_ov = numpy.einsum('xyr,xria->yria', wfxc*2.0, rho_ov_b2a)
                iajb = lib.einsum('xria,xrjb->iajb', w_ov, rho_ov_b2a)
                a_b2a += iajb
                iajb = lib.einsum('xria,xrjb->iajb', w_ov, rho_ov_a2b)
                b_b2a += iajb

                w_ov = numpy.einsum('xyr,xria->yria', wfxc*2.0, rho_ov_a2b)
                iajb = lib.einsum('xria,xrjb->iajb', w_ov, rho_ov_a2b)
                a_a2b += iajb
                iajb = lib.einsum('xria,xrjb->iajb', w_ov, rho_ov_b2a)
                b_a2b += iajb

        elif xctype == 'HF':
            pass

        elif xctype == 'NLC':
            raise NotImplementedError('NLC')

        elif xctype == 'MGGA':
            ao_deriv = 1
            for ao, mask, weight, coords \
                    in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
                fxc = cache_xc_kernel_sf(ni, mol, mf.grids, mf.xc, mo_coeff, mo_occ, 1)[2]
                wfxc = fxc * weight
                rho_oa = lib.einsum('xrp,pi->xri', ao, orbo_a)
                rho_ob = lib.einsum('xrp,pi->xri', ao, orbo_b)
                rho_va = lib.einsum('xrp,pi->xri', ao, orbv_a)
                rho_vb = lib.einsum('xrp,pi->xri', ao, orbv_b)
                rho_ov_b2a = numpy.einsum('xri,ra->xria', rho_ob, rho_va[0])
                rho_ov_a2b = numpy.einsum('xri,ra->xria', rho_oa, rho_vb[0])
                rho_ov_b2a[1:4] += numpy.einsum('ri,xra->xria', rho_ob[0], rho_va[1:4])
                rho_ov_a2b[1:4] += numpy.einsum('ri,xra->xria', rho_oa[0], rho_vb[1:4])
                tau_ov_b2a = numpy.einsum('xri,xra->ria', rho_ob[1:4], rho_va[1:4]) * .5
                tau_ov_a2b = numpy.einsum('xri,xra->ria', rho_oa[1:4], rho_vb[1:4]) * .5
                rho_ov_b2a = numpy.vstack([rho_ov_b2a, tau_ov_b2a[numpy.newaxis]])
                rho_ov_a2b = numpy.vstack([rho_ov_a2b, tau_ov_a2b[numpy.newaxis]])

                w_ov = numpy.einsum('xyr,xria->yria', wfxc*2.0, rho_ov_b2a)
                iajb = lib.einsum('xria,xrjb->iajb', w_ov, rho_ov_b2a)
                a_b2a += iajb
                iajb = lib.einsum('xria,xrjb->iajb', w_ov, rho_ov_a2b)
                b_b2a += iajb

                w_ov = numpy.einsum('xyr,xria->yria', wfxc*2.0, rho_ov_a2b)
                iajb = lib.einsum('xria,xrjb->iajb', w_ov, rho_ov_a2b)
                a_a2b += iajb
                iajb = lib.einsum('xria,xrjb->iajb', w_ov, rho_ov_b2a)
                b_a2b += iajb
    else:
        add_hf_(a, b)

    return a, b

@lib.with_doc(rhf.TDA.__doc__)
class TDA_SF(TDBase):
    extype = getattr(__config__, 'tdscf_uhf_sf_SF-TDA_extype', 0)
    collinear_samples = getattr(__config__, 'tdscf_uhf_sf_SF-TDA_collinear_samples', 200)

    _keys = {'extype','collinear_samples'}

    def __init__(self,mf,extype=0,collinear_samples=200):
        TDBase.__init__(self,mf)
        # extype is used to determine which spin flip excitation will be calculated.
        # spin flip up: exytpe=0, spin flip down: exytpe=1.
        self.extype=extype
        # collinear_samples controls the 1d spin sample points in TDDFT/TDA.
        self.collinear_samples = collinear_samples

    def gen_vind(self, mf=None,extype=None):
        '''Generate function to compute Ax'''
        if mf is None:
            mf = self._scf
        if extype is None:
            extype = self.extype
        return gen_tda_hop_sf(mf, wfnsym=self.wfnsym,extype=self.extype,
                              collinear_samples=self.collinear_samples)

    def init_guess0(self, mf, nstates=None, wfnsym=None,extype=None):
        if nstates is None: nstates = self.nstates
        if wfnsym is None: wfnsym = self.wfnsym

        mol = mf.mol
        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ

        if wfnsym is not None and mol.symmetry:
            raise NotImplementedError("UKS Spin Flip TDA/ TDDFT haven't taken symmetry\
                                      into account.")
        if self.extype==0:
            occidxb = numpy.where(mo_occ[1]>0)[0]
            viridxa = numpy.where(mo_occ[0]==0)[0]

            e_ia_b2a = (mo_energy[0][viridxa,None] - mo_energy[1][occidxb]).T
            e_ia_b2a = e_ia_b2a.ravel()

            nov_b2a = e_ia_b2a.size
            nstates = min(nstates, nov_b2a)
            e_threshold = numpy.sort(e_ia_b2a)[nstates-1]
            e_threshold += self.deg_eia_thresh

            idx = numpy.where(e_ia_b2a <= e_threshold)[0]
            x0 = numpy.zeros((idx.size, nov_b2a))
            for i, j in enumerate(idx):
                x0[i, j] = 1  # Koopmans' excitations

        elif self.extype==1:
            occidxa = numpy.where(mo_occ[0]>0)[0]
            viridxb = numpy.where(mo_occ[1]==0)[0]

            e_ia_a2b = (mo_energy[1][viridxb,None] - mo_energy[0][occidxa]).T
            e_ia_a2b = e_ia_a2b.ravel()

            nov_a2b = e_ia_a2b.size
            nstates = min(nstates, nov_a2b)

            e_threshold = numpy.sort(e_ia_a2b)[nstates-1]
            e_threshold += self.deg_eia_thresh

            idx = numpy.where(e_ia_a2b <= e_threshold)[0]
            x0 = numpy.zeros((idx.size, nov_a2b))
            for i, j in enumerate(idx):
                x0[i, j] = 1  # Koopmans' excitations
        return x0

    def init_guess(self, mf, nstates=None, wfnsym=None):
        if nstates is None: nstates = self.nstates
        if wfnsym is None: wfnsym = self.wfnsym

        mol = mf.mol
        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        occidxa = numpy.where(mo_occ[0]>0)[0]
        occidxb = numpy.where(mo_occ[1]>0)[0]
        viridxa = numpy.where(mo_occ[0]==0)[0]
        viridxb = numpy.where(mo_occ[1]==0)[0]
        e_ia_b2a = (mo_energy[0][viridxa,None] - mo_energy[1][occidxb]).T
        e_ia_a2b = (mo_energy[1][viridxb,None] - mo_energy[0][occidxa]).T

        if wfnsym is not None and mol.symmetry:
            raise NotImplementedError("UKS Spin Flip TDA/ TDDFT haven't taken symmetry\
                                      into account.")

        e_ia_b2a = e_ia_b2a.ravel()
        e_ia_a2b = e_ia_a2b.ravel()
        nov_b2a = e_ia_b2a.size
        nov_a2b = e_ia_a2b.size

        if self.extype==0:
            nstates = min(nstates, nov_b2a)
            e_threshold = numpy.sort(e_ia_b2a)[nstates-1]
            e_threshold += self.deg_eia_thresh

            idx = numpy.where(e_ia_b2a <= e_threshold)[0]
            x0 = numpy.zeros((idx.size, nov_b2a))
            for i, j in enumerate(idx):
                x0[i, j] = 1  # Koopmans' excitations

            y0 = numpy.zeros((len(idx),nov_a2b))
            z0 = numpy.concatenate((x0,y0),axis=1)

        elif self.extype==1:
            nstates = min(nstates, nov_a2b)
            e_threshold = numpy.sort(e_ia_a2b)[nstates-1]
            e_threshold += self.deg_eia_thresh

            idx = numpy.where(e_ia_a2b <= e_threshold)[0]
            x0 = numpy.zeros((idx.size, nov_a2b))
            for i, j in enumerate(idx):
                x0[i, j] = 1  # Koopmans' excitations

            y0 = numpy.zeros((len(idx),nov_b2a))
            z0 = numpy.concatenate((x0,y0),axis=1)
        return z0

    def kernel(self, x0=None, nstates=None, extype=None):
        '''SF_TDA diagonalization solver
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

        vind, hdiag = self.gen_vind(self._scf,extype=extype)
        precond = hdiag

        if x0 is None:
            x0 = self.init_guess0(self._scf, self.nstates,extype=extype)

        nstates_new = x0.shape[0]
        self.converged, self.e, x1 = \
                lib.davidson1(vind, x0, precond,
                              tol=self.conv_tol,
                              nroots=nstates_new, lindep=self.lindep,
                              max_cycle=self.max_cycle,
                              max_space=self.max_space,
                              verbose=log)

        nmo = self._scf.mo_occ[0].size
        nocca = (self._scf.mo_occ[0]>0).sum()
        noccb = (self._scf.mo_occ[1]>0).sum()
        nvira = nmo - nocca
        nvirb = nmo - noccb

        if self.extype==0:
            self.xy = [((xi[:noccb*nvira].reshape(noccb,nvira),0),  # X_alpha_beta
                        (0,0))  # (Y_beta_alpha)
                        for xi in x1]

        elif self.extype==1:
            self.xy = [((0,xi[:nocca*nvirb].reshape(nocca,nvirb)),  # X_beta_alpha
                        (0, 0))  # (Y_beta_alpha)
                        for xi in x1]

        if self.chkfile:
            lib.chkfile.save(self.chkfile, 'tddft/e', self.e)
            lib.chkfile.save(self.chkfile, 'tddft/xy', self.xy)

        log.timer('TDA_SF', *cpu0)
        self._finalize()
        return self.e, self.xy

    # this function should be moved into uhf.py
    def get_ab_sf(self, mf=None):
        if mf is None: mf = self._scf
        return get_ab_sf(mf)

scf.uhf.UHF.TDA_SF = lib.class_as_method(TDA_SF)
