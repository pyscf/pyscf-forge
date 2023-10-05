#!/usr/bin/env python
# Copyright 2014-2023 The PySCF Developers. All Rights Reserved.
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
# Author: Matthew Hennefarth <mhennefarth@uchicago.com>

from pyscf import lib
from pyscf.grad import rks as rks_grad
from pyscf.dft import gen_grid
from pyscf.lib import logger, tag_array, pack_tril, current_memory
from pyscf.mcscf import casci, mc1step, newton_casscf
from pyscf.grad import sacasscf
from pyscf.mcscf.casci import cas_natorb

from pyscf.mcpdft.otpd import get_ontop_pair_density, _grid_ao2mo
from pyscf.mcpdft.tfnal_derivs import contract_fot, unpack_vot, contract_vot
from pyscf.mcpdft import _dms
import pyscf.grad.mcpdft as mcpdft_grad

import numpy as np
import gc

BLKSIZE = gen_grid.BLKSIZE


def get_ontop_response(mc, ot, state, atmlst, casdm1, casdm1_0, mo_coeff=None, ci=None, max_memory=None):
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    if max_memory is None: max_memory = mc.max_memory

    t0 = (logger.process_clock(), logger.perf_counter())

    mol = mc.mol
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    nelecas = mc.nelecas
    nao, nmo = mo_coeff.shape

    dvxc = np.zeros((3, nao))
    de_grid = np.zeros((len(atmlst), 3))
    de_wgt = np.zeros((len(atmlst), 3))

    mo_coeff_0, ci_0, mo_occup_0 = cas_natorb(mc, mo_coeff=mo_coeff, ci=ci, casdm1=casdm1_0)
    mo_coeff, ci, mo_occup = cas_natorb(mc, mo_coeff=mo_coeff, ci=ci, casdm1=casdm1)

    mo_occ = mo_coeff[:, :nocc]
    mo_cas = mo_coeff[:, ncore:nocc]

    mo_occ_0 = mo_coeff_0[:, :nocc]
    mo_cas_0 = mo_coeff_0[:, ncore:nocc]

    # Need to regenerate these with the updated ci values....
    casdm1s = mc.make_one_casdm1s(ci=ci, state=state)
    casdm1 = casdm1s[0] + casdm1s[1]
    casdm2 = mc.make_one_casdm2(ci=ci, state=state)
    dm1s = _dms.casdm1s_to_dm1s(mc, casdm1s, mo_coeff=mo_coeff, ncore=ncore, ncas=ncas)
    dm1 = dm1s[0] + dm1s[1]
    #dm1 = tag_array(dm1, mo_coeff=mo_coeff, mo_occ=mo_occup[:nocc])

    casdm1s_0, casdm2_0 = mc.get_casdm12_0(ci=ci_0)
    casdm1_0 = casdm1s_0[0] + casdm1s_0[1]
    dm1s_0 = _dms.casdm1s_to_dm1s(mc, casdm1s_0, mo_coeff=mo_coeff_0, ncore=ncore, ncas=ncas)
    dm1_0 = dm1s_0[0] + dm1s_0[1]
    #dm1_0 = tag_array(dm1_0, mo_coeff=mo_coeff_0, mo_occ=mo_occup_0[:nocc])

    cascm2 = _dms.dm2_cumulant(casdm2, casdm1)
    cascm2_0 = _dms.dm2_cumulant(casdm2_0, casdm1_0)

    make_rho = ot._numint._gen_rho_evaluator(mol, dm1, 1)[0]
    make_rho_0 = ot._numint._gen_rho_evaluator(mol, dm1_0, 1)[0]

    idx = np.array([[1, 4, 5, 6], [2, 5, 7, 8], [3, 6, 8, 9]], dtype=np.int_)
    # For addressing particular ao derivatives
    if ot.xctype == 'LDA': idx = idx[:, 0:1]  # For LDAs, no second derivatives

    casdm2_0_pack = mcpdft_grad.pack_casdm2(cascm2_0, ncas)
    casdm2_pack = mcpdft_grad.pack_casdm2(cascm2, ncas)

    full_atmlst = -np.ones(mol.natm, dtype=np.int_)
    for k, ia in enumerate(atmlst):
        full_atmlst[ia] = k

    t1 = logger.timer(mc, 'L-PDFT HlFn quadrature setup', *t0)

    ndao = (1, 4)[ot.dens_deriv]
    ndpi = (1, 4)[ot.Pi_deriv]
    ncols = 1.05 * 3 * (ndao * nao + nocc) + max(ndao * nao, ndpi * ncas * ncas)
    # I have no idea if this is actually the correct number of columns, but I have a feeling it is not since I should
    # be accounting for the extra rows from feff stuff...

    for ia, (coords, w0, w1) in enumerate(rks_grad.grids_response_cc(ot.grids)):
        gc.collect()
        ngrids = coords.shape[0]
        remaining_floats = (max_memory - current_memory()[0]) * 1e6 / 8
        blksize = int(remaining_floats / (ncols * BLKSIZE)) * BLKSIZE
        blksize = max(BLKSIZE, min(blksize, ngrids, BLKSIZE * 1200))
        t1 = logger.timer(mc, 'L-PDFT HlFn quadrature atom {} mask and memory setup'.format(ia), *t1)
        for ip0 in range(0, ngrids, blksize):
            ip1 = min(ngrids, ip0 + blksize)
            mask = gen_grid.make_mask(mol, coords[ip0:ip1])
            logger.info(mc, 'L-PDFT gradient atom {} slice {}-{} of {} total'.format(ia, ip0, ip1, ngrids))
            ao = ot._numint.eval_ao(mol, coords[ip0:ip1], deriv=ot.dens_deriv + 1, non0tab=mask)

            t1 = logger.timer(mc, 'L-PDFT HlFn quadrature atom {} ao grids'.format(ia), *t1)

            if ot.xctype == "LDA":
                aoval = ao[0]

            if ot.xctype == "GGA":
                aoval = ao[:4]

            rho = make_rho(0, aoval, mask, ot.xctype) / 2.0
            rho = np.stack((rho,) * 2, axis=0)

            rho_0 = make_rho_0(0, aoval, mask, ot.xctype) / 2.0
            rho_0 = np.stack((rho_0,) * 2, axis=0)

            delta_rho = rho - rho_0

            t1 = logger.timer(mc, 'L-PDFT HlFn quadrature atom {} rho calc'.format(ia), *t1)

            Pi = get_ontop_pair_density(ot, rho, aoval, cascm2, mo_cas, ot.dens_deriv, mask)
            Pi_0 = get_ontop_pair_density(ot, rho_0, aoval, cascm2_0, mo_cas_0, ot.dens_deriv, mask)
            delta_Pi = Pi - Pi_0
            t1 = logger.timer(mc, 'L-PDFT HlFn quadrature atom {} Pi calc'.format(ia), *t1)

            if ot.xctype == "LDA":
                aoval = ao[:1]

            moval_occ = _grid_ao2mo(mol, aoval, mo_occ, mask)
            moval_occ_0 = _grid_ao2mo(mol, aoval, mo_occ_0, mask)
            t1 = logger.timer(mc, 'L-PDFT HlFn quadrature atom {} ao2mo grids'.format(ia), *t1)

            aoval = np.ascontiguousarray([ao[ix].transpose(0, 2, 1)
                                          for ix in idx[:, :ndao]]).transpose(0, 1, 3, 2)
            ao = None
            t1 = logger.timer(mc, 'L-PDFT HlFn quadrature atom {} ao grid reshape'.format(ia), *t1)

            eot, vot, fot = ot.eval_ot(rho_0, Pi_0, weights=w0[ip0:ip1], dderiv=2, _unpack_vot=False)
            fot = contract_fot(ot, fot, rho_0, Pi_0, delta_rho, delta_Pi, unpack=True, vot_packed=vot)
            vot = unpack_vot(vot, rho_0, Pi_0)
            # See the equations...
            eot += contract_vot(vot, delta_rho, delta_Pi)
            t1 = logger.timer(mc, 'PDFT HlFn quadrature atom {} eval_ot'.format(ia), *t1)

            puvx_mem = 2 * ndpi * (ip1 - ip0) * ncas * ncas * 8 / 1e6
            remaining_mem = max_memory - current_memory()[0]
            logger.info(mc, (
                'L-PDFT gradient memory note: working on {} grid points: estimated puvx usage = {:.1f} of {:.1f} '
                'remaining MB').format(
                (ip1 - ip0), puvx_mem, remaining_mem))

            # Weight response
            de_wgt += np.tensordot(eot, w1[atmlst, ..., ip0:ip1], axes=(0, 2))
            t1 = logger.timer(mc, 'L-PDFT HlFn quadrature atom {} weight response'.format(ia), *t1)

            # The mo_occup values might be screwing me here...
            # rho_delta * fot * drho_SA/dx
            tmp_df = mcpdft_grad.xc_response(ot, fot, rho_0, Pi_0, w0[ip0:ip1], moval_occ_0, aoval, mo_occ_0,
                                             mo_occup_0, ncore, nocc, casdm2_0_pack, ndpi, mo_cas_0)
            # vot * drho_Gamma/dx
            tmp_dv = mcpdft_grad.xc_response(ot, vot, rho, Pi, w0[ip0:ip1], moval_occ, aoval, mo_occ, mo_occup,
                                             ncore, nocc, casdm2_pack, ndpi, mo_cas)

            tmp_dxc = tmp_df + tmp_dv

            # Find the atoms that are part of the atomlist
            # grid correction shouldn't be added if they arent there
            k = full_atmlst[ia]
            if k >= 0:
                de_grid[k] += 2 * tmp_dxc.sum(1)  # Grid response

            dvxc -= tmp_dxc  # XC response

            tmp_dvxc = tmp_df = tmp_dv = None
            t1 = logger.timer(mc, 'L-PDFT HlFn quadrature atom {}'.format(ia), *t1)

            rho_0 = Pi_0 = rho = Pi = delta_rho = delta_Pi = None
            eot = vot = fot = aoval = moval_occ = moval_occ_0 = None
            gc.collect()

    return dvxc, de_wgt, de_grid


def lpdft_HellmanFeynman_grad(mc, ot, state, feff1, feff2, mo_coeff=None, ci=None, atmlst=None, mf_grad=None,
                              verbose=None,
                              max_memory=None):
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    if mf_grad is None: mf_grad = mc._scf.nuc_grad_method()
    if mc.frozen is not None:
        raise NotImplementedError
    mol = mc.mol
    if atmlst is None:
        atmlst = range(mol.natm)

    t0 = (logger.process_clock(), logger.perf_counter())

    # Specific state density
    casdm1s = mc.make_one_casdm1s(ci=ci, state=state)
    casdm1 = casdm1s[0] + casdm1s[1]
    dm1s = _dms.casdm1s_to_dm1s(mc, casdm1s=casdm1s)
    dm1 = dm1s[0] + dm1s[1]
    casdm2 = mc.make_one_casdm2(ci=ci, state=state)

    # The model-space density (or state-average density)
    casdm1s_0, casdm2_0 = mc.get_casdm12_0()
    dm1s_0 = _dms.casdm1s_to_dm1s(mc, casdm1s=casdm1s_0)
    dm1_0 = dm1s_0[0] + dm1s_0[1]
    casdm1_0 = casdm1s_0[0] + casdm1s_0[1]

    # Generate the Generalized Fock Component
    gfock_expl = mcpdft_grad.gfock_sym(mc, mo_coeff, casdm1, casdm2, mc.get_lpdft_hcore(), mc.veff2)
    gfock_impl = mcpdft_grad.gfock_sym(mc, mo_coeff, casdm1_0, casdm2_0, feff1, feff2)
    gfock = gfock_expl + gfock_impl

    dme0 = mo_coeff @ (0.5 * (gfock + gfock.T)) @ mo_coeff.T
    del gfock, gfock_impl, gfock_expl
    t0 = logger.timer(mc, 'L-PDFT HlFn gfock', *t0)

    # Coulomb potential derivatives generated from zero-order density
    vj = mf_grad.get_jk(dm=dm1)[0]
    vj_0 = mf_grad.get_jk(dm=dm1_0)[0]

    dvxc, de_wgt, de_grid = get_ontop_response(mc, ot, state, atmlst, casdm1, casdm1_0, mo_coeff=mo_coeff.copy(), ci=ci.copy(),
                                               max_memory=max_memory)

    delta_dm1 = dm1 - dm1_0

    def coul_term(p0, p1):
        return 2 * (np.tensordot(vj_0[:, p0:p1], delta_dm1[p0:p1]) + np.tensordot(vj[:, p0:p1], dm1_0[p0:p1]))

    de_hcore, de_coul, de_xc, de_nuc, de_renorm = mcpdft_grad.sum_terms(mf_grad, mol, atmlst, dm1, dme0, coul_term,
                                                                        dvxc)

    logger.debug(mc, "L-PDFT Hellmann-Feynman nuclear:\n{}".format(de_nuc))
    logger.debug(mc, "L-PDFT Hellmann-Feynman hcore component:\n{}".format(de_hcore))
    logger.debug(mc, "L-PDFT Hellmann-Feynman coulomb component:\n{}".format(de_coul))
    logger.debug(mc, "L-PDFT Hellmann-Feynman xc component:\n{}".format(de_xc))
    logger.debug(mc, "L-PDFT Hellmann-Feynman quadrature point component:\n{}".format(de_grid))
    logger.debug(mc, "L-PDFT Hellmann-Feynman quadrature weight component:\n{}".format(de_wgt))
    logger.debug(mc, "L-PDFT Hellmann-Feynman renorm component:\n{}".format(de_renorm))

    de = de_nuc + de_hcore + de_coul + de_renorm + de_xc + de_grid + de_wgt

    t1 = logger.timer(mc, 'L-PDFT HlFn total', *t0)

    return de


class Gradients(sacasscf.Gradients):
    def __init(self, pdft, state=None):
        super().__init__(pdft, state=state)

        if self.state is None and self.nroots == 1:
            self.state = 0

        self.e_mcscf = self.base.e_mcscf
        self._not_implemented_check()

    def _not_implemented_check(self):
        name = self.__class__.__name__
        if (isinstance(self.base, casci.CASCI) and not isinstance(self.base, mc1step.CASSCF)):
            raise NotImplementedError(
                "{} for CASCI-based MC-PDFT".format(name)
            )
        ot, otxc, nelecas = self.base.otfnal, self.base.otxc, self.base.nelecas
        spin = abs(nelecas[0] - nelecas[1])
        omega, alpha, hyb = ot._numint.rsh_and_hybrid_coeff(
            otxc, spin=spin)
        hyb_x, hyb_c = hyb
        if hyb_x or hyb_c:
            raise NotImplementedError(
                "{} for hybrid MC-PDFT functionals".format(name)
            )
        if omega or alpha:
            raise NotImplementedError(
                "{} for range-separated MC-PDFT functionals".format(name)
            )

    def kernel(self, **kwargs):
        state = kwargs['state'] if 'state' in kwargs else self.state
        if state is None:
            raise NotImplementedError('Gradient of LPDFT state-average energy')
        self.state = state
        mo = kwargs['mo'] if 'mo' in kwargs else self.base.mo_coeff
        ci = kwargs['ci'] if 'ci' in kwargs else self.base.ci
        if isinstance(ci, np.ndarray): ci = [ci]  # hack hack hack????? idk
        kwargs['ci'] = ci
        # need to compute feff1, feff2 if not already in kwargs
        if ('feff1' not in kwargs) or ('feff2' not in kwargs):
            kwargs['feff1'], kwargs['feff2'] = self.get_otp_gradient_response(mo, ci, state)

        return super().kernel(**kwargs)

    def get_wfn_response(self, state=None, verbose=None, mo=None, ci=None, feff1=None, feff2=None, nlag=None, **kwargs):
        if state is None: state = self.state
        if verbose is None: verbose = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if verbose is None: verbose = self.verbose
        if nlag is None: nlag = self.nlag
        if (feff1 is None) or (feff2 is None):
            feff1, feff2 = self.get_otp_gradient_response(mo, ci, state)

        log = lib.logger.new_logger(self, verbose)

        ndet = self.na_states[state] * self.nb_states[state]
        fcasscf = self.make_fcasscf(state)

        # Exploit (hopefully) the fact that the zero-order density is
        # really just the State Average Density!
        fcasscf_sa = self.make_fcasscf_sa()

        fcasscf.mo_coeff = mo
        fcasscf.ci = ci[state]

        fcasscf.get_hcore = self.base.get_lpdft_hcore
        fcasscf_sa.get_hcore = lambda: feff1

        g_all_explicit = newton_casscf.gen_g_hop(fcasscf, mo, ci[state], self.base.veff2, verbose)[0]
        g_all_implicit = newton_casscf.gen_g_hop(fcasscf_sa, mo, ci, feff2, verbose)[0]

        # Debug
        log.debug("g_all explicit mo:\n{}".format(g_all_explicit[:self.ngorb]))
        log.debug("g_all explicit CI:\n{}".format(g_all_explicit[self.ngorb:]))
        log.debug("g_all implicit mo:\n{}".format(g_all_implicit[:self.ngorb]))
        log.debug("g_all implicit CI:\n{}".format(g_all_implicit[self.ngorb:]))

        g_all = np.zeros(nlag)
        g_all[:self.ngorb] = g_all_explicit[:self.ngorb] + g_all_implicit[:self.ngorb]

        # Need to remove the SA-SA rotations from g_all_implicit CI contributions
        spin_states = np.asarray(self.spin_states)
        for root in range(self.nroots):
            idx_spin = spin_states == spin_states[root]
            idx = np.where(idx_spin)[0]

            offs = sum([na * nb for na, nb in zip(self.na_states[:root], self.nb_states[:root])]) if root > 0 else 0
            gci_root = g_all_implicit[self.ngorb:][offs:][:ndet]
            if root == state:
                gci_root += g_all_explicit[self.ngorb:]

            assert (root in idx)
            ci_proj = np.asarray([ci[i].ravel() for i in idx])
            gci_sa = np.dot(ci_proj, gci_root)
            gci_root -= np.dot(gci_sa, ci_proj)

            g_all[self.ngorb:][offs:][:ndet] = gci_root

        log.debug("g_mo component:\n{}".format(g_all[:self.ngorb]))
        log.debug("g_ci component:\n{}".format(g_all[self.ngorb:]))

        return g_all

    def get_ham_response(self, state=None, atmlst=None, verbose=None, mo=None, ci=None, eris=None, mf_grad=None,
                         feff1=None, feff2=None, **kwargs):
        if state is None: state = self.state
        if atmlst is None: atmlst = self.atmlst
        if verbose is None: verbose = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if (feff1 is None) or (feff2 is None):
            assert (False), kwargs

        return lpdft_HellmanFeynman_grad(self.base, self.base.otfnal, state, feff1=feff1, feff2=feff2, mo_coeff=mo,
                                         ci=ci, atmlst=atmlst, mf_grad=mf_grad, verbose=verbose)

    def get_otp_gradient_response(self, mo=None, ci=None, state=0):
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if state is None: state = self.state

        # This is the zero-order density
        casdm1s_0, casdm2_0 = self.base.get_casdm12_0()
        dm1s_0 = _dms.casdm1s_to_dm1s(self.base, casdm1s_0)

        # This is the density of the state we are differentiating with respect to
        casdm1s = self.base.make_one_casdm1s(ci=ci, state=state)
        casdm2 = self.base.make_one_casdm2(ci=ci, state=state)
        dm1s = _dms.casdm1s_to_dm1s(self.base, casdm1s)

        # We contract and have the coulomb generated from the "delta" density!
        delta_dm1s = dm1s - dm1s_0

        delta_casdm1s = (casdm1s[0] - casdm1s_0[0], casdm1s[1] - casdm1s_0[1])
        delta_casdm2 = casdm2 - casdm2_0
        delta_cascm2 = _dms.dm2_cumulant(delta_casdm2, delta_casdm1s)

        feff1, feff2= self.base.get_pdft_feff(mo=mo, ci=ci,
                                       casdm1s=casdm1s_0,
                                       casdm2=casdm2_0,
                                       c_dm1s=delta_dm1s,
                                       c_cascm2=delta_cascm2,
                                       jk_pc=True,
                                       paaa_only=True,
                                       incl_coul=True)

        return feff1, feff2


if __name__ == '__main__':
    from pyscf import scf, gto
    from pyscf import mcpdft

    xyz = '''O  0.00000000   0.08111156   0.00000000
             H  0.78620605   0.66349738   0.00000000
             H -0.78620605   0.66349738   0.00000000'''
    mol = gto.M(atom=xyz, basis='6-31g', symmetry=False, output='lpdft.log',
                verbose=5)
    mf = scf.RHF(mol).run()
    # mc = mcpdft.CASSCF (mf, 'tLDA,VWN3', 4, 4)
    mc = mcpdft.CASSCF(mf, 'ftPBE', 4, 4)
    mc.fix_spin_(ss=0)  # often necessary!
    mc = mc.multi_state([1.0 / 3, ] * 3, 'lin').run()
    mc_grad = Gradients(mc)
    de = np.stack([mc_grad.kernel(state=i) for i in range(1)], axis=0)
