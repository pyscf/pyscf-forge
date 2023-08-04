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
# Author: Matthew R. Hermes
# Author: Matthew Hennefarth <mhennefarth@uchicago.com>

import numpy as np

from pyscf.lib import logger
from pyscf.mcpdft.otpd import _grid_ao2mo

# TODO: outcore implementation; can I use properties instead of copying?
class _ERIS(object):
    '''Stores two-body PDFT on-top effective integral arrays in a form
    compatible with existing MC-SCF kernel and derivative functions.
    Unlike actual eris, PDFT 2-electron effective integrals have 24-fold
    permutation symmetry, so j_pc = k_pc and ppaa = papa.transpose
    (0,2,1,3). The mcscf _ERIS is currently undocumented so I won't
    spend more time documenting this for now.
    '''
    def __init__(self, mol, mo_coeff, ncore, ncas, method='incore',
            paaa_only=False, aaaa_only=False, jk_pc=False, verbose=0, stdout=None):
        self.mol = mol
        self.mo_coeff = mo_coeff
        self.nao, self.nmo = mo_coeff.shape
        self.ncore = ncore
        self.ncas = ncas
        self.vhf_c = np.zeros ((self.nmo, self.nmo), dtype=mo_coeff.dtype)
        self.method = method
        self.paaa_only = paaa_only
        self.aaaa_only = aaaa_only
        self.jk_pc = jk_pc
        self.verbose = verbose
        self.stdout = stdout
        if method == 'incore':
            self.papa = np.zeros ((self.nmo, ncas, self.nmo, ncas),
                dtype=mo_coeff.dtype)
            self.j_pc = np.zeros ((self.nmo, ncore), dtype=mo_coeff.dtype)
        else:
            raise NotImplementedError ("method={} for pdft_eff2".format (
                self.method))

    def _accumulate (self, ot, rho, Pi, ao, weight, rho_c, rho_a, vPi,
            non0tab=None, shls_slice=None, ao_loc=None):
        args = [ot,rho,Pi,ao,weight,rho_c,rho_a,vPi,non0tab,shls_slice,ao_loc]
        self._accumulate_vhf_c (*args)
        if self.method.lower () == 'incore':
            self._accumulate_ppaa_incore (*args)
        else:
            raise NotImplementedError ("method={} for pdft_eff2".format (
                self.method))
        self._accumulate_j_pc (*args)

    def _accumulate_vhf_c (self, ot, rho, Pi, ao, weight, rho_c, rho_a, vPi,
            non0tab, shls_slice, ao_loc):
        mo_coeff = self.mo_coeff
        ncore, ncas = self.ncore, self.ncas
        nocc = ncore + ncas

        vrho_c = _contract_vot_rho (vPi, rho_c)
        self.vhf_c += mo_coeff.conjugate ().T @ ot.get_veff_1body (rho, Pi, ao,
            weight, non0tab=non0tab, shls_slice=shls_slice, ao_loc=ao_loc,
            hermi=1, kern=vrho_c) @ mo_coeff
        vrho_c = None
        self.energy_core = np.trace (self.vhf_c[:ncore,:ncore])/2
        if self.paaa_only:
            # 1/2 v_aiuv D_ii D_uv = v^ai_uv D_uv -> F_ai, F_ia
            # needs to be in here since it would otherwise be calculated using
            # ppaa and papa. This is harmless to the CI problem because the
            # elements in the active space and core-core sector are ignored
            # below.
            vrho_a = _contract_vot_rho (vPi, rho_a)
            vhf_a = ot.get_veff_1body (rho, Pi, ao, weight, non0tab=non0tab,
                shls_slice=shls_slice, ao_loc=ao_loc, hermi=1, kern=vrho_a)
            vhf_a = mo_coeff.conjugate ().T @ vhf_a @ mo_coeff
            vhf_a[ncore:nocc,:] = vhf_a[:,ncore:nocc] = 0.0
            vrho_a = None
            self.vhf_c += vhf_a

    def _ftpt_vhf_c (self):
        return self.nao+1

    def _accumulate_ppaa_incore (self, ot, rho, Pi, ao, weight, rho_c, rho_a,
            vPi, non0tab, shls_slice, ao_loc):
        # ao is here stored in row-major order = deriv,AOs,grids regardless of
        # what the ndarray object thinks
        mo_coeff = self.mo_coeff
        ncore, ncas = self.ncore, self.ncas
        nocc = ncore + ncas
        nderiv = vPi.shape[0]
        mo_cas = _grid_ao2mo (self.mol, ao[:nderiv], mo_coeff[:,ncore:nocc],
            non0tab)
        if self.aaaa_only:
            aaaa = ot.get_veff_2body (rho, Pi, [mo_cas, mo_cas, mo_cas,
                mo_cas], weight, aosym='s1', kern=vPi)
            self.papa[ncore:nocc,:,ncore:nocc,:] += aaaa
        elif self.paaa_only:
            paaa = ot.get_veff_2body (rho, Pi, [ao, mo_cas, mo_cas, mo_cas],
                weight, aosym='s1', kern=vPi)
            paaa = np.tensordot (mo_coeff.T, paaa, axes=1)
            self.papa[:,:,ncore:nocc,:] += paaa
            self.papa[ncore:nocc,:,:,:] += paaa.transpose (2,3,0,1)
            self.papa[ncore:nocc,:,ncore:nocc,:] -= paaa[ncore:nocc,:,:,:]
        else:
            papa = ot.get_veff_2body (rho, Pi, [ao, mo_cas, ao, mo_cas],
                weight, aosym='s1', kern=vPi)
            papa = np.tensordot (mo_coeff.T, papa, axes=1)
            self.papa += np.tensordot (mo_coeff.T, papa,
                axes=((1),(2))).transpose (1,2,0,3)

    def _ftpt_ppaa_incore (self):
        nao, ncas = self.nao, self.ncas
        ncol = 1 + 2*ncas
        ij_aa = int (self.aaaa_only)
        kl_aa = int (ij_aa or self.paaa_only)
        ncol += (ij_aa + kl_aa) * (nao - ncas)
        return ncol*ncas

    def _accumulate_j_pc (self, ot, rho, Pi, ao, weight, rho_c, rho_a, vPi,
            non0tab, shls_slice, ao_loc):
        mo_coeff = self.mo_coeff
        ncore = self.ncore
        nderiv = vPi.shape[0]
        if self.jk_pc:
            mo = _square_ao (_grid_ao2mo (self.mol, ao[:nderiv], mo_coeff,
                non0tab))
            mo_core = mo[:,:,:ncore]
            self.j_pc += ot.get_veff_1body (rho, Pi, [mo, mo_core], weight,
                kern=vPi)

    def _ftpt_j_pc (self):
        return self.nao + self.ncore + self.ncas + 1

    def _accumulate_ftpt (self):
        ''' memory footprint of _accumulate, divided by nderiv_Pi*ngrids '''
        ncol = 0
        ftpt_fns = [self._ftpt_vhf_c]
        if self.method.lower () == 'incore':
            ftpt_fns.append (self._ftpt_ppaa_incore)
        else:
            raise NotImplementedError ("method={} for pdft_eff2".format (
                self.method))
        if self.verbose > logger.DEBUG:
            ftpt_fns.append (self._ftpt_j_pc)
        ncol = 0
        for fn in ftpt_fns: ncol = max (ncol, fn ())
        return ncol

    def _finalize (self):
        if self.method == 'incore':
            self.ppaa = np.ascontiguousarray (self.papa.transpose (0,2,1,3))
            self.k_pc = self.j_pc.copy ()
        else:
            raise NotImplementedError ("method={} for pdft_eff2".format (
                self.method))
        self.k_pc = self.j_pc.copy ()


def _contract_vot_rho (vot, rho, add_vrho=None):
    ''' Make a jk-like vrho from vot and a density. k = j so it's just
        vot * vrho / 2 , but the product rule needs to be followed '''
    if rho.ndim == 1: rho = rho[None,:]
    nderiv = vot.shape[0]
    vrho = vot * rho[0]
    if nderiv > 1:
        vrho[0] += (vot[1:4] * rho[1:4]).sum (0)
    vrho /= 2
    # vot involves lower derivatives than vrho in original translation
    # make sure vot * rho gets added to only the proper component(s)
    if add_vrho is not None:
        add_vrho[:nderiv] += vrho
        vrho = add_vrho
    return vrho

def _square_ao (ao):
    # On a grid, square each element of an AO or MO array, but preserve the
    # chain rule so that columns 1 to 4 are still the first derivative of
    # the squared AO value, etc.
    nderiv = ao.shape[0]
    ao_sq = ao * ao[0]
    if nderiv > 1:
        ao_sq[1:4] *= 2
    if nderiv > 4:
        ao_sq[4:10] += ao[1:4]**2
        ao_sq[4:10] *= 2
    return ao_sq
