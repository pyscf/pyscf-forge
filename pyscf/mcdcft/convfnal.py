#!/usr/bin/env python
# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
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
from pyscf.mcpdft.otfnal import otfnal, t_hybrid_coeff, t_nlc_coeff, t_rsh_coeff, t_eval_xc, t_xc_type
from pyscf.lib import logger
import numpy as np
import copy


class convfnal(otfnal):
    def __init__ (self, ks, **kwargs):
        otfnal.__init__(self, ks.mol, **kwargs)
        self.otxc = 'c' + ks.xc
        self._numint = copy.copy(ks._numint)
        self.grids = copy.copy(ks.grids)
        self._numint.hybrid_coeff = t_hybrid_coeff.__get__(self._numint)
        self._numint.nlc_coeff = t_nlc_coeff.__get__(self._numint)
        self._numint.rsh_coeff = t_rsh_coeff.__get__(self._numint)
        self._numint.eval_xc = t_eval_xc.__get__(self._numint)
        self._numint._xc_type = t_xc_type.__get__(self._numint)
        self._init_info()
        self.ms = 0.0

    def _set_natorb(self, natorb, occ):
        self.natorb = natorb
        self.occ = occ

    def get_E_ot (self, rho, D, weight):
        r''' E_ot[rho, Pi] = V_xc[rho_translated]

            Args:
                rho : ndarray of shape (2,*,ngrids)
                    containing spin-density [and derivatives]
                D : ndarray with shape (4, ngrids)
                    containing unpaired density and derivatives
                weight : ndarray of shape (ngrids)
                    containing numerical integration weights

            Returns : float
                The on-top exchange-correlation energy, for an on-top xc functional
                which uses a translated density with an otherwise standard xc functional
        '''
        assert (rho.shape[-1] == D.shape[-1]), f"rho.shape={rho.shape}, D.shape={D.shape}"
        if rho.ndim == 2:
            rho = np.expand_dims (rho, 1)

        rho_t = self.get_rho_converted(rho, D)
        rho = np.squeeze(rho)

        dexc_ddens = self._numint.eval_xc(
            self.otxc, (rho_t[0,:,:], rho_t[1,:,:]), spin=1, relativity=0, deriv=0, verbose=self.verbose)[0]
        rho = rho_t[:,0,:].sum(0)
        rho *= weight
        dexc_ddens *= rho

        ms = np.dot(rho_t[0,0,:] - rho_t[1,0,:], weight) / 2.0
        self.ms += ms
        if self.verbose >= logger.DEBUG:
            nelec = rho.sum()
            logger.debug(self, 'MC-DCFT: Total number of electrons in (this chunk of) the total density = %s', nelec)
            logger.debug(self,
                         'MC-DCFT: Total ms = (neleca - nelecb) / 2 in (this chunk of) the translated density = %s',
                         ms)

        return dexc_ddens.sum()

    def get_rho_converted(self, rho, D):
        r''' converted rho
        rho_c[0] = {(rho[0] + rho[1]) / 2} + D / 2
        rho_c[1] = {(rho[0] + rho[1]) / 2} - D / 2

            Args:
                rho : ndarray of shape (2, *, ngrids)
                    containing spin density [and derivatives]
                D : ndarray of shape (*, ngrids)
                    containing on-top pair density [and derivatives]

            Returns: ndarray of shape (2, *, ngrids)
                containing converted unpaired density (and derivatives)
        '''
        rho_avg = (rho[0] + rho[1]) / 2.0
        D_half = D / 2.0

        rho_c = np.empty_like(rho)
        rho_c[0] = rho_avg + D_half
        rho_c[1] = rho_avg - D_half

        # if rho_beta is negative or small, set rho_beta and its gradient to zero
        negative = rho_c[1,0] < 1e-10
        rho_c[1,:,negative] = 0.

        return rho_c
