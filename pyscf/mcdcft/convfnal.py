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
from pyscf.lib import logger
from pyscf.dft2.libxc import XCFunctional
from pyscf import dft
import numpy as np
import copy


class convfnal:
    def __init__ (self, mol, xc_code, hyb_x=0., grids=None, display_name=None, **kwargs):
        self.mol = mol
        self.otxc = xc_code
        self.hyb_x = hyb_x
        self.display_name = 'c' + xc_code if display_name is None else display_name
        xcfunc = XCFunctional(xc_code, 1)
        ni = dft.numint.NumInt()
        ni.eval_xc = xcfunc.eval_xc
        ni.hybrid_coeff = lambda *x: 0.
        ni.rsh_coeff = lambda *x: (0., 0., 0.)
        ni._xc_type = xcfunc.xc_type_
        self._numint = ni
        self.xcfunc = xcfunc
        self.xctype = xcfunc.xc_type()
        self.dens_deriv = ['LDA', 'GGA', 'MGGA'].index(self.xctype)
        self.grids = dft.grid.Grids(mol).build() if grids is None else grids
        self.xcfunc = xcfunc
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
            (rho_t[0,:,:], rho_t[1,:,:]), spin=1, relativity=0, deriv=0, verbose=self.verbose)[0]
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
