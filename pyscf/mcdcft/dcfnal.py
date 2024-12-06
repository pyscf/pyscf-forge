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

'''
Multiconfiguration Density-Coherence Functional Theory

Reference:
[1] Multiconfiguration Density-Coherence Functional Theory
    Dayou Zhang, Matthew R. Hermes, Laura Gagliardi and Donald G. Truhlar
    J. Chem. Theory and Comput. 2021 17 (5), 2775-2782
    DOI: 10.1021/acs.jctc.0c01346
[2] DC24: A new density coherence functional for multiconfiguration
    density-coherence functional theory.
    Dayou Zhang, Yinan Shu and Donald G. Truhlar
    J. Comput. Chem. 2024.
    DOI: 10.1002/jcc.27522
'''

from pyscf.lib import logger
from pyscf import dft, lib
from pyscf.dft2 import libxc
import numpy as np
import copy


def f_v1(occ):
    return (occ * (3.0 - occ)), (occ * (-1.0 + occ))

def get_f_v2_m(m):
    def f(occ):
        c = (1/m + m - np.sqrt(2 + m**(-2) + m**2 + 4*(-2 + occ)*occ))/2.
        return (occ + c), (occ - c)
    return f

f_v2 = get_f_v2_m(0.96)

def get_converted_rho(natorb, occ, ao, type_id, f=f_v2, negative_rho=False):
    '''  Calculate rho used in eval_xc
         This rho may contain the converted density, density derivitive, and kinetic
         energy density.

        Args:
            natorb : ndarray of shape (nao, nao)
                natural orbital
            occ : ndarray with shape (nao,)
                natural orbital occupation number
            ao : ndarray of shape (ngrids, nao) for LDA or (4, ngrids, nao) for GGA
                and MGGA
                magnitude of atomic basis function [and gradients]
            type_id : int
                0 for LDA; 1 for GGA; 2 for MGGA
            f : callable
                a callable that evaluates f(occ)±occ, where f(occ) is the effective
                numbers of unpaired electrons of a natural orbital with occupation
                number `occ`. It takes one argument `occ` as input, and returns a
                2-tuple, with the larger value being the first tuple element.
            negative_rho : bool
                if set to True, set negative converted rho values and their gradients
                to zero

        Returns : two ndarrays with shape (ngrids,) for LDA or (4, ngrids) for GGA
            or (6, ngrids) for MGGA
            This varible may be passed to eval_xc to evaluate the nonclassical energy
    '''

    if ao.ndim == 3:  # GGA
        ao_magnitude = ao[0]
        ao_magnitude_grad = ao[1:4]
    else:  # LDA
        ao_magnitude = ao

    # phi is the Magnitude of natural orbital and gradient
    phi = np.matmul(ao_magnitude, natorb)

    c_a, c_b = f(occ)
    phi_squared = phi * phi
    rhos_size = (1, 4, 6)[type_id]
    ngrids = ao.shape[-2]
    rhos_a = np.zeros((rhos_size, ngrids))
    rhos_b = np.zeros((rhos_size, ngrids))
    rho_a = rhos_a[0]
    rho_b = rhos_b[0]
    np.matmul(phi_squared, c_a, out=rho_a)
    np.matmul(phi_squared, c_b, out=rho_b)
    rho_a *= 0.5
    rho_b *= 0.5
    if type_id >= 1: # grad for GGA and MGGA
        phi_grad = np.matmul(ao_magnitude_grad, natorb)
        phi_phigrad = phi[None, :, :] * phi_grad
        np.matmul(phi_phigrad, c_a, out=rhos_a[1:4])
        np.matmul(phi_phigrad, c_b, out=rhos_b[1:4])
        if type_id >= 2: # tau for MGGA
            phigrad_phigrad = phi_grad * phi_grad
            np.einsum('a,dga->g', c_a, phigrad_phigrad, out=rhos_a[5])
            np.einsum('a,dga->g', c_b, phigrad_phigrad, out=rhos_b[5])
    if negative_rho:
        idx = rho_b < 0.
        rhos_b[:, idx] = 0.
    if rhos_size == 1:
        rhos_a = rhos_a[0]
        rhos_b = rhos_b[0]
    return (rhos_a, rhos_b)

# ALIAS for preset DC functionals
# `xc_code` and `hyb_x` are required keys
DC_PRESETS = {
        'DC24': {
            'xc_code': 164,
            'hyb_x': 4.525671e-01,
            'params': {164: [8.198942e-01, 4.106753e+00, -3.716774e+01, 1.100812e+02, -9.600026e+01,
                             1.352989e+01, -6.881959e+01, 2.371350e+02, -3.433615e+02, 1.720927e+02,
                             1.134169e+00, 1.148509e+01, -2.210990e+01, -1.006682e+02, 1.477906e+02]},
            'args': {'f': f_v2},
        },
}

_REGISTERED_PRESETS = {}

DEFAULT_RHO_ARGS = dict(f=f_v2)

_LIBXC_REGISTER_PREFIX = '_MC-DCFT_'

def register_dcfnal_(dc_code, preset):
    '''
    Register a new density coherence functional. Once registered,
    users may use the functional by passing `dc_code` to any MC-DCFT module.
    If `dc_code` has previously been registered, the older functional will
    be replaced with the new definition. New `dcfnal` isinstances must be
    re-constructed after updating a functional to avoid inconsistencies.

    Args:
    dc_code : str
        The string identifier of the density coherence functional
    preset : dict
        A dict that defines the functional. It can contain the following keys-value pairs:
        xc_code : int or str (required)
            the xc_code of the underlying Kohn-Sham functional. It follows the format of
            xc_code in LibXC interface
        hyb_x : float (required)
            mixing factor of MCSCF exchange-correlation energy
        display_name : str
            display name of the functional
        ext_params : dict, with LibXC functional integer ID as key, and an array-like
            object containing the functional parameters as value.
            Set the external parameters of the LibXC functional componet from the dict.
        args : dict
            keyword arguments to be passed to `get_converted_rho`. See `get_converted_rho`
            for a description of the keyword arguments.
    '''
    libxc_register_code = _LIBXC_REGISTER_PREFIX + dc_code
    libxc_base_code = preset['xc_code']

    # register in libxc module
    ext_params = preset.get('params')
    libxc.register_custom_functional_(libxc_register_code, libxc_base_code, ext_params=ext_params, hyb=0.)
    _REGISTERED_PRESETS[dc_code] = preset

def unregister_dcfnal_(dc_code):
    '''
    Unregister a density coherence functional with name `dc_code` that was previously registered
    through `register_dcfnal_`.
    '''
    libxc_register_code = _LIBXC_REGISTER_PREFIX + dc_code
    libxc.unregister_custom_functional_(libxc_register_code)
    del _REGISTERED_PRESETS[dc_code]

def _get_dc(dc_code):
    try:
        return _REGISTERED_PRESETS[dc_code]
    except KeyError:
        preset = DC_PRESETS.get(dc_code)
        if preset is None:
            raise KeyError(f'DC functional {dc_code} not defined.')
        register_dcfnal_(dc_code, preset)
        return preset

class dcfnal(lib.StreamObject):
    def __init__ (self, mol, dc_code, grids_level=None, verbose=0, **kwargs):
        self.mol = mol
        self.dc_code = dc_code
        preset = _get_dc(dc_code)
        self.hyb_x = preset.get('hyb_x', 0.)
        self.display_name = preset.get('display_name', dc_code)
        self.libxc_code = _LIBXC_REGISTER_PREFIX + dc_code
        self.get_converted_rho_args = preset.get('args', DEFAULT_RHO_ARGS)
        self.get_converted_rho = preset.get('get_converted_rho', get_converted_rho)
        ni = dft.numint.NumInt()
        ni.eval_xc = libxc.eval_xc
        ni.hybrid_coeff = lambda *x: 0.
        ni.rsh_coeff = lambda *x: (0., 0., 0.)
        self._numint = ni
        self.xctype = libxc.xc_type(self.libxc_code)
        ni._xc_type = lambda *x: self.xctype
        self.dens_deriv = ['LDA', 'GGA', 'MGGA'].index(self.xctype)
        self.grids = dft.grid.Grids(mol)
        if grids_level is not None:
            if isinstance(grids_level, int):
                self.grids.level = grids_level
            elif isinstance(grids_level, tuple):
                self.grids.atom_grid = grids_level
        self.grids.build()
        self.verbose = verbose
        if self.verbose >= logger.DEBUG:
            self.ms = 0.0

    def get_E_dc(self, natorb, occ, ao, weight):
        ''' E_dc[dm] = V_xc[rho_converted]

            Args:
                natorb : ndarray of shape (nao, nao)
                    generated by natorb
                occ : ndarray with shape (nao,)
                    occupation numbers of natorb
                ao : ndarray of shape (ngrids, nao) for LDA or (4, ngrids, nao) for GGA and MGGA
                    magnitude of atomic basis function [and gradients]
                weight : ndarray of shape (ngrids)
                    containing numerical integration weights

            Returns : float
                The density-coherence exchange-correlation energy
        '''
        assert (natorb.shape[1] == occ.shape[0]), \
                f"natorb.shape = {natorb.shape}, occ.shape = {occ.shape}"

        rho_c = self.get_converted_rho(natorb, occ, ao, self.dens_deriv, **self.get_converted_rho_args)

        dexc_ddens = self._numint.eval_xc(self.libxc_code,
            rho_c, spin=1, relativity=0, deriv=0, verbose=self.verbose)[0]
        rho = rho_c[0][0] + rho_c[1][0]
        rho *= weight

        if self.verbose >= logger.DEBUG:
            ms = np.dot(rho_c[0,0,:] - rho_c[1,0,:], weight) * 0.5
            self.ms += ms
            nelec = rho.sum()
            logger.debug(self, 'MC-DCFT: Total number of electrons in (this chunk of) the total density = %s', nelec)
            logger.debug(self,
                         'MC-DCFT: Total ms = (neleca - nelecb) / 2 in (this chunk of) the translated density = %s',
                         ms)

        return np.dot(dexc_ddens, rho)

