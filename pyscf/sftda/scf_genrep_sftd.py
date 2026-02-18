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

# This file can be merged into pyscf.scf._response_functions.py

import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import uhf
from pyscf.dft import KohnShamDFT, numint, numint2c

# import fcuntion
from pyscf.sftda.numint2c_sftd import cache_xc_kernel_sf

def gen_uhf_response_sf(mf, mo_coeff=None, mo_occ=None, hermi=0,
                        collinear_samples=200, max_memory=None):
    '''
    Generate a function to compute the product of Spin-flip UKS response function
    and UKS density matrices.
    '''
    assert isinstance(mf, (uhf.UHF))
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    mol = mf.mol

    if isinstance(mf, KohnShamDFT):
        ni = numint2c.NumInt2C()
        ni.collinear = 'mcol'
        ni.collinear_samples = collinear_samples
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)

        if mf.nlc or ni.libxc.is_nlc(mf.xc):
            logger.warn(mf, 'NLC functional found in DFT object. Its contribution is '
            'not included in the TDDFT response function.')
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
        hybrid = ni.libxc.is_hybrid_xc(mf.xc)

        if collinear_samples >= 0:
            fxc = 2.0 * cache_xc_kernel_sf(ni, mol, mf.grids, mf.xc, mo_coeff, mo_occ, deriv=2, spin=1)[2]

        dm0 = None

        if max_memory is None:
            max_memory = mf.max_memory
        max_memory = max_memory * 0.8 - lib.current_memory()[0]

        def vind(dm1):
            if collinear_samples < 0:
                v1 = np.zeros_like(dm1)
            else:
                in2 = numint.NumInt()
                v1 = in2.nr_rks_fxc(mol, mf.grids, mf.xc, dm0, dm1, 0, hermi,
                                    None, None, fxc, max_memory=max_memory)
            if hybrid:
                # j = 0 in spin flip part.
                if omega == 0:
                    vk = mf.get_k(mol, dm1, hermi) * hyb
                elif alpha == 0: # LR=0, only SR exchange
                    vk = mf.get_k(mol, dm1, hermi, omega=-omega) * hyb
                elif hyb == 0: # SR=0, only LR exchange
                    vk = mf.get_k(mol, dm1, hermi, omega=omega) * alpha
                else: # SR and LR exchange with different ratios
                    vk = mf.get_k(mol, dm1, hermi) * hyb
                    vk += mf.get_k(mol, dm1, hermi, omega=omega) * (alpha-hyb)
                v1 -= vk
            return v1
        return vind
    else: # HF
        def vind(dm1):
            vk = mf.get_k(mol, dm1, hermi)
            return -vk
        return vind
