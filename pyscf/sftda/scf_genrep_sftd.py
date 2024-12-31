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

# This file can be merged into pyscf.scf._response_functions.py

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import uhf
from pyscf.dft import numint,numint2c

# import fcuntion
from pyscf.sftda.numint_sftd import nr_uks_fxc_sf,nr_uks_fxc_sf_tda
from pyscf.sftda.numint2c_sftd import cache_xc_kernel_sf

def _gen_uhf_response_sf(mf, mo_coeff=None, mo_occ=None, hermi=0, extype=0, collinear_samples=200, max_memory=None):
    '''Generate a function to compute the product of Spin Flip UKS response function
    and UKS density matrices.
    '''
    assert isinstance(mf, (uhf.UHF))
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    mol = mf.mol

    ni = numint2c.NumInt2C()
    ni.collinear = 'mcol'
    ni.collinear_samples = collinear_samples
    ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
    if mf.nlc or ni.libxc.is_nlc(mf.xc):
        logger.warn(mf, 'NLC functional found in DFT object.  Its second '
                    'deriviative is not available. Its contribution is '
                    'not included in the response function.')
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    hybrid = ni.libxc.is_hybrid_xc(mf.xc)

    # mf can be pbc.dft.UKS object with multigrid
    if (not hybrid and
        'MultiGridFFTDF' == getattr(mf, 'with_df', None).__class__.__name__):
        raise NotImplementedError("Spin Flip TDDFT doesn't support pbc calculations.")


    fxc = cache_xc_kernel_sf(ni,mol, mf.grids, mf.xc, mo_coeff, mo_occ, 1)[2]
    dm0 = None

    if max_memory is None:
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, mf.max_memory*.8-mem_now)

    def vind(dm1):
        in2 = numint.NumInt()
        v1 = nr_uks_fxc_sf(in2,mol, mf.grids, mf.xc, dm0, dm1, 0, hermi,
                               None, None, fxc, extype=extype,max_memory=max_memory)
        if not hybrid:
            # No with_j because = 0 in spin flip part.
            pass
        else:
            vk = mf.get_k(mol, dm1, hermi=hermi)
            vkT= mf.get_k(mol, dm1.transpose(0,1,3,2), hermi=hermi)
            vk *= hyb
            vkT*= hyb
            if omega > 1e-10:  # For range separated Coulomb
                vk += mf.get_k(mol, dm1, hermi, omega) * (alpha-hyb)
                vkT+= mf.get_k(mol, dm1.transpose(0,1,3,2), hermi, omega) * (alpha-hyb)

            vk1A_b2a,vk1A_a2b = vk
            vk1B_a2b,vk1B_b2a = vkT
            vk1 = numpy.asarray((vk1A_b2a,vk1A_a2b,vk1B_b2a,vk1B_a2b))
            v1 -= vk1
        return v1
    return vind

def _gen_uhf_tda_response_sf(mf, mo_coeff=None, mo_occ=None, hermi=0, collinear_samples=200, max_memory=None):
    '''Generate a function to compute the product of Spin Flip UKS response function
    and UKS density matrices.
    '''
    assert isinstance(mf, (uhf.UHF))
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    mol = mf.mol

    ni = numint2c.NumInt2C()
    ni.collinear = 'mcol'
    ni.collinear_samples = collinear_samples
    ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
    if mf.nlc or ni.libxc.is_nlc(mf.xc):
        logger.warn(mf, 'NLC functional found in DFT object.  Its second '
                    'deriviative is not available. Its contribution is '
                    'not included in the response function.')
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    hybrid = ni.libxc.is_hybrid_xc(mf.xc)

    # mf can be pbc.dft.UKS object with multigrid
    if (not hybrid and
        'MultiGridFFTDF' == getattr(mf, 'with_df', None).__class__.__name__):
        raise NotImplementedError("Spin Flip TDDFT doesn't support pbc calculations.")


    fxc = cache_xc_kernel_sf(ni,mol, mf.grids, mf.xc, mo_coeff, mo_occ, 1)[2]
    dm0 = None

    if max_memory is None:
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, mf.max_memory*.8-mem_now)

    def vind(dm1):
        in2 = numint.NumInt()
        v1 = nr_uks_fxc_sf_tda(in2,mol, mf.grids, mf.xc, dm0, dm1, 0, hermi,
                               None, None, fxc, max_memory=max_memory)
        if not hybrid:
            # No with_j because = 0 in spin flip part.
            pass
        else:
            vk = mf.get_k(mol, dm1, hermi=hermi)
            vk *= hyb
            if omega > 1e-10:  # For range separated Coulomb
                vk += mf.get_k(mol, dm1, hermi, omega) * (alpha-hyb)
            v1 -= vk
        return v1
    return vind
