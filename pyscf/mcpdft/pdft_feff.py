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

import numpy as np
from pyscf.lib import logger, tag_array, current_memory
from pyscf.mcpdft.otpd import get_ontop_pair_density
from pyscf.mcpdft.pdft_eff import get_eff_1body, get_eff_2body, get_eff_2body_kl
from pyscf.mcpdft.tfnal_derivs import contract_fot


def kernel(ot, dm1s, cascm2, c_dm1s, c_cascm2, mo_coeff, ncore, ncas, max_memory=2000, hermi=1, paaa_only=False, aaaa_only=False, jk_pc=False):
    return lazy_kernel(ot, dm1s, cascm2, c_dm1s, c_cascm2, mo_coeff[:,ncore:ncore+ncas], hermi=hermi, max_memory=max_memory)


def lazy_kernel(ot, dm1s, cascm2, c_dm1s, c_cascm2, mo_cas, hermi=1, max_memory=2000):
    ni, xctype, dens_deriv = ot._numint, ot.xctype, ot.dens_deriv
    nao = mo_cas.shape[0]

    feff1 = np.zeros_like(dm1s[0])
    feff2 = np.zeros((nao, nao, nao, nao), dtype=feff1.dtype)

    t0 = (logger.process_clock(), logger.perf_counter())
    make_rho = tuple(ni._gen_rho_evaluator(ot.mol, dm1s[i,:,:], hermi) for i in range(2))
    make_crho = tuple(ni._gen_rho_evaluator(ot.mol, c_dm1s[i,:,:], hermi) for i in range(2))
    for ao, mask, weight, coords in ni.block_loop(ot.mol, ot.grids, nao, dens_deriv, max_memory):
        rho = np.asarray([m[0](0, ao, mask, xctype) for m in make_rho])
        crho = np.asarray([m[0](0, ao, mask, xctype) for m in make_crho])
        t0 = logger.timer(ot, 'untransformed density', *t0)
        Pi = get_ontop_pair_density(ot, rho, ao, cascm2, mo_cas, dens_deriv, mask)
        cPi = get_ontop_pair_density(ot, crho, ao, c_cascm2, mo_cas, dens_deriv, mask)
        t0 = logger.timer(ot, 'on-top pair density calculation', *t0)
        fot = ot.eval_ot(rho, Pi, weights=weight, dderiv=2)[2]
        frho, fPi = contract_fot(ot, fot, rho, Pi, crho, cPi)

        t0 = logger.timer(ot, 'effective gradient response kernel calculation', *t0)
        if ao.ndim == 2:
            ao = ao[None, :, :]

        feff1 += get_eff_1body(ot, ao, weight, frho)
        t0 = logger.timer(ot, '1-body effective gradient response calculation', *t0)

        feff2 += get_eff_2body(ot, ao, weight, fPi, aosym=1)
        t0 = logger.timer(ot, '2-body effective gradient response calculation', *t0)

    return feff1, feff2


def get_feff_1body(otfnal, ao, rho, Pi, crho, cPi, weight, kern=None, non0tab=None,
        shls_slice=None, ao_loc=None, hermi=0, **kwargs):
    if kern is None:
        if rho.ndim == 2:
            rho = np.expand_dims(rho, 1)
            Pi = np.expand_dims(Pi, 0)

        if crho.ndim == 2:
            crho = np.expand_dims(crho, 1)
            cPi = np.expand_dims(cPi, 0)

        fot = otfnal.eval_ot(rho, Pi, dderiv=2, **kwargs)[2]
        kern = contract_fot(otfnal, fot, rho, Pi, crho, cPi)[0]
        rho = np.squeeze(rho)
        Pi = np.squeeze(Pi)
        crho = np.squeeze(crho)
        cPi = np.squeeze(cPi)

    return get_eff_1body(otfnal, ao, weight, kern=kern, non0tab=non0tab, shls_slice=shls_slice, ao_loc=ao_loc, hermi=hermi)

def get_feff_2body(otfnal, rho, Pi, crho, cPi, ao, weight, aosym='s4', kern=None, fao=None, **kwargs):
    if kern is None:
        if rho.ndim == 2:
            rho = np.expand_dims(rho, 1)
            Pi = np.expand_dims(Pi, 0)

        if crho.ndim == 2:
            crho = np.expand_dims(crho, 1)
            cPi = np.expand_dims(cPi, 0)

        fot = otfnal.eval_ot(rho, Pi, dderiv=2, **kwargs)[2]
        kern = contract_fot(otfnal, fot, rho, Pi, crho, cPi)[1]
        rho = np.squeeze(rho)
        Pi = np.squeeze(Pi)
        crho = np.squeeze(crho)
        cPi = np.squeeze(cPi)

    return get_eff_2body(otfnal, ao, weight, kern, aosym=aosym, eff_ao=fao)

def get_feff_2body_kl(otfnal, rho, Pi, crho, cPi, ao_k, ao_l, weight, symm=False, kern=None, **kwargs):
    if kern is None:
        if rho.ndim == 2:
            rho = np.expand_dims(rho, 1)
            Pi = np.expand_dims(Pi, 0)

        if crho.ndim == 2:
            crho = np.expand_dims(crho, 1)
            cPi = np.expand_dims(cPi, 0)

        fot = otfnal.eval_ot(rho, Pi, dderiv=2, **kwargs)[2]
        kern = contract_fot(otfnal, fot, rho, Pi, crho, cPi)[1]

    return get_eff_2body_kl(ao_k, ao_l, weight, kern=kern, symm=symm)