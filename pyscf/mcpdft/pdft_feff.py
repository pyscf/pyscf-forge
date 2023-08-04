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
from pyscf.mcpdft._pdft_eris import _ERIS

def kernel(ot, dm1s, cascm2, contract_dm1s, contract_cascm2, mo_coeff, ncore, ncas, max_memory=2000, hermi=1, paaa_only=False, aaaa_only=False, jk_pc=False):
    nocc = ncore + ncas
    ni, xctype, dens_deriv = ot._numint, ot.xctype, ot.dens_deriv
    nao = mo_coeff.shape[0]
    mo_core = mo_coeff[:,:ncore]
    mo_cas = mo_coeff[:,ncore:nocc]
    shls_slice = (0, ot.mol.nbas)
    ao_loc = ot.mol.ao_loc_nr()

    omega, alpha, hyb = ot._numint.rsh_and_hybrid_coeff(ot.otxc)
    hyb_x, hyb_c = hyb
    if abs(omega) > 1e-11:
        raise NotImplementedError("range-separated on-top functionals")

    if drop_mcwfn:
        if abs(hyb_x - hyb_c) > 1e-11:
            raise NotImplementedError(
                "effective potential for hybrid functionals with different exchange, correlations components")

    elif abs(hyb_x) > 1e-11 or abs(hyb_c) > 1e-11:
        raise NotImplementedError("effective potential for hybrid functionals")

    feff1 = np.zeros((nao, nao), dtype=dm1s.dtype)
    feff2 = _ERIS(ot.mol, mo_coeff, ncore, ncas, paaa_only=paaa_only, aaaa_only=aaaa_only, jk_pc=jk_pc, verbose=ot.verbose, stdout=ot.stdout)
