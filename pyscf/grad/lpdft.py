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

from pyscf.mcscf import casci, mc1step
from pyscf.grad import sacasscf

import numpy as np

class Gradients (sacasscf.Gradients):
    def __init(self, pdft, state=None):
        super().__init__(pdft, state=state)

        if self.state is None and self.nroots == 1:
            self.state = 0

        self.e_mcscf = self.base.e_mcscf
        self._not_implemented_check()

    def _not_implemented_check(self):
        name = self.__class__.__name__
        if (isinstance (self.base, casci.CASCI) and not
            isinstance (self.base, mc1step.CASSCF)):
            raise NotImplementedError (
                "{} for CASCI-based MC-PDFT".format (name)
            )
        ot, otxc, nelecas = self.base.otfnal, self.base.otxc, self.base.nelecas
        spin = abs (nelecas[0]-nelecas[1])
        omega, alpha, hyb = ot._numint.rsh_and_hybrid_coeff (
            otxc, spin=spin)
        hyb_x, hyb_c = hyb
        if hyb_x or hyb_c:
            raise NotImplementedError (
                "{} for hybrid MC-PDFT functionals".format (name)
            )
        if omega or alpha:
            raise NotImplementedError (
                "{} for range-separated MC-PDFT functionals".format (name)
            )

    def kernel(self, **kwargs):
        state = kwargs['state'] if 'state' in kwargs else self.state
        if state is None:
            raise NotImplementedError('Gradient of LPDFT state-average energy')
        self.state = state
        mo = kwargs['mo'] if 'mo' in kwargs else self.base.mo_coeff
        ci = kwargs['ci'] if 'ci' in kwargs else self.base.ci
        if isinstance(ci, np.ndarray): ci = [ci] #hack hack hack????? idk
        kwargs['ci'] = ci
        # need to compute feff1, feff2 if not already in kwargs
        return super().kernel(**kwargs)