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
# Author: Matthew Hennefarth <mhennefarth@uchicago.edu>

import numpy as np
import unittest

from pyscf import scf, gto, mcscf
from pyscf import mcpdft
from pyscf.data.nist import BOHR


def setUpModule():
    global h2, lih
    h2 = scf.RHF(gto.M(atom='H 0 0 0; H 1.5 0 0', basis='sto-3g',
                       output='lpdft.log', verbose=5)).run()
    lih = scf.RHF(gto.M(atom='Li 0 0 0; H 1.2 0 0', basis='sto-3g',
                        output='/dev/null', verbose=0)).run()


def tearDownModule():
    global h2, lih
    h2.mol.stdout.close()
    lih.mol.stdout.close()
    del h2, lih

def get_de(scanner, delta):
    xyz_forward = f'H 0 0 0; H {1.5+delta} 0 0'
    xyz_backward = f'H 0 0 0; H {1.5-delta} 0 0'
    scanner(xyz_forward)
    e_forward = np.asarray(scanner.e_states)
    scanner(xyz_backward)
    e_backward = np.asarray(scanner.e_states)

    return (e_forward - e_backward)/(2*delta)


class KnownValues(unittest.TestCase):
    def test_h2_sto3g(self):
        # There is a problem with Lagrange multiplier stuff with tPbe and 4 states for L-PDFT...

        mc = mcpdft.CASSCF(h2, 'ftPBE', 2, 2, grids_level=1)
        nstates = 4
        weights = [1.0 / nstates, ] * nstates

        lpdft = mc.multi_state(weights, method='lin')
        lpdft.run()

        mc = mc.state_average(weights)
        mc.run()

        mc_grad = mc.nuc_grad_method()
        lpdft_grad = lpdft.nuc_grad_method()

        mc_scanner = mc.as_scanner()
        lpdft_scanner = lpdft.as_scanner()
        e = []
        for p in range(19, 20):
            delta = 1.0/2**p
            e.append(get_de(mc_scanner, delta))

        e = np.array(e)
        print(e[-1, 0])
        print(mc_grad.kernel(state=0)/BOHR)

        print("LPDFT STUFF NOW")
        e = []
        for p in range(19, 20):
            delta = 1.0/2**p
            e.append(get_de(lpdft_scanner, delta))

        e = np.array(e)
        print(e[-1, 0])
        print(lpdft_grad.kernel(state=0)/BOHR)


if __name__ == "__main__":
    print("Full Tests for L-PDFT gradients API")
    unittest.main()
