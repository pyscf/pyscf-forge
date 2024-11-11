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

import numpy
import unittest
from pyscf import gto
from pyscf import scf
from pyscf import lib
from pyscf.pv.energy import Epv_molecule

def setUpModule():
    global mol, mf

#   alanine molecule
    mol = gto.M(
    atom = '''
    O   -0.000008 0.000006 0.473161
    O   -0.498429 1.617953 -0.942950
    N   -2.916494 2.018558 0.304530
    C   -2.245961 0.738717 0.446378
    C   -2.933825 -0.437589 -0.265779
    C   -0.836260 0.869228 -0.089564
    H   -2.164332 0.502686 1.502658
    H   -2.396710 -1.368611 -0.107150
    H   -3.940684 -0.559206 0.124631
    H   -3.002345 -0.251817 -1.334665
    H   -3.903065 1.915512 0.431392
    H   -2.755346 2.398021 -0.608200
    H   0.850292 0.091697 0.064913
    ''',
    basis = 'sto3g',
    verbose = 3)

    mf = scf.DHF(mol)
    mf.conv_tol = 1e-9
    mf.with_ssss=False
    mf.kernel()

def tearDownModule():
    global mol, mf
    del mol, mf

class KnownValues(unittest.TestCase):
    def test_pv(self):
#       Reference results (for PV) compared with DIRAC24
        Epv = Epv_molecule(mol, mf)
        Epv_Oxigen = numpy.sum(Epv, axis=1)[0]
        self.assertAlmostEqual(Epv_Oxigen, -2.745821e-21, 5)
        self.assertAlmostEqual(mf.e_tot,-317.831176378,8)


if __name__ == "__main__":
    print("Full Tests for Parity violating quantities")
    unittest.main()


