#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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
# Author: Hong-Zhou Ye <hzyechem@gmail.com>
#

'''Plane wave-based Hartree-Fock for periodic systems
'''

from pyscf.pbc.pwscf import khf, kuhf

PWKRHF = KRHF = khf.PWKRHF
PWKUHF = KUHF = kuhf.PWKUHF

from pyscf.pbc.pwscf import krks, kuks

PWKRKS = KRKS = krks.PWKRKS
PWKUKS = KUKS = kuks.PWKUKS

from pyscf.pbc.pwscf import kmp2, kump2
PWKRMP2 = KRMP2 = PWKMP2 = KMP2 = kmp2.PWKRMP2
PWKUMP2 = KUMP2 = kump2.PWKUMP2

from pyscf.pbc.pwscf import kccsd_rhf
PWKRCCSD = KRCCSD = PWKCCSD = KCCSD = kccsd_rhf.PWKRCCSD
