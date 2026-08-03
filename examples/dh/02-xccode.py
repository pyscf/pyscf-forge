#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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
# Authors:
#          Shirong Wang <srwang20@fudan.edu.cn>
#

"""Example: xccode — how XC strings work in double-hybrid calculations."""

from pyscf import gto
from pyscf.dh import DFDH
from pyscf.dh.xccode import parse_xc_dh, xc_equal, describe_xc_dh

mol = gto.M(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="cc-pVDZ")

# 1. DH functionals can be any name from the JSON database
for name in ["B2PLYP", "XYG3", "PBE0-DH", "DSD-PBEP86-D3BJ"]:
    mf = DFDH(mol, xc=name).run()
    print(f"{name:25s} {mf.e_tot:.10f}")

# 2. bDH functionals: separate OS and SS MP2 terms
mf = DFDH(mol, xc="revDSD-PBEP86-D3BJ").run()
print(f"revDSD-PBEP86-D3BJ: {mf.e_tot:.10f}")

# Same bDH via full code string with D3 XC= name
code = "0.69*HF + 0.31*PBE, 0.4296*P86 + 0.5785*MP2_OS + 0.0799*MP2_SS + DFTD3(BJ, XC=revdsdpbep86)"
mf = DFDH(mol, xc=code).run()
print(f"revDSD-PBEP86-D3BJ (code): {mf.e_tot:.10f}")

# 3. xDH via 2-tuple (code_scf, code_eng)
xc_xdh = ("B3LYPg", "0.8033*HF - 0.0140*LDA + 0.2107*B88, 0.6789*LYP + 0.3211*MP2")
mf = DFDH(mol, xc=xc_xdh).run()
print(f"custom XYG3 (2-tuple): {mf.e_tot:.10f}")

# 4. Inspect any DH functional with describe_xc_dh
for name in ["MP2", "B2PLYP", "XYG3", "DSD-PBEP86-D3BJ", "revDSD-PBEP86-D3BJ"]:
    describe_xc_dh(name)

# 5. XC token comparison via xc_equal
print(f"\nxc_equal('HF', 'HF,') = {xc_equal('HF', 'HF,')}")
print(f"xc_equal('B3LYPG', 'B3LYPg') = {xc_equal('B3LYPG', 'B3LYPg')}")
print(f"xc_equal('PBE0', 'B3LYPG') = {xc_equal('PBE0', 'B3LYPG')}")
