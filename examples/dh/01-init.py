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

"""Example: DFDH initialization"""

from pyscf import gto, dft
from pyscf.dh import DFDH, to_dh

mol = gto.M(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="cc-pVDZ")

# 1. Init with gto.Mole 
mf = DFDH(mol, xc="B2PLYP").run() # J. Chem. Phys. 2006, 124 (3), 034108.

print(f"B2PLYP via mol:   {mf.e_tot:.8f}")

# 2. Init with converged KS SCF 
# DFDH allows initialization mf_or_mol, but it only accepts and reuses mf 
# when its xc exactly matches the DH's SCF part ("0.53*HF + 0.47*B88, 0.73*LYP" for B2PLYP)
# It refuses to continue when SCF mismatches
mf_ks = dft.KS(mol, xc="0.53*HF + 0.47*B88, 0.73*LYP").density_fit().run()
mf = DFDH(mf_ks, xc="B2PLYP").run()
print(f"B2PLYP via KS:    {mf.e_tot:.8f}")

# example for xDH — pre-converge B3LYPg SCF for XYG3 (Proc. Natl. Acad. Sci. 2009, 106 (13), 4963–4968.)
mf_ks = dft.KS(mol, xc="B3LYPg").density_fit().run()
mf = DFDH(mf_ks, xc="XYG3").run()
print(f"XYG3 via B3LYPg:  {mf.e_tot:.8f}")

# 3. to_dh — reuse when SCF matches (the same as DFDH initialization)
mf_ks = dft.KS(mol, xc="B3LYPg").density_fit().run()
mf = to_dh(mf_ks, xc="XYG3").run()
print(f"to_dh reuse XYG3: {mf.e_tot:.8f}")

# to_dh also supports auto-convert when SCF mismatches 
mf_ks = dft.KS(mol, xc="PBE0").density_fit().run()
mf = to_dh(mf_ks, xc="B2PLYP").run()
print(f"to_dh conv PBE0 -> B2PLYP: {mf.e_tot:.8f}")


