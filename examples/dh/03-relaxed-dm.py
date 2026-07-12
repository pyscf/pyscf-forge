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

"""Example: Relaxed density matrix 
"""

from pyscf import gto
from pyscf.dh import DFDH
import numpy as np

mol = gto.M(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="cc-pVDZ")

# 1. Run DH energy
mf = DFDH(mol, xc="B2PLYP").run()
print(f"B2PLYP energy: {mf.e_tot:.10f}")

# 2. Access relaxed DM via nuc_grad_method() 
mf_g = mf.nuc_grad_method()

# Relaxed DM (MO basis)
rdm1_mo = mf_g.make_rdm1_relaxed()
D_r_mo = mf_g.tensors["D_r"]

# 3. Relaxed DM (AO basis)
rdm1_ao = mf_g.make_rdm1_relaxed(ao_repr=True)

# 4. .dipole() method on Gradients uses relaxed DM internally
dip = mf_g.dipole()
print(f"Dipole (via .dipole):   {np.linalg.norm(dip):.6f} a.u.")
