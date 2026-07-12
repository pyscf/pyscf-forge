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

# 4. .dipole() method uses relaxed DM internally
dip = mf.dipole()
print(f"Dipole (via .dipole):   {np.linalg.norm(dip):.6f} a.u.")
