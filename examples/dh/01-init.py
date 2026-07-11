"""Example: DFDH initialization — mol, SCF object, and to_dh conversion."""

from pyscf import gto, dft
from pyscf.dh import DFDH, to_dh

mol = gto.M(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="cc-pVDZ")

# 1. Init with gto.Mole 
mf = DFDH(mol, xc="B2PLYP").run()
print(f"B2PLYP via mol:   {mf.e_tot:.8f}")

# 2. Init with converged KS SCF — reuses orbitals, skips SCF
mf_ks = dft.KS(mol, xc="0.53*HF + 0.47*B88, 0.73*LYP").density_fit().run()
mf = DFDH(mf_ks, xc="B2PLYP").run()
print(f"B2PLYP via KS:    {mf.e_tot:.8f}")

# For xDH — pre-converge B3LYPg SCF for XYG3
mf_ks = dft.KS(mol, xc="B3LYPg").density_fit().run()
mf = DFDH(mf_ks, xc="XYG3").run()
print(f"XYG3 via B3LYPg:  {mf.e_tot:.8f}")

# 3. to_dh — reuse when SCF matches 
mf_ks = dft.KS(mol, xc="B3LYPg").density_fit().run()
mf = to_dh(mf_ks, xc="XYG3").run()
print(f"to_dh reuse XYG3: {mf.e_tot:.8f}")

# to_dh — auto-convert when SCF mismatches 
mf_ks = dft.KS(mol, xc="PBE0").density_fit().run()
mf = to_dh(mf_ks, xc="B2PLYP").run()
print(f"to_dh conv PBE0 -> B2PLYP: {mf.e_tot:.8f}")


