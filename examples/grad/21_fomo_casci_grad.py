#!/usr/bin/env python
"""
Example 21: Nuclear Gradients for FOMO-CASCI (Both Methods)
===========================================================

This example demonstrates gradient calculations for FOMO-CASCI using
both analytical and numerical methods.

Author:
    Arshad Mehmood, IACS, Stony Brook University
    31 December 2025
"""

from pyscf import gto, scf, mcscf
from pyscf.scf import fomoscf
from pyscf.mcscf import dft_corrected_casci
import numpy as np
import time

mol = gto.M(
    atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
    basis='cc-pvdz', unit='Angstrom', verbose=3
)

ncas, nelecas = 6, 6
ncore = (mol.nelectron - nelecas) // 2

print("="*70)
print("FOMO-CASCI Nuclear Gradients: Analytical vs Numerical")
print("="*70)

# Standard RHF
mf_rhf = scf.RHF(mol).run()

# FOMO-SCF
mf_fomo = fomoscf.fomo_scf(
    mf_rhf, 
    temperature=0.25, 
    method='gaussian', 
    restricted=(ncore, ncas)
)
mf_fomo.kernel()
print(f"\nFOMO occupations: {mf_fomo.mo_occ}")

# Standard CASCI for comparison
print("\n" + "-"*70)
print("1. Standard CASCI (RHF orbitals)")
print("-"*70)
mc_std = mcscf.CASCI(mf_rhf, ncas, nelecas)
mc_std.kernel()
print(f"Energy: {mc_std.e_tot:.10f} Ha")

# Standard CASCI gradient (only analytical available)
g_std = mc_std.Gradients().kernel()

# FOMO-CASCI
print("\n" + "-"*70)
print("2. FOMO-CASCI (FOMO orbitals, LDA core)")
print("-"*70)
mc_fomo = dft_corrected_casci.CASCI(mf_fomo, ncas, nelecas, xc='LDA')
mc_fomo.kernel()
print(f"Energy: {mc_fomo.e_tot:.10f} Ha")

# Analytical gradient
print("\n2a. Analytical gradient:")
t0 = time.time()
g_fomo_analytical = mc_fomo.Gradients(method='analytical').kernel()
t_analytical = time.time() - t0
print(f"    Time: {t_analytical:.2f} seconds")

# Numerical gradient
print("\n2b. Numerical gradient:")
t0 = time.time()
g_fomo_numerical = mc_fomo.Gradients(method='numerical').kernel()
t_numerical = time.time() - t0
print(f"    Time: {t_numerical:.2f} seconds")

# Comparison
print("\n" + "="*70)
print("Gradient Comparison")
print("="*70)

atoms = ['O', 'H1', 'H2']
coords = ['x', 'y', 'z']

print(f"\n{'Atom':<6} {'Coord':<6} {'Standard':<15} {'FOMO-Ana':<15} {'FOMO-Num':<15}")
print("-"*57)

for i, atom in enumerate(atoms):
    for j, coord in enumerate(coords):
        print(f"{atom:<6} {coord:<6} {g_std[i,j]:>14.8f} {g_fomo_analytical[i,j]:>14.8f} {g_fomo_numerical[i,j]:>14.8f}")

# Analysis
diff_std_fomo = np.abs(g_std - g_fomo_analytical).max()
diff_methods = np.abs(g_fomo_analytical - g_fomo_numerical).max()

print("\nAnalysis:")
print(f"  Max difference (Standard vs FOMO analytical): {diff_std_fomo:.6f} Ha/Bohr")
print(f"  Max difference (Analytical vs Numerical):     {diff_methods:.6f} Ha/Bohr")
print(f"  Speedup (Analytical over Numerical):          {t_numerical/t_analytical:.1f}x")