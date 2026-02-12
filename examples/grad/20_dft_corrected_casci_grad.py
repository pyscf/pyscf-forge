#!/usr/bin/env python
"""
Example 20: Nuclear Gradients for DFT-Corrected CASCI 
=====================================================

This example demonstrates nuclear gradient calculations for CASCI with
DFT-corrected core energy using BOTH analytical and numerical methods.

Gradient Methods:
1. Analytical: Fast (~1-2e-2 Ha/Bohr accuracy)
2. Numerical: Slow but accurate (~1e-6 Ha/Bohr accuracy)
3. Auto: Tries analytical, falls back to numerical if needed

Author:
    Arshad Mehmood, IACS, Stony Brook University
    arshad.mehmood@stonybrook.edu
    31 December 2025
"""

from pyscf import gto, scf, mcscf
from pyscf.mcscf import dft_corrected_casci
import numpy as np
import time

# Build water molecule
mol = gto.M(
    atom='''
    O   0.000   0.000   0.000
    H   0.000   0.757   0.587
    H   0.000  -0.757   0.587
    ''',
    basis='cc-pvdz',
    unit='Angstrom',
    verbose=3
)

ncas, nelecas = 6, 6

print("="*70)
print("DFT-corrected CASCI Nuclear Gradients: Analytical vs Numerical")
print("="*70)

# =============================================================================
# Part 1: RHF Reference - Compare Methods
# =============================================================================
print("\n" + "="*70)
print("Part 1: RHF Reference - Method Comparison")
print("="*70)

mf_rhf = scf.RHF(mol).run()
mc = dft_corrected_casci.CASCI(mf_rhf, ncas, nelecas, xc='PBE')
mc.kernel()

print(f"\nCASCI Energy: {mc.e_tot:.10f} Ha")

# Analytical gradient
print("\n1a. Analytical Gradient (fast, ~1-2e-2 Ha/Bohr accuracy):")
t0 = time.time()
g_analytical = mc.Gradients(method='analytical').kernel()
t_analytical = time.time() - t0
print(f"    Time: {t_analytical:.2f} seconds")

# Numerical gradient
print("\n1b. Numerical Gradient (slow, ~1e-6 Ha/Bohr accuracy):")
t0 = time.time()
g_numerical = mc.Gradients(method='numerical', step_size=1e-4).kernel()
t_numerical = time.time() - t0
print(f"    Time: {t_numerical:.2f} seconds")
print(f"    Speedup: {t_numerical/t_analytical:.1f}x faster with analytical")

# Auto mode
print("\n1c. Auto Mode (tries analytical, falls back to numerical):")
g_auto = mc.Gradients(method='auto').kernel()

# =============================================================================
# Part 2: Detailed Comparison
# =============================================================================
print("\n" + "="*70)
print("Part 2: Gradient Comparison (Ha/Bohr)")
print("="*70)

atoms = ['O', 'H1', 'H2']
coords = ['x', 'y', 'z']

print(f"\n{'Atom':<6} {'Coord':<6} {'Analytical':<15} {'Numerical':<15} {'|Difference|':<15}")
print("-"*66)

max_diff = 0.0
for i, atom in enumerate(atoms):
    for j, coord in enumerate(coords):
        diff = abs(g_analytical[i,j] - g_numerical[i,j])
        max_diff = max(max_diff, diff)
        print(f"{atom:<6} {coord:<6} {g_analytical[i,j]:>14.8f} {g_numerical[i,j]:>14.8f} {diff:>14.8f}")

print("-"*66)
print(f"Maximum difference: {max_diff:.2e} Ha/Bohr")
print(f"RMS difference:     {np.sqrt(np.mean((g_analytical - g_numerical)**2)):.2e} Ha/Bohr")

# =============================================================================
# Part 3: UHF Reference (Numerical Only)
# =============================================================================
print("\n" + "="*70)
print("Part 3: UHF Reference (Numerical Gradients)")
print("="*70)

mf_uhf = scf.UHF(mol).run()
mc_uhf = dft_corrected_casci.UCASCI(mf_uhf, ncas, nelecas, xc='PBE')
mc_uhf.kernel()

print(f"\nUCASCI Energy: {mc_uhf.e_tot:.10f} Ha")
print("\nNote: UCASCI only supports numerical gradients")

t0 = time.time()
g_uhf = mc_uhf.Gradients(method='numerical').kernel()
t_uhf = time.time() - t0
print(f"Time: {t_uhf:.2f} seconds")

print(f"\n{'Atom':<6} {'Coord':<6} {'Gradient':<15}")
print("-"*27)
for i, atom in enumerate(atoms):
    for j, coord in enumerate(coords):
        print(f"{atom:<6} {coord:<6} {g_uhf[i,j]:>14.8f}")

# =============================================================================
# Part 4: Different XC Functionals
# =============================================================================
print("\n" + "="*70)
print("Part 4: Gradient Comparison Across XC Functionals")
print("="*70)

functionals = ['LDA', 'PBE', 'B3LYP']
results = []

for xc in functionals:
    mc_xc = dft_corrected_casci.CASCI(mf_rhf, ncas, nelecas, xc=xc)
    mc_xc.kernel()
    g_xc = mc_xc.Gradients(method='analytical').kernel()
    results.append((xc, mc_xc.e_tot, g_xc))

print(f"\n{'Functional':<12} {'Energy (Ha)':<18} {'Max Gradient':<15}")
print("-"*45)
for xc, energy, grad in results:
    max_g = np.abs(grad).max()
    print(f"{xc:<12} {energy:<18.10f} {max_g:<15.8f}")

# =============================================================================
# Part 5: Performance Recommendations
# =============================================================================
print("\n" + "="*70)
print("Performance and Accuracy Summary")
print("="*70)

print("\nMethod Characteristics:")
print(f"{'Method':<15} {'Speed':<12} {'Accuracy':<20} {'Use Case'}")
print("-"*70)
print(f"{'Analytical':<15} {'Fast':<12} {'~1-2e-2 Ha/Bohr':<20} {'MD, optimization (loose)'}")
print(f"{'Numerical':<15} {'Slow':<12} {'~1e-6 Ha/Bohr':<20} {'Validation, tight opt'}")
print(f"{'Auto':<15} {'Fast*':<12} {'Variable':<20} {'General purpose'}")

print("\nRecommendations:")
print("  • Use analytical for: Molecular dynamics, exploratory optimization")
print("  • Use numerical for: Final geometries, benchmark calculations")
print("  • Use auto for: General purpose (safe default)")
print(f"  • Current speedup: Analytical is {t_numerical/t_analytical:.1f}x faster")

print("\n" + "="*70)