#!/usr/bin/env python
"""
Example 83: FOMO-CASCI with DFT Core Embedding
==============================================

This example combines FOMO (Floating Occupation Molecular Orbital) orbitals
with DFT-corrected core energy.

Author:
    Arshad Mehmood, IACS, Stony Brook University
    arshad.mehmood@stonybrook.edu
    30 December 2025

References:
    P. Slavicek and T. J. Martinez, J. Chem. Phys. 132, 234102 (2010)
    S. Pijeau and E. G. Hohenstein, J. Chem. Theory Comput. 2017, 13, 1130
"""

from pyscf import gto, scf, mcscf
from pyscf.scf import fomoscf
from pyscf.mcscf import dft_corrected_casci
import numpy as np

# Build water molecule
mol = gto.M(
    atom='''
    O   0.000   0.000   0.000
    H   0.000   0.757   0.587
    H   0.000  -0.757   0.587
    ''',
    basis='cc-pvdz',
    unit='Angstrom',
    verbose=3  # Reduced verbosity
)

# Define active space
ncas = 6
nelecas = 6
ncore = (mol.nelectron - nelecas) // 2

print("="*70)
print("FOMO-CASCI with DFT Core Embedding")
print("="*70)
print(f"Active space: ({nelecas}e, {ncas}o)")
print(f"Core orbitals: {ncore}")
print(f"FOMO temperature: 0.25 eV (Gaussian smearing)")

# Run base RHF for FOMO
mf_rhf = scf.RHF(mol)
mf_rhf.verbose = 0
mf_rhf.run()

# =============================================================================
# Part 1: Standard CASCI (RHF orbitals, HF core)
# =============================================================================
print("\n" + "-"*70)
print("1. Standard CASCI (RHF orbitals, HF core)")
print("-"*70)
mc_std = mcscf.CASCI(mf_rhf, ncas, nelecas)
mc_std.verbose = 0
mc_std.kernel()
print(f"   Energy: {mc_std.e_tot:.10f} Ha")

# =============================================================================
# Part 2: FOMO-CASCI (FOMO orbitals, HF core)
# =============================================================================
print("\n" + "-"*70)
print("2. FOMO-CASCI (FOMO orbitals, HF core)")
print("-"*70)
mf_fomo = fomoscf.fomo_scf(
    mf_rhf, 
    temperature=0.25, 
    method='gaussian',
    restricted=(ncore, ncas)
)
mf_fomo.verbose = 0
mf_fomo.kernel()
mc_fomo = mcscf.CASCI(mf_fomo, ncas, nelecas)
mc_fomo.verbose = 0
mc_fomo.kernel()
print(f"   Energy: {mc_fomo.e_tot:.10f} Ha")

# =============================================================================
# Part 3: DFT-corrected CASCI (RHF orbitals, DFT core)
# =============================================================================
print("\n" + "-"*70)
print("3. DFT-corrected CASCI (RHF orbitals, PBE core)")
print("-"*70)
mc_dft = dft_corrected_casci.CASCI(mf_rhf, ncas, nelecas, xc='PBE')
mc_dft.verbose = 0
mc_dft.kernel()
print(f"   Energy: {mc_dft.e_tot:.10f} Ha")

# =============================================================================
# Part 4: FOMO-DFT-corrected CASCI (FOMO orbitals, DFT core) - The Full Method
# =============================================================================
print("\n" + "-"*70)
print("4. FOMO-DFT-corrected CASCI (FOMO orbitals, PBE core)")
print("-"*70)
mc_fomo_dft = dft_corrected_casci.CASCI(mf_fomo, ncas, nelecas, xc='PBE')
mc_fomo_dft.verbose = 0
mc_fomo_dft.kernel()
print(f"   Energy: {mc_fomo_dft.e_tot:.10f} Ha")

# =============================================================================
# Part 5: Try Different XC Functionals with FOMO
# =============================================================================
print("\n" + "-"*70)
print("5. FOMO-DFT-corrected CASCI with different XC functionals")
print("-"*70)

functionals = ['LDA', 'PBE', 'B3LYP', 'PBE0']

# Collect results first
xc_results = []
for xc in functionals:
    mc_xc = dft_corrected_casci.CASCI(mf_fomo, ncas, nelecas, xc=xc)
    mc_xc.verbose = 0
    mc_xc.kernel()
    delta_e = (mc_xc.e_tot - mc_fomo.e_tot) * 1000
    xc_results.append((xc, mc_xc.e_tot, delta_e))

# Now print the table
print(f"\n{'Functional':<15} {'Energy (Ha)':<20} {'ΔE from HF (mHa)':<20}")
print("-"*55)
for xc, energy, delta in xc_results:
    print(f"{xc:<15} {energy:<20.10f} {delta:<+20.4f}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*70)
print("Summary: Energy Comparison")
print("="*70)
print(f"{'Method':<40} {'Energy (Ha)':<20}")
print("-"*60)
print(f"{'Standard CASCI (RHF + HF core)':<40} {mc_std.e_tot:<20.10f}")
print(f"{'FOMO-CASCI (FOMO + HF core)':<40} {mc_fomo.e_tot:<20.10f}")
print(f"{'DFT-corrected CASCI (RHF + PBE core)':<40} {mc_dft.e_tot:<20.10f}")
print(f"{'FOMO-DFT-corrected CASCI (FOMO + PBE core)':<40} {mc_fomo_dft.e_tot:<20.10f}")
print("="*70)

# Analysis
print("\nEnergy contributions:")
print(f"  FOMO effect (HF core):     {(mc_fomo.e_tot - mc_std.e_tot)*1000:+.4f} mHa")
print(f"  DFT effect (RHF orbitals): {(mc_dft.e_tot - mc_std.e_tot)*1000:+.4f} mHa")
print(f"  Combined (FOMO + DFT):     {(mc_fomo_dft.e_tot - mc_std.e_tot)*1000:+.4f} mHa")

print("\nNote: FOMO-DFT-corrected CASCI is recommended for:")
print("  - Multireference systems requiring accurate core treatment")
print("  - Photochemistry and excited state dynamics")
print("  - Systems where both static and dynamic correlation are important")