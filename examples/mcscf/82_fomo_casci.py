#!/usr/bin/env python
"""
Example 82: FOMO-CASCI (Floating Occupation Molecular Orbital CASCI)
====================================================================

This example demonstrates FOMO-CASCI calculations where the orbitals are
obtained from a FOMO-SCF (Fractional Occupation) calculation instead of
standard RHF.

FOMO-SCF uses a smearing function to assign fractional occupations to
orbitals near the Fermi level, which helps avoid SCF convergence issues
in systems with near-degeneracies.

Author:
    Arshad Mehmood, IACS, Stony Brook University
    arshad.mehmood@stonybrook.edu
    30 December 2025

Reference:
    P. Slavicek and T. J. Martinez,
    J. Chem. Phys. 132, 234102 (2010)
    https://doi.org/10.1063/1.3436501
"""

from pyscf import gto, scf, mcscf
from pyscf.scf import fomoscf
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
ncore = (mol.nelectron - nelecas) // 2  # Number of core orbitals

print("="*70)
print("FOMO-CASCI: Floating Occupation Molecular Orbital CASCI")
print("="*70)
print(f"Active space: ({nelecas}e, {ncas}o)")
print(f"Core orbitals: {ncore}")

# =============================================================================
# Part 1: Standard RHF + CASCI for Comparison
# =============================================================================
print("\n" + "-"*70)
print("1. Standard RHF + CASCI")
print("-"*70)
mf_rhf = scf.RHF(mol).run()
mc_std = mcscf.CASCI(mf_rhf, ncas, nelecas)
mc_std.verbose = 0
mc_std.kernel()
print(f"   RHF Energy:   {mf_rhf.e_tot:.10f} Ha")
print(f"   CASCI Energy: {mc_std.e_tot:.10f} Ha")

# =============================================================================
# Part 2: FOMO-SCF + CASCI with Gaussian Smearing
# =============================================================================
print("\n" + "-"*70)
print("2. FOMO-SCF (Gaussian smearing, T=0.25 eV) + CASCI")
print("-"*70)

# Create FOMO-SCF object
mf_fomo_gauss = fomoscf.fomo_scf(
    mf_rhf, 
    temperature=0.25,      # Smearing temperature in eV
    method='gaussian',     # Gaussian smearing
    restricted=(ncore, ncas)  # Restrict fractional occupations to active space
)
mf_fomo_gauss.verbose = 0
mf_fomo_gauss.kernel()

print(f"   FOMO-SCF Energy: {mf_fomo_gauss.e_tot:.10f} Ha")
print(f"   Orbital occupations: {mf_fomo_gauss.mo_occ}")

# FOMO-CASCI
mc_fomo_gauss = mcscf.CASCI(mf_fomo_gauss, ncas, nelecas)
mc_fomo_gauss.verbose = 0
mc_fomo_gauss.kernel()
print(f"   FOMO-CASCI Energy: {mc_fomo_gauss.e_tot:.10f} Ha")

# =============================================================================
# Part 3: FOMO-SCF + CASCI with Fermi-Dirac Smearing
# =============================================================================
print("\n" + "-"*70)
print("3. FOMO-SCF (Fermi-Dirac smearing, T=0.25 eV) + CASCI")
print("-"*70)

mf_fomo_fermi = fomoscf.fomo_scf(
    scf.RHF(mol),
    temperature=0.25,
    method='fermi',        # Fermi-Dirac smearing
    restricted=(ncore, ncas)
)
mf_fomo_fermi.verbose = 0
mf_fomo_fermi.kernel()

print(f"   FOMO-SCF Energy: {mf_fomo_fermi.e_tot:.10f} Ha")
print(f"   Orbital occupations: {mf_fomo_fermi.mo_occ}")

mc_fomo_fermi = mcscf.CASCI(mf_fomo_fermi, ncas, nelecas)
mc_fomo_fermi.verbose = 0
mc_fomo_fermi.kernel()
print(f"   FOMO-CASCI Energy: {mc_fomo_fermi.e_tot:.10f} Ha")

# =============================================================================
# Part 4: Effect of Temperature on FOMO
# =============================================================================
print("\n" + "-"*70)
print("4. Effect of smearing temperature on FOMO-CASCI")
print("-"*70)

temperatures = [0.1, 0.25, 0.5, 1.0]

# First, collect all results
results = []
for temp in temperatures:
    mf_temp = fomoscf.fomo_scf(
        scf.RHF(mol),
        temperature=temp,
        method='gaussian',
        restricted=(ncore, ncas)
    )
    mf_temp.verbose = 0
    mf_temp.kernel()
    
    mc_temp = mcscf.CASCI(mf_temp, ncas, nelecas)
    mc_temp.verbose = 0
    mc_temp.kernel()
    
    results.append((temp, mf_temp.e_tot, mc_temp.e_tot))

# Now print the table
print(f"\n{'Temperature (eV)':<20} {'FOMO-SCF Energy':<20} {'FOMO-CASCI Energy':<20}")
print("-"*60)
for temp, e_scf, e_casci in results:
    print(f"{temp:<20.2f} {e_scf:<20.10f} {e_casci:<20.10f}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*70)
print("Summary")
print("="*70)
print(f"{'Method':<35} {'Energy (Ha)':<20}")
print("-"*55)
print(f"{'Standard RHF + CASCI':<35} {mc_std.e_tot:<20.10f}")
print(f"{'FOMO-CASCI (Gaussian, T=0.25eV)':<35} {mc_fomo_gauss.e_tot:<20.10f}")
print(f"{'FOMO-CASCI (Fermi-Dirac, T=0.25eV)':<35} {mc_fomo_fermi.e_tot:<20.10f}")
print("="*70)

print("\nNote: FOMO-CASCI is particularly useful for:")
print("  - Systems with near-degenerate orbitals")
print("  - Transition states and conical intersections")
print("  - Improving SCF convergence in difficult cases")
print("  - Providing better initial orbitals for multireference calculations")
print("\nNote: FOMO is currently only implemented for RHF references.")