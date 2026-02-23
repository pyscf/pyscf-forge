#!/usr/bin/env python
"""
Example 81: DFT-Corrected CASCI with DFT Core Embedding
========================================================

This example demonstrates CASCI calculations with DFT-corrected core energy
for both RHF and UHF references.

Author:
    Arshad Mehmood, IACS, Stony Brook University
    arshad.mehmood@stonybrook.edu
    30 December 2025

Reference:
    S. Pijeau and E. G. Hohenstein,
    J. Chem. Theory Comput. 2017, 13, 1130-1146
    https://doi.org/10.1021/acs.jctc.6b00893
"""

from pyscf import gto, scf, mcscf
from pyscf.mcscf import dft_corrected_casci

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

# Define active space: 6 electrons in 6 orbitals
ncas = 6
nelecas = 6

print("="*70)
print("DFT-Corrected CASCI: RHF vs UHF Reference")
print("="*70)

# =============================================================================
# Part 1: RHF Reference
# =============================================================================
print("\n" + "="*70)
print("Part 1: RHF Reference")
print("="*70)

mf_rhf = scf.RHF(mol)
mf_rhf.verbose = 0
mf_rhf.run()

# Standard CASCI (HF core)
print("\n1a. Standard CASCI (RHF, HF core energy):")
mc_rhf_hf = mcscf.CASCI(mf_rhf, ncas, nelecas)
mc_rhf_hf.verbose = 0
mc_rhf_hf.kernel()
print(f"    Total Energy: {mc_rhf_hf.e_tot:.10f} Ha")

# DFT-corrected CASCI with different functionals
print("\n1b. DFT-corrected CASCI (RHF reference) with different XC functionals:")
rhf_results = []
for xc in ['LDA', 'PBE', 'B3LYP']:
    mc = dft_corrected_casci.CASCI(mf_rhf, ncas, nelecas, xc=xc)
    mc.verbose = 0
    mc.kernel()
    delta_e = (mc.e_tot - mc_rhf_hf.e_tot) * 1000
    rhf_results.append((xc, mc.e_tot, delta_e))

for xc, energy, delta in rhf_results:
    print(f"    {xc:8s}: {energy:.10f} Ha  (ΔE = {delta:+.4f} mHa)")

# =============================================================================
# Part 2: UHF Reference
# =============================================================================
print("\n" + "="*70)
print("Part 2: UHF Reference")
print("="*70)

mf_uhf = scf.UHF(mol)
mf_uhf.verbose = 0
mf_uhf.run()

# Standard UCASCI (HF core)
print("\n2a. Standard UCASCI (UHF, HF core energy):")
mc_uhf_hf = mcscf.UCASCI(mf_uhf, ncas, nelecas)
mc_uhf_hf.verbose = 0
mc_uhf_hf.kernel()
print(f"    Total Energy: {mc_uhf_hf.e_tot:.10f} Ha")

# DFT-corrected UCASCI with different functionals
print("\n2b. DFT-corrected UCASCI (UHF reference) with different XC functionals:")
uhf_results = []
for xc in ['LDA', 'PBE', 'B3LYP']:
    mc = dft_corrected_casci.UCASCI(mf_uhf, ncas, nelecas, xc=xc)
    mc.verbose = 0
    mc.kernel()
    delta_e = (mc.e_tot - mc_uhf_hf.e_tot) * 1000
    uhf_results.append((xc, mc.e_tot, delta_e))

for xc, energy, delta in uhf_results:
    print(f"    {xc:8s}: {energy:.10f} Ha  (ΔE = {delta:+.4f} mHa)")

# =============================================================================
# Part 3: Using Factory Function
# =============================================================================
print("\n" + "="*70)
print("Part 3: Using DFCASCI Factory Function")
print("="*70)

print("\n3a. DFCASCI auto-detects RHF:")
mc_auto_rhf = dft_corrected_casci.DFCASCI(mf_rhf, ncas, nelecas, xc='PBE')
mc_auto_rhf.verbose = 0
mc_auto_rhf.kernel()
print(f"    Type: {type(mc_auto_rhf).__name__}")
print(f"    Energy: {mc_auto_rhf.e_tot:.10f} Ha")

print("\n3b. DFCASCI auto-detects UHF:")
mc_auto_uhf = dft_corrected_casci.DFCASCI(mf_uhf, ncas, nelecas, xc='PBE')
mc_auto_uhf.verbose = 0
mc_auto_uhf.kernel()
print(f"    Type: {type(mc_auto_uhf).__name__}")
print(f"    Energy: {mc_auto_uhf.e_tot:.10f} Ha")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*70)
print("Summary")
print("="*70)
print(f"{'Method':<30} {'Energy (Ha)':<20}")
print("-"*50)
print(f"{'RHF-CASCI (HF core)':<30} {mc_rhf_hf.e_tot:<20.10f}")
print(f"{'RHF-CASCI (PBE core)':<30} {mc_auto_rhf.e_tot:<20.10f}")
print(f"{'UHF-CASCI (HF core)':<30} {mc_uhf_hf.e_tot:<20.10f}")
print(f"{'UHF-CASCI (PBE core)':<30} {mc_auto_uhf.e_tot:<20.10f}")
print("="*70)

print("\nNote: CI coefficients are identical across all methods for the same")
print("reference because the active space embedding uses HF-like potential.")
print("Only the core energy differs (DFT vs HF treatment).")