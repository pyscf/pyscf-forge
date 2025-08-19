#!/usr/bin/env python

"""
OCCRI with k-points: Usage Examples

This example demonstrates how to use OCCRI for k-point calculations.
OCCRI provides efficient exact exchange evaluation for periodic systems
with k-point sampling, making it ideal for band structure calculations
and solid-state systems.

Key features demonstrated:
- Setting up k-point calculations with OCCRI
- Different SCF methods (RHF, UHF, RKS, UKS)
- Configuration options
- Performance considerations and best practices
"""

import numpy
from pyscf.occri import OCCRI
from pyscf.pbc import gto, scf

print('=== OCCRI k-point Usage Examples ===')
print('This example shows how to use OCCRI for different k-point calculations.\n')

# Set up a simple diamond structure
cell = gto.Cell()
cell.atom = """
    C 0.000000 0.000000 0.000000
    C 0.890186 0.890186 0.890186
"""
cell.basis = 'gth-szv'  # Compact basis for faster demonstration
cell.pseudo = 'gth-pbe'  # Pseudopotentials for efficiency
cell.a = numpy.eye(3) * 3.5607  # Diamond lattice parameter
cell.mesh = [20] * 3  # FFT mesh
cell.verbose = 0
cell.build()

print(f'System: Diamond structure with {cell.natm} atoms')
print(f'FFT mesh: {cell.mesh} (total {numpy.prod(cell.mesh)} points)')

# =============================================================================
# Example 1: Basic k-point setup
# =============================================================================
print('\n' + '=' * 60)
print('Example 1: Basic k-point Hartree-Fock')
print('=' * 60)

# Define k-point mesh - start with small mesh for demonstration
kmesh = [2, 2, 2]
kpts = cell.make_kpts(kmesh)

print(f'k-point mesh: {kmesh} ({len(kpts)} k-points total)')
print('k-point coordinates:')
for i, kpt in enumerate(kpts):
    print(f'  k{i + 1}: [{kpt[0]:8.4f}, {kpt[1]:8.4f}, {kpt[2]:8.4f}]')

# Set up KRHF calculation with OCCRI
mf = scf.KRHF(cell, kpts)
mf.with_df = OCCRI.from_mf(mf)

print('\nRunning KRHF calculation...')
energy = mf.kernel()
print(f'KRHF energy: {energy:.6f} Hartree')


mf = scf.KRHF(cell, kpts)
mf.with_df = OCCRI.from_mf(mf, disable_c=True)

print('\nRunning KRHF calculation...')
energy = mf.kernel()
print(f'KRHF energy: {energy:.6f} Hartree')


# =============================================================================
# Example 2: Different SCF methods
# =============================================================================
print('\n' + '=' * 60)
print('Example 2: Different SCF methods with OCCRI')
print('=' * 60)

# RHF - Restricted (closed shell)
print('\n2a. Restricted Hartree-Fock (RHF)')
mf_rhf = scf.KRHF(cell, kpts)
mf_rhf.with_df = OCCRI.from_mf(mf_rhf)
e_rhf = mf_rhf.kernel()
print(f'    Energy: {e_rhf:.6f} Ha')

# UHF - Unrestricted (open shell capable)
print('\n2b. Unrestricted Hartree-Fock (UHF)')
mf_uhf = scf.KUHF(cell, kpts)
mf_uhf.with_df = OCCRI.from_mf(mf_uhf)
e_uhf = mf_uhf.kernel()
print(f'    Energy: {e_uhf:.6f} Ha')

# DFT with hybrid functional
print('\n2c. DFT with PBE0 hybrid functional')
mf_dft = scf.KRKS(cell, kpts)
mf_dft.xc = 'pbe0'  # 25% exact exchange + PBE correlation
mf_dft.with_df = OCCRI.from_mf(mf_dft)
e_dft = mf_dft.kernel()
print(f'    Energy: {e_dft:.6f} Ha')

# =============================================================================
# Example 3: Configuration options
# =============================================================================
print('\n' + '=' * 60)
print('Example 3: OCCRI configuration options')
print('=' * 60)

import time

# Force Python implementation
print('\n3a. Python implementation (disable_c=True)')
mf_python = scf.KRHF(cell, kpts)
mf_python.with_df = OCCRI.from_mf(mf_python, disable_c=True)
t0 = time.time()
e_python = mf_python.kernel()
print(f'    Energy (Python): {e_python:.6f} Ha')
print(f'    Time (Python): {time.time() - t0}')

# Use C extension if available (default)
print('\n3b. C extension (default, disable_c=False)')
mf_c = scf.KRHF(cell, kpts)
mf_c.with_df = OCCRI.from_mf(mf_c, disable_c=False)
t0 = time.time()
e_c = mf_c.kernel()
print(f'    Energy (C ext):  {e_c:.6f} Ha')
print(f'    Time (C ext): {time.time() - t0}')


# =============================================================================
# Example 5: Gamma point vs k-point comparison
# =============================================================================
print('\n' + '=' * 60)
print('Example 5: Gamma point vs k-point comparison')
print('=' * 60)

# Gamma point only (equivalent to molecular calculation)
print('\n5a. Gamma point only')
mf_gamma = scf.RHF(cell)  # Note: RHF (not KRHF) for gamma point
mf_gamma.with_df = OCCRI.from_mf(mf_gamma)
e_gamma = mf_gamma.kernel()
print(f'    Gamma point energy: {e_gamma:.6f} Ha')

# k-point sampling
print(f'\n5b. k-point sampling ({kmesh})')
print(f'    k-point energy:     {e_rhf:.6f} Ha')
print(f'    k-point correction: {e_rhf - e_gamma:.6f} Ha')
