#!/usr/bin/env python

"""
OCCRI Gamma Point Examples: Getting Started

This example demonstrates the basic usage of OCCRI (Occupied Orbital Coulomb
Resolution of Identity) for efficient exact exchange evaluation in periodic
systems at the Gamma point (single k-point).

OCCRI provides significant speedup over standard FFTDF while maintaining
chemical accuracy for hybrid DFT and Hartree-Fock calculations.

Key concepts covered:
- Basic OCCRI setup and usage
- Different SCF methods (RHF, UHF, RKS, UKS)
- How to set up periodic systems
- Performance and accuracy considerations
"""

import numpy
from pyscf.occri import OCCRI
from pyscf.pbc import gto, scf

print('=== OCCRI Gamma Point Tutorial ===')
print('This example shows basic OCCRI usage for single k-point calculations.\n')

# =============================================================================
# System Setup
# =============================================================================
print('Setting up diamond structure...')

# Set up diamond structure (2 carbon atoms per unit cell)
cell = gto.Cell()
cell.atom = """
    C 0.000000 0.000000 0.000000
    C 0.890186 0.890186 0.890186
"""
cell.basis = 'gth-szv'  # Compact basis set
cell.pseudo = 'gth-pbe'  # Pseudopotentials
cell.a = numpy.eye(3) * 3.5607  # Diamond lattice parameter (Å)
cell.mesh = [20] * 3  # FFT mesh
cell.verbose = 0
cell.build()

print(f'System: {" ".join(cell.atom_symbol(i) for i in range(cell.natm))} ({cell.natm} atoms)')
print(f'Basis: {cell.basis}')
print(f'Lattice parameter: {cell.a[0, 0]:.3f} Å')
print(f'FFT mesh: {cell.mesh} ({numpy.prod(cell.mesh)} total points)')

# =============================================================================
# Example 1: Basic OCCRI usage
# =============================================================================
print('\n' + '=' * 50)
print('Example 1: Basic OCCRI Setup')
print('=' * 50)

print('\n1a. Restricted Hartree-Fock (RHF)')

# Standard syntax: attach OCCRI to mean-field object
mf_rhf = scf.RHF(cell)
mf_rhf.with_df = OCCRI.from_mf(mf_rhf)  # This line enables OCCRI
e_rhf = mf_rhf.kernel()

print(f'    Energy: {e_rhf:.6f} Hartree')

print('\n1b. Unrestricted Hartree-Fock (UHF)')
mf_uhf = scf.UHF(cell)
mf_uhf.with_df = OCCRI.from_mf(mf_uhf)
e_uhf = mf_uhf.kernel()
print(f'    Energy: {e_uhf:.6f} Hartree')

# =============================================================================
# Example 2: Hybrid DFT calculations
# =============================================================================
print('\n' + '=' * 50)
print('Example 2: Hybrid DFT with OCCRI')
print('=' * 50)

print('\n2a. PBE0 (25% exact exchange)')
mf_pbe0 = scf.RKS(cell)
mf_pbe0.xc = 'pbe0'
mf_pbe0.with_df = OCCRI.from_mf(mf_pbe0)  # OCCRI handles exact exchange
e_pbe0 = mf_pbe0.kernel()
print(f'    PBE0 energy: {e_pbe0:.6f} Hartree')

print('\n2b. HSE06 range-separated hybrid')
mf_hse = scf.RKS(cell)
mf_hse.xc = 'hse06'  # 25% short-range exact exchange
mf_hse.with_df = OCCRI.from_mf(mf_hse)
e_hse = mf_hse.kernel()
print(f'    HSE06 energy: {e_hse:.6f} Hartree')

# =============================================================================
# Example 3: Configuration options
# =============================================================================
print('\n' + '=' * 50)
print('Example 3: OCCRI Configuration')
print('=' * 50)

print('\n3a. Python implementation')
mf_python = scf.RHF(cell)
mf_python.with_df = OCCRI.from_mf(mf_python, disable_c=True)
e_python = mf_python.kernel()
print(f'    Python: {e_python:.6f} Ha')

print('\n3b. C extension')
mf_c = scf.RHF(cell)
mf_c.with_df = OCCRI.from_mf(mf_c, disable_c=False)
e_c = mf_c.kernel()
print(f'    C extension: {e_c:.6f} Ha')
print(f'    Difference: {abs(e_python - e_c):.2e} Ha')

# =============================================================================
# Example 4: FFT mesh convergence study
# =============================================================================
print('\n' + '=' * 60)
print('Example 4: FFT mesh convergence study')
print('=' * 60)

print('OCCRI accuracy depends on FFT mesh density. This example shows')
print('how to converge the mesh size for reliable results.\n')

# Test different mesh sizes - keep k-points fixed
mesh_sizes = [25, 27, 29, 31]  # FFT mesh dimensions
energies = []

for mesh_size in mesh_sizes:
    print(f'4.{mesh_size}: [{mesh_size}]³ mesh ({mesh_size**3} total points)')

    # Create new cell with different mesh
    test_cell = cell.copy()
    test_cell.mesh = [mesh_size] * 3
    test_cell.build()

    mf_test = scf.RHF(test_cell)
    mf_test.with_df = OCCRI.from_mf(mf_test)
    mf_test.verbose = 1  # Reduce output for cleaner display

    e_test = mf_test.kernel()
    energies.append(e_test)
    print(f'    Energy: {e_test:.8f} Ha')

    if len(energies) > 1:
        diff = e_test - energies[-2]
        print(f'    Change: {diff:.8f} Ha ({abs(diff) * 1000:.2f} mHa)')

        # Check convergence
        if abs(diff) < 1e-6:
            print('    ✓ Converged to μHa accuracy')
        elif abs(diff) < 5e-6:
            print('    ✓ Converged to 5 μHa accuracy')
        else:
            print('    ⚠ Not yet converged')
    print()

print('Mesh convergence guidelines:')
print('• Energy differences < 1-5 μHa/atom typically sufficient')
print('• Denser meshes → higher accuracy but slower calculation')

# =============================================================================
# Usage guide
# =============================================================================
print('\n' + '=' * 50)
print('OCCRI Usage Guide')
print('=' * 50)

print(
    """
Quick start:
  mf = scf.RHF(cell)           # Create SCF object
  mf.with_df = OCCRI.from_mf(mf)       # Enable OCCRI
  energy = mf.kernel()         # Run calculation

When to use OCCRI:
  • Hartree-Fock calculations (exact exchange)
  • Hybrid functionals (PBE0, HSE06, etc.)
  • When standard FFTDF is too slow
  • Large basis sets

Configuration options:
  OCCRI.from_mf(mf)                    # Default (use C if available)
  OCCRI.from_mf(mf, disable_c=True)    # Force Python implementation
  OCCRI.from_mf(mf, disable_c=False)   # Force C implementation

Compatible methods:
  • scf.RHF, scf.UHF          # Hartree-Fock
  • scf.RKS, scf.UKS          # DFT (any functional)
  • Gamma point calculations (use 02-kpoint for k-points)

Performance tips:
  • Converge FFT mesh: start low, increase until energy changes < 1-5 μHa/atom
  • OCCRI scaling: O(N_occ²) vs FFTDF O(N_AO²)
  • Most beneficial when N_AO >> N_occ (large basis, few electrons)
"""
)

print('Example completed successfully!')
