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
- Configuration options (C extension vs Python, different k-meshes)
- Performance considerations and best practices
"""

import numpy

from pyscf.occri import OCCRI
from pyscf.pbc import gto, scf

print("=== OCCRI k-point Usage Examples ===")
print("This example shows how to use OCCRI for different k-point calculations.\n")

# Set up a simple diamond structure
cell = gto.Cell()
cell.atom = """
    C 0.000000 0.000000 0.000000
    C 0.890186 0.890186 0.890186
"""
cell.basis = "gth-szv"  # Compact basis for faster demonstration
cell.pseudo = "gth-pbe"  # Pseudopotentials for efficiency
cell.a = numpy.eye(3) * 3.5607  # Diamond lattice parameter
cell.mesh = [20] * 3  # FFT mesh
cell.verbose = 0
cell.build()

print(f"System: Diamond structure with {cell.natm} atoms")
print(f"FFT mesh: {cell.mesh} (total {numpy.prod(cell.mesh)} points)")

# =============================================================================
# Example 1: Basic k-point setup
# =============================================================================
print("\n" + "=" * 60)
print("Example 1: Basic k-point Hartree-Fock")
print("=" * 60)

# Define k-point mesh - start with small mesh for demonstration
kmesh = [2, 2, 2]
kpts = cell.make_kpts(kmesh)

print(f"k-point mesh: {kmesh} ({len(kpts)} k-points total)")
print("k-point coordinates:")
for i, kpt in enumerate(kpts):
    print(f"  k{i+1}: [{kpt[0]:8.4f}, {kpt[1]:8.4f}, {kpt[2]:8.4f}]")

# Set up KRHF calculation with OCCRI
mf = scf.KRHF(cell, kpts)
mf.with_df = OCCRI(mf)  # Specify kmesh

print("\nRunning KRHF calculation...")
energy = mf.kernel()
print(f"KRHF energy: {energy:.6f} Hartree")
print(f"Converged: {mf.converged}")

# =============================================================================
# Example 2: Different SCF methods
# =============================================================================
print("\n" + "=" * 60)
print("Example 2: Different SCF methods with OCCRI")
print("=" * 60)

# RHF - Restricted (closed shell)
print("\n2a. Restricted Hartree-Fock (RHF)")
mf_rhf = scf.KRHF(cell, kpts)
mf_rhf.with_df = OCCRI(mf_rhf)
e_rhf = mf_rhf.kernel()
print(f"    Energy: {e_rhf:.6f} Ha")

# UHF - Unrestricted (open shell capable)
print("\n2b. Unrestricted Hartree-Fock (UHF)")
mf_uhf = scf.KUHF(cell, kpts)
mf_uhf.with_df = OCCRI(mf_uhf)
e_uhf = mf_uhf.kernel()
print(f"    Energy: {e_uhf:.6f} Ha")

# DFT with hybrid functional
print("\n2c. DFT with PBE0 hybrid functional")
mf_dft = scf.KRKS(cell, kpts)
mf_dft.xc = "pbe0"  # 25% exact exchange + PBE correlation
mf_dft.with_df = OCCRI(mf_dft)
e_dft = mf_dft.kernel()
print(f"    Energy: {e_dft:.6f} Ha")

# =============================================================================
# Example 3: Configuration options
# =============================================================================
print("\n" + "=" * 60)
print("Example 3: OCCRI configuration options")
print("=" * 60)

# Force Python implementation (useful for debugging)
print("\n3a. Python implementation (disable_c=True)")
mf_python = scf.KRHF(cell, kpts)
mf_python.with_df = OCCRI(mf_python, disable_c=True)
e_python = mf_python.kernel()
print(f"    Energy (Python): {e_python:.6f} Ha")

# Use C extension if available (default)
print("\n3b. C extension (default, disable_c=False)")
mf_c = scf.KRHF(cell, kpts)
mf_c.with_df = OCCRI(mf_c, disable_c=False)
e_c = mf_c.kernel()
print(f"    Energy (C ext):  {e_c:.6f} Ha")


# =============================================================================
# Example 5: Gamma point vs k-point comparison
# =============================================================================
print("\n" + "=" * 60)
print("Example 5: Gamma point vs k-point comparison")
print("=" * 60)

# Gamma point only (equivalent to molecular calculation)
print("\n5a. Gamma point only")
mf_gamma = scf.RHF(cell)  # Note: RHF (not KRHF) for gamma point
mf_gamma.with_df = OCCRI(mf_gamma)  # No kmesh needed for gamma point
e_gamma = mf_gamma.kernel()
print(f"    Gamma point energy: {e_gamma:.6f} Ha")

# k-point sampling
print(f"\n5b. k-point sampling ({kmesh})")
print(f"    k-point energy:     {e_rhf:.6f} Ha")
print(f"    k-point correction: {e_rhf - e_gamma:.6f} Ha")

# =============================================================================
# Usage tips and best practices
# =============================================================================
print("\n" + "=" * 60)
print("OCCRI Usage Tips")
print("=" * 60)

print(
    """
Best practices for OCCRI k-point calculations:

1. FFT mesh convergence (most important):
   - Increase mesh size until energy changes < 1-5 μHa/atom
   - Example: [15]³ → [17]³ → [19]³ → [21]³
   - Dense meshes improve accuracy but increase cost significantly

2. k-point sampling:
   - Start with 2×2×2 or 3×3×3 k-point mesh
   - Increase until total energy converges
   - More k-points = higher accuracy but O(N_k²) cost scaling

3. Performance:
   - C extension provides ~5-10× speedup when available
   - Use disable_c=True for debugging or if C extension fails
   - Memory usage scales as O(N_k × N_occ × N_grid)
   - OCCRI scales as O(N_occ²) vs FFTDF O(N_AO²)
   - Preferable for large basis sets (>~100 AOs), slower for small basis (<~100 AOs)

4. Method selection:
   - KRHF: Fast, suitable for closed-shell systems
   - KUHF: Handles open-shell systems, slightly more expensive
   - KRKS/KUKS: Include electron correlation via DFT

5. Troubleshooting:
   - If SCF doesn't converge, try different initial guess
   - For large energy differences, check mesh size and k-point convergence
   - Use standard PySCF as reference for validation
"""
)

print("Example completed successfully!")
