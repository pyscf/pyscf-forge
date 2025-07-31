#!/usr/bin/env python

"""
OCCRI with k-points: Multiple k-point sampling

This example demonstrates OCCRI for k-point calculations, where Bloch functions
and complex phase factors require special handling. OCCRI supports arbitrary
k-point meshes with proper momentum conservation.

The k-point implementation uses complex FFTs and handles phase factors 
exp(-i(k-k')·r) for all k-point pair interactions.
"""

import numpy
from pyscf.pbc import gto, scf
from pyscf.occri import OCCRI

# Set up diamond structure with appropriate unit cell for k-point sampling
cell = gto.Cell()
cell.atom = '''
    C 0.000000 0.000000 0.000000
    C 0.890186 0.890186 0.890186
'''
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pbe'
cell.a = numpy.eye(3) * 3.5607
cell.mesh = [25] * 3  # Dense mesh required for k-point accuracy
cell.build()

# Generate k-point mesh: 2×2×2 = 8 k-points
kmesh = [2, 2, 2]
kpts = cell.make_kpts(kmesh)

print("=== OCCRI k-point Examples ===")
print(f"System: {' '.join(cell.atom_symbol(i) for i in range(cell.natm))} ({cell.natm} atoms)")
print(f"k-point mesh: {kmesh} ({len(kpts)} k-points)")
print(f"k-points (first 3):")
for i, kpt in enumerate(kpts[:3]):
    print(f"   k{i+1}: [{kpt[0]:6.3f}, {kpt[1]:6.3f}, {kpt[2]:6.3f}]")
if len(kpts) > 3:
    print(f"   ... and {len(kpts)-3} more")

# Example 1: k-point Restricted Hartree-Fock (KRHF)
print("\n1. k-point Restricted Hartree-Fock (KRHF)")
mf_krhf = scf.KRHF(cell, kpts)
mf_krhf.with_df = OCCRI(mf_krhf, kmesh=kmesh)
e_krhf = mf_krhf.kernel()
print(f"   KRHF energy: {e_krhf:.8f} Hartree")

# Example 2: k-point Unrestricted Hartree-Fock (KUHF)
print("\n2. k-point Unrestricted Hartree-Fock (KUHF)")
mf_kuhf = scf.KUHF(cell, kpts)
mf_kuhf.with_df = OCCRI(mf_kuhf, kmesh=kmesh) 
e_kuhf = mf_kuhf.kernel()
print(f"   KUHF energy: {e_kuhf:.8f} Hartree")
print(f"   KRHF-KUHF difference: {abs(e_krhf - e_kuhf):.2e} Hartree")

# Example 3: k-point Restricted Kohn-Sham with PBE0
print("\n3. k-point Restricted Kohn-Sham with PBE0")
mf_krks = scf.KRKS(cell, kpts)
mf_krks.xc = 'pbe0'
mf_krks.with_df = OCCRI(mf_krks, kmesh=kmesh)
e_krks = mf_krks.kernel()
print(f"   KRKS/PBE0 energy: {e_krks:.8f} Hartree")

# Example 4: k-point Unrestricted Kohn-Sham with PBE0
print("\n4. k-point Unrestricted Kohn-Sham with PBE0")
mf_kuks = scf.KUKS(cell, kpts)
mf_kuks.xc = 'pbe0'
mf_kuks.with_df = OCCRI(mf_kuks, kmesh=kmesh)
e_kuks = mf_kuks.kernel()
print(f"   KUKS/PBE0 energy: {e_kuks:.8f} Hartree")

# Compare with Gamma point calculation to show k-point convergence
print("\n=== k-point vs Gamma Point Comparison ===")
mf_gamma = scf.RKS(cell)
mf_gamma.xc = 'pbe0'
mf_gamma.with_df = OCCRI(mf_gamma)
e_gamma = mf_gamma.kernel()

print(f"Gamma point energy:  {e_gamma:.8f} Hartree")
print(f"k-point energy:      {e_krks:.8f} Hartree")
print(f"k-point correction:  {e_krks - e_gamma:.6f} Hartree")

print("\n=== k-point Technical Notes ===")
print("• k-point OCCRI handles complex Bloch functions: ψ(k,r) = e^(ik·r) u(k,r)")
print("• Complex FFTs evaluate exchange integrals with phase factors exp(-i(k-k')·r)")
print("• All k-point pairs (k,k') contribute to exchange matrix at each k-point")
print("• Computational cost scales as O(N_k^2) where N_k is number of k-points")
print("• Dense FFT meshes recommended for production accuracy (ensure that the")
print("  error from the finite plane-wave cutoff is less than 5 μHa per atom)")