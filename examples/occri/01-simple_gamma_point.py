#!/usr/bin/env python

"""
Simple OCCRI example: Gamma point calculations

This example demonstrates the basic usage of OCCRI (Occupied Orbital Coulomb 
Resolution of Identity) for efficient exact exchange evaluation in periodic 
systems at the Gamma point.

OCCRI provides significant speedup over standard FFTDF while maintaining
chemical accuracy for hybrid DFT and Hartree-Fock calculations.
"""

import numpy
from pyscf.pbc import gto, scf
from pyscf.occri import OCCRI

# Set up diamond structure (2 carbon atoms per unit cell)
cell = gto.Cell()
cell.atom = '''
    C 0.000000 0.000000 0.000000
    C 0.890186 0.890186 0.890186
'''
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pbe'
cell.a = numpy.eye(3) * 3.5607  # 3.56 Å lattice parameter
cell.mesh = [25] * 3  # Dense mesh for high accuracy
cell.build()

print("=== OCCRI Gamma Point Examples ===")
print(f"System: {' '.join(cell.atom_symbol(i) for i in range(cell.natm))} ({cell.natm} atoms)")
print(f"Basis: {cell.basis}")
print(f"Lattice parameter: {cell.a[0,0]:.3f} Å")

# Example 1: Restricted Hartree-Fock (RHF)
print("\n1. Restricted Hartree-Fock (RHF)")
mf_rhf = scf.RHF(cell)
mf_rhf.with_df = OCCRI(mf_rhf)
e_rhf = mf_rhf.kernel()
print(f"   RHF energy: {e_rhf:.8f} Hartree")

# Example 2: Unrestricted Hartree-Fock (UHF) 
print("\n2. Unrestricted Hartree-Fock (UHF)")
mf_uhf = scf.UHF(cell)
mf_uhf.with_df = OCCRI(mf_uhf)
e_uhf = mf_uhf.kernel()
print(f"   UHF energy: {e_uhf:.8f} Hartree")
print(f"   RHF-UHF difference: {abs(e_rhf - e_uhf):.2e} Hartree (should be small for closed shell)")

# Example 3: Restricted Kohn-Sham with PBE0 hybrid functional
print("\n3. Restricted Kohn-Sham with PBE0 (25% exact exchange)")
mf_rks = scf.RKS(cell)
mf_rks.xc = 'pbe0'
mf_rks.with_df = OCCRI(mf_rks)
e_rks = mf_rks.kernel()
print(f"   RKS/PBE0 energy: {e_rks:.8f} Hartree")

# Example 4: Unrestricted Kohn-Sham with PBE0 hybrid functional
print("\n4. Unrestricted Kohn-Sham with PBE0")
mf_uks = scf.UKS(cell)
mf_uks.xc = 'pbe0'
mf_uks.with_df = OCCRI(mf_uks)
e_uks = mf_uks.kernel()
print(f"   UKS/PBE0 energy: {e_uks:.8f} Hartree")
print(f"   RKS-UKS difference: {abs(e_rks - e_uks):.2e} Hartree")

print("\n=== Performance Note ===")
print("OCCRI provides significant speedup over FFTDF, especially for systems")
print("with many occupied orbitals. The C extension with FFTW provides optimal performance.")

print("\n=== Accuracy Note ===")
print("OCCRI accuracy depends on FFT mesh density. Dense meshes (~1e-5 grid accuracy or higher)")
print("are recommended for high-accuracy calculations.")