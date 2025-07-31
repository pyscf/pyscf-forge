#!/usr/bin/env python

"""
OCCRI Performance Comparison

This example compares the performance and accuracy of OCCRI against standard
FFTDF for exact exchange evaluation. OCCRI provides significant speedup 
while maintaining chemical accuracy.

The performance advantage is most pronounced for systems with many occupied
orbitals and when using the optimized C extension with FFTW and OpenMP.
"""

import time
import numpy
from pyscf.pbc import gto, scf, df
from pyscf.occri import OCCRI

# Set up a moderately sized system for performance comparison
cell = gto.Cell()
cell.atom = '''
    C 0.000000 0.000000 1.780373
    C 0.890186 0.890186 2.670559
    C 0.000000 1.780373 0.000000
    C 0.890186 2.670559 0.890186
'''
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pbe'
cell.a = numpy.array([
    [3.560745, 0.000000, 0.000000],
    [0.000000, 3.560745, 0.000000], 
    [0.000000, 0.000000, 3.560745],
])
cell.mesh = [20] * 3  # Moderate mesh for timing comparison
cell.build()

print("=== OCCRI Performance Comparison ===")
print(f"System: {' '.join(cell.atom_symbol(i) for i in range(cell.natm))} ({cell.natm} atoms, {cell.nao} AOs)")
print(f"Basis: {cell.basis}")
print(f"Mesh: {cell.mesh}")

# Example 1: Compare FFTDF vs OCCRI for RHF
print("\n1. Restricted Hartree-Fock: FFTDF vs OCCRI")

# FFTDF calculation
print("   Running FFTDF reference...")
start_time = time.time()
mf_fftdf = scf.RHF(cell)
# Default df is FFTDF
e_fftdf = mf_fftdf.kernel()
fftdf_time = time.time() - start_time

# OCCRI calculation  
print("   Running OCCRI...")
start_time = time.time()
mf_occri = scf.RHF(cell)
mf_occri.with_df = OCCRI(mf_occri)
e_occri = mf_occri.kernel()
occri_time = time.time() - start_time

# Results
energy_diff = abs(e_fftdf - e_occri)
speedup = fftdf_time / occri_time

print(f"   FFTDF energy:     {e_fftdf:.8f} Hartree ({fftdf_time:.2f}s)")
print(f"   OCCRI energy:     {e_occri:.8f} Hartree ({occri_time:.2f}s)")
print(f"   Energy difference: {energy_diff:.2e} Hartree")
print(f"   OCCRI speedup:    {speedup:.2f}x")

# Example 2: Hybrid DFT performance comparison
print("\n2. Hybrid DFT (PBE0): FFTDF vs OCCRI")

print("   Running FFTDF/PBE0...")
start_time = time.time()
mf_fftdf_pbe0 = scf.RKS(cell)
mf_fftdf_pbe0.xc = 'pbe0'
e_fftdf_pbe0 = mf_fftdf_pbe0.kernel()
fftdf_pbe0_time = time.time() - start_time

print("   Running OCCRI/PBE0...")
start_time = time.time()
mf_occri_pbe0 = scf.RKS(cell)
mf_occri_pbe0.xc = 'pbe0'
mf_occri_pbe0.with_df = OCCRI(mf_occri_pbe0)
e_occri_pbe0 = mf_occri_pbe0.kernel()
occri_pbe0_time = time.time() - start_time

energy_diff_pbe0 = abs(e_fftdf_pbe0 - e_occri_pbe0)
speedup_pbe0 = fftdf_pbe0_time / occri_pbe0_time

print(f"   FFTDF/PBE0 energy: {e_fftdf_pbe0:.8f} Hartree ({fftdf_pbe0_time:.2f}s)")
print(f"   OCCRI/PBE0 energy: {e_occri_pbe0:.8f} Hartree ({occri_pbe0_time:.2f}s)")
print(f"   Energy difference: {energy_diff_pbe0:.2e} Hartree")
print(f"   OCCRI speedup:    {speedup_pbe0:.2f}x")

# Example 3: Memory usage comparison
print("\n3. Memory Usage Analysis")
from pyscf import lib

# Estimate memory usage for both methods
mem_fftdf = lib.current_memory()[0]
print(f"   Baseline memory usage: {mem_fftdf:.1f} MB")

# Test get_jk memory usage
dm = mf_fftdf.make_rdm1()

# FFTDF get_jk
mem_before = lib.current_memory()[0]
_, vk_fftdf = mf_fftdf.get_jk(dm, with_k=True)
mem_after_fftdf = lib.current_memory()[0]
fftdf_mem = mem_after_fftdf - mem_before

# OCCRI get_jk  
mem_before = lib.current_memory()[0]
_, vk_occri = mf_occri.get_jk(dm, with_k=True)
mem_after_occri = lib.current_memory()[0]
occri_mem = mem_after_occri - mem_before

print(f"   FFTDF get_jk memory: {fftdf_mem:.1f} MB")
print(f"   OCCRI get_jk memory: {occri_mem:.1f} MB")

# Verify accuracy
vk_diff = numpy.abs(vk_fftdf - vk_occri).max()
print(f"   K matrix difference: {vk_diff:.2e}")

print("\n=== Performance Summary ===")
print(f"• OCCRI provides {speedup:.1f}x speedup for RHF")
print(f"• OCCRI provides {speedup_pbe0:.1f}x speedup for hybrid DFT")
print(f"• Energy accuracy: ~{max(energy_diff, energy_diff_pbe0):.0e} Hartree")
print(f"• Exchange matrix accuracy: ~{vk_diff:.0e}")

print("\n=== Optimization Notes ===")
try:
    from pyscf.occri import _OCCRI_C_AVAILABLE
    if _OCCRI_C_AVAILABLE:
        print("✓ Using optimized C extension with FFTW and OpenMP")
        print("  - Compiled C code provides ~5-10x base speedup")
        print("  - FFTW optimized FFTs for best performance")
        print("  - OpenMP parallelization scales with CPU cores")
    else:
        print("⚠ Using Python fallback implementation")
        print("  - Install FFTW, BLAS, and OpenMP for optimal performance")
        print("  - C extension provides significant additional speedup")
except ImportError:
    print("⚠ OCCRI module information not available")

print("\n=== Scaling Notes ===")
print("• OCCRI advantage increases with system size (more occupied orbitals)")
print("• k-point calculations show even greater speedup benefits")
print("• Dense FFT meshes recommended for production accuracy (ensure that the")
print("  error from the finite plane-wave cutoff is less than 5 μHa per atom)")