#!/usr/bin/env python

"""
OCCRI Performance Demonstration

This example demonstrates OCCRI's performance characteristics and shows
how to benchmark it against standard FFTDF. OCCRI provides significant
speedup while maintaining chemical accuracy.

Key topics covered:
- Timing OCCRI vs FFTDF calculations
- Performance scaling considerations
- When OCCRI provides the most benefit
- How to optimize OCCRI performance
"""

import time

import numpy

from pyscf.occri import OCCRI
from pyscf.pbc import df, gto, scf
from pyscf.pbc.tools.pbc import super_cell

# Set up a moderately sized system for performance comparison
cell = gto.Cell()
cell.atom = """
    C 0.000000 0.000000 1.780373
    C 0.890186 0.890186 2.670559
    C 0.000000 1.780373 0.000000
    C 0.890186 2.670559 0.890186
"""
cell.basis = "gth-cc-tzvp"
cell.pseudo = "gth-pbe"
cell.a = numpy.array(
    [
        [3.560745, 0.000000, 0.000000],
        [0.000000, 3.560745, 0.000000],
        [0.000000, 0.000000, 3.560745],
    ]
)
cell.mesh = [25] * 3
cell.verbose = 0
cell.build()
kmesh = [1,1,1]
kpts = cell.make_kpts(kmesh)
cell = super_cell(cell, [1,1,2])

print("=== OCCRI Performance Comparison ===")
print(
    f"System: {' '.join(cell.atom_symbol(i) for i in range(cell.natm))} ({cell.natm} atoms, {cell.nao} AOs)"
)
print(f"Basis: {cell.basis}")
print(f"Mesh: {cell.mesh}")

# Example 1: Compare K matrix construction: FFTDF vs OCCRI
print("\n1. K matrix construction timing comparison")

# Set up common density matrix for fair comparison
print("   Setting up test density matrix...")
mf_ref = scf.KRHF(cell, kpts=kpts)
mf_ref.max_cycle = 1  # Store MO Coeff for comparison
mf_ref.kernel()
dm = mf_ref.make_rdm1(kpts=kpts)

# Time FFTDF K matrix construction only
print("   Timing FFTDF K matrix construction...")
mf_ref = scf.KRHF(cell, kpts=kpts)
mf_ref._is_mem_enough = lambda: False  # Turn off 'incore' for small demo
start_time = time.time()
_, vk_fftdf = mf_ref.get_jk(dm_kpts=dm, with_j=False, with_k=True, kpts=kpts)
fftdf_k_time = time.time() - start_time

# # Time OCCRI K matrix construction only
# print("   Timing OCCRI K matrix construction...")
# mf_occri = scf.KRHF(cell, kpts=kpts)
# mf_occri.with_df = OCCRI(mf_occri, disable_c=True, kmesh=kmesh)
# mf_occri.with_df.scf_iter = 1  # Don't rebuild MOs for timing

# start_time = time.time()
# _, vk_occri = mf_occri.get_jk(dm=dm, with_j=False, with_k=True, kpts=kpts)
# occri_k_time = time.time() - start_time

# Results
k_energy_fftdf = numpy.einsum("kij,kji", vk_fftdf, dm) * 0.5
# k_energy_occri = numpy.einsum("kij,kji", vk_occri, dm) * 0.5
# energy_diff = abs(k_energy_fftdf - k_energy_occri)
# k_speedup = fftdf_k_time / occri_k_time

print(f"   FFTDF K matrix:   {k_energy_fftdf:.8f} Ha ({fftdf_k_time:.3f}s)")
# print(f"   OCCRI K matrix:   {k_energy_occri:.8f} Ha ({occri_k_time:.3f}s)")
# print(f"   Energy difference: {energy_diff:.2e} Hartree")
# print(f"   K matrix speedup: {k_speedup:.2f}x")

# # Example 2: K matrix timing for multiple calls (realistic usage)
# print("\n2. Multiple K matrix evaluations (typical in SCF)")

# print("   Testing with 7 K matrix evaluations...")
# n_calls = 7

# # Time multiple FFTDF K matrix calls
# print("   Timing FFTDF...")
# start_time = time.time()
# for i in range(n_calls):
#     _, vk_fftdf = mf_ref.get_jk(dm_kpts=dm, with_j=False, with_k=True, kpts=kpts)
# fftdf_multi_time = time.time() - start_time

# # Time multiple OCCRI K matrix calls
# print("   Timing OCCRI...")
# start_time = time.time()
# for i in range(n_calls):
#     _, vk_occri = mf_occri.get_jk(dm=dm, with_j=False, with_k=True, kpts=kpts)
# occri_multi_time = time.time() - start_time

# multi_speedup = fftdf_multi_time / occri_multi_time

# print(
#     f"   FFTDF: {n_calls} calls in {fftdf_multi_time:.3f}s ({fftdf_multi_time/n_calls:.3f}s per call)"
# )
# print(
#     f"   OCCRI: {n_calls} calls in {occri_multi_time:.3f}s ({occri_multi_time/n_calls:.3f}s per call)"
# )
# print(f"   Average K speedup: {multi_speedup:.2f}x")


# print("\n=== Performance Summary ===")
# print(f"• K matrix construction speedup: {k_speedup:.1f}x (single call)")
# print(f"• K matrix construction speedup: {multi_speedup:.1f}x (multiple calls)")
# print(f"• Exchange energy accuracy: ~{energy_diff:.0e} Hartree")

# print("\n=== Optimization Notes ===")
# try:
#     from pyscf.occri import _OCCRI_C_AVAILABLE

#     if _OCCRI_C_AVAILABLE:
#         print("✓ Using optimized C extension with FFTW and OpenMP")
#         print("  - Compiled C code provides ~5-10x base speedup")
#         print("  - FFTW optimized FFTs for best performance")
#         print("  - OpenMP parallelization scales with CPU cores")
#     else:
#         print("⚠ Using Python fallback implementation")
#         print("  - Install FFTW, BLAS, and OpenMP for optimal performance")
#         print("  - C extension provides significant additional speedup")
# except ImportError:
#     print("⚠ OCCRI module information not available")

# print("\n=== Benchmarking Tips ===")
# print("To properly benchmark OCCRI:")
# print("• Run multiple trials and average timings for statistical significance")
# print("• Use representative system sizes (OCCRI benefits scale with system size)")
# print("• Test both C extension and Python implementations")
# print("• Consider memory usage in addition to timing")
# print("• Verify energy accuracy remains within acceptable thresholds")

# print("\n=== When to Use OCCRI ===")
# print("OCCRI provides most benefit for:")
# print("• Large basis sets: cc-pVTZ, aug-cc-pVDZ, gth-cc-tzvp")
# print("• Systems where N_AO >> N_occ (wide band gap insulators)")
# print("• Hybrid DFT calculations requiring exact exchange")
# print("• k-point calculations (see 02-kpoint_calculations.py)")
# print("• Production calculations where FFTDF becomes a bottleneck")
# print("")
# print("OCCRI may be slower for:")
# print("• Small basis sets: STO-3G, 6-31G, gth-szv")
# print("• Metallic systems where N_occ ≈ N_AO/2")
# print("• Quick test calculations with minimal basis sets")

# print("\n=== Critical Performance Scaling Insight ===")
# print("K matrix construction complexity (the bottleneck OCCRI optimizes):")
# print("• FFTDF K matrix: O(N_k² × N_AO² × N_grid × log(N_grid))")
# print("• OCCRI K matrix: O(N_k² × N_occ² × N_grid × log(N_grid))")
# print(
#     f"• Theoretical K matrix speedup: N_AO²/N_occ² = {cell.nao**2/(cell.nelectron//2)**2:.1f}x"
# )
# print("")

# print(f"\nCurrent system ({cell.basis}):")
# print(f"• {cell.nao} AOs, {cell.nelectron//2} occupied orbitals")
# print(f"• Theoretical K speedup limit: {cell.nao**2/(cell.nelectron//2)**2:.1f}x")
# print("• Practical K speedup: typically achieves 10-30% of limit")

# print("\n=== Additional Scaling Factors ===")
# print("• k-point calculations: O(N_k²) scaling favors OCCRI even more")
# print("• C extension: provides additional ~5-10x speedup")
# print("• Memory: OCCRI scales as O(N_occ) vs FFTDF O(N_AO)")

# print(
#     "\nExample completed! Try different basis sets (gth-szv, gth-dzvp, gth-cc-tzvp) to see scaling."
# )
