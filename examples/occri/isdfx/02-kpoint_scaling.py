#!/usr/bin/env python
"""
ISDFX K-point Scaling and Accuracy Analysis

Demonstrates how ISDFX accuracy and computational cost scale with k-point density.
Shows convergence behavior and timing comparison vs. traditional methods.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from pyscf.occri.isdfx import ISDFX
from pyscf.pbc import df, gto, scf

print('=== ISDFX K-point Scaling Analysis ===')

# Silicon test system (simple, well-behaved)
cell = gto.Cell()
cell.atom = """
    Li 0.0 0.0 0.0
    H  2.0 0.0 0.0
"""
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pbe'
cell.a = np.eye(3) * 6.0
cell.mesh = [25] * 3
cell.verbose = 0
cell.build()

# K-point meshes to test (increasing density)
kmesh_list = [
    [1, 1, 1],  # 1 k-point (gamma)
    [1, 1, 2],  # 2 k-points
    [1, 2, 2],  # 4 k-points
    [2, 2, 2],  # 8 k-points
    [2, 2, 3],  # 12 k-points
    [2, 3, 3],  # 18 k-points
    [3, 3, 3],  # 27 k-points
]

# Storage for results
results = {
    'kmesh': [],
    'nkpts': [],
    'e_isdfx': [],
    'e_fftdf': [],
    'err_isdfx': [],
    'time_isdfx': [],
    'time_fftdf': [],
    'time_build': [],
}

print('\n=== K-point Convergence Study ===')
print(f'{"Mesh":<8} {"Nk":<4} {"T_ISDFX":<8} {"T_FFTDF":<8} {"T_BUILD":<8}')
print('-' * 50)

for kmesh in kmesh_list:
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)
    mesh_str = f'{kmesh[0]}x{kmesh[1]}x{kmesh[2]}'

    # FFTDF calculation (for comparison)
    mf_fftdf = scf.KRHF(cell, kpts=kpts)
    mf_fftdf.max_cycle = 1
    mf_fftdf.kernel()

    dm = mf_fftdf.make_rdm1(kpts=kpts)
    t0 = time.time()
    _, vk = mf_fftdf.get_jk(dm_kpts=dm, with_j=False, with_k=True, kpts=kpts)
    t_fftdf = time.time() - t0

    # ISDFX calculation
    mf_isdfx = scf.KRHF(cell, kpts=kpts)
    t0 = time.time()
    mf_isdfx.with_df = ISDFX.from_mf(mf_isdfx)
    mf_isdfx.with_df.scf_iter = 1
    t_build = time.time() - t0

    t0 = time.time()
    _, vk = mf_isdfx.get_jk(dm_kpts=dm, with_j=False, with_k=True, kpts=kpts)
    t_isdfx = time.time() - t0

    # Store results
    results['kmesh'].append(mesh_str)
    results['nkpts'].append(nkpts)
    results['time_isdfx'].append(t_isdfx)
    results['time_fftdf'].append(t_fftdf)
    results['time_build'].append(t_build)

    print(f'{mesh_str:<8} {nkpts:<4} {t_isdfx:<8.4f} {t_fftdf:<8.4f} {t_build:<8.4f}')

print('\nTimes in seconds')

# === Scaling Analysis ===
print('\n=== Scaling Analysis ===')

# Check if ISDFX error scales linearly with k-point spacing
nk_array = np.array(results['nkpts'])


# Computational scaling
print('\nComputational time scaling:')
print(f'{"Nk":<6} {"ISDFX":<10} {"FFTDF":<10} {"BUILD":<10} {"Speedup":<10}')
print('-' * 55)
for i in range(len(nk_array)):
    speedup = results['time_fftdf'][i] / results['time_isdfx'][i]
    print(
        f'{nk_array[i]:<6} {results["time_isdfx"][i]:<10.3f} {results["time_fftdf"][i]:<10.3f} \
            {results["time_build"][i]:<10.3f} {speedup:<10.2f}x'
    )


# === Log-Log Scaling Analysis ===
print('\n=== Computational Scaling Analysis ===')

# Extract timing data (skip gamma point which may have different scaling behavior)
nk_scaling = np.array(results['nkpts'][1:])  # Skip gamma point
time_isdfx_scaling = np.array(results['time_isdfx'][1:])
time_fftdf_scaling = np.array(results['time_fftdf'][1:])
time_build_scaling = np.array(results['time_build'][1:])

if len(nk_scaling) >= 3:
    # Log-log fit: log(time) = m * log(nk) + b  =>  time ∝ nk^m
    log_nk = np.log(nk_scaling)
    log_time_isdfx = np.log(time_isdfx_scaling)
    log_time_fftdf = np.log(time_fftdf_scaling)
    log_time_build = np.log(time_build_scaling)

    # Linear fit in log-log space: y = mx + b
    isdfx_fit = np.polyfit(log_nk, log_time_isdfx, 1)
    fftdf_fit = np.polyfit(log_nk, log_time_fftdf, 1)
    build_fit = np.polyfit(log_nk, log_time_build, 1)

    # Extract scaling exponents (slopes in log-log plot)
    isdfx_scaling = isdfx_fit[0]  # m in y = mx + b
    fftdf_scaling = fftdf_fit[0]
    build_scaling = build_fit[0]

    # R-squared for goodness of fit
    isdfx_predicted = isdfx_fit[0] * log_nk + isdfx_fit[1]
    fftdf_predicted = fftdf_fit[0] * log_nk + fftdf_fit[1]
    build_predicted = build_fit[0] * log_nk + build_fit[1]

    isdfx_r2 = 1 - np.sum((log_time_isdfx - isdfx_predicted) ** 2) / np.sum(
        (log_time_isdfx - np.mean(log_time_isdfx)) ** 2
    )
    fftdf_r2 = 1 - np.sum((log_time_fftdf - fftdf_predicted) ** 2) / np.sum(
        (log_time_fftdf - np.mean(log_time_fftdf)) ** 2
    )
    build_r2 = 1 - np.sum((log_time_build - build_predicted) ** 2) / np.sum(
        (log_time_build - np.mean(log_time_build)) ** 2
    )

    print('Scaling analysis from log-log fit:')
    print(f'  ISDFX: time ∝ Nk^{isdfx_scaling:.2f} (R² = {isdfx_r2:.3f})')
    print(f'  FFTDF: time ∝ Nk^{fftdf_scaling:.2f} (R² = {fftdf_r2:.3f})')
    print(f'  BUILD: time ∝ Nk^{build_scaling:.2f} (R² = {build_r2:.3f})')

    # Interpret scaling exponents
    def interpret_scaling(exponent):
        if abs(exponent - 1.0) < 0.1:
            return 'Linear (≈ O(Nk))'
        elif abs(exponent - 2.0) < 0.1:
            return 'Quadratic (≈ O(Nk²))'
        elif abs(exponent - 3.0) < 0.1:
            return 'Cubic (≈ O(Nk³))'
        elif exponent < 1.0:
            return f'Sub-linear (O(Nk^{exponent:.2f}))'
        elif exponent < 2.0:
            return f'Between linear and quadratic (O(Nk^{exponent:.2f}))'
        elif exponent < 3.0:
            return f'Between quadratic and cubic (O(Nk^{exponent:.2f}))'
        else:
            return f'Higher than cubic (O(Nk^{exponent:.2f}))'

    print('\nScaling interpretation:')
    print(f'  ISDFX: {interpret_scaling(isdfx_scaling)}')
    print(f'  FFTDF: {interpret_scaling(fftdf_scaling)}')
    print(f'  BUILD: {interpret_scaling(build_scaling)}')

    # Compare scaling
    if isdfx_scaling < fftdf_scaling:
        advantage = fftdf_scaling - isdfx_scaling
        print(f'  ✓ ISDFX has better scaling by {advantage:.2f} orders')
        print(f'    For large Nk, ISDFX advantage grows as Nk^{advantage:.2f}')
    elif isdfx_scaling > fftdf_scaling:
        disadvantage = isdfx_scaling - fftdf_scaling
        print(f'  ⚠ FFTDF has better scaling by {disadvantage:.2f} orders')
        print('    ISDFX may have higher overhead or need larger Nk to show advantage')
    else:
        print('  ≈ Both methods have similar scaling')


else:
    print('Need at least 3 data points for reliable scaling analysis')

print('\n=== ISDFX Scaling Notes ===')
print('• Log-log analysis reveals computational scaling behavior')
print('• Scaling exponents determine efficiency for large systems')
print('• ISDFX should show linear scaling with k-points')
