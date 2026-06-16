#!/usr/bin/env python
"""
Example 22: FOMO-DFT-corrected CASCI Gradients - Complete Comparison
==========================================================

This example demonstrates all combinations of:
- Orbitals: RHF vs FOMO
- Core: HF vs DFT
- Gradient: Analytical vs Numerical

Author:
    Arshad Mehmood, IACS, Stony Brook University
    31 December 2025
"""

from pyscf import gto, scf, mcscf
from pyscf.scf import fomoscf
from pyscf.mcscf import dft_corrected_casci
import numpy as np
import time

mol = gto.M(
    atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
    basis='cc-pvdz', unit='Angstrom', verbose=3
)

ncas, nelecas = 6, 6
ncore = (mol.nelectron - nelecas) // 2

print("="*70)
print("Complete FOMO-DFT-corrected CASCI Gradient Comparison")
print("="*70)

# Setup
mf_rhf = scf.RHF(mol).run()
mf_fomo = fomoscf.fomo_scf(mf_rhf, temperature=0.25, method='gaussian', restricted=(ncore, ncas))
mf_fomo.kernel()

# All combinations
configs = [
    ('RHF + HF core',     mf_rhf,  None),
    ('RHF + PBE core',    mf_rhf,  'PBE'),
    ('FOMO + LDA core',   mf_fomo, 'LDA'),
    ('FOMO + PBE core',   mf_fomo, 'PBE'),
]

results = []

for name, mf, xc in configs:
    print(f"\n{'-'*70}")
    print(f"Configuration: {name}")
    print(f"{'-'*70}")
    
    if xc is None:
        # Standard CASCI (only analytical gradient)
        mc = mcscf.CASCI(mf, ncas, nelecas)
        mc.kernel()
        
        t0 = time.time()
        g = mc.Gradients().kernel()
        t_grad = time.time() - t0
        
        method_used = 'analytical (standard)'
        energy = mc.e_tot
        
    else:
        # DFT-corrected CASCI (both methods available)
        mc = dft_corrected_casci.CASCI(mf, ncas, nelecas, xc=xc)
        mc.kernel()
        energy = mc.e_tot
        
        # Try both methods
        print("  Computing analytical gradient...")
        t0 = time.time()
        g_ana = mc.Gradients(method='analytical').kernel()
        t_ana = time.time() - t0
        
        print("  Computing numerical gradient...")
        t0 = time.time()
        g_num = mc.Gradients(method='numerical').kernel()
        t_num = time.time() - t0
        
        # Use analytical for results
        g = g_ana
        t_grad = t_ana
        method_used = f'analytical ({t_num/t_ana:.1f}x faster than numerical)'
        
        diff = np.abs(g_ana - g_num).max()
        print(f"  Analytical vs Numerical difference: {diff:.2e} Ha/Bohr")
    
    print(f"  Energy: {energy:.10f} Ha")
    print(f"  Gradient method: {method_used}")
    print(f"  Time: {t_grad:.2f} seconds")
    
    results.append((name, energy, g, t_grad))

# Summary comparison
print("\n" + "="*70)
print("Energy Summary")
print("="*70)
print(f"{'Configuration':<25} {'Energy (Ha)':<18} {'Rel. to Std (mHa)':<20}")
print("-"*63)

e_ref = results[0][1]  # Standard CASCI energy
for name, energy, _, _ in results:
    delta = (energy - e_ref) * 1000
    print(f"{name:<25} {energy:<18.10f} {delta:>19.4f}")

# Gradient comparison table
print("\n" + "="*70)
print("Gradient Comparison (Maximum absolute component in Ha/Bohr)")
print("="*70)
print(f"{'Configuration':<25} {'Max |Gradient|':<18} {'Time (s)':<12}")
print("-"*55)

for name, _, grad, time_val in results:
    max_g = np.abs(grad).max()
    print(f"{name:<25} {max_g:<18.8f} {time_val:<12.2f}")

# Detailed gradient table for one component
print("\n" + "="*70)
print("Detailed Gradient: O atom, z-component (Ha/Bohr)")
print("="*70)

for name, _, grad, _ in results:
    g_oz = grad[0, 2]  # O atom, z component
    print(f"  {name:<25} {g_oz:>14.8f}")

# Method recommendation
print("\n" + "="*70)
print("Recommendations")
print("="*70)
print("""
For FOMO-DFT-corrected CASCI calculations:

1. **Molecular Dynamics / Exploratory Work:**
   - Use: Analytical gradients (method='analytical')
   - Accuracy: ~1-2e-2 Ha/Bohr
   - Speed: Fast
   - Command: mc.Gradients(method='analytical').kernel()

2. **High-Precision / Benchmarks:**
   - Use: Numerical gradients (method='numerical')
   - Accuracy: ~1e-6 Ha/Bohr
   - Speed: Slow (but reliable)
   - Command: mc.Gradients(method='numerical').kernel()

3. **General Purpose / Safe Default:**
   - Use: Auto mode (tries analytical, falls back to numerical)
   - Accuracy: Variable
   - Speed: Usually fast
   - Command: mc.Gradients(method='auto').kernel()

4. **Critical Geometries (e.g., transition states):**
   - Validate analytical with numerical spot checks
   - Use numerical for final optimization steps
""")

print("="*70)