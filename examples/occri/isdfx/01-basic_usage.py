#!/usr/bin/env python
"""
Basic ISDFX Usage Examples

Shows ISDFX integration with different SCF methods for various material types.
ISDFX provides efficient exact exchange evaluation for periodic systems.
"""

import numpy as np
from pyscf.occri.isdfx import ISDFX
from pyscf.pbc import gto, scf

# === Diamond (Semiconductor) with RHF ===
print('=== Diamond (Semiconductor) ===')

cell_diamond = gto.Cell()
cell_diamond.atom = """
    C 0.0 0.0 0.0
    C 0.89 0.89 0.89
"""
cell_diamond.basis = 'gth-dzvp'
cell_diamond.pseudo = 'gth-pbe'
cell_diamond.a = np.eye(3) * 3.57
cell_diamond.mesh = [15] * 3  # See Example 4 in occri/01-simple_gamma_point.py for mesh selection.
cell_diamond.verbose = 3
cell_diamond.build()

# Standard RHF with ISDFX
mf = scf.RHF(cell_diamond)
mf.with_df = ISDFX.from_mf(mf)
e_diamond = mf.kernel()
print(f'Diamond RHF energy: {e_diamond:.6f} Ha')

# === Graphite (Metal) with UHF ===
print('\n=== Graphite (Metal) ===')

cell_graphite = gto.Cell()
cell_graphite.atom = """
    C 0.0000 0.0000 0.0000
    C 1.2300 0.7100 0.0000
"""
cell_graphite.basis = 'gth-dzvp'
cell_graphite.pseudo = 'gth-pbe'
cell_graphite.a = [[2.46, 0, 0], [-1.23, 2.13, 0], [0, 0, 6.70]]
cell_graphite.mesh = [15] * 3  # See Example 4 in occri/01-simple_gamma_point.py for mesh selection.
cell_graphite.verbose = 0
cell_graphite.build()

# UHF for potential magnetic ordering
kpts = cell_graphite.make_kpts([1, 1, 2])  # 2D material
mf = scf.KUHF(cell_graphite, kpts=kpts)
mf.with_df = ISDFX.from_mf(mf)
e_graphite = mf.kernel()
print(f'Graphite UHF energy: {e_graphite:.6f} Ha')

# === Silicon (Semiconductor) with UKS ===
print('\n=== Silicon (Semiconductor) ===')

cell_si = gto.Cell()
cell_si.atom = """
    Si 0.0000 0.0000 0.0000
    Si 1.3575 1.3575 1.3575
"""
cell_si.basis = 'gth-dzvp'
cell_si.pseudo = 'gth-pbe'
cell_si.a = np.eye(3) * 5.43
cell_si.mesh = [15] * 3  # See Example 4 in occri/01-simple_gamma_point.py for mesh selection.
cell_si.verbose = 0
cell_si.build()

# DFT with exact exchange via ISDFX
kpts = cell_si.make_kpts([1, 1, 2])
mf = scf.KUKS(cell_si, kpts=kpts)
mf.xc = 'pbe0'  # Hybrid functional
mf.with_df = ISDFX.from_mf(mf)
e_si = mf.kernel()
print(f'Silicon PBE0 energy: {e_si:.6f} Ha')

# === Aluminum (Metal) with RKS ===
print('\n=== Aluminum (Metal) ===')

cell_al = gto.Cell()
cell_al.atom = """
    Al 0.0 0.0 0.0
"""
cell_al.basis = 'gth-dzvp'
cell_al.pseudo = 'gth-pbe'
cell_al.a = np.eye(3) * 4.05  # FCC lattice
cell_al.spin = 1  # Avoid closed-shell for odd electrons
cell_al.mesh = [15] * 3  # See Example 4 in occri/01-simple_gamma_point.py for mesh selection.
cell_al.verbose = 0
cell_al.build()

# RKS for metallic system
kpts = cell_al.make_kpts([1, 1, 2])
mf = scf.KRKS(cell_al, kpts=kpts)
mf.xc = 'b3lyp'  # Hybrid functional
mf.with_df = ISDFX.from_mf(mf)
e_al = mf.kernel()
print(f'Aluminum B3LYP energy: {e_al:.6f} Ha')

print('\n=== ISDFX Configuration Notes ===')
print('• Works with (K)RHF/(K)UHF for Hartree-Fock theory')
print('• Works with (K)RKS/(K)UKS for hybrid DFT (PBE0, B3LYP, etc.)')
print('• Efficient for metals, semiconductors, and insulators')
print('• Scales better than traditional methods for large k-point sets')
