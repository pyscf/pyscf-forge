#!/usr/bin/env python

"""
"""

import numpy

from pyscf.occri.isdfx import ISDF
from pyscf.pbc import gto, scf

print("=== OCCRI Gamma Point Tutorial ===")
print("This example shows basic OCCRI usage for single k-point calculations.\n")

# =============================================================================
# System Setup
# =============================================================================
print("Setting up diamond structure...")

# Set up diamond structure (2 carbon atoms per unit cell)
cell = gto.Cell()
cell.atom = """
    C 0.000000 0.000000 0.000000
    C 0.890186 0.890186 0.890186
"""
cell.basis = "unc-gth-dzvp"  # Compact basis set
cell.pseudo = "gth-pbe"  # Pseudopotentials
cell.a = numpy.eye(3) * 3.5607  # Diamond lattice parameter (Å)
cell.mesh = [25] * 3  # FFT mesh
cell.verbose = 4
cell.build()

print(
    f"System: {' '.join(cell.atom_symbol(i) for i in range(cell.natm))} ({cell.natm} atoms)"
)
print(f"Basis: {cell.basis}")
print(f"Lattice parameter: {cell.a[0,0]:.3f} Å")
print(f"FFT mesh: {cell.mesh} ({numpy.prod(cell.mesh)} total points)")

# =============================================================================
# Example 1: Basic OCCRI usage
# =============================================================================
print("\n" + "=" * 50)
print("Example 1: Basic OCCRI Setup")
print("=" * 50)

print("\n1a. Restricted Hartree-Fock (RHF)")
print("    How to enable OCCRI for exact exchange:")

# Standard syntax: attach OCCRI to mean-field object
mf_rhf = scf.RHF(cell)
mf_rhf.with_df = ISDF(mf_rhf)  # This line enables OCCRI
e_rhf = mf_rhf.kernel()