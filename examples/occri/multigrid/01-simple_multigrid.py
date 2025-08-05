#!/usr/bin/env python

"""
Simple Multigrid OccRI Example

This example demonstrates basic usage of the multigrid-enhanced OCCRI
method for efficient exact exchange evaluation in periodic systems.

Key features demonstrated:
- Basic multigrid OCCRI setup
- Comparison with standard OCCRI
- Performance and accuracy considerations
- Different multigrid parameters
"""

import numpy

# Import standard OCCRI for comparison
from pyscf.occri import OCCRI
from pyscf.pbc import gto, scf

# Import multigrid OCCRI (when available)
try:
    from pyscf.occri.multigrid import MultigridOccRI
    MULTIGRID_AVAILABLE = True
except ImportError:
    print("Multigrid OccRI not yet implemented - this is a placeholder example")
    MULTIGRID_AVAILABLE = False

print("=== Simple Multigrid OccRI Example ===")

# =============================================================================
# System Setup
# =============================================================================
print("\nSetting up diamond structure...")

cell = gto.Cell()
cell.atom = """
    C 0.000000 0.000000 0.000000
    C 0.890186 0.890186 0.890186
"""
cell.basis = "gth-szv"
cell.pseudo = "gth-pbe"
cell.a = numpy.eye(3) * 3.5607
cell.mesh = [24] * 3  # Relatively fine mesh for accuracy
cell.verbose = 0
cell.build()

print(f"System: {cell.natm} carbon atoms")
print(f"FFT mesh: {cell.mesh} ({numpy.prod(cell.mesh)} points)")

# =============================================================================
# Standard OCCRI Calculation (Reference)
# =============================================================================
print("\n" + "=" * 50)
print("Reference: Standard OCCRI")
print("=" * 50)

mf_ref = scf.RHF(cell)
mf_ref.with_df = OCCRI(mf_ref)
e_ref = mf_ref.kernel()

print(f"Standard OCCRI energy: {e_ref:.8f} Ha")
print(f"Converged: {mf_ref.converged}")

# =============================================================================
# Multigrid OccRI Calculations
# =============================================================================
if MULTIGRID_AVAILABLE:
    print("\n" + "=" * 50)
    print("Multigrid OccRI")
    print("=" * 50)
    
    # Example 1: Basic multigrid with default parameters
    print("\n1. Basic multigrid (3 levels, factor 2)")
    mf_mg1 = scf.RHF(cell)
    mf_mg1.with_df = MultigridOccRI(mf_mg1)
    e_mg1 = mf_mg1.kernel()
    
    print(f"   Energy: {e_mg1:.8f} Ha")
    print(f"   Difference from reference: {e_mg1 - e_ref:.2e} Ha")
    print(f"   Converged: {mf_mg1.converged}")
    
    # Example 2: More aggressive coarsening
    print("\n2. Aggressive coarsening (4 levels, factor 3)")
    mf_mg2 = scf.RHF(cell)
    mf_mg2.with_df = MultigridOccRI(
        mf_mg2,
        mg_levels=4,
        coarsening_factor=3,
        mg_method='vcycle'
    )
    e_mg2 = mf_mg2.kernel()
    
    print(f"   Energy: {e_mg2:.8f} Ha")
    print(f"   Difference from reference: {e_mg2 - e_ref:.2e} Ha")
    print(f"   Converged: {mf_mg2.converged}")
    
    # Example 3: Full Multigrid method
    print("\n3. Full Multigrid (FMG) method")
    mf_mg3 = scf.RHF(cell)
    mf_mg3.with_df = MultigridOccRI(
        mf_mg3,
        mg_levels=3,
        coarsening_factor=2,
        mg_method='fmg'
    )
    e_mg3 = mf_mg3.kernel()
    
    print(f"   Energy: {e_mg3:.8f} Ha")
    print(f"   Difference from reference: {e_mg3 - e_ref:.2e} Ha")
    print(f"   Converged: {mf_mg3.converged}")
    
    # =============================================================================
    # Performance Comparison
    # =============================================================================
    print("\n" + "=" * 50)
    print("Performance Summary")
    print("=" * 50)
    
    print(f"Standard OCCRI:     {e_ref:.8f} Ha")
    if 'e_mg1' in locals():
        print(f"Multigrid (basic):  {e_mg1:.8f} Ha  (Δ = {e_mg1-e_ref:.2e})")
    if 'e_mg2' in locals():
        print(f"Multigrid (aggr.):  {e_mg2:.8f} Ha  (Δ = {e_mg2-e_ref:.2e})")
    if 'e_mg3' in locals():
        print(f"Multigrid (FMG):    {e_mg3:.8f} Ha  (Δ = {e_mg3-e_ref:.2e})")
    
    print("\nMultigrid benefits:")
    print("• Faster convergence for large systems")
    print("• Better scaling with basis set size")
    print("• Reduced memory requirements")
    print("• Systematic error control")

else:
    print("\n" + "=" * 50)
    print("Multigrid OccRI - Placeholder")
    print("=" * 50)
    
    print("Multigrid OccRI is not yet implemented.")
    print("When available, it will provide:")
    print("• Accelerated convergence through hierarchical grids")
    print("• Better performance for large basis sets")
    print("• Systematic control of discretization errors")
    print("• V-cycle and Full Multigrid solution methods")

# =============================================================================
# Usage Guide
# =============================================================================
print("\n" + "=" * 50)
print("Multigrid OccRI Usage Guide")
print("=" * 50)

print("""
Basic usage:
  from pyscf.occri.multigrid import MultigridOccRI
  
  mf = scf.RHF(cell)
  mf.with_df = MultigridOccRI(mf, mg_levels=3, coarsening_factor=2)
  energy = mf.kernel()

Parameters:
  mg_levels         : Number of multigrid levels (2-5 recommended)
  coarsening_factor : Grid coarsening ratio (2-3 typical)
  mg_method         : Solution method ('vcycle', 'fmg')

When to use multigrid OCCRI:
  • Large basis sets (>100 AOs)
  • Dense k-point sampling
  • When standard OCCRI is memory-limited
  • Systems requiring high accuracy

Performance tips:
  • Start with mg_levels=3, coarsening_factor=2
  • Use 'fmg' for better initial guess
  • Monitor convergence vs. mg_levels
  • Balance accuracy vs. computational cost
""")

print("Example completed successfully!")