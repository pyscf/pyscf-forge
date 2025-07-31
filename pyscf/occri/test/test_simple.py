#!/usr/bin/env python
"""
Simple test to verify k-point generation and basic functionality.
"""

import numpy
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf
from pyscf.occri import OCCRI

def test_kpoint_generation():
    """Test that k-points are generated correctly"""
    # Setup cell
    cell = pgto.Cell()
    cell.atom = '''
    C 0.0 0.0 0.0
    C 0.9 0.9 0.9
    '''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pbe'
    cell.a = numpy.eye(3) * 3.5607
    cell.mesh = [12] * 3
    cell.build()
    
    # Generate k-points properly
    kpts = cell.make_kpts([2,2,2])
    print(f"Generated {len(kpts)} k-points")
    print(f"K-points shape: {kpts.shape}")
    print(f"First few k-points:\n{kpts[:3]}")
    
    # Test OCCRI with k-points
    mf = scf.KRHF(cell, kpts)
    mf.with_df = OCCRI(mf, kmesh=[2,2,2])
    
    # Just test initialization
    dm = mf.get_init_guess()
    print(f"Density matrix shape: {dm.shape}")
    print(f"Density matrix dtype: {dm.dtype}")
    
    print("Basic k-point test passed!")
    return True

if __name__ == '__main__':
    test_kpoint_generation()