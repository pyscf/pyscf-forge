"""
Test suite for OCCRI (Occupied Orbital Coulomb Resolution of Identity) method.

This module contains regression tests to validate the OCCRI implementation
against reference FFTDF calculations for various mean-field methods in 
periodic systems. The tests ensure that OCCRI maintains chemical accuracy
while providing computational speedup.

Test Coverage:
    - Restricted Hartree-Fock (RHF)
    - Unrestricted Hartree-Fock (UHF) 
    - Restricted Kohn-Sham DFT (RKS) with PBE0 functional
    - Unrestricted Kohn-Sham DFT (UKS) with PBE0 functional

The test system is a diamond structure with 8 carbon atoms, chosen as a
representative covalent solid for validating periodic exchange evaluation.

Tolerance:
    All tests use TOL = 1e-8 Hartree/atom for energy differences, which
    ensures sub-mHartree accuracy per atom typical for chemical applications.

Usage:
    Run directly: python test_get_k.py
    Or via pytest: pytest test_get_k.py
"""

import numpy
import pyscf
from pyscf.pbc import gto
from pyscf.occri import OCCRI

# Tolerance for energy comparison (Hartree per atom)
TOL = 1.e-8

if __name__ == "__main__":
    # Set up test system: diamond structure (8 carbon atoms)
    # This represents a typical covalent solid for testing periodic exchange
    refcell = gto.Cell()
    refcell.atom = """ 
        C 0.000000 0.000000 1.780373
        C 0.890186 0.890186 2.670559
        C 0.000000 1.780373 0.000000
        C 0.890186 2.670559 0.890186
        C 1.780373 0.000000 0.000000
        C 2.670559 0.890186 0.890186
        C 1.780373 1.780373 1.780373
        C 2.670559 2.670559 2.670559
    """
    # Lattice vectors for diamond structure (3.56 Å lattice parameter)
    refcell.a = numpy.array(
        [
            [3.560745, 0.000000, 0.000000],
            [0.000000, 3.560745, 0.000000],
            [0.000000, 0.000000, 3.560745],
        ]
    )
    refcell.basis = "gth-cc-dzvp"      # Double-zeta valence basis with polarization
    refcell.pseudo = "gth-pbe"         # Goedecker-Teter-Hutter pseudopotentials
    refcell.ke_cutoff = 70             # Kinetic energy cutoff in Hartree
    refcell.verbose = 0                # Suppress SCF output for cleaner test logs
    refcell.build()

    # Test 1: Restricted Hartree-Fock (RHF)
    # Reference energy from standard FFTDF calculation  
    en_fftdf = -43.9399339901445
    mf = pyscf.pbc.scf.RHF(refcell)
    mf.with_df = OCCRI(mf)
    en = mf.kernel()
    en_diff = abs(en - en_fftdf) / refcell.natm
    if en_diff < TOL:
        print("RHF occRI passed", en_diff)
    else:
        print("RHF occRI FAILED!!!", en_diff)


    # Test 2: Unrestricted Hartree-Fock (UHF)
    # For closed-shell diamond, UHF should give same energy as RHF
    en_fftdf = -43.9399339901445
    mf = pyscf.pbc.scf.UHF(refcell)
    mf.with_df = OCCRI(mf)
    en = mf.kernel()
    en_diff = abs(en - en_fftdf) / refcell.natm
    if en_diff < TOL:
        print("UHF occRI passed", en_diff)
    else:
        print("UHF occRI FAILED!!!", en_diff)    


    # Test 3: Restricted Kohn-Sham DFT with PBE0 hybrid functional
    # PBE0 contains 25% exact exchange, testing hybrid DFT capability
    en_fftdf = -45.0265010261793
    mf = pyscf.pbc.scf.RKS(refcell)
    mf.xc = 'pbe0'  # 25% exact exchange + 75% PBE exchange + 100% PBE correlation
    mf.with_df = OCCRI(mf)
    en = mf.kernel()
    en_diff = abs(en - en_fftdf) / refcell.natm
    if en_diff < TOL:
        print("RKS occRI passed", en_diff)
    else:
        print("RKS occRI FAILED!!!", en_diff)


    # Test 4: Unrestricted Kohn-Sham DFT with PBE0 hybrid functional  
    # Tests spin-unrestricted hybrid DFT (should match RKS for closed shell)
    en_fftdf = -45.0265009589458
    mf = pyscf.pbc.scf.UKS(refcell)
    mf.xc = 'pbe0'  # Same functional as RKS test
    mf.with_df = OCCRI(mf)
    en = mf.kernel()
    en_diff = abs(en - en_fftdf) / refcell.natm
    if en_diff < TOL:
        print("UKS occRI passed", en_diff)
    else:
        print("UKS occRI FAILED!!!", en_diff)            