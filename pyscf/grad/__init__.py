#!/usr/bin/env python
"""
Gradient extensions for PySCF-Forge

Extensions to pyscf.mcscf providing:
- FOMO-SCF: Floating occupation molecular orbital SCF
- DFT-CASCI: CASCI with DFT-corrected core energy
"""

from pyscf.grad import casci_dft

__all__ = ['casci_dft']
