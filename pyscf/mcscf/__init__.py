#!/usr/bin/env python
"""
MCSCF extensions for PySCF-Forge

Extensions to pyscf.mcscf providing:
- FOMO-SCF: Floating occupation molecular orbital SCF
- DFT-CASCI: CASCI with DFT-corrected core energy
"""

from pyscf.mcscf import addons
from pyscf.mcscf import casci_dft

__all__ = ['addons', 'casci_dft']
