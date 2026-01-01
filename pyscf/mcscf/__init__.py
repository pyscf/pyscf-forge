#!/usr/bin/env python
"""
MCSCF extensions for PySCF-Forge

Extensions to pyscf.mcscf providing:
- DFT-Corrected-CASCI: CASCI with DFT-corrected core energy
"""

from pyscf.mcscf import dft_corrected_casci

__all__ = ['dft_corrected_casci']
