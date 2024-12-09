#!/usr/bin/env/python
from pyscf import gto, scf, mcpdft

mol = gto.M (
    atom = 'O 0 0 0; O 0 0 1.2',
    basis = 'ccpvdz',
    spin = 2)

mf = scf.RHF (mol).run ()

# The translation of Meta-GGAs and hybrid-Meta-GGAs [ChemRxiv. 2024; doi:10.26434/chemrxiv-2024-5hc8g-v2]

# Translated-Meta-GGA
mc = mcpdft.CASCI(mf, 'tM06L', 6, 8).run ()

# Hybrid-Translated-Meta-GGA
tM06L0 = 't' + mcpdft.hyb('M06L',0.25, hyb_type='average')
mc = mcpdft.CASCI(mf, tM06L0, 6, 8).run ()
