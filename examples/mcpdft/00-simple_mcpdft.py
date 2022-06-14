#!/usr/bin/env/python

from pyscf import gto, scf, mcpdft

mol = gto.M (
    atom = 'O 0 0 0; O 0 0 1.2',
    basis = 'ccpvdz',
    spin = 2)

mf = scf.RHF (mol).run ()

# 1. CASCI density

mc0 = mcpdft.CASCI (mf, 'tPBE', 6, 8).run ()

# 2. CASSCF density
# Note that the MC-PDFT energy may not be lower, even though
# E(CASSCF)<=E(CASCI).

mc1 = mcpdft.CASSCF (mf, 'tPBE', 6, 8).run ()

