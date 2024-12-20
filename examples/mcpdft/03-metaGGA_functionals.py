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

# MC23 on-top functional [ChemRxiv. 2024; doi:10.26434/chemrxiv-2024-5hc8g-v2]

# MC23 = { '0.2952*HF + (1-0.2952)*rep-M06L, 0.2952*HF + (1-0.2952)*rep-M06L'}}
# To calculate the MC23 energies,
# First: Calculate the MCPDFT Energies using repM06L
# Second: Add the CAS Contributation

# State-Specific
mc = mcpdft.CASCI(mf, 'trepM06L', 6, 8)
e_pdft = mc.kernel()[0]
emc23 = 0.2952*mc.e_mcscf + (1-0.2952)*e_pdft
print("MC-PDFT state %d Energy at %s OT Functional: %d" % (0, "MC23", emc23))

# State-average
nroots=2
mc = mcpdft.CASCI(mf, 'trepM06L', 6, 8)
mc.fcisolver.nroots=nroots
e_pdft = mc.kernel()[0]

for i in range(nroots):
    emc23 = 0.2952*mc.e_mcscf[i] + (1-0.2952)*e_pdft[i]
    print("MC-PDFT state %d Energy at %s OT Functional: %d" % (i, "MC23", emc23))
