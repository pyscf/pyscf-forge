import numpy as np
from pyscf import siso
from pyscf.mcscf import avas
from pyscf import gto, scf, mcscf, mcpdft, lib

# SISO: SO-L-PDFT: J. Chem. Theory Comput. 2026, 22, 1, 318–333

# In this example: Computing lower Excited States of [CeCl6]^(3-) (1e, 7o) (2F5/2 and 2F7/2 states)

# 1. Molecular structure and basis set
mol = gto.Mole()
mol.atom='''
 Ce  -1.20285418    0.12742100    0.000
 Cl  -1.20285418    2.96742100    0.000
 Cl   1.63714582    0.12742100    0.000
 Cl  -1.20285418   -2.71257900    0.000
 Cl  -4.04285418    0.12742100    0.000
 Cl  -1.20285418    0.12742100    2.840
 Cl  -1.20285418    0.12742100   -2.840
'''
mol.basis={'Ce': 'ano@8s7p5d3f2g1h', 'Cl': 'ano@5s4p2d1f'}
mol.spin = 1
mol.charge = -3
mol.verbose = 4
mol.max_memory = 10000
mol.output = 'CeCl6.out'
mol.build()

# 2. SCF calculation
mf = scf.ROHF(mol).sfx2c1e().density_fit()
mf.chkfile='CeCl6.chk'
mf.max_cycle = 100
mf.kernel()

# 3. Active space selection via AVAS:
mo_coeff = avas.kernel(mf, ['Ce 4f',], minao=mol.basis)[2]

# 4. State-averaged CASSCF followed by L-PDFT:
mc = mcpdft.CASSCF(mf, 'tPBE0', 7, 1)
mc = siso.state_average_solver(mc, [(7, 2), ], ms='lin') # Model-space: 7 doublets: (2F States)
mc.max_cycle_macro = 100
mc.kernel(mo_coeff)

'''
CASCI energy for each state
  State 0 weight 0.142857  E = -11622.397455428 S^2 = 0.7500000
  State 1 weight 0.142857  E = -11622.396298347 S^2 = 0.7500000
  State 2 weight 0.142857  E = -11622.396298345 S^2 = 0.7500000
  State 3 weight 0.142857  E = -11622.396298344 S^2 = 0.7500000
  State 4 weight 0.142857  E = -11622.393971381 S^2 = 0.7500000
  State 5 weight 0.142857  E = -11622.393971375 S^2 = 0.7500000
  State 6 weight 0.142857  E = -11622.393971375 S^2 = 0.7500000

LINPDFT (final) states:
  State 0 weight 0.142857  ELPDFT = -11627.0044607867  S^2 = 0.7500000
  State 1 weight 0.142857  ELPDFT = -11627.0032278632  S^2 = 0.7500000
  State 2 weight 0.142857  ELPDFT = -11627.0032278625  S^2 = 0.7500000
  State 3 weight 0.142857  ELPDFT = -11627.0032278624  S^2 = 0.7500000
  State 4 weight 0.142857  ELPDFT = -11627.0004679539  S^2 = 0.7500000
  State 5 weight 0.142857  ELPDFT = -11627.0004679530  S^2 = 0.7500000
  State 6 weight 0.142857  ELPDFT = -11627.0004679525  S^2 = 0.7500000
'''

# Print the orbitals:
from pyscf.tools import molden
molden.from_mo(mol, mf.chkfile.rstrip('chk')+'molden', mc.mo_coeff[:, mc.ncore:mc.ncore+mc.ncas])

# State interaction: SO-L-PDFT, the difference between this and the SO-MC-PDFT is
# that the SO Hamiltonian is constructed in the L-PDFT states instead of the CASSCF states.
mysiso = siso.SISO(mc,  [(7, 2), ], ham='DKH', amf=True)
mysiso.kernel()


'''
*** Relative Spin Orbit Coupling Energetics ***
SO State       Relative Energy(au)   Relative Energy(eV)   Relative Energy(cm$^{-1}$)
 0                   0.000000000              0.00000              0.00000
 1                   0.000000000              0.00000              0.00000
 2                   0.002165955              0.05894            475.37219
 3                   0.002165955              0.05894            475.37219
 4                   0.002165959              0.05894            475.37313
 5                   0.002165959              0.05894            475.37313
 6                   0.010810547              0.29417           2372.64075
 7                   0.010810547              0.29417           2372.64075
 8                   0.012647061              0.34414           2775.70895
 9                   0.012647061              0.34414           2775.70895
 10                  0.012647065              0.34414           2775.70999
 11                  0.012647065              0.34414           2775.70999
 12                  0.014215367              0.38682           3119.91238
 13                  0.014215367              0.38682           3119.91238
'''
