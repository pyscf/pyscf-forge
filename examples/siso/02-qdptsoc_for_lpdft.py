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
mol.basis={'Ce': 'ano@7s6p4d2f', 'Cl': 'ano@4s3p'} # ANO-MB
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
  State 0 weight 0.142857  E = -11622.3670314309 S^2 = 0.7500000
  State 1 weight 0.142857  E = -11622.3660216499 S^2 = 0.7500000
  State 2 weight 0.142857  E = -11622.3660216499 S^2 = 0.7500000
  State 3 weight 0.142857  E = -11622.3660216498 S^2 = 0.7500000
  State 4 weight 0.142857  E = -11622.3637494851 S^2 = 0.7500000
  State 5 weight 0.142857  E = -11622.3637494851 S^2 = 0.7500000
  State 6 weight 0.142857  E = -11622.3637494851 S^2 = 0.7500000

LINPDFT (final) states:
  State 0 weight 0.142857  ELPDFT = -11626.9790857768  S^2 = 0.7500000
  State 1 weight 0.142857  ELPDFT = -11626.9778927226  S^2 = 0.7500000
  State 2 weight 0.142857  ELPDFT = -11626.9778927225  S^2 = 0.7500000
  State 3 weight 0.142857  ELPDFT = -11626.9778927224  S^2 = 0.7500000
  State 4 weight 0.142857  ELPDFT = -11626.9751321296  S^2 = 0.7500000
  State 5 weight 0.142857  ELPDFT = -11626.9751321294  S^2 = 0.7500000
  State 6 weight 0.142857  ELPDFT = -11626.9751321293  S^2 = 0.7500000
'''

# Print the orbitals:
from pyscf.tools import molden
molden.from_mo(mol, mf.chkfile.rstrip('chk')+'molden', mc.mo_coeff[:, mc.ncore:mc.ncore+mc.ncas])

# State interaction: SO-L-PDFT, the difference between this and the SO-MC-PDFT is
# that the SO Hamiltonian is constructed in the L-PDFT states instead of the CASSCF states.
mysiso = siso.SISO(mc, ham='DKH', amf=True)
mysiso.kernel()


'''
*** Relative Spin Orbit Coupling Energetics ***
SO State       Relative Energy(au)   Relative Energy(eV)   Relative Energy(cm$^{-1}$)
 0                   0.000000000              0.00000              0.00000
 1                   0.000000000              0.00000              0.00000
 2                   0.002148618              0.05847            471.56703
 3                   0.002148618              0.05847            471.56704
 4                   0.002148618              0.05847            471.56710
 5                   0.002148618              0.05847            471.56710
 6                   0.010806104              0.29405           2371.66577
 7                   0.010806104              0.29405           2371.66577
 8                   0.012620583              0.34342           2769.89779
 9                   0.012620583              0.34342           2769.89779
 10                  0.012620583              0.34342           2769.89786
 11                  0.012620583              0.34342           2769.89787
 12                  0.014189232              0.38611           3114.17656
 13                  0.014189232              0.38611           3114.17656
'''
