import numpy as np
from pyscf.mcscf import avas
from pyscf import gto, scf, mcscf
from pyscf import siso

# In this script, there are couple of implemented functions which 
# can be used to analyze the SOC states.

# 1. Define the molecule
mol = gto.Mole(atom="O 0 0 0; O 0 0 1.21",
               spin=2,
               max_memory=10000,
               basis="ano@4s3p2d1f", #=ANO_RCC_VTZP
               verbose=4,
               output='O2.out')
mol.build()

# 2. Mean-field calculation
mf = scf.ROHF(mol).sfx2c1e()
mf.chkfile = 'O2.chk'
mf.kernel()

# Active space selection:
mo_coeff = avas.kernel(mf, ['O 2s', 'O 2p'], minao=mol.basis)[2]

# 3. State-average Calculation.
mc = mcscf.CASSCF(mf, 8, 12)
mc = siso.sacasscf_solver(mc, [(3, 1), (1, 3)])
mc.max_cycle_macro = 100
mc.conv_tol = 1e-8
mc.kernel(mo_coeff)

exit()


'''
CASCI energy for each state
  State 0 weight 0.333333  E = -242.329497518035 S^2 = 0.7500000
  State 1 weight 0.333333  E = -242.329497505989 S^2 = 0.7500000
  State 2 weight 0.333333  E = -242.329497505961 S^2 = 0.7500000
'''

# 4. State interaction
# Note, the model-space for the SA-CASSCF and SISO should be the same. You can also define the different model-space
# for the SA-CASSCF and SISO, but then you need to reconstruct your mc object. There are two Hamiltonian options for
# the SOC calculations: Breit-Pauli (BP) and Douglas-Kroll-Hess (DKH).

# amf: is the AMFI integrals.

mysiso = siso.SISO(mc,  [(3, 1), (1, 3)], ham='BP', amf=True)
mysiso.kernel()

# 2P1/2 and 2P3/2 States:
'''
******** Relative Spin Orbit Coupling Energetics ********
SO State       Relative Energy(au)   Relative Energy(eV)   Relative Energy(cm^-1)
 0                   0.000000000              0.00000              0.00000
 1                   0.000000000              0.00000              0.00000
 2                   0.000473949              0.01290            104.01970
 3                   0.000473949              0.01290            104.01970
 4                   0.000473957              0.01290            104.02147
 5                   0.000473957              0.01290            104.02147
'''
