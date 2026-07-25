import numpy as np
from pyscf.mcscf import avas
from pyscf import gto, scf, mcscf
from pyscf import siso

# Example to perform the two-step state interaction calculation including the
# spin-orbit coupling effects. The first step is to perform a state-averaged CASSCF
# calculation, and the second step is to perform the QDPT-SOC calculation using the
# state-averaged CASSCF wave functions as the model space.

# 1. Define the molecule
mol = gto.Mole(atom="Al 0 0 0",
               spin=1,
               max_memory=10000,
               basis="ano@5s4p2d1f", #=ANO_RCC_VTZP
               verbose=4,
               output='Al.out')
mol.build()

# Note, scalar relativistic effects are included via the spin-free X2C Hamiltonian, and
# that too for only the 1e part. There is an option to include the scalar relativistic
# effects via scalar DKH Hamiltonian, currently that code is accessible at:
# https://github.com/MatthewRHermes/mrh/blob/master/examples/dkh/run_dkh.py

# 2. Mean-field calculation
mf = scf.ROHF(mol).sfx2c1e()
mf.kernel()

# Active space selection:
mo_coeff = avas.kernel(mf, ['Al 3s', 'Al 3p'], minao=mol.basis)[2]

# SA-CASSCF Calculation. In PySCF, you can inject different approximate FCI solvers like DMRG, SCI etc.
# or different types of FCI solvers like direct_spin*. However, here, I have hard coded this approach,
# state interaction approach with exact FCI solver and CSFSolver as the FCISolver. CSFSolver will make sure
# we are getting spin-adapated wave functions.

# Note: to select the model-space which is N_i states of SM_i spin multiplicity,
# you need to define the tuple in this format [(N_i, SM_i,), (N_j, SM_j), ...].
# Additionally, you can also specify the symmetry of the target states
# as [(N_i, SM_i, wfnsym_i), (N_j, SM_j, wfnsym_j), ...].
# If the symmetry is not specified, it will be set to None by default.

# 3. State-average Calculation.
mc = mcscf.CASSCF(mf, 4, 3)
mc = siso.sacasscf_solver(mc, [(3, 2),]) # Here, I am defining a model space of 3 doublets. (N_i=3, SM_i=2): (2P States)
mc.max_cycle_macro = 100
mc.conv_tol = 1e-8
mc.kernel(mo_coeff)

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

mysiso = siso.SISO(mc,  [(3, 2),], ham='BP', amf=True)
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

# Computing the L and J values:
from pyscf.siso.addons import soc_analysis, generate_siso_data

modelspace = [(3, 2),]
mydata = generate_siso_data(mol, mc, modelspace, mysiso, 
                            origin='CHARGE_CENTER', ham='DKH')

mysiso_analysis = soc_analysis(mysiso, mydata, )

# Now compute the L-values.
mysiso_analysis.compute_L_values()

# Now compute the J-values.
mysiso_analysis.compute_J_values()
