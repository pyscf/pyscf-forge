from pyscf.mcscf import avas
from pyscf import gto, scf, mcscf, lib
from pyscf import siso

# In this script, there are couple of implemented functions which
# can be used to analyze the SOC states.

# 1. Define the molecule
mol = gto.Mole(atom="O 0 0 0; O 0 0 1.21",
               spin=2,
               basis="ano@3s2p1d", #=ANO_RCC_VDZP
               verbose=4,
               output='O2.out')
mol.build()

# 2. Mean-field calculation
mf = scf.ROHF(mol).sfx2c1e()
mf.kernel()

# Active space selection:
mo_coeff = avas.kernel(mf, ['O 2s', 'O 2p'], minao=mol.basis)[2]

# 3. State-average Calculation.
modelspace = [(3, 1), (1, 3)]
mc = mcscf.CASSCF(mf, 8, 12)
mc = siso.sacasscf_solver(mc, modelspace)
mc.max_cycle_macro = 100
mc.conv_tol = 1e-8
mc.kernel(mo_coeff)
'''
CASCI energy for each state
  State 0 weight 0.25  E = -149.828698971524 S^2 = 0.0000000
  State 1 weight 0.25  E = -149.828698971524 S^2 = -0.0000000
  State 2 weight 0.25  E = -149.809434105444 S^2 = 0.0000000
  State 3 weight 0.25  E = -149.863555677146 S^2 = 2.0000000
'''

mysiso = siso.SISO(mc, ham='DKH', amf=True)
mysiso.kernel()
'''
******** Relative Spin Orbit Coupling Energetics ********
SO State       Relative Energy(au)   Relative Energy(eV)   Relative Energy(cm^-1)
 0                   0.000000000              0.00000              0.00000
 1                   0.000012263              0.00033              2.69150
 2                   0.000012263              0.00033              2.69150
 3                   0.034868969              0.94883           7652.85412
 4                   0.034868969              0.94883           7652.85412
 5                   0.054146098              1.47339          11883.69500
'''

# Analysis Functions for SISO calculation:
from pyscf.siso.addons import soc_analysis, generate_siso_data

mydata = generate_siso_data(mol, mc, mysiso=mysiso,
                            origin='CHARGE_CENTER', ham='DKH')

mysiso_analysis = soc_analysis(mysiso, mydata, )

# Compute the Lambda values, which is the projection of the orbital
# angular momentum along the principal axis.
mysiso_analysis.compute_L_values_for_diatomics(axis='z')
'''
******** SOC energies and effective orbital L projections ********
Projection axis: L_z
Degeneracy tolerance: 1.000e-06 Hartree
  State         Energy (au)         Lambda-value    Block
------------------------------------------------------------
  0        -149.8635679405           0.0000        0
  1        -149.8635556771           0.0000        1
  2        -149.8635556771           0.0000        1
  3        -149.8286989715           2.0000        2
  4        -149.8286989715           2.0000        2
  5        -149.8094218421           0.0000        3
'''

# Compute the omega values for the SO states: Omega is the
# projection of the total angular momentum along the principal axis.
mysiso_analysis.compute_omega_values(axis='z')
'''
******** SOC energies and effective Omega values ********
Projection axis: J_z
Degeneracy tolerance: 1.000e-06 Hartree
  State         Energy (au)         Omega-value     Block
------------------------------------------------------------
  0        -149.8635679405           0.0000        0
  1        -149.8635556771           1.0000        1
  2        -149.8635556771           1.0000        1
  3        -149.8286989715           2.0000        2
  4        -149.8286989715           2.0000        2
  5        -149.8094218421           0.0000        3
'''

# Do the SO State decomposition in the spin-free basis.
mysiso_analysis.soc_state_analysis(state=(0, 1, 2, 3, 4, 5), threshold=1e-3)
'''
SOC-state decomposition in the spin-free basis

SOC state 0, energy = -149.8635679405 au
  spin-free state 3: total weight = 0.99977351
    state 3, m_s = +0.0: weight = 0.99977351

SOC state 1, energy = -149.8635556771 au
  spin-free state 3: total weight = 1.00000000
    state 3, m_s = -1.0: weight = 0.11232255
    state 3, m_s = +1.0: weight = 0.88767745

SOC state 2, energy = -149.8635556771 au
  spin-free state 3: total weight = 1.00000000
    state 3, m_s = -1.0: weight = 0.88767745
    state 3, m_s = +1.0: weight = 0.11232255

SOC state 3, energy = -149.8286989715 au
  spin-free state 0: total weight = 1.00000000
    state 0, m_s = -0.0: weight = 1.00000000

SOC state 4, energy = -149.8286989715 au
  spin-free state 1: total weight = 1.00000000
    state 1, m_s = -0.0: weight = 1.00000000

SOC state 5, energy = -149.8094218421 au
  spin-free state 2: total weight = 0.99977351
    state 2, m_s = -0.0: weight = 0.99977351
'''
