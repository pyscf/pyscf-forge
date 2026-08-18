from pyscf import gto, scf, mcpdft
from pyscf.mcscf import avas
from pyscf import siso

# Two-step state interaction calculation with dynamical correlation effects included via MC-PDFT.
# Alternatively, you can also include the dynamical correlation effects via NEVPT2.

mol = gto.Mole(atom="P 0 0 0",
               spin=3,
               max_memory=120000,
               basis="ano@4s3p1d",#=ANO-RCC-VDZP
               verbose=4,
               output='P.out')
mol.build()

mf = scf.ROHF(mol).sfx2c1e().density_fit()
mf.kernel()

mo_coeff = avas.kernel(mf, ['P 3s', 'P 3p', 'P 3d',], minao=mol.basis)[2]

# Here, I am defining a model space of 5 doublets and 1 quartet. (4S, and 2D States)
# i.e.: (N_i=5, SM_i=2) and (N_j=1, SM_j=4) = [(5, 2), (1,4)]

mc = mcpdft.CASSCF(mf,'tPBE0', 9, 5)

# Note: state_average_solver is a wrapper function that generates multiple FCI
# solvers with specified spin multiplicities, symmetries, and numbers of roots.
#
# mc = siso.state_average_solver(mc, [(5, 2), (1, 4)]) is equivalent to:
#
# from pyscf.csf_fci import csf_solver
# nroots_i, smult_i, wfnsym_i = 5, 2, None
# solver_i = csf_solver(mol, smult=smult_i)
# solver_i.wfnsym = wfnsym_i
# solver_i.nroots = nroots_i
# solver_i.spin = smult_i - 1
#
# nroots_j, smult_j, wfnsym_j = 1, 4, None
# solver_j = csf_solver(mol, smult=smult_j)
# solver_j.wfnsym = wfnsym_j
# solver_j.nroots = nroots_j
# solver_j.spin = smult_j - 1
#
# solvers = [solver_i, solver_j]
# weights = [1 / 6] * 6
# mc = mcscf.state_average_mix_(mc, solvers, weights)

mc = siso.state_average_solver(mc, [(5, 2), (1,4)], )
mc.max_cycle_macro = 100
mc.kernel(mo_coeff)

'''
CASCI energy for each state
  State 0 weight 0.166667  E = -341.536802633453 S^2 = 0.7500000
  State 1 weight 0.166667  E = -341.536802631798 S^2 = 0.7500000
  State 2 weight 0.166667  E = -341.536802618748 S^2 = 0.7500000
  State 3 weight 0.166667  E = -341.536802607231 S^2 = 0.7500000
  State 4 weight 0.166667  E = -341.536802605068 S^2 = 0.7500000
  State 5 weight 0.166667  E = -341.602704771996 S^2 = 3.7500000

MC-PDFT state 0 E = -341.7844757903552, Eot(t0.25*HF + 0.75*PBE, 0.25*HF + 0.75*PBE) = -17.33842791348752
MC-PDFT state 1 E = -341.7849306642770, Eot(t0.25*HF + 0.75*PBE, 0.25*HF + 0.75*PBE) = -17.33888276870781
MC-PDFT state 2 E = -341.7849333716372, Eot(t0.25*HF + 0.75*PBE, 0.25*HF + 0.75*PBE) = -17.338885534485637
MC-PDFT state 3 E = -341.7849301299142, Eot(t0.25*HF + 0.75*PBE, 0.25*HF + 0.75*PBE) = -17.338882260181094
MC-PDFT state 4 E = -341.7843719056824, Eot(t0.25*HF + 0.75*PBE, 0.25*HF + 0.75*PBE) = -17.338324018465634
MC-PDFT state 5 E = -341.8357909022529, Eot(t0.25*HF + 0.75*PBE, 0.25*HF + 0.75*PBE) = -17.37025694337651
'''

# State interaction:
mysiso = siso.SISO(mc, ham='DKH', amf=True)
mysiso.kernel()

'''
******** Relative Spin Orbit Free Energetics ********
State         Relative Energy(au)   Relative Energy(eV)   Relative Energy(cm^-1)
 0                   0.000000000              0.00000              0.00000
 1                   0.050857531              1.38390          11161.93778
 2                   0.050860238              1.38398          11162.53198
 3                   0.050860772              1.38399          11162.64926
 4                   0.051315112              1.39636          11262.36527
 5                   0.051418997              1.39918          11285.16532
'''

# compute_nevpt2_energies returns the NEVPT2 total energies in the same order
# as the model-space states. When modelspace is omitted, it is read from the
# state-average FCI solvers attached to mc. The function does not modify
# mc.e_states. During the calculation, the logger reports the NEVPT2 energy
# for each state.
nevpt2_energies = siso.compute_nevpt2_energies(mc)

# Apply the diagonal approximation explicitly. The CASSCF model-space wave
# functions, and therefore the off-diagonal spin-orbit couplings, are retained,
# while the diagonal spin-free energies are replaced by the NEVPT2 energies.
mc.e_states[:] = nevpt2_energies

'''
NEVPT2 Energies
******** Spin Orbit Free Energetics ********
 State 0 Total Energy = -341.6192897060 S^2 = 3.75
 State 1 Total Energy = -341.5547952144 S^2 = 0.75
 State 2 Total Energy = -341.5547951896 S^2 = 0.75
 State 3 Total Energy = -341.5547565827 S^2 = 0.75
 State 4 Total Energy = -341.5547563004 S^2 = 0.75
 State 5 Total Energy = -341.5547553869 S^2 = 0.75
'''

# State interaction:
mysiso = siso.SISO(mc, ham='DKH', amf=True)
mysiso.kernel()

'''
******** Relative Spin Orbit Coupling Energetics ********
SO State       Relative Energy(au)   Relative Energy(eV)   Relative Energy(cm^-1)
 0                   0.000000000              0.00000              0.00000
 1                   0.000000000              0.00000              0.00000
 2                   0.000000000              0.00000              0.00000
 3                   0.000000000              0.00000              0.00000
 4                   0.064494491              1.75498          14154.90469
 5                   0.064494491              1.75498          14154.90469
 6                   0.064494516              1.75499          14154.91017
 7                   0.064494516              1.75499          14154.91017
 8                   0.064533114              1.75604          14163.38145
 9                   0.064533114              1.75604          14163.38145
 10                  0.064533412              1.75604          14163.44687
 11                  0.064533412              1.75604          14163.44687
 12                  0.064534322              1.75607          14163.64655
 13                  0.064534322              1.75607          14163.64655
'''
