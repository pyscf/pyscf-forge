from pyscf import gto, scf, mcpdft
from pyscf.mcscf import avas
from pyscf import siso

# Two-step state interaction calculation with dynamical correlation effects included via MC-PDFT.
# Alternatively, you can also include the dynamical correlation effects via NEVPT2.

mol = gto.Mole(atom="P 0 0 0",
               spin=3,
               max_memory=120000,
               basis="ano@5s4p2d1f",#=ANO_RCC_VTZP
               verbose=4,
               output='P.out')
mol.build()

mf = scf.ROHF(mol).sfx2c1e().density_fit()
mf.kernel()

mo_coeff = avas.kernel(mf, ['P 3s', 'P 3p', 'P 3d', 'P 4s', 'P 4p'], minao=mol.basis)[2]

# Here, I am defining a model space of 8 doublets and 1 quartet. (4S, 2D, and 2P States)
# i.e.: (N_i=8, SM_i=2) and (N_j=1, SM_j=4) = [(8, 2), (1,4)]

mc = mcpdft.CASSCF(mf,'tPBE0', 13, 5)

# Note: state_average_solver is a wrapper function that generates multiple FCI
# solvers with specified spin multiplicities, symmetries, and numbers of roots.
#
# mc = siso.state_average_solver(mc, [(8, 2), (1, 4)]) is equivalent to:
#
# from pyscf.csf_fci import csf_solver
# nroots_i, smult_i, wfnsym_i = 8, 2, None
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
# weights = [1 / 9] * 9
# mc = mcscf.state_average_mix_(mc, solvers, weights)

mc = siso.state_average_solver(mc, [(8, 2), (1,4)], )
mc.max_cycle_macro = 100
mc.kernel(mo_coeff)

'''
CASCI energy for each state
  State 0 weight 0.111111  E = -341.56020464031  S^2 = 0.7500000
  State 1 weight 0.111111  E = -341.56020463948  S^2 = 0.7500000
  State 2 weight 0.111111  E = -341.56020463643  S^2 = 0.7500000
  State 3 weight 0.111111  E = -341.56020463204  S^2 = 0.7500000
  State 4 weight 0.111111  E = -341.56020462993  S^2 = 0.7500000
  State 5 weight 0.111111  E = -341.52663106582  S^2 = 0.7500000
  State 6 weight 0.111111  E = -341.52663106166  S^2 = 0.7500000
  State 7 weight 0.111111  E = -341.52663105728  S^2 = 0.7500000
  State 8 weight 0.111111  E = -341.62300041540  S^2 = 3.7500000

MC-PDFT state 0 E = -341.7887722059603, Eot(t0.25*HF + 0.75*PBE, 0.25*HF + 0.75*PBE) = -17.33756155114337
MC-PDFT state 1 E = -341.7886743999867, Eot(t0.25*HF + 0.75*PBE, 0.25*HF + 0.75*PBE) = -17.33746379933534
MC-PDFT state 2 E = -341.7886056913069, Eot(t0.25*HF + 0.75*PBE, 0.25*HF + 0.75*PBE) = -17.33739504085652
MC-PDFT state 3 E = -341.7887339883463, Eot(t0.25*HF + 0.75*PBE, 0.25*HF + 0.75*PBE) = -17.33752333956303
MC-PDFT state 4 E = -341.7886722132114, Eot(t0.25*HF + 0.75*PBE, 0.25*HF + 0.75*PBE) = -17.33746162441176
MC-PDFT state 5 E = -341.7611493313987, Eot(t0.25*HF + 0.75*PBE, 0.25*HF + 0.75*PBE) = -17.32352966777335
MC-PDFT state 6 E = -341.7611499103902, Eot(t0.25*HF + 0.75*PBE, 0.25*HF + 0.75*PBE) = -17.32353024620902
MC-PDFT state 7 E = -341.7611497689359, Eot(t0.25*HF + 0.75*PBE, 0.25*HF + 0.75*PBE) = -17.32353011152015
MC-PDFT state 8 E = -341.8383313515934, Eot(t0.25*HF + 0.75*PBE, 0.25*HF + 0.75*PBE) = -17.37763730304456
'''

# State interaction:
mysiso = siso.SISO(mc, [(8, 2),(1,4)], ham='DKH', amf=True)
mysiso.kernel()

'''
******** Relative Spin Orbit Coupling Energetics ********
SO State       Relative Energy(au)   Relative Energy(eV)   Relative Energy(cm^-1)
 0                   0.000000000              0.00000              0.00000
 1                   0.000000000              0.00000              0.00000
 2                   0.000000000              0.00000              0.00001
 3                   0.000000000              0.00000              0.00001
 4                   0.049534892              1.34791          10871.65210
 5                   0.049534892              1.34791          10871.65210
 6                   0.049585591              1.34929          10882.77936
 7                   0.049585591              1.34929          10882.77936
 8                   0.049640414              1.35078          10894.81154
 9                   0.049640414              1.35078          10894.81154
 10                  0.049677824              1.35180          10903.02213
 11                  0.049677824              1.35180          10903.02213
 12                  0.049732014              1.35328          10914.91543
 13                  0.049732014              1.35328          10914.91543
 14                  0.077167944              2.09985          16936.40611
 15                  0.077167944              2.09985          16936.40611
 16                  0.077258343              2.10231          16956.24632
 17                  0.077258343              2.10231          16956.24632
 18                  0.077258692              2.10232          16956.32290
 19                  0.077258692              2.10232          16956.32290
'''

# Compute NEVPT2 energies for the model space and update mc.e_states.
siso.compute_nevpt2_energies(mc, [(8, 2), (1, 4)])

'''
NEVPT2 Energies
******** Spin Orbit Free Energetics ********
 State 0 Total Energy = -341.6535332810 S^2 = 3.75
 State 1 Total Energy = -341.5970503972 S^2 = 0.75
 State 2 Total Energy = -341.5970429332 S^2 = 0.75
 State 3 Total Energy = -341.5970429067 S^2 = 0.75
 State 4 Total Energy = -341.5970405174 S^2 = 0.75
 State 5 Total Energy = -341.5970399780 S^2 = 0.75
 State 6 Total Energy = -341.5630640073 S^2 = 0.75
 State 7 Total Energy = -341.5630639026 S^2 = 0.75
 State 8 Total Energy = -341.5630638737 S^2 = 0.75
'''

# State interaction:
mysiso = siso.SISO(mc, [(8, 2),(1,4)], ham='DKH', amf=True)
mysiso.kernel()

'''
******** Relative Spin Orbit Coupling Energetics ********
SO State       Relative Energy(au)   Relative Energy(eV)   Relative Energy(cm^-1)
 0                   0.000000000              0.00000              0.00000
 1                   0.000000000              0.00000              0.00000
 2                   0.000000000              0.00000              0.00000
 3                   0.000000000              0.00000              0.00000
 4                   0.056442770              1.53589          12387.75618
 5                   0.056442770              1.53589          12387.75618
 6                   0.056447096              1.53600          12388.70565
 7                   0.056447096              1.53600          12388.70565
 8                   0.056510342              1.53772          12402.58648
 9                   0.056510342              1.53772          12402.58648
 10                  0.056514826              1.53785          12403.57062
 11                  0.056514826              1.53785          12403.57062
 12                  0.056516581              1.53789          12403.95573
 13                  0.056516581              1.53789          12403.95573
 14                  0.090454010              2.46138          19852.36055
 15                  0.090454010              2.46138          19852.36055
 16                  0.090535049              2.46358          19870.14649
 17                  0.090535049              2.46358          19870.14649
 18                  0.090535132              2.46359          19870.16471
 19                  0.090535132              2.46359          19870.16471
'''
