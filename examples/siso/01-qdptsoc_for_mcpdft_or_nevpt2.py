from pyscf import gto, scf, mcpdft
from pyscf.mcscf import avas
from pyscf import siso

# Author: Bhavnesh Jangid

# Two-step state interaction calculation with dynamical correlation effects included via MC-PDFT.
# Alternatively, you can also include the dynamical correlation effects via NEVPT2.

mol = gto.Mole()
mol.atom='''
Cr 0 0 0
H 0 0 1.69'''
mol.basis= basis = {'Cr': 'ano@4s3p1d', 'H' : 'ano@1s'} # ANO-MB
mol.spin = 5
mol.charge = 0
mol.verbose = 4
mol.output = 'CrH.out'
mol.build()

mf = scf.ROHF(mol).sfx2c1e().density_fit()
mf.kernel()

mo_coeff = avas.kernel(mf, ['Cr 3d', 'Cr 4s', 'H 1s'], minao=mol.basis)[2]

modelspace = [(5,4), (5,6)]
# Here, I am defining a model space of 8 doublets and 1 quartet, just from demonstration purposes.
# i.e.: (N_i=5, SM_i=4) and (N_j=5, SM_j=6) = [(5, 4), (5, 6)]

mc = mcpdft.CASSCF(mf,'tPBE0', 7, 7)
mc = siso.state_average_solver(mc, modelspace)
# Note: state_average_solver is a wrapper function that generates multiple FCI
# solvers with specified spin multiplicities, symmetries, and numbers of roots.
#
# mc = siso.state_average_solver(mc, [(5, 4), (5, 6)]) is equivalent to:
#
# from pyscf.csf_fci import csf_solver
# nroots_i, smult_i, wfnsym_i = 5, 4, None
# solver_i = csf_solver(mol, smult=smult_i)
# solver_i.wfnsym = wfnsym_i
# solver_i.nroots = nroots_i
# solver_i.spin = smult_i - 1
#
# nroots_j, smult_j, wfnsym_j = 5, 6, None
# solver_j = csf_solver(mol, smult=smult_j)
# solver_j.wfnsym = wfnsym_j
# solver_j.nroots = nroots_j
# solver_j.spin = smult_j - 1
#
# solvers = [solver_i, solver_j]
# weights = [1 / 10] * 10
# mc = mcscf.state_average_mix_(mc, solvers, weights)
mc.max_cycle_macro = 200
mc.kernel(mo_coeff)

'''
CASCI energy for each state
  State 0 weight 0.1  E = -1050.10963650892 S^2 = 3.7500000
  State 1 weight 0.1  E = -1050.09422101535 S^2 = 3.7500000
  State 2 weight 0.1  E = -1050.09422101461 S^2 = 3.7500000
  State 3 weight 0.1  E = -1050.08988019818 S^2 = 3.7500000
  State 4 weight 0.1  E = -1050.08988019816 S^2 = 3.7500000
  State 5 weight 0.1  E = -1050.18056241694 S^2 = 8.7500000
  State 6 weight 0.1  E = -1050.11540775232 S^2 = 8.7500000
  State 7 weight 0.1  E = -1050.11494595141 S^2 = 8.7500000
  State 8 weight 0.1  E = -1050.11494595049 S^2 = 8.7500000
  State 9 weight 0.1  E = -1050.10773852682 S^2 = 8.7500000

MC-PDFT state 0 E = -1050.806933172248, Eot(t0.25*HF + 0.75*PBE, 0.25*HF + 0.75*PBE) = -36.821674503654656
MC-PDFT state 1 E = -1050.795930017495, Eot(t0.25*HF + 0.75*PBE, 0.25*HF + 0.75*PBE) = -36.77461237351305
MC-PDFT state 2 E = -1050.795930016996, Eot(t0.25*HF + 0.75*PBE, 0.25*HF + 0.75*PBE) = -36.774612374017444
MC-PDFT state 3 E = -1050.793040153555, Eot(t0.25*HF + 0.75*PBE, 0.25*HF + 0.75*PBE) = -36.77615011897173
MC-PDFT state 4 E = -1050.793040485211, Eot(t0.25*HF + 0.75*PBE, 0.25*HF + 0.75*PBE) = -36.7761501792654
MC-PDFT state 5 E = -1050.882406036268, Eot(t0.25*HF + 0.75*PBE, 0.25*HF + 0.75*PBE) = -37.03252482545145
MC-PDFT state 6 E = -1050.811936394242, Eot(t0.25*HF + 0.75*PBE, 0.25*HF + 0.75*PBE) = -36.99044835563739
MC-PDFT state 7 E = -1050.785536946272, Eot(t0.25*HF + 0.75*PBE, 0.25*HF + 0.75*PBE) = -36.75560906277122
MC-PDFT state 8 E = -1050.785536945457, Eot(t0.25*HF + 0.75*PBE, 0.25*HF + 0.75*PBE) = -36.755609062852024
MC-PDFT state 9 E = -1050.775953624509, Eot(t0.25*HF + 0.75*PBE, 0.25*HF + 0.75*PBE) = -36.751566750182846
'''

# State interaction:
mysiso = siso.SISO(mc, ham='DKH', amf=True)
mysiso.kernel()

'''
******** Relative Spin Orbit Free Energetics ********
SO State       Relative Energy(au)   Relative Energy(eV)   Relative Energy(cm^-1)
 0                   0.000000000              0.00000              0.00000
 1                   0.000000000              0.00000              0.00000
 2                   0.000000034              0.00000              0.00744
 3                   0.000000034              0.00000              0.00744
 4                   0.000000102              0.00000              0.02236
 5                   0.000000102              0.00000              0.02236
 6                   0.070456931              1.91723          15463.50899
 7                   0.070456931              1.91723          15463.50899
 8                   0.070461242              1.91735          15464.45512
 9                   0.070461242              1.91735          15464.45512
 10                  0.070469874              1.91758          15466.34965
 11                  0.070469874              1.91758          15466.34965
 12                  0.075435935              2.05272          16556.27413
 13                  0.075435935              2.05272          16556.27413
 14                  0.075458681              2.05334          16561.26618
 15                  0.075458681              2.05334          16561.26618
 16                  0.086071506              2.34212          18890.51201
 17                  0.086071506              2.34212          18890.51201
 18                  0.086264197              2.34737          18932.80287
 19                  0.086264197              2.34737          18932.80287
 20                  0.086522761              2.35440          18989.55117
 21                  0.086522761              2.35440          18989.55117
 22                  0.086829031              2.36274          19056.76948
 23                  0.086829031              2.36274          19056.76948
 24                  0.088643968              2.41213          19455.10231
 25                  0.088643968              2.41213          19455.10231
 26                  0.089213152              2.42761          19580.02373
 27                  0.089213152              2.42761          19580.02373
 28                  0.089726284              2.44158          19692.64318
 29                  0.089726284              2.44158          19692.64318
 30                  0.090194891              2.45433          19795.49036
 31                  0.090194891              2.45433          19795.49036
 32                  0.096400547              2.62319          21157.47456
 33                  0.096400547              2.62319          21157.47456
 34                  0.096586888              2.62826          21198.37156
 35                  0.096586888              2.62826          21198.37156
 36                  0.096775747              2.63340          21239.82129
 37                  0.096775747              2.63340          21239.82129
 38                  0.096963939              2.63852          21281.12474
 39                  0.096963939              2.63852          21281.12474
 40                  0.097158376              2.64381          21323.79886
 41                  0.097158376              2.64381          21323.79886
 42                  0.097351139              2.64906          21366.10542
 43                  0.097351139              2.64906          21366.10542
 44                  0.106467960              2.89714          23367.01637
 45                  0.106467960              2.89714          23367.01637
 46                  0.106484710              2.89760          23370.69257
 47                  0.106484710              2.89760          23370.69258
 48                  0.106493051              2.89782          23372.52306
 49                  0.106493051              2.89782          23372.52306
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
 State 0 Total Energy = -1050.2247075926 S^2 = 8.75
 State 1 Total Energy = -1050.1614230694 S^2 = 8.75
 State 2 Total Energy = -1050.1614230693 S^2 = 8.75
 State 3 Total Energy = -1050.1585321199 S^2 = 3.75
 State 4 Total Energy = -1050.1577237134 S^2 = 8.75
 State 5 Total Energy = -1050.1553263264 S^2 = 8.75
 State 6 Total Energy = -1050.1485861100 S^2 = 3.75
 State 7 Total Energy = -1050.1485861100 S^2 = 3.75
 State 8 Total Energy = -1050.1451310244 S^2 = 3.75
 State 9 Total Energy = -1050.1451101929 S^2 = 3.75
'''

# State interaction:
mysiso = siso.SISO(mc, ham='DKH', amf=True)
mysiso.kernel()

'''
******** Relative Spin Orbit Coupling Energetics ********
SO State       Relative Energy(au)   Relative Energy(eV)   Relative Energy(cm^-1)
 0                   0.000000000              0.00000              0.00000
 1                   0.000000000              0.00000              0.00000
 2                   0.000000394              0.00001              0.08637
 3                   0.000000394              0.00001              0.08637
 4                   0.000001181              0.00003              0.25918
 5                   0.000001181              0.00003              0.25918
 6                   0.062757547              1.70772          13773.68948
 7                   0.062757547              1.70772          13773.68948
 8                   0.062912679              1.71194          13807.73697
 9                   0.062912679              1.71194          13807.73697
 10                  0.063116002              1.71747          13852.36129
 11                  0.063116002              1.71747          13852.36129
 12                  0.063300501              1.72249          13892.85416
 13                  0.063300501              1.72249          13892.85416
 14                  0.063520509              1.72848          13941.14021
 15                  0.063520509              1.72848          13941.14021
 16                  0.063758470              1.73496          13993.36659
 17                  0.063758470              1.73496          13993.36659
 18                  0.066155979              1.80020          14519.55920
 19                  0.066155979              1.80020          14519.55920
 20                  0.066184610              1.80097          14525.84299
 21                  0.066184610              1.80097          14525.84299
 22                  0.067031592              1.82402          14711.73385
 23                  0.067031592              1.82402          14711.73385
 24                  0.067084381              1.82546          14723.31983
 25                  0.067084381              1.82546          14723.31983
 26                  0.067109530              1.82614          14728.83930
 27                  0.067109530              1.82614          14728.83930
 28                  0.069401505              1.88851          15231.86980
 29                  0.069401505              1.88851          15231.86980
 30                  0.069426903              1.88920          15237.44384
 31                  0.069426903              1.88920          15237.44384
 32                  0.069439599              1.88955          15240.23044
 33                  0.069439599              1.88955          15240.23044
 34                  0.075723957              2.06055          16619.48745
 35                  0.075723957              2.06055          16619.48745
 36                  0.075936007              2.06632          16666.02705
 37                  0.075936007              2.06632          16666.02705
 38                  0.076194544              2.07336          16722.76936
 39                  0.076194544              2.07336          16722.76936
 40                  0.076491351              2.08144          16787.91106
 41                  0.076491351              2.08144          16787.91106
 42                  0.078849807              2.14561          17305.53232
 43                  0.078849807              2.14561          17305.53232
 44                  0.079418724              2.16109          17430.39520
 45                  0.079418724              2.16109          17430.39520
 46                  0.079940684              2.17530          17544.95222
 47                  0.079940684              2.17530          17544.95222
 48                  0.080423951              2.18845          17651.01693
 49                  0.080423951              2.18845          17651.01693
'''
