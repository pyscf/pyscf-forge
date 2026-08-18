import numpy as np
from pyscf import siso
from pyscf.mcscf import avas
from pyscf import gto, scf, mcscf, mcpdft, lib

# SISO: SO-L-PDFT: J. Chem. Theory Comput. 2026, 22, 1, 318–333

# 1. Molecular structure and basis set
mol = gto.Mole(atom="P 0 0 0",
               spin=3,
               max_memory=10000,
               basis="ano@4s3p1d",#=ANO-RCC-VDZP
               verbose=4,
               output='P.out')
mol.build()

# 2. SCF calculation
mf = scf.ROHF(mol).sfx2c1e().density_fit()
mf.chkfile='P.chk'
mf.max_cycle = 100
mf.kernel()

# 3. Active space selection via AVAS:
mo_coeff = avas.kernel(mf, ['P 3s', 'P 3p', 'P 3d','P 4s', 'P 4p'], minao=mol.basis)[2]

# 4. State-averaged CASSCF followed by L-PDFT:
mc = mcpdft.CASSCF(mf,'tPBE0', 13, 5)
mc = siso.state_average_solver(mc, [(8, 2), (1,4)], ms='lin')
mc.max_cycle_macro = 100
mc.kernel(mo_coeff)

'''
CASCI energy for each state
  State 0 weight 0.111111  E = -341.557379802129 S^2 = 0.7500000
  State 1 weight 0.111111  E = -341.557379802128 S^2 = 0.7500000
  State 2 weight 0.111111  E = -341.557379802127 S^2 = 0.7500000
  State 3 weight 0.111111  E = -341.557379802125 S^2 = 0.7500000
  State 4 weight 0.111111  E = -341.557379802125 S^2 = 0.7500000
  State 5 weight 0.111111  E = -341.524474660536 S^2 = 0.7500000
  State 6 weight 0.111111  E = -341.524474660534 S^2 = 0.7500000
  State 7 weight 0.111111  E = -341.524474660531 S^2 = 0.7500000
  State 8 weight 0.111111  E = -341.620374849748 S^2 = 3.7500000

LINPDFT (final) states:
  State 0 weight 0.111111  ELPDFT = -341.787242041527  S^2 = 0.7500000
  State 1 weight 0.111111  ELPDFT = -341.787242038511  S^2 = 0.7500000
  State 2 weight 0.111111  ELPDFT = -341.787242037149  S^2 = 0.7500000
  State 3 weight 0.111111  ELPDFT = -341.787242035418  S^2 = 0.7500000
  State 4 weight 0.111111  ELPDFT = -341.787242031175  S^2 = 0.7500000
  State 5 weight 0.111111  ELPDFT = -341.760666598525  S^2 = 0.7500000
  State 6 weight 0.111111  ELPDFT = -341.760666594007  S^2 = 0.7500000
  State 7 weight 0.111111  ELPDFT = -341.760666587515  S^2 = 0.7500000
  State 8 weight 0.111111  ELPDFT = -341.837242618884  S^2 = 3.7500000
'''

# State interaction: SO-L-PDFT, the difference between this and the SO-MC-PDFT is
# that the SO Hamiltonian is constructed in the L-PDFT states instead of the CASSCF states.
mysiso = siso.SISO(mc, ham='DKH', amf=True)
mysiso.kernel()


'''
*** Relative Spin Orbit Coupling Energetics ***
SO State       Relative Energy(au)   Relative Energy(eV)   Relative Energy(cm^-1)
 0                   0.000000000              0.00000              0.00000
 1                   0.000000000              0.00000              0.00000
 2                   0.000000000              0.00000              0.00000
 3                   0.000000000              0.00000              0.00000
 4                   0.049943505              1.35903          10961.33233
 5                   0.049943505              1.35903          10961.33233
 6                   0.049943508              1.35903          10961.33301
 7                   0.049943508              1.35903          10961.33301
 8                   0.050029353              1.36137          10980.17385
 9                   0.050029353              1.36137          10980.17385
 10                  0.050029355              1.36137          10980.17423
 11                  0.050029355              1.36137          10980.17423
 12                  0.050029359              1.36137          10980.17508
 13                  0.050029359              1.36137          10980.17508
 14                  0.076546434              2.08293          16800.00032
 15                  0.076546434              2.08293          16800.00032
 16                  0.076661013              2.08605          16825.14760
 17                  0.076661013              2.08605          16825.14760
 18                  0.076661019              2.08605          16825.14898
 19                  0.076661019              2.08605          16825.14898
'''
