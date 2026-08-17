from pyscf import gto, scf, mcscf
from pyscf.mcscf import avas
from pyscf import siso

# Example to perform a symmetry-resolved state-interaction calculation for the
# OH diatomic. The three O 2p-like doublet states transform as A1, B1, and B2 in
# the C2v subgroup. Spin-orbit coupling splits the two components
# of the 2-Pi state while the higher 2-Sigma state remains in the model space.

# If you are using this code, please consider citing:
# J. Chem. Theory Comput. 2026, 22, 1, 318-333

# 1. Define the molecule. Symmetry must be enabled before wfnsym labels can be
# used in the model-space specification.
mol = gto.Mole(atom='O 0 0 0; H 0 0 0.9697',
               spin=1,
               max_memory=10000,
               basis='ano@4s3p2d1f',  # ANO-RCC-VTZP
               symmetry='C2v',
               verbose=4,
               output='OH.out')
mol.build()

# Scalar relativistic effects are included through the spin-free X2C one-electron
# Hamiltonian. The spin-orbit interaction is added in the subsequent SISO step.

# 2. Mean-field calculation.
mf = scf.ROHF(mol).sfx2c1e()
mf.kernel()

# 3. Select the three O 2p-like orbitals. With five active electrons, this gives
# one A1, one B1, and one B2 doublet state.
mo_coeff = avas.kernel(mf, ['O 2p'], minao=mol.basis)[2]

# Each model-space entry has the form (number of roots, spin multiplicity,
# wave-function symmetry). Entries with the same spin multiplicity but different
# irreps are separate state-average solvers and are all retained by SISO.
modelspace = [(1, 2, 'A1'),
              (1, 2, 'B1'),
              (1, 2, 'B2')]

# 4. Symmetry-resolved state-averaged CASSCF calculation.
mc = mcscf.CASSCF(mf, 3, 5)
mc = siso.sacasscf_solver(mc, modelspace)
mc.max_cycle_macro = 100
mc.conv_tol = 1e-8
mc.kernel(mo_coeff)

'''
CASCI energy for each state
  State 0 weight 0.333333  E = -75.313997154567 S^2 = 0.7500000  (A1)
  State 1 weight 0.333333  E = -75.471878612036 S^2 = 0.7500000  (B1)
  State 2 weight 0.333333  E = -75.471878612858 S^2 = 0.7500000  (B2)
'''

# 5. State interaction. SISO reads the symmetry-resolved model space from mc.
mysiso = siso.SISO(mc, ham='BP', amf=True)
mysiso.kernel()

# The lowest four states are the two Kramers pairs arising from the 2-Pi state.
'''
******** Relative Spin Orbit Coupling Energetics ********
SO State       Relative Energy(au)   Relative Energy(eV)   Relative Energy(cm^-1)
 0                   0.000000000              0.00000              0.00000
 1                   0.000000000              0.00000              0.00000
 2                   0.000613132              0.01668            134.56690
 3                   0.000613132              0.01668            134.56690
 4                   0.158189238              4.30455          34718.52000
 5                   0.158189238              4.30455          34718.52000
'''
