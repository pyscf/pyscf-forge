import numpy as np
from pyscf import gto, scf, mcpdft, mcscf
from pyscf.mcscf import avas
from pyscf import siso
from pyscf.siso.anisoaddons import generate_aniso_data, write_aniso_file


# Example to perform the two-step state interaction calculation followed by generating
# the data for the single_aniso module. That module can be used to compute various magnetic
# properties such as g-tensors, zero-field splitting parameters, etc.

# 1. Define the molecule
mol = gto.Mole()
mol.atom='''
Dy 0 0 0
O 0 0 1.74'''
mol.basis= {'Dy': 'ano@7s6p4d2f1g', 'O': 'ano@3s2p1d'}
mol.spin = 5
mol.charge = 1
mol.verbose = 4
mol.max_memory = 120000
mol.output = 'DyO.out'
mol.build()

# 2. SCF calculation
mf = scf.RHF(mol).sfx2c1e().density_fit()
mf.chkfile='DyO.chk'
mf.init_guess = 'atom'
mf.kernel()

# 3. Active space selection via AVAS:
mo_coeff = avas.kernel(mf, ['Dy 4f',], minao=mol.basis)[2]

# 4. State-averaged CASSCF followed by state interaction:
modelspace = [(21,6),]
mc = mcscf.CASSCF(mf, 7, 9)
mc = siso.sacasscf_solver(mc, modelspace) # 21 Sextets
mc.max_cycle_macro = 200
mc.kernel(mo_coeff)

# 5. State interaction: SO-CASSCF, similar to previous examples 
# (00-qdptsoc_for_cas.py, 01-qdptsoc_for_mcpdft.py, 02-qdptsoc_for_lpdft.py)
# one can also use different methods for above calculation.
mysiso = siso.SISO(mc, ham='DKH', amf=True)
mysiso.kernel()

# 6. Generate the data for the single_aniso module:
mydata = generate_aniso_data(
    mol, mc, mysiso=mysiso, origin='CHARGE_CENTER', ham='DKH')
write_aniso_file('DyO.aniso', data = mydata, backend='Orca')

# 7. To run the single_aniso module, you will need to create the input file for the single_aniso, in addition to
# the aniso data file created above (DyO.aniso) and binaries of the single_aniso module. Currently you can access these via
# OpenMolcas or Orca.
# For more details on how to use SingleANISO: see: https://www.faccts.de/docs/orca/6.0/manual/contents/detailed/single_aniso.html
# https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/single_aniso.html
