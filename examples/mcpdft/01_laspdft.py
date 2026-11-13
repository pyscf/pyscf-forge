#!/usr/bin/env python

'''
LAS-PDFT: multiconfiguration pair-density functional theory on a localized active space (LASSCF or
LASCI) wave function.

The LAS-PDFT constructors live in pyscf.mcpdft.laspdft, in PySCF-Forge. Importing that module
also adds them to the pyscf.mcpdft namespace, next to mcpdft.CASSCF and mcpdft.CASCI, so the
import below is required even though the name laspdft is not used afterwards.

(mrh's mrh.my_pyscf.mcpdft has its own LASSCF constructor, which also handles periodic systems,
asynchronous LASSCF and LASSI-PDFT. Importing it also makes mcpdft.LASSCF available.)
'''

from pyscf import gto, scf, mcpdft
from pyscf.mcscf import las
from pyscf.mcpdft import laspdft # Required: adds mcpdft.LASSCF and mcpdft.LASCI

# Two H2 molecules, far apart, one fragment each
mol = gto.M (atom='''H 0.0 0.0 0.0
                     H 0.0 0.0 0.74
                     H 3.0 0.0 0.0
                     H 3.0 0.0 0.74''',
             basis='6-31g', verbose=4, output='01_laspdft.log')
mf = scf.RHF (mol).run ()
frag_atom_list = ([0, 1], [2, 3])

# Option 1: LAS-PDFT from scratch. The arguments after the functional are those of las.LASSCF.
mc = mcpdft.LASSCF (mf, 'tPBE', (2, 2), (2, 2))
mo0 = mc.localize_init_guess (frag_atom_list)
mc.kernel (mo0)
print ('E(LASSCF)   =', mc.e_mcscf)
print ('E(LAS-tPBE) =', mc.e_tot)

# Option 2: LAS-PDFT on an existing LASSCF calculation; its orbitals and CI vectors are copied
my_las = las.LASSCF (mf, (2, 2), (2, 2))
my_las.kernel (my_las.localize_init_guess (frag_atom_list))
mc = mcpdft.LASSCF (my_las, 'tPBE')
mc.compute_pdft_energy_ ()
print ('E(LAS-tPBE) from an existing LASSCF =', mc.e_tot)

# LASCI-PDFT: the CI vectors are optimized at fixed orbitals (here, the LASSCF ones)
mc = mcpdft.LASCI (mf, 'tPBE', (2, 2), (2, 2))
mc.kernel (my_las.mo_coeff)
print ('E(LASCI-tPBE) at the LASSCF orbitals =', mc.e_tot)
