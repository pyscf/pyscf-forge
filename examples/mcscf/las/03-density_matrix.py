#!/usr/bin/env python

'''
Density matrices of a LAS wave function.

Methods whose names end in "_sub" return one density matrix per fragment, in the active orbitals
of that fragment. The others return density matrices of the whole active space, or of the whole
molecule in the AO basis, like the CASSCF methods of the same names.
'''

import numpy as np
from pyscf import gto, scf
from pyscf.mcscf import las

# C2H6N4: two H-N=N- groups joined by a CH2-CH2 bridge, with both N=N bonds stretched by 0.3
# Angstrom so that the N=N pi and pi* orbitals are the frontier orbitals of the RHF calculation
mol = gto.M (atom='''H -0.145525  2.756831  1.397652
                     N  0.586282  2.907265  0.655805
                     N  0.586282  1.768962 -0.376701
                     C -0.376894  0.666619 -0.222182
                     H -0.986448  0.788749  0.689250
                     H -1.039496  0.688384 -1.095107
                     C  0.376894 -0.666619 -0.222182
                     H  0.986448 -0.788749  0.689250
                     H  1.039496 -0.688384 -1.095107
                     N -0.586282 -1.768962 -0.376701
                     N -0.586282 -2.907265  0.655805
                     H  0.145525 -2.756831  1.397652''',
             basis='6-31g')
mf = scf.RHF (mol).run ()

# Triplet fragment 0 (M = +1) and singlet fragment 1
mc = las.LASSCF (mf, (2, 2), ((2, 0), (1, 1)), spin_sub=(3, 1))
mo0 = mc.localize_init_guess (([0, 1, 2], [9, 10, 11]), mf.mo_coeff)
mc.kernel (mo0)

# Fragment density matrices in the fragment active orbitals.
# casdm1s_sub[i] has shape (2, ncas_sub[i], ncas_sub[i]) (alpha, beta), and
# casdm2_sub[i] has shape (ncas_sub[i],)*4 (spin-summed, Mulliken order, as in CASSCF).
casdm1s_sub = mc.make_casdm1s_sub ()
casdm2_sub = mc.make_casdm2_sub ()
for i, (dm1s, dm2) in enumerate (zip (casdm1s_sub, casdm2_sub)):
    print ('Fragment {}: (nalpha, nbeta) = ({:.4f}, {:.4f}); dm2 shape = {}'.format (
        i, np.trace (dm1s[0]), np.trace (dm1s[1]), dm2.shape))
    print ('  natural occupation numbers:', np.linalg.eigvalsh (dm1s.sum (0))[::-1].round (4))

# Density matrices of the whole active space (4 orbitals here). The 1-RDM is block-diagonal by
# fragment, and in the 2-RDM, a block with two indices on each fragment is a product of fragment
# 1-RDMs, because the LAS wave function is a product of fragment wave functions.
casdm1 = mc.make_casdm1 ()
casdm2 = mc.make_casdm2 ()
dm1_0 = casdm1s_sub[0].sum (0)
dm1_1 = casdm1s_sub[1].sum (0)
print ('Whole active space: casdm1 shape = {}, casdm2 shape = {}'.format (casdm1.shape,
                                                                        casdm2.shape))
print ('casdm1 off-diagonal fragment block is zero:', np.allclose (casdm1[:2,2:], 0))
print ('casdm2 fragment (0,0,1,1) block is a product:',
       np.allclose (casdm2[:2,:2,2:,2:], np.multiply.outer (dm1_0, dm1_1)))

# Density matrices in the AO basis. make_rdm1s_sub and make_rdm1_sub give the AO-basis density
# of the active orbitals of each fragment; make_rdm1s and make_rdm1 give the density of the
# whole molecule, including the inactive orbitals.
dm1_sub = mc.make_rdm1_sub ()
dm1s = mc.make_rdm1s ()
print ('Electrons in the active orbitals of each fragment:',
       [round (float (np.trace (dm @ mf.get_ovlp ())), 4) for dm in dm1_sub])

# For instance, the Mulliken spin population shows that the unpaired electrons are on fragment 0
from pyscf.scf import uhf
spin_pop = uhf.mulliken_spin_pop (mol, dm1s, verbose=0)[1]
print ('Mulliken spin population by atom:', spin_pop.round (3))
