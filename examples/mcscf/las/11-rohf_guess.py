#!/usr/bin/env python

'''
Build a LAS initial guess that contains a high-spin ROHF wave function, so that the LASCI energy
is bounded from above by the ROHF energy.

A high-spin ROHF determinant is a LAS product state if each fragment's active orbitals are
spanned by its own ROHF orbitals, with the right number of doubly and singly occupied ones. The
default localize_init_guess mixes active orbitals with inactive and virtual ones (and singly with
doubly occupied ones), which breaks this; the kwargs below preserve it.
'''

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
             basis='6-31g', spin=4, verbose=4, output='11-rohf_guess.log')
mf = scf.ROHF (mol).run ()
print ('E(ROHF) =', mf.e_tot)
frag_atom_list = ([0, 1, 2], [9, 10, 11])

# 1. Only singly occupied orbitals in the active space: each fragment has the pi and pi* orbitals
#    of its N=N bond, both singly occupied (a triplet fragment), and the 4 active orbitals are the
#    4 singly occupied ROHF orbitals. freeze_cas_spaces=True mixes the active orbitals only among
#    themselves, so the ROHF determinant is in the LAS space and LASCI can only go down from it.
mc = las.LASCI (mf, (2, 2), ((2, 0), (2, 0)), spin_sub=(3, 3))
mo0 = mc.localize_init_guess (frag_atom_list, mf.mo_coeff, freeze_cas_spaces=True)
mc.kernel (mo0)
print ('E(LASCI), SOMOs only, freeze_cas_spaces=True =', mc.e_tot)

# 2. Doubly occupied and empty orbitals in the active space too: add one doubly occupied and one 
#    virtual orbital to each fragment, (4 orbitals, 3 alpha + 1 beta electrons).
#    freeze_cas_spaces=True keeps the active space as a whole, but it lets the localization mix
#    ROHF orbitals with different occupancies, so the guess no longer contains the ROHF
#    determinant.
mc = las.LASCI (mf, (4, 4), ((3, 1), (3, 1)), spin_sub=(3, 3))
mo0 = mc.localize_init_guess (frag_atom_list, mf.mo_coeff, freeze_cas_spaces=True)
mc.kernel (mo0)
print ('E(LASCI), with doubly occupied, freeze_cas_spaces=True =', mc.e_tot)

# 3. Passing mo_occ=mf.mo_occ only mixes orbitals with the same ROHF occupation (and implies
#    freeze_cas_spaces=True). With the 'aufbau' CI guess, which occupies the orbitals of each
#    fragment in order, the initial LAS state is the ROHF determinant.
mc = las.LASCI (mf, (4, 4), ((3, 1), (3, 1)), spin_sub=(3, 3))
mc.init_guess_ci = 'aufbau'
mo0 = mc.localize_init_guess (frag_atom_list, mf.mo_coeff, mo_occ=mf.mo_occ)
mc.kernel (mo0)
print ('E(LASCI), with doubly occupied, mo_occ=mf.mo_occ =', mc.e_tot)
