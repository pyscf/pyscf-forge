#!/usr/bin/env python

'''
Frozen orbitals and frozen CI vectors in LASSCF.

frozen works as in CASSCF: the listed orbitals (indices into mo_coeff) are not rotated in the
orbital optimization. frozen_ci lists fragments whose CI vectors are not optimized: they keep
their initial guess, either the one passed to the kernel or the default one.
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
             basis='6-31g', verbose=4, output='17-frozen_orbital_ci.log')
mf = scf.RHF (mol).run ()
frag_atom_list = ([0, 1, 2], [9, 10, 11])

mc = las.LASSCF (mf, (2, 2), (2, 2))
mo0 = mc.localize_init_guess (frag_atom_list, mf.mo_coeff)
mc.kernel (mo0)
e_ref = mc.e_tot
print ('E(LASSCF) =', e_ref)

# 1. Freeze the 1s core orbitals of the N and C atoms (the 6 lowest orbitals). They stay exactly
#    as they are in mo0.
mc = las.LASSCF (mf, (2, 2), (2, 2), frozen=list (range (6)))
mc.kernel (mo0)
print ('E(LASSCF), frozen 1s cores =', mc.e_tot, '; change =', mc.e_tot - e_ref)
print ('Largest change in the frozen orbitals:', abs (mc.mo_coeff[:,:6] - mo0[:,:6]).max ())

# 2. Freeze the CI vector of fragment 1 at its default initial guess, the closed-shell (pi)^2
#    determinant, while fragment 0 and the orbitals are optimized. Fragment 1 stays uncorrelated.
mc = las.LASSCF (mf, (2, 2), (2, 2), frozen_ci=[1])
mc.kernel (mo0)
print ('E(LASSCF), fragment 1 CI frozen =', mc.e_tot, '; change =', mc.e_tot - e_ref)
print ('Fragment 1 natural occupation numbers:',
       np.linalg.eigvalsh (mc.make_casdm1s_sub ()[1].sum (0))[::-1].round (4))
