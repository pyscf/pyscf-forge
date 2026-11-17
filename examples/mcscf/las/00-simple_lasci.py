#!/usr/bin/env python

'''
A simple example to run a LASCI calculation.

In a localized active space (LAS) calculation, the active space is divided into fragments and the
wave function is a product of one CI vector per fragment. LASCI optimizes those CI vectors, each
in the mean field of the others, at fixed orbitals. See 00-simple_lasscf.py to optimize the
orbitals as well.
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
             basis='6-31g', verbose=4)
mf = scf.RHF (mol).run ()

# Two fragments, each with 2 active orbitals (pi and pi* of N=N) and 2 active electrons
mc = las.LASCI (mf, (2, 2), (2, 2))

# The LAS object doesn't know which atoms belong to which fragment. The active orbitals are only
# "localized" if the initial orbitals are. localize_init_guess projects the active orbitals of
# mf.mo_coeff onto the atoms of each fragment; see 01-localize_init_guess.py.
frag_atom_list = ([0, 1, 2], [9, 10, 11])
mo0 = mc.localize_init_guess (frag_atom_list, mf.mo_coeff)
mc.kernel (mo0)
print ('E(LASCI) =', mc.e_tot)

# The CI vectors are stored per fragment, then per state: mc.ci[ifrag][istate]
print ('Shape of the CI vector of fragment 0:', mc.ci[0][0].shape)
