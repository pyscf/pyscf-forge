#!/usr/bin/env python

'''
A simple example to run a LASSCF calculation.

LASSCF optimizes the orbitals as well as the CI vectors of each fragment (see 00-simple_lasci.py
for LASCI). The orbital optimization includes rotations between the active orbitals of
different fragments, but not within a fragment.
'''

from pyscf import gto, scf, mcscf
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
mc = las.LASSCF (mf, (2, 2), (2, 2))
frag_atom_list = ([0, 1, 2], [9, 10, 11])
mo0 = mc.localize_init_guess (frag_atom_list, mf.mo_coeff)
mc.kernel (mo0)
print ('E(LASSCF) =', mc.e_tot)

# For comparison, CASSCF with the same total active space. Since the LAS wave function is a
# product of fragment wave functions, it is a variational approximation to this one.
mc_cas = mcscf.CASSCF (mf, 4, 4)
mc_cas.kernel (mc.mo_coeff)
print ('E(CASSCF) =', mc_cas.e_tot)
