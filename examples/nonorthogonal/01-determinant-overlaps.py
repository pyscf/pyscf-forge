#!/usr/bin/env python

'''
Overlap and Hamiltonian matrix elements between excitations of two
nonorthogonal reference determinants (RHF and broken-symmetry UHF).
'''

import numpy
from pyscf import gto, scf, nonorthogonal

mol = gto.M(atom='H 0 0 0; H 0 0 2.0', basis='cc-pvdz', verbose=0)

mf0 = scf.RHF(mol).run()
# Broken-symmetry UHF: localise the bonding and antibonding RHF orbitals
# onto opposite atoms for the two spins
mo = mf0.mo_coeff
left = (mo[:, 0] + mo[:, 1]) / 2**.5
right = (mo[:, 0] - mo[:, 1]) / 2**.5
mf1 = scf.UHF(mol)
mf1.kernel((numpy.outer(left, left), numpy.outer(right, right)))

wicks = nonorthogonal.WickEngine(mol, [mf0, mf1])

# The unexcited references
d0 = nonorthogonal.Determinant(0)
d1 = nonorthogonal.Determinant(1)
# An alpha HOMO -> LUMO single excitation of the UHF reference
d2 = nonorthogonal.Determinant(1, alpha_holes=[0], alpha_particles=[1])

print('<RHF|UHF>         =', wicks.overlap(d0, d1))
h, s = wicks.matrix_element(d0, d2)
print('<RHF|H|UHF_0^1>   = %.10f   <RHF|UHF_0^1> = %.10f' % (h, s))
print('<UHF|H|UHF>       = %.10f   E(UHF)        = %.10f' %
      (wicks.matrix_element(d1, d1)[0], mf1.e_tot))

H, S = wicks.build_matrices([d0, d1, d2])
print('H =\n', H)
print('S =\n', S)
