#!/usr/bin/env python

'''
Example script for native maximally localized Wannier functions in PBC.
'''

import numpy

from pyscf.pbc import gto, scf
from pyscf.pbc.lo import mlwf


cell = gto.Cell()
cell.unit = 'Bohr'
cell.atom = 'H 0 0 0; H 1.5 0 0'
cell.a = numpy.eye(3) * 8.0
cell.basis = 'sto-3g'
cell.verbose = 0
cell.build()

kpts = cell.make_kpts([2, 2, 2])
kmf = scf.KRHF(cell, kpts=kpts)
kmf.conv_tol = 1e-10
kmf.kernel()

proj_guess = [
    {'center': (0.0, 0.0, 0.0), 'l': 0, 'm': 0, 'zeta': 1.0},
    {'center': (1.5, 0.0, 0.0), 'l': 0, 'm': 0, 'zeta': 1.0},
]

mo_coeff, centers, spreads, omega_i, omega_d, omega_od, converged = mlwf.kernel(
    kmf, proj_guess, max_iter=80, conv_tol=1e-12)

print('MLWF converged:', converged)
print('Localized coefficient shape:', mo_coeff.shape)
print('Centers / Bohr')
print(centers)
print('Spreads / Bohr^2')
print(spreads)

assert mo_coeff.shape == (len(kpts), cell.nao_nr(), len(proj_guess))
assert abs(omega_i + omega_d + omega_od - spreads.sum()) < 1e-8
