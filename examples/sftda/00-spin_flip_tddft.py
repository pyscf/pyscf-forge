#!/usr/bin/env/python

'''
Spin flip TDA/ TDDFT. The kwarg 'extype' is used to control which
kind of excited energy to be calculated, 0 for spin flip up, 1 for
spin flip down.
'''

from pyscf import lib, gto, dft
from pyscf import sftda

mol = gto.Mole()
mol.verbose = 6
mol.output = None
mol.spin = 2
mol.atom = 'O 0 0 2.07; O 0 0 0'
mol.unit = 'B'
mol.basis = '631g'
mol.build()

mf = dft.UKS(mol)
mf.xc = 'svwn' # blyp, b3lyp, tpss
mf.kernel()

#
# 1. spin flip up TDA
#
mftd1 = sftda.TDA_SF(mf)
mftd1.nstates = 5 # the number of excited states
mftd1.extype = 0 # 0 for spin flip up excited energies
# the spin sample points in multicollinear approach, which
# can be increased by users.
mftd1.collinear_samples=200
mftd1.kernel()

mftd1.e # to get the excited energies
mftd1.xy # to get the transition vectors

#
# 2. spin flip down TDA
#
mftd2 = sftda.uks_sf.TDA_SF(mf)
mftd2.nstates = 5 # the number of excited states
mftd2.extype = 1 # 1 for spin flip down excited energies
mftd2.collinear_samples=200
mftd2.kernel()

mftd2.e # to get the excited energies
mftd2.xy # to get the transition vectors

#
# 3. spin flip up TDDFT, which can not converged.
#
mftd3 = sftda.TDDFT_SF(mf) # equal to mftd3 = mf.TDDFT_SF()
mftd3.nstates = 4
mftd3.extype = 0
mftd3.collinear_samples=200
mftd3.kernel()

mftd3.e
mftd3.xy

#
# 4. spin flip down TDDFT, which can not converged.
#
mftd4 = sftda.uks_sf.CasidaTDDFT(mf)
mftd4.nstates = 4
mftd4.extype = 1
mftd4.collinear_samples=200
mftd4.kernel()

mftd4.e
mftd4.xy

#
# 5. get_ab_sf()
# Besides, users can use get_ab_sf() to construct the whole TDDFT matrix
#          to get all excited energies, if the system is small.
# a, b = sftda.TDA_SF(mf).get_ab_sf()
a, b = sftda.TDDFT_SF(mf).get_ab_sf()
# List a has two items: (A_baba,A_abab) with A[i,a,j,b].
# List b has two items: (B_baab,B_abba) with B[i,a,j,b].
