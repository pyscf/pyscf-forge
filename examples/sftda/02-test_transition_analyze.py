#!/usr/bin/env/python

'''
Test transition_analyze function in sftda.tool_td.py.
'''

from pyscf import lib, gto, dft
from pyscf import sftda
from pyscf.sftda import tools_td

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

mftd1 = sftda.TDA_SF(mf)
mftd1.nstates = 5
mftd1.extype = 0
mftd1.collinear_samples=200
mftd1.kernel()

e = mftd1.e
xy = mftd1.xy

#
# 1. Print tansition analyze for the first spin-flip up excited state.
#
tools_td.transition_analyze(mf, mftd1, e[0], xy[0], tdtype='TDA')


mftd2 = sftda.uks_sf.TDA_SF(mf)
mftd2.nstates = 5
mftd2.extype = 1
mftd2.collinear_samples=200
mftd2.kernel()

#
# 2. Print tansition analyze for the first spin-flip down excited state.
#
tools_td.transition_analyze(mf, mftd1, e[0], xy[0], tdtype='TDA')