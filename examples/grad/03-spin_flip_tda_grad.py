#!/usr/bin/env/python

'''
Spin flip TDA (TDDFT) graditnt. The kwarg 'extype' is used to control which
kind of excited energy to be calculated, 0 for spin flip up, 1 for
spin flip down.

Note: pyscf.sftda_grad is implemented under the spin-flip TDDFT framework,
that can deal with both spin-flip TDA and TDDFT results. However, the Davi-
dson solver to spin-flip TDDFT hasn't been implemented, and no direct exam-
ples are displayed here.
'''

import numpy as np
from pyscf import lib, gto, dft
from pyscf import sftda
from pyscf.grad import tduks_sf # this import is necessary.
try:
    import mcfun # mcfun>=0.2.5 must be used.
except ImportError:
    mcfun = None

mol = gto.Mole()
mol.verbose = 6
mol.output = None
mol.spin = 2
mol.atom = 'O 0 0 2.07; O 0 0 0'
mol.unit = 'B'
mol.basis = '6-31g'
mol.build()

# UKS object
mf = dft.UKS(mol)
mf.xc = 'svwn' # blyp, b3lyp, tpss
mf.kernel()

# TDA_SF object
mftd1 = sftda.TDA_SF(mf)
mftd1.nstates = 5 # the number of excited states
mftd1.extype = 1  # 1 for spin flip down excited energies
mftd1.collinear_samples=200
mftd1.kernel()

# TDA_SF_Grad object
mftdg1 = mftd1.Gradients()
# mftdg1 = mftd1.nuc_grad_method() # or build object by it
g1 = mftdg1.kernel(state=2) # state=i means the i^th state

print(g1)
