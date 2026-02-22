#!/usr/bin/env python

"""
Expection value of <S^2> and oscillator strengths between
excited states using spin-flip TDDFT/TDA
"""

import numpy as np
from pyscf import gto, sftda

atom = '''
 C                 -0.31474382    0.01213719   -0.02117853
 N                  0.19004517   -1.31047912    0.37476865
 O                  0.15651526    0.99794522    0.90131667
 H                 -0.13695077   -1.53063275    1.29379315
 H                 -0.13950663   -1.99985536   -0.27033289
 H                 -1.38468136    0.00166093   -0.01628742
 F                  0.12670070    0.30934460   -1.26186161
 H                 -0.17314286    1.86169465    0.64273892
'''
mol = gto.M(atom=atom, charge=0, spin=2, basis='6-31G*')

mf = mol.UKS(xc='B3LYP').run()
td = mf.TDDFT_SF()
td.extype = 1
td.collinear_samples = 20
td.nstates = 5
td.kernel()
td.analyze()

# 1. Spin Purity (S^2)
# The <S^2> value helps identify the spin multiplicity of the excited states.
print(td.spin_square(state=[1, 2, 3])) # state=[1, 2, 3] for the 1st, 2nd, and 3rd excited states

# 2. Oscillator Strengths between Excited States
# ref:   The "starting" state (Reference state).
#        Here ref=1 refers to the first excited state.
# state: The "target" states.
#        state=[2, 3] means we calculate transitions: 1 -> 2 and 1 -> 3.
print(td.oscillator_strength(ref=1, state=[2, 3]))
