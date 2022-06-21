#!/usr/bin/env python

'''
Optimize the geometry of excited states using CASSCF-based MC-PDFT

Note when optiming the excited states, states may flip and this may cause
convergence issue in geometry optimizer.
'''

from pyscf import gto
from pyscf import scf, mcpdft

mol = gto.Mole()
mol.atom="N; N 1, 1.1"
mol.basis= "6-31g"
mol.build()

mf = scf.RHF(mol).run()

#
# 1. Ground state
#
mc = mcpdft.CASSCF(mf, 'tPBE', 4,4).run ()
excited_grad = mc.nuc_grad_method().as_scanner()
mol1 = excited_grad.optimizer().kernel()

#
# 2. Geometry optimization of the 3rd of 4 states
#
mc = mcpdft.CASSCF(mf, 'tPBE', 4, 4)
mc.state_average_([0.25, 0.25, 0.25, 0.25]).run ()
excited_grad = mc.nuc_grad_method().as_scanner(state=2)
mol1 = excited_grad.optimizer().kernel()

#
# 3. Geometry optimization of the triplet state
# In a triplet-singlet state average
#
import copy
mc = mcpdft.CASSCF(mf, 'tPBE', 4, 4)
solver1 = mc.fcisolver
solver2 = copy.copy(mc.fcisolver)
solver2.spin = 2
mc = mc.state_average_mix ([solver1, solver2], (.5, .5)).run ()
excited_grad = mc.nuc_grad_method().as_scanner(state=1)
mol1 = excited_grad.optimizer().kernel()



