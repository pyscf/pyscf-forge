#!/usr/bin/env python

'''
Gradients for multi-state PDFT
'''

from pyscf import gto, scf, mcpdft

mol = gto.M(
    atom = [
        ['Li', ( 0., 0.    , 0.   )],
        ['H', ( 0., 0., 1.7)]
    ], basis = 'sto-3g',
    symmetry = 0 # symmetry enforcement is not recommended for MS-PDFT
    )

mf = scf.RHF(mol)
mf.kernel()

mc = mcpdft.CASSCF(mf, 'tpbe', 2, 2)
mc.fix_spin_(ss=0) # often necessary!
mc = mc.multi_state ([.5,.5]).run ()

mc_grad = mc.nuc_grad_method ()
de0 = mc_grad.kernel (state=0)
de1 = mc_grad.kernel (state=1)
print ("Gradient of ground state:\n",de0)
print ("Gradient of first singlet excited state:\n",de1)

