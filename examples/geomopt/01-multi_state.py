#!/usr/bin/env python

'''
Geometry optimization for multi-state PDFT
'''

from pyscf import gto, scf, mcpdft
from pyscf.data.nist import BOHR
from scipy import linalg

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

mc_opt = mc_grad.as_scanner (state=1).optimizer ()
carts = mc_opt.kernel ().atom_coords ()
rLiH1 = linalg.norm (carts[1]-carts[0]) * BOHR

mc_opt = mc_grad.as_scanner (state=0).optimizer ()
carts = mc_opt.kernel ().atom_coords ()
rLiH0 = linalg.norm (carts[1]-carts[0]) * BOHR

print ("Equilibrium distance of LiH S0 state:", round (rLiH0, 2), 'Angstrom')
print ("Equilibrium distance of LiH S1 state:", round (rLiH1, 2), 'Angstrom')
print ("(S1 potential energy surface is very flat; precision here is not great)")
