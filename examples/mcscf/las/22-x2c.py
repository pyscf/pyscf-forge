#!/usr/bin/env python

'''
Scalar-relativistic LASSCF with the spin-free X2C Hamiltonian.

LASCI and LASSCF get the one-electron Hamiltonian from the SCF object (mf.get_hcore), so using an
X2C SCF object is all that is needed, as for CASSCF.
'''

import numpy as np
from pyscf import gto, scf
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
             basis='6-31g', verbose=4, output='22-x2c.log')
frag_atom_list = ([0, 1, 2], [9, 10, 11])

# Non-relativistic reference
mf = scf.RHF (mol).run ()
mc = las.LASSCF (mf, (2, 2), (2, 2))
mc.kernel (mc.localize_init_guess (frag_atom_list, mf.mo_coeff))
e_nr = mc.e_tot

# Spin-free X2C
mf_x2c = scf.RHF (mol).x2c ().run ()
mc = las.LASSCF (mf_x2c, (2, 2), (2, 2))
print ('LAS uses the X2C one-electron Hamiltonian:',
       np.allclose (mc.get_hcore (), mf_x2c.get_hcore ()))
mc.kernel (mc.localize_init_guess (frag_atom_list, mf_x2c.mo_coeff))
print ('E(LASSCF), non-relativistic =', e_nr)
print ('E(LASSCF), sf-X2C           =', mc.e_tot)
print ('E(sf-X2C RHF) - E(RHF)       =', mf_x2c.e_tot - mf.e_tot)
print ('E(sf-X2C LASSCF) - E(LASSCF) =', mc.e_tot - e_nr)
