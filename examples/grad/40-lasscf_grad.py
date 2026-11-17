#!/usr/bin/env python

'''
LASSCF analytical nuclear gradients

The gradients are available for single-state LASSCF without density fitting (see
pyscf/mcscf/las and examples/mcscf/las for LASSCF itself).
'''

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
             basis='6-31g', verbose=4)
mf = scf.RHF (mol).run ()

mc = las.LASSCF (mf, (2, 2), (2, 2))
mo0 = mc.localize_init_guess (([0, 1, 2], [9, 10, 11]), mf.mo_coeff)
mc.kernel (mo0)

# Analytical gradient, in Hartree/Bohr, with shape (natm, 3)
de = mc.nuc_grad_method ().kernel ()

# Check one component against a finite difference. The LASSCF scanner starts each calculation
# from the orbitals and CI vectors of the previous one.
mc_scanner = mc.as_scanner ()
coords = mol.atom_coords ()  # Bohr
delta = 1e-4
e = []
for sign in (1, -1):
    x = coords.copy ()
    x[1,2] += sign * delta
    e.append (mc_scanner (mol.set_geom_ (x, unit='Bohr', inplace=False)))
print ('dE/dz of atom 1: analytical = {:.6f}, finite difference = {:.6f}'.format (
    de[1,2], (e[0]-e[1]) / (2*delta)))
