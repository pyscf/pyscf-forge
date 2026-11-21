#!/usr/bin/env python

'''
Write LASSCF orbitals to a molden file.

At the end of a LASSCF calculation (with the default canonicalization = True), the orbitals in
mc.mo_coeff are the natural orbitals of each fragment's active space, in order of fragment, and
canonical inactive and virtual orbitals. The canonicalize method returns them together with
their energies and occupation numbers, which is all that molden.from_mo needs.
'''

from pyscf import gto, scf
from pyscf.mcscf import las
from pyscf.tools import molden

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
             basis='6-31g')
mf = scf.RHF (mol).run ()

mc = las.LASSCF (mf, (2, 2), (2, 2))
mo0 = mc.localize_init_guess (([0, 1, 2], [9, 10, 11]), mf.mo_coeff)
mc.kernel (mo0)

# LAS "canonicalize" always generates natural orbitals in the active space, which cannot mix
# orbitals between different fragments.
mo_coeff, mo_energy, mo_occ = mc.canonicalize ()[:3]
# mo_occ is 2 for inactive orbitals, the natural occupation numbers for active orbitals, and 0
# for virtual orbitals. mo_energy holds the Fock-matrix eigenvalues of the inactive and virtual
# orbitals, and zero for the active orbitals.
molden.from_mo (mol, '04-molden.molden', mo_coeff, ene=mo_energy, occ=mo_occ)

# The active orbitals of fragment i are columns i0:i1 of mo_coeff below. To write only one
# fragment's active orbitals, e.g. to look at them in a viewer:
for i in range (mc.nfrags):
    i0 = mc.ncore + sum (mc.ncas_sub[:i])
    i1 = i0 + mc.ncas_sub[i]
    molden.from_mo (mol, '04-molden_frag{}.molden'.format (i), mo_coeff[:,i0:i1],
                    ene=mo_energy[i0:i1], occ=mo_occ[i0:i1])
    print ('Fragment {} natural occupation numbers:'.format (i), mo_occ[i0:i1].round (4))
