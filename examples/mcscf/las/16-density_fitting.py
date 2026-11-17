#!/usr/bin/env python

'''
Density fitting in LASCI and LASSCF.

If the SCF object uses density fitting (has with_df), las.LASCI and las.LASSCF use the same
density-fitting object for the LAS calculation. A LAS object built from a conventional SCF object
can also be given its own density fitting with pyscf.mcscf.las.lasci.density_fit. The
density-fitting error in the LASSCF energy is similar to that in the SCF energy.

LASSCF analytical nuclear gradients are not available with density fitting.
'''

from pyscf import gto, scf, df
from pyscf.mcscf import las
from pyscf.mcscf.las.lasci import density_fit

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
frag_atom_list = ([0, 1, 2], [9, 10, 11])

# Reference: no density fitting
mf = scf.RHF (mol).run ()
mc = las.LASSCF (mf, (2, 2), (2, 2))
mc.kernel (mc.localize_init_guess (frag_atom_list, mf.mo_coeff))
e_ref = mc.e_tot
print ('E(LASSCF) =', e_ref)

# 1. Density-fitted SCF (default auxiliary basis): the LAS object inherits mf.with_df
mf_df = scf.RHF (mol).density_fit ().run ()
print ('DF error in E(RHF) =', mf_df.e_tot - mf.e_tot)
mc = las.LASSCF (mf_df, (2, 2), (2, 2))
mc.kernel (mc.localize_init_guess (frag_atom_list, mf_df.mo_coeff))
print ('E(DF-LASSCF), DF SCF =', mc.e_tot, '; error =', mc.e_tot - e_ref)

# 2. Conventional SCF, density-fitted LAS only, with its own DF object
mc = density_fit (las.LASSCF (mf, (2, 2), (2, 2)),
                  with_df=df.DF (mol, auxbasis='def2-universal-jkfit'))
mc.kernel (mc.localize_init_guess (frag_atom_list, mf.mo_coeff))
print ('E(DF-LASSCF), conventional SCF =', mc.e_tot, '; error =', mc.e_tot - e_ref)
