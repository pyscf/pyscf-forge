#!/usr/bin/env python

'''
Save a LASSCF calculation to a chkfile, and load it again, e.g. to restart it.

At the end of the kernel, LASSCF writes its configuration (ncas_sub, nelecas_sub, ...), its state
list (weights, charges, spins, smults, wfnsyms), and its results (energies, orbitals and CI
vectors) to mc.chkfile, under the key 'las'. By default, mc.chkfile is the chkfile of the SCF
object. load_chk_ reads all of that back into a LAS object.
'''

from pyscf import gto, scf, lib
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
mc.chkfile = '13-load_chkfile.chk'
mo0 = mc.localize_init_guess (([0, 1, 2], [9, 10, 11]), mf.mo_coeff)
mc.kernel (mo0)
print ('E(LASSCF) =', mc.e_tot)

# Later, e.g. in another script: rebuild the molecule from the chkfile, and load the LASSCF
# results into a new LAS object. The SCF calculation does not need to be repeated: passing a Mole
# instead of an SCF object wraps it in an RHF object that is not run.
mol = lib.chkfile.load_mol ('13-load_chkfile.chk')
mc = las.LASSCF (mol, (2, 2), (2, 2))
mc.load_chk_ (chkfile='13-load_chkfile.chk')
print ('E(LASSCF) from the chkfile =', mc.e_tot)

# The loaded orbitals and CI vectors are used as the initial guess for a new kernel call, which
# converges immediately. The same works for restarting a calculation that did not converge.
mc.kernel ()
print ('E(LASSCF) restarted from the chkfile =', mc.e_tot)
