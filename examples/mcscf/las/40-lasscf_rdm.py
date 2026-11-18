#!/usr/bin/env python

'''
LASSCF with fragment solvers that return density matrices instead of CI vectors.

pyscf.mcscf.las.lasscf_rdm.LASSCF alternates between an orbital step and a call to each
fragment's solver, and it only needs each solver's energy and 1- and 2-RDMs. Any function with the
call and return signature of my_kernel below can be plugged in, e.g. to use a solver that does not
produce a CI vector in PySCF's format (a DMRG, selected-CI, or quantum-computing solver). Without a
kernel, the default solver is a CSF-basis FCI solver.

Only single-state calculations without point-group symmetry are supported. Since it does not
optimize CI vectors and orbitals together, this algorithm usually needs more cycles than LASSCF.
'''

import numpy as np
from pyscf import gto, scf, fci, ao2mo
from pyscf.mcscf import las
from pyscf.mcscf.las import lasscf_rdm

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
frag_atom_list = ([0, 1, 2], [9, 10, 11])

# Reference: the CI-vector LASSCF algorithm
mc = las.LASSCF (mf, (2, 2), (2, 2))
mo0 = mc.localize_init_guess (frag_atom_list, mf.mo_coeff)
mc.kernel (mo0)
print ('E(LASSCF), CI algorithm =', mc.e_tot)

# 1. RDM algorithm with the default solver
mc_rdm = lasscf_rdm.LASSCF (mf, (2, 2), (2, 2))
mc_rdm.kernel (mo0)
print ('E(LASSCF), RDM algorithm, default solver =', mc_rdm.e_tot)

# 2. RDM algorithm with a user-defined solver. The arguments are the number of orbitals and
#    electrons of the fragment, and the fragment's effective Hamiltonian: the constant h0, the
#    spin-separated 1-electron part h1s with shape (2, norb, norb), and the 2-electron part h2
#    with shape (norb,)*4. It must return the energy (including h0), the spin-separated 1-RDM
#    with shape (2, norb, norb), and the spin-summed 2-RDM with shape (norb,)*4, in the same
#    (Mulliken) index order as CASSCF's 2-RDM. Here, the solver is PySCF's spin-unrestricted FCI.
def my_kernel (norb, nelec, h0, h1s, h2):
    solver = fci.direct_uhf.FCISolver (mol)
    e, ci = solver.kernel ((h1s[0], h1s[1]), (h2, h2, h2), norb, nelec, ecore=h0)
    (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = solver.make_rdm12s (ci, norb, nelec)
    dm1s = np.stack ([dm1a, dm1b], axis=0)
    dm2 = dm2aa + dm2ab + dm2ab.transpose (2, 3, 0, 1) + dm2bb
    return e, dm1s, dm2

# 2a. get_ham_eff_sub returns the effective Hamiltonian of each fragment, (h0, h1, h2), exactly as
#     the LASSCF kernel passes it to the fragment solvers. h0 and h1 have one element per state;
#     h2 is packed with 8-fold permutation symmetry. At the converged LASSCF solution, solving each
#     fragment's effective Hamiltonian with my_kernel reproduces the LASSCF fragment RDMs.
casdm1s_sub = mc.make_casdm1s_sub ()
casdm2_sub = mc.make_casdm2_sub ()
for ifrag, (h0, h1, h2) in enumerate (mc.get_ham_eff_sub ()):
    norb = mc.ncas_sub[ifrag]
    h2 = ao2mo.restore (1, h2, norb)
    e, dm1s, dm2 = my_kernel (norb, (1, 1), h0[0], h1[0], h2)
    print ('Fragment {}: my_kernel reproduces the LASSCF 1-RDM: {}; 2-RDM: {}'.format (
        ifrag, np.allclose (dm1s, casdm1s_sub[ifrag], atol=1e-6),
        np.allclose (dm2, casdm2_sub[ifrag], atol=1e-6)))

# 2b. Plug my_kernel into the RDM algorithm
mc_rdm = lasscf_rdm.LASSCF (mf, (2, 2), (2, 2))
# make_fcibox wraps the function for one fragment
mc_rdm.fciboxes = [lasscf_rdm.make_fcibox (mol, kernel=my_kernel)
                   for ifrag in range (2)]
mc_rdm.kernel (mo0)
print ('E(LASSCF), RDM algorithm, user-defined solver =', mc_rdm.e_tot)

# The fragment RDMs are stored on the object, as lists with one element per fragment:
# casdm1frs[i] has shape (1, 2, norb, norb) and casdm2fr[i] has shape (1, norb, norb, norb, norb)
print ('Shapes of the fragment 0 RDMs:', mc_rdm.casdm1frs[0].shape, mc_rdm.casdm2fr[0].shape)
