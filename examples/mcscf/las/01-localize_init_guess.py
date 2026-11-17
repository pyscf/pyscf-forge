#!/usr/bin/env python

'''
Use localize_init_guess to build localized initial guess orbitals for LASSCF.

The LAS object has no notion of which atoms belong to which fragment. It only knows the number of
active orbitals and electrons in each fragment, and it treats the active orbitals of the initial
guess in sequence: columns ncore:ncore+ncas_sub[0] of mo_coeff are the active orbitals of
fragment 0, the next ncas_sub[1] columns are those of fragment 1, and so on. The active space is
"localized" only if those orbitals are actually localized on the fragments.

localize_init_guess takes the active orbitals of the mo_coeff that you pass, projects them onto
the AOs of each fragment, and rebuilds the inactive and virtual orbitals in the orthogonal
complement. The result is only as good as the active orbitals you start from: mo_coeff[:,
ncore:nocc] must span (approximately) the orbitals you want to localize. If the frontier orbitals
of the mean-field calculation are not the ones you want, reorder them first, for instance with
mcscf.sort_mo, or select them with AVAS (see 05-avas.py), as in CASSCF. Here, at the equilibrium
geometry, the two highest occupied RHF orbitals would be N lone pairs rather than the N=N pi
orbitals; stretching the N=N bonds slightly brings the pi orbitals to the frontier.
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

# 1. Fragments as lists of atom indices. The first list goes with ncas_sub[0], and so on.
frag_atom_list = ([0, 1, 2], [9, 10, 11])
mc = las.LASSCF (mf, (2, 2), (2, 2))
mo0 = mc.localize_init_guess (frag_atom_list, mf.mo_coeff)
mc.kernel (mo0)
print ('E(LASSCF), fragments by atom index =', mc.e_tot)

# 2. Fragments as lists of AO labels, which are passed to mol.search_ao_label. This lets you
#    choose a subset of the AOs of an atom. Here, the active orbitals of each fragment are built
#    only from the inner 2p AOs of its two N atoms. (The outer valence p shell of 6-31G is
#    labeled 3p; print mol.ao_labels () to see all labels.)
frag_ao_list = (['1 N 2p', '2 N 2p'], ['9 N 2p', '10 N 2p'])
mc = las.LASSCF (mf, (2, 2), (2, 2))
mo0 = mc.localize_init_guess (frag_ao_list, mf.mo_coeff)
mc.kernel (mo0)
print ('E(LASSCF), fragments by AO label =', mc.e_tot)

# 3. By default, the AOs are orthonormalized with meta-Lowdin before the projection. Other
#    localized orthonormal orbitals can be passed as lo_coeff, e.g. IAOs completed with PAOs.
from pyscf import lo
lo_coeff = lo.orth_ao (mol, 'meta_lowdin')
mo0 = mc.localize_init_guess (frag_atom_list, mf.mo_coeff, lo_coeff=lo_coeff)

# 4. freeze_cas_spaces=True mixes the active orbitals only among themselves, which leaves the
#    inactive and virtual orbitals of mo_coeff unchanged. The fragment orbitals are then less
#    localized, but no inactive or virtual character is mixed into the active space. See
#    11-rohf_guess.py for a case where this matters.
mc = las.LASSCF (mf, (2, 2), (2, 2))
mo0 = mc.localize_init_guess (frag_atom_list, mf.mo_coeff, freeze_cas_spaces=True)
mc.kernel (mo0)
print ('E(LASSCF), freeze_cas_spaces=True =', mc.e_tot)

# All three guesses converge to the same LASSCF solution in this simple case. The choice matters
# more when the fragments are close together or when the frontier orbitals are delocalized.
