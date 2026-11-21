#!/usr/bin/env python

'''
Point-group symmetry in LASSCF.

If mol.symmetry is set, las.LASCI and las.LASSCF return classes that use point-group symmetry.
The fragments must not break the point group: every operation of mol.groupname must map each
fragment onto itself, so that all of the active orbitals of each fragment can be labeled by
irreps of mol.groupname. A point group whose operations exchange fragments cannot be used, even
if it is the full point group of the molecule. In that case, set mol.symmetry to the largest
subgroup that leaves every fragment unchanged.
'''

from pyscf import gto, scf, symm
from pyscf.mcscf import las

# C2H4N4 at its equilibrium geometry (see 05-avas.py). Its full point group is C2h, but the C2
# axis and the inversion center exchange the two H-N=N-CH fragments. Only the molecular plane
# maps each fragment onto itself, so use the subgroup Cs. With symmetry=True or symmetry='C2h',
# localize_init_guess fails, because the AOs of one fragment do not span a C2h-symmetric space.
mol = gto.M (atom='''H -2.772250 -1.103953 0.0
                     N -2.875835 -0.058207 0.0
                     N -1.761797  0.508543 0.0
                     C -0.604255 -0.293583 0.0
                     H -0.704824 -1.383428 0.0
                     C  0.604255  0.293583 0.0
                     H  0.704824  1.383428 0.0
                     N  1.761797 -0.508543 0.0
                     N  2.875835  0.058207 0.0
                     H  2.772250  1.103953 0.0''',
             basis='6-31g', symmetry='Cs', verbose=4, output='15-symmetry.log')
print ('Point group used:', mol.groupname, '(full point group: {})'.format (mol.topgroup))
mf = scf.RHF (mol).run ()

# Each fragment gets the sigma, pi, pi* and sigma* orbitals of its N=N bond: 4 orbitals and 4
# electrons. The canonical RHF orbitals are delocalized over both fragments, so each of these
# appears as a pair of near-degenerate orbitals (in-phase and out-of-phase combinations of the two
# fragments). They are not at the frontier, so pick them with sort_mo, which works for LAS objects
# the same way as for CASSCF (1-based indices):
#   13, 14: N=N sigma (A')    16, 18: N=N pi (A")
#   23, 24: N=N pi* (A")      33, 34: N=N sigma* (A')
# Use mf.analyze () or a molden file (see 04-molden.py) to identify orbitals like these. With
# poorly chosen orbitals, localize_init_guess can produce fragments that LASSCF struggles to
# converge.
mc = las.LASSCF (mf, (4, 4), (4, 4))
mo = mc.sort_mo ([13, 14, 16, 18, 23, 24, 33, 34], mf.mo_coeff)
mo0 = mc.localize_init_guess (([0, 1, 2, 3, 4], [5, 6, 7, 8, 9]), mo)
mc.kernel (mo0)
print ('E(LASSCF, Cs) =', mc.e_tot)

# The orbitals carry irrep labels, as in CASSCF with symmetry. Each fragment has two A' (sigma)
# and two A" (pi) active orbitals.
orbsym = mc.mo_coeff.orbsym
for i in range (mc.nfrags):
    i0 = mc.ncore + sum (mc.ncas_sub[:i])
    i1 = i0 + mc.ncas_sub[i]
    print ('Fragment {} active orbital irreps:'.format (i),
           [symm.irrep_id2name (mol.groupname, x) for x in orbsym[i0:i1]])

# wfnsym_sub sets the irrep of each fragment's CI vector, like wfnsym in CASSCF. For instance,
# a singlet sigma->pi* excitation on fragment 0 (A' x A" = A") with fragment 1 in its ground
# state:
mc1 = las.LASSCF (mf, (4, 4), (4, 4), wfnsym_sub=('A"', "A'"))
mc1.kernel (mc.mo_coeff)
print ('E(LASSCF, fragment 0 A") - E(LASSCF) =', mc1.e_tot - mc.e_tot)
