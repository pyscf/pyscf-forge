#!/usr/bin/env python

'''
State-averaged LASSCF and LASCI.

In CAS, state averaging only means solving for several roots of one CI problem and averaging
their energies. A LAS state is a product of fragment states, each with its own charge, spin
multiplicity, spin projection and point-group irrep, so a state-averaged LAS calculation is an
average over a list of product states that you choose, each solved with its own CI vectors.
In LASSCF, the orbitals are then optimized for the weighted average energy, as in SA-CASSCF.
The same input can be used with LASCI to access multiple states with frozen orbitals. (Note
that simply setting las.fcisolver.nroots > 1, as done in CASCI, does not work, because states
cannot be ordered in any sort of overall energy order.)

Here, the central C-C bond of C2H6N4 is broken, leaving two doublet radicals H-N=N-CH2*. A
single LAS state with M = +1/2 on one fragment and M = -1/2 on the other is a broken-symmetry
state: it has spin density +1 on one fragment and -1 on the other. Averaging it with the state
with the fragment spins swapped gives a spin-symmetric average.
'''

from pyscf import gto, scf
from pyscf.mcscf import las
from pyscf.scf import uhf

# C2H6N4 with N=N bonds stretched by 0.3 Angstrom, as in 00-simple_lasscf.py, and in addition
# the C-C bond stretched by 1.5 Angstrom (to 3.03 Angstrom)
mol = gto.M (atom='''H  -0.514649   3.409707   1.397652
                     N   0.217158   3.560141   0.655805
                     N   0.217158   2.421838  -0.376701
                     C  -0.746018   1.319495  -0.222182
                     H  -1.355572   1.441625   0.689250
                     H  -1.408620   1.341260  -1.095107
                     C   0.746018  -1.319495  -0.222182
                     H   1.355572  -1.441625   0.689250
                     H   1.408620  -1.341260  -1.095107
                     N  -0.217158  -2.421838  -0.376701
                     N  -0.217158  -3.560141   0.655805
                     H   0.514649  -3.409707   1.397652''',
             basis='6-31g', verbose=4)
mf = scf.RHF (mol).run ()

# Each fragment is one half of the molecule, with 3 active orbitals (the radical orbital on C and
# the N=N pi and pi* orbitals) and 3 active electrons, in a doublet state.
frag_atom_list = (list (range (6)), list (range (6, 12)))

def spin_pop (mc):
    # Mulliken spin population summed over the atoms of each fragment
    pop = uhf.mulliken_spin_pop (mol, mc.make_rdm1s (), verbose=0)[1]
    return [round (float (pop[f].sum ()), 3) for f in frag_atom_list]

# 1. A single state: M = +1/2 on fragment 0 and M = -1/2 on fragment 1
mc = las.LASSCF (mf, (3, 3), ((2, 1), (1, 2)), spin_sub=(2, 2))
mo0 = mc.localize_init_guess (frag_atom_list, mf.mo_coeff)
mc.kernel (mo0)
print ('Single state (ab): E = {:.8f}; fragment spin populations = {}'.format (
    mc.e_tot, spin_pop (mc)))

# 2. State-averaged LASSCF: the ab state above and the ba state, with equal weights. For the jth
#    fragment in the ith state, spins[i][j] = 2M (number of alpha minus beta electrons) and
#    smults[i][j] = 2S+1. state_average returns a new object; state_average_ changes mc in place.
mc_sa = mc.state_average (weights=[0.5, 0.5], spins=[[1, -1], [-1, 1]], smults=[[2, 2], [2, 2]])
mc_sa.kernel (mo0)
print ('SA-LASSCF: E(average) = {:.8f}; E(states) = {}'.format (mc_sa.e_tot, mc_sa.e_states))
print ('SA-LASSCF: fragment spin populations of the average density =', spin_pop (mc_sa))

# 3. State-averaged LASCI: the same states with orbitals frozen
mc = las.LASCI (mf, (3, 3), ((2, 1), (1, 2)), spin_sub=(2, 2))
mc_sa = mc.state_average (weights=[0.5, 0.5], spins=[[1, -1], [-1, 1]], smults=[[2, 2], [2, 2]])
mc_sa.kernel (mo0)
print ('SA-LASCI: E(average) = {:.8f}; E(states) = {}'.format (mc_sa.e_tot, mc_sa.e_states))
print ('SA-LASCI: fragment spin populations of the average density =', spin_pop (mc_sa))
