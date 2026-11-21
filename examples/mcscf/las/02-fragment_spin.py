#!/usr/bin/env python

'''
Control the spin of each fragment with nelecas_sub and spin_sub.

Each fragment of a LAS wave function has its own number of electrons, spin multiplicity (S) and
spin projection (M). Pass nelecas_sub as a sequence of (nalpha, nbeta) pairs to set M for each
fragment, and spin_sub as a sequence of 2S+1 values to set S. By default, each fragment has the
lowest spin consistent with its electron count (and M = S).

The whole LAS wave function is a product of fragment wave functions with definite S and M. It
has a definite total M (the sum over fragments), but it is not generally an eigenfunction of the
total S^2: coupling two triplets gives singlet, triplet and quintet components. To get spin
eigenstates for the whole molecule, use a method that couples several LAS states, such as LASSI
(in mrh).
'''

from pyscf import gto, scf, fci
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
             basis='6-31g')
mf = scf.RHF (mol).run ()
# Each fragment has the 2 electrons of an N=N pi bond in its pi and pi* orbitals, which can be
# coupled into a singlet or a triplet.
frag_atom_list = ([0, 1, 2], [9, 10, 11])

def total_spin (mc):
    # Expand the LAS product into a CAS CI vector, and compute <S^2> and 2S+1 of the whole thing.
    # las2cas_civec returns lists with one element per state.
    ci, nelec = las.addons.las2cas_civec (mc)
    return fci.spin_op.spin_square (ci[0], mc.ncas, nelec[0])

# 1. Singlet-singlet. Same as the default for (2, 2).
mc = las.LASSCF (mf, (2, 2), (2, 2), spin_sub=(1, 1))
mo0 = mc.localize_init_guess (frag_atom_list, mf.mo_coeff)
mc.kernel (mo0)
e_ss = mc.e_tot
print ('E(singlet-singlet)      = {:.8f}; <S^2> = {:.4f}'.format (e_ss, total_spin (mc)[0]))

# 2. Triplet-triplet with total M = 0: fragment 0 has M = +1 and fragment 1 has M = -1.
mc = las.LASSCF (mf, (2, 2), ((2, 0), (0, 2)), spin_sub=(3, 3))
mc.kernel (mo0)
e_tt0 = mc.e_tot
print ('E(triplet-triplet, M=0) = {:.8f}; <S^2> = {:.4f}'.format (e_tt0, total_spin (mc)[0]))
# <S^2> = 2 is not S(S+1) for any S: this product state is 1/3 singlet, 1/2 triplet and 1/6
# quintet (Clebsch-Gordan coefficients for |1,+1>|1,-1>).

# 3. Triplet-triplet with total M = 2: both fragments have M = +1. This state is the M = 2
#    component of a pure quintet.
mc = las.LASSCF (mf, (2, 2), ((2, 0), (2, 0)), spin_sub=(3, 3))
mc.kernel (mo0)
e_tt2 = mc.e_tot
print ('E(triplet-triplet, M=2) = {:.8f}; <S^2> = {:.4f}'.format (e_tt2, total_spin (mc)[0]))

# The M = 0 and M = 2 triplet-triplet states differ in energy only through the exchange
# interaction between the fragments' spin densities, which is small for well-separated fragments.
print ('E(TT, M=0) - E(TT, M=2) = {:.2e} Eh'.format (e_tt0 - e_tt2))
