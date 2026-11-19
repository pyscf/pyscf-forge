#!/usr/bin/env python

'''
Select the active orbitals with AVAS, then localize them with localize_init_guess.

When the frontier orbitals of the mean-field calculation are not the ones you want in the active
space, AVAS (see pyscf/examples/mcscf/43-avas.py) can select them by their overlap with a set of
AOs. AVAS returns the size of the whole active space and orbitals with the selected ones in the
active columns. Those orbitals are generally delocalized over all the fragments, so pass them to
localize_init_guess, which only needs the active columns to span the right space.
'''

from pyscf import gto, scf, mcscf
from pyscf.mcscf import las, avas

# C2H4N4 at its equilibrium geometry: two H-N=N- groups joined by a C=C bond. The molecule lies in
# the xy plane, so its pi orbitals are made of pz AOs. Here the frontier RHF orbitals include the
# C=C pi orbitals, which we do not want in the active space.
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
             basis='6-31g')
mf = scf.RHF (mol).run ()

# Select the orbitals that overlap with the N 2pz AOs: the pi and pi* orbitals of the two N=N
# bonds, i.e., 4 orbitals and 4 electrons
ncas, nelecas, mo_avas = avas.kernel (mf, ['N 2pz'], minao='sto-3g')
print ('AVAS active space: ({}e, {}o)'.format (nelecas, ncas))

# Divide them into two fragments, one per N=N bond
mc = las.LASSCF (mf, (2, 2), (2, 2))
mo0 = mc.localize_init_guess (([0, 1, 2], [7, 8, 9]), mo_avas)
mc.kernel (mo0)

# For comparison, CASSCF in the same active space
mc_cas = mcscf.CASSCF (mf, ncas, nelecas)
mc_cas.kernel (mo_avas)
