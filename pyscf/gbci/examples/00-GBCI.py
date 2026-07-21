#!/usr/bin/env python
#
# NH3-F2 GBCI example.

import numpy as np

from pyscf import gto, scf, gbci
from pyscf.mcscf import addons


ATOM = """
H 2.490675 0.939499 -0.000156
N 2.085280 0.000000 0.000003
H 2.490739 -0.469633 0.813668
H 2.490627 -0.469860 -0.813586
F -0.222963 0.000000 0.000001
F -1.710402 0.000000 0.000000
"""

BASIS = "ccpvdz"
NCAS = 3
NELECAS = (2, 2)
CAS_LIST = [9, 14, 15]
GROUP_A = {"atom": [[0, 1, 2, 3], [4, 5]]}
NROOTS = 4
TARGET_S2 = 0.0


mol = gto.M(
    atom=ATOM,
    basis=BASIS,
    charge=0,
    spin=0,
    verbose=5,
)
mf = scf.ROHF(mol).run()

mc = gbci.gbci(mf, NCAS, NELECAS, group_a=GROUP_A)
mc.fcisolver.nroots = NROOTS
mc.fcisolver.spin = 0
mc.fix_spin_(ss=TARGET_S2)

mo = addons.sort_mo(mc, mf.mo_coeff, CAS_LIST, 1)
e_tot, e_cas, ci = mc.kernel(mo)

print("GBCI energies:")
print(np.asarray(e_tot))
