from pyscf.pbc import gto
from pyscf.pbc.pwscf.khf import PWKRHF
from pyscf.pbc.pwscf.kpt_symm import KsymAdaptedPWKRHF
import numpy as np
import time

"""
Demonstrate the speedup of symmetry-adapted HF over
non-symmetry-adapted HF.
"""

cell = gto.Cell(
    atom = "C 0 0 0; C 0.89169994 0.89169994 0.89169994",
    a = np.asarray([
            [0.       , 1.78339987, 1.78339987],
            [1.78339987, 0.        , 1.78339987],
            [1.78339987, 1.78339987, 0.        ]]),
    basis="gth-szv",
    ke_cutoff=50,
    pseudo="gth-pade",
    verbose=4,
    space_group_symmetry=True,
    symmorphic=True,
)
cell.build()

kmesh = [2, 2, 2]
center = [0, 0, 0]
kpts = cell.make_kpts(kmesh)
skpts = cell.make_kpts(
    kmesh,
    scaled_center=center,
    space_group_symmetry=True,
    time_reversal_symmetry=True,
)

mf = PWKRHF(cell, kpts, ecut_wf=40)
mf.nvir = 4
t0 = time.monotonic()
mf.kernel()
t1 = time.monotonic()

mf2 = KsymAdaptedPWKRHF(cell, skpts, ecut_wf=20)
mf2.damp_type = "simple"
mf2.damp_factor = 0.7
mf2.nvir = 4
t2 = time.monotonic()
mf2.kernel()
t3 = time.monotonic()

print(mf.e_tot, mf2.e_tot)
mf.dump_scf_summary()
mf2.dump_scf_summary()
print("nkpts in BZ and IBZ", skpts.nkpts, skpts.nkpts_ibz)
print("Runtime without symmmetry", t1 - t0)
print("Runtime with symmetry", t3 - t2)
