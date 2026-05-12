from pyscf.pbc import gto, pwscf
from pyscf.pbc.pwscf.kmp2 import PWKRMP2
import numpy as np

"""
Simple MP2 calculation
"""

atom = "H 0 0 0; H 0.9 0 0"
a = np.eye(3) * 3
basis = "gth-szv"
pseudo = "gth-pade"

ke_cutoff = 50

cell = gto.Cell(atom=atom, a=a, basis=basis, pseudo=pseudo,
                ke_cutoff=ke_cutoff)
cell.build()
cell.verbose = 6

nk = 2
kmesh = [nk] * 3
kpts = cell.make_kpts(kmesh)
nkpts = len(kpts)

pwmf = pwscf.PWKRHF(cell, kpts)
pwmf.nvir = 20
pwmf.kernel()

es = {"5": -0.01363871, "10": -0.01873622, "20": -0.02461560}

pwmp = PWKRMP2(pwmf)
pwmp.kernel(nvir_lst=[5,10,20])
pwmp.dump_mp2_summary()
nvir_lst = pwmp.mp2_summary["nvir_lst"]
ecorr_lst = pwmp.mp2_summary["e_corr_lst"]
for nvir,ecorr in zip(nvir_lst,ecorr_lst):
    err = abs(ecorr - es["%d"%nvir])
    print(err)
    assert(err < 1e-5)
