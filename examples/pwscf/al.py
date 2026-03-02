from pyscf.pbc import gto
from pyscf.pbc.pwscf.smearing import smearing_
from pyscf.pbc.pwscf import kpt_symm
import numpy as np

"""
Simple examples of running DFT for FCC Al. Uses
k-point symmetrt and smearing
"""

cell = gto.Cell()
cell.a = 4.0 * np.eye(3)
x = 2.0
cell.atom = f"""
Al 0 0 0
Al 0 {x} {x}
Al {x} 0 {x}
Al {x} {x} 0
"""
cell.pseudo = 'gth-pade'
cell.basis = 'gth-szv'
cell.verbose = 4
cell.space_group_symmetry = True
cell.symmorphic = True
cell.build()

kpts = cell.make_kpts(
    [4, 4, 4],
    time_reversal_symmetry=True,
    space_group_symmetry=True
)
kmf = kpt_symm.KsymAdaptedPWKRKS(cell, kpts, ecut_wf=40)
kmf = smearing_(kmf, sigma=0.01, method='gauss')
kmf.xc = "PBE"
kmf.nvir = 2
kmf.conv_tol = 1e-7
kmf.kernel()
kmf.dump_scf_summary()
