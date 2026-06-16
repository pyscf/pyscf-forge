from pyscf.pbc import gto
from pyscf.gto.basis import parse_cp2k_pp
from pyscf.pbc.pwscf import kpt_symm, khf, smearing
import numpy as np

"""
Simple examples of running HF and DFT for BCC Li.
Both calculations run with smearing.
The DFT calculation handles a larger k-mesh with
k-point symmetry.
"""

cell = gto.Cell()
a = 3.4393124531669552
cell.a = a * np.eye(3)
x = 0.5 * a
cell.atom = f"""
Li 0   0   0
Li {x} {x} {x}
"""
cell.pseudo = {'Li': parse_cp2k_pp.parse("""
#PSEUDOPOTENTIAL
Li GTH2-HF-q1
    1    0    0    0
    0.75910286326041       2   -1.83343584669401    0.32295157976066
       2
    0.66792517034256       1    1.83367870276199
    1.13098354939590       1   -0.00004141168540
""")}
cell.basis = 'gth-szv'
cell.verbose = 4
cell.space_group_symmetry = True
cell.symmorphic = True
cell.mesh = [10, 10, 10]
cell.build()

# Center at the Baldereshi point
kpts = cell.make_kpts(
    [2, 2, 2],
    scaled_center=[1.0/6, 1.0/6, 0.5]
)
kmf = khf.PWKRHF(cell, kpts, ecut_wf=40)
kmf = smearing.smearing_(kmf, sigma=0.02, method='gauss')
kmf.xc = "PBE"
kmf.conv_tol = 1e-7
kmf.conv_tol_grad = 2e-3
ehf = kmf.kernel()

kpts = cell.make_kpts(
    [4, 4, 4],
    scaled_center=[1.0/6, 1.0/6, 0.5],
    time_reversal_symmetry=True,
    space_group_symmetry=True
)
kmf = kpt_symm.KsymAdaptedPWKRKS(cell, kpts, ecut_wf=40)
kmf = smearing.smearing_(kmf, sigma=0.01, method='gauss')
kmf.xc = "PBE"
kmf.conv_tol = 1e-7
kmf.conv_tol_grad = 2e-3
ehf = kmf.kernel()

