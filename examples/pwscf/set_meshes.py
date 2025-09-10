from pyscf.pbc import gto
from pyscf.pbc.pwscf.krks import PWKRKS
import numpy as np

"""
This example demonstrates converging the energy with respect to
the mesh size. Note that 
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
)

kmesh = [2, 1, 1]
kpts = cell.make_kpts(kmesh)

mf = PWKRKS(cell, kpts)

# defaults
print(cell.mesh, mf.wf_mesh, mf.xc_mesh)
mf.kernel()

mf.set_meshes(wf_mesh=[15, 15, 15], xc_mesh=[33, 33, 33])
print(mf.wf_mesh, mf.xc_mesh)
mf.kernel()
