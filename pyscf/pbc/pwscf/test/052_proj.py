""" When orbitals from init guess uses a different grid mesh than the current
calculation, perform a projection.
"""

import tempfile
import numpy as np

from pyscf import lib
from pyscf.pbc import gto, pwscf


def make_cell(atom, a, pseudo, ke_cutoff):
    cell = gto.Cell(
        atom=atom,
        a=a,
        basis="gth-szv",
        pseudo=pseudo,
        ke_cutoff=ke_cutoff
    )
    cell.build()
    cell.verbose = 6
    return cell


def make_mf(cell, kmesh):
    kpts = cell.make_kpts(kmesh)
    mf = pwscf.KRHF(cell, kpts)
    return mf


if __name__ == "__main__":
    kmesh = [2,1,1]
    ke_cutoffs = [30,40,50]
    pseudo = "gth-pade"
    atom = "C 0 0 0; C 0.89169994 0.89169994 0.89169994"
    a = np.asarray(
        [[0.       , 1.78339987, 1.78339987],
        [1.78339987, 0.        , 1.78339987],
        [1.78339987, 1.78339987, 0.        ]])

    cells = [make_cell(atom, a, pseudo, ke) for ke in ke_cutoffs]
    mfs = [make_mf(cell, kmesh) for cell in cells]

    erefs = [-10.6754924867542, -10.6700816768958, -10.6734527455548]

# tempfile
    swapfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    chkfile = swapfile.name
    swapfile = None
    for mf in mfs:
        mf.chkfile = chkfile

# run ke1
    mfs[1].kernel()
    assert(abs(mfs[1].e_tot-erefs[1]) < 1e-5)
# run ke0 with ke1 init guess (projection down)
    mfs[0].init_guess = "chk"
    mfs[0].kernel()
    assert(abs(mfs[0].e_tot-erefs[0]) < 1e-5)
# run ke2 with ke0 init guess (projection up)
    mfs[2].init_guess = "chk"
    mfs[2].kernel()
    assert(abs(mfs[2].e_tot-erefs[2]) < 1e-5)
