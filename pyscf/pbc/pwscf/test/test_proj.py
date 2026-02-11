""" When orbitals from init guess uses a different grid mesh than the current
calculation, perform a projection.
"""

import tempfile
import numpy as np
from numpy.testing import assert_allclose

from pyscf import lib
from pyscf.pbc import gto, pwscf

import unittest


def make_cell(atom, a, pseudo, ke_cutoff, mesh=None):
    if mesh is None:
        mesh = [12, 12, 12]
    cell = gto.Cell(
        atom=atom,
        a=a,
        basis="gth-szv",
        pseudo=pseudo,
        ke_cutoff=ke_cutoff,
        mesh=mesh,
    )
    cell.build()
    cell.verbose = 0
    return cell


def make_mf(cell, kmesh):
    kpts = cell.make_kpts(kmesh)
    mf = pwscf.KRHF(cell, kpts)
    return mf


class KnownValues(unittest.TestCase):
    def test_proj(self):
        kmesh = [2,1,1]
        ke_cutoffs = [30,40,50]
        pseudo = "gth-pade"
        atom = "C 0 0 0; C 0.89169994 0.89169994 0.89169994"
        a = np.asarray(
            [[0.       , 1.78339987, 1.78339987],
            [1.78339987, 0.        , 1.78339987],
            [1.78339987, 1.78339987, 0.        ]])

        meshes = [
            [10, 10, 10],
            [12, 12, 12],
            [13, 13, 13],
        ]
        cells = [
            make_cell(atom, a, pseudo, ke, mesh)
            for ke, mesh in zip(ke_cutoffs, meshes)
        ]
        mfs = [make_mf(cell, kmesh) for cell in cells]

        # Reference energies for pyscf<=2.10
        # erefs = [-10.6754924867542, -10.6700816768958, -10.6734527455548]
        erefs = [-10.6755254339103, -10.6700884878958, -10.6734569703190]

        # tempfile
        swapfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        chkfile = swapfile.name
        swapfile = None
        for mf in mfs:
            mf.chkfile = chkfile

        # run ke1
        mfs[1].kernel()
        assert_allclose(mfs[1].e_tot, erefs[1], atol=1e-5, rtol=0)
        # run ke0 with ke1 init guess (projection down)
        mfs[0].init_guess = "chk"
        mfs[0].kernel()
        assert_allclose(mfs[0].e_tot, erefs[0], atol=1e-5, rtol=0)
        # run ke2 with ke0 init guess (projection up)
        mfs[2].init_guess = "chk"
        mfs[2].kernel()
        assert_allclose(mfs[2].e_tot, erefs[2], atol=1e-5, rtol=0)


if __name__ == "__main__":
    unittest.main()
