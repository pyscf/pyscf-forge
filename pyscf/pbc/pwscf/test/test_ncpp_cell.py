import unittest
from pyscf.pbc.gto.cell import Cell
from pyscf.pbc.pwscf.ncpp_cell import NCPPCell, DEFAULT_SG15_PATH
from pyscf.pbc.pwscf.upf import get_nc_data_from_upf
from pyscf.pbc.pwscf.krks import PWKRKS
from pyscf.pbc.pwscf.kuks import PWKUKS
from pyscf.pbc.pwscf import kpt_symm
import pyscf.pbc
import numpy as np
import os
from numpy.testing import assert_allclose


pyscf.pbc.DEBUG = False
HAVE_SG15 = DEFAULT_SG15_PATH is not None and os.path.exists(DEFAULT_SG15_PATH)


def setUpModule():
    global CELL, KPTS, KPTS2, ATOM, KPT1
    CELL = NCPPCell(
        atom = "C 0 0 0; C 0.89169994 0.89169994 0.89169994",
        a = np.asarray([
                [0.       , 1.78339987, 1.78339987],
                [1.78339987, 0.        , 1.78339987],
                [1.78339987, 1.78339987, 0.        ]]),
        basis="gth-szv",
        verbose=0,
    )
    if HAVE_SG15:
        CELL.build()

    kmesh = [2, 2, 2]
    KPTS = CELL.make_kpts(kmesh)

    kmesh2 = [1, 1, 3]
    KPTS2 = CELL.make_kpts(kmesh2)

    ATOM = NCPPCell(
        atom = "C 0 0 0",
        a = np.eye(3) * 4,
        basis="gth-szv",
        spin=2,
        verbose=0,
    )
    if HAVE_SG15:
        ATOM.build()

    nk = 1
    kmesh = (nk,)*3
    KPT1 = ATOM.make_kpts(kmesh)


def tearDownModule():
    global CELL, ATOM, KPTS, KPTS2, KPT1
    del CELL, ATOM, KPTS, KPTS2, KPT1


@unittest.skipIf(not HAVE_SG15, "Missing SG15 pseudos")
class KnownValues(unittest.TestCase):
    def test_energy(self):
        ecut_wf = 18.38235294
        e_ref2 = -10.5957823763498
        e_ref = -11.044064472796734
        mf = PWKRKS(CELL, KPTS2, xc="PBE", ecut_wf=ecut_wf)
        mf.nvir = 4 # converge first 4 virtual bands
        mf.kernel()
        assert_allclose(mf.e_tot, e_ref2, atol=1e-7)
        mf = PWKUKS(CELL, KPTS2, xc="PBE", ecut_wf=ecut_wf)
        mf.nvir = 4
        mf.kernel()
        assert_allclose(mf.e_tot, e_ref2, atol=1e-7)
        mf = kpt_symm.KsymAdaptedPWKRKS(CELL, KPTS, xc="PBE", ecut_wf=ecut_wf)
        mf.nvir = 4
        mf.kernel()
        assert_allclose(mf.e_tot, e_ref, atol=1e-7)

        # check loading and unloading the cell
        cell2 = NCPPCell.loads(CELL.dumps())
        mf2 = kpt_symm.KsymAdaptedPWKRKS(
            cell2, KPTS, xc="PBE", ecut_wf=ecut_wf
        )
        mf2.nvir = 4
        mf2.init_pp()
        mf2.init_jk()
        assert_allclose(
            mf2.energy_tot(mf.mo_coeff, mf.mo_occ), e_ref, atol=1e-7
        )
        # make sure original cell was not affected
        assert_allclose(
            mf.energy_tot(mf.mo_coeff, mf.mo_occ), e_ref, atol=1e-7
        )

        # make sure a ghost atom doesn't mess anything up
        gcell = NCPPCell(
            atom = """
            C 0 0 0
            C 0.89169994 0.89169994 0.89169994
            ghost:C -0.9 -0.9 -0.9
            """,
            a = np.asarray([
                    [0.       , 1.78339987, 1.78339987],
                    [1.78339987, 0.        , 1.78339987],
                    [1.78339987, 1.78339987, 0.        ]]),
            basis="gth-szv",
            verbose=0,
        )
        gcell.build()
        mf2 = PWKRKS(gcell, KPTS2, xc="PBE", ecut_wf=ecut_wf)
        mf2.kernel()
        assert_allclose(mf2.e_tot, e_ref2, atol=1e-6, rtol=0)


if __name__ == "__main__":
    unittest.main()

