""" Check PW-KRHF, PW-KRMP2 and read init guess from chkfile
"""


import h5py
import tempfile
import numpy as np
from numpy.testing import assert_allclose

from pyscf.pbc import gto, pwscf
from pyscf import lib

import unittest


class KnownValues(unittest.TestCase):
    def _run_test(self, pseudo, atom, a, e_tot0, e_corr0, mesh=None):
        kmesh = [2,1,1]
        ke_cutoff = 30

        # cell
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

        # kpts
        kpts = cell.make_kpts(kmesh)
        nkpts = len(kpts)

        # tempfile
        swapfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        chkfile = swapfile.name
        swapfile = None

        # krhf
        pwmf = pwscf.KRHF(cell, kpts)
        pwmf.nvir = 10 # request 10 virtual states
        pwmf.chkfile = chkfile
        pwmf.kernel(save_ccecp_kb=True)
        assert_allclose(pwmf.e_tot, e_tot0, atol=1.e-6, rtol=0)

        # krhf init from chkfile
        pwmf.init_guess = "chkfile"
        pwmf.kernel()
        assert_allclose(pwmf.e_tot, e_tot0, atol=1.e-6, rtol=0)

        # input C0
        pwmf.kernel(C0=pwmf.mo_coeff)
        assert_allclose(pwmf.e_tot, e_tot0, atol=1.e-6, rtol=0)

        # krmp2
        pwmp = pwscf.KMP2(pwmf)
        pwmp.kernel()
        assert_allclose(pwmp.e_corr, e_corr0, atol=1.e-4, rtol=0)

        pwmf = pwscf.KRHF(cell, kpts, ecut_wf=20)
        pwmf.nvir = 10 # request 10 virtual states
        pwmf.chkfile = chkfile
        pwmf.kernel(save_ccecp_kb=True)

        pwmp = pwscf.KMP2(pwmf)
        pwmp.kernel()
        # higher relative error threshold because the PW basis is different
        assert_allclose(abs((pwmp.e_corr - e_corr0) / e_corr0),
                       0, atol=5.e-2, rtol=0)


    def test_alle(self):
        atom = "He 0 0 0"
        a = np.eye(3) * 2
        mesh = [10, 10, 10]
        # e0 = -3.01953411844147 for pyscf<=2.10
        e0 = -3.01955958753843
        self._run_test(
            None, atom, a, e0, -0.0184642869417647, mesh
        )

    def test_ccecp(self):
        atom = "C 0 0 0; C 0.89169994 0.89169994 0.89169994"
        a = np.asarray(
            [[0.        , 1.78339987, 1.78339987],
             [1.78339987, 0.        , 1.78339987],
             [1.78339987, 1.78339987, 0.        ]]
        )
        mesh = [10, 10, 10]
        # e0 = -10.6261884956522 for pyscf<=2.10
        e0 = -10.6262216801107
        self._run_test(
            "ccecp", atom, a, e0, -0.136781915070538, mesh
        )

    def test_gth(self):
        atom = "C 0 0 0; C 0.89169994 0.89169994 0.89169994"
        a = np.asarray(
            [[0.        , 1.78339987, 1.78339987],
             [1.78339987, 0.        , 1.78339987],
             [1.78339987, 1.78339987, 0.        ]]
        )
        mesh = [10, 10, 10]
        # e0 = -10.6754927046184 for pyscf<=2.10
        e0 = -10.675524900499157
        self._run_test(
            "gth-pade", atom, a, e0, -0.139309030515543, mesh
        )


if __name__ == "__main__":
    unittest.main()
