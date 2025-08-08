""" Check PW-KRHF, PW-KRMP2 and read init guess from chkfile
"""


import h5py
import tempfile
import numpy as np

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
        print(pwmf.e_tot, e_tot0)
        assert(abs(pwmf.e_tot - e_tot0) < 1.e-6)

        # krhf init from chkfile
        pwmf.init_guess = "chkfile"
        pwmf.kernel()
        assert(abs(pwmf.e_tot - e_tot0) < 1.e-6)

        # input C0
        pwmf.kernel(C0=pwmf.mo_coeff)
        assert(abs(pwmf.e_tot - e_tot0) < 1.e-6)

        # krmp2
        pwmp = pwscf.KMP2(pwmf)
        pwmp.kernel()
        print(pwmp.e_corr)
        assert(abs(pwmp.e_corr - e_corr0) < 1.e-4)

    def test_alle(self):
        atom = "He 0 0 0"
        a = np.eye(3) * 2
        mesh = [10, 10, 10]
        self._run_test(
            None, atom, a, -3.01953411844147, -0.0184642869417647, mesh
        )

    def test_ccecp(self):
        atom = "C 0 0 0; C 0.89169994 0.89169994 0.89169994"
        a = np.asarray(
            [[0.        , 1.78339987, 1.78339987],
             [1.78339987, 0.        , 1.78339987],
             [1.78339987, 1.78339987, 0.        ]]
        )
        mesh = [10, 10, 10]
        self._run_test(
            "ccecp", atom, a, -10.6261884956522, -0.136781915070538, mesh
        )
    
    def test_gth(self):
        atom = "C 0 0 0; C 0.89169994 0.89169994 0.89169994"
        a = np.asarray(
            [[0.        , 1.78339987, 1.78339987],
             [1.78339987, 0.        , 1.78339987],
             [1.78339987, 1.78339987, 0.        ]]
        )
        mesh = [10, 10, 10]
        self._run_test(
            "gth-pade", atom, a, -10.6754927046184, -0.139309030515543, mesh
        )


if __name__ == "__main__":
    unittest.main()
