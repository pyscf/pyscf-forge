""" Check PW-KUHF, PW-KUMP2 and read init guess from chkfile
"""


import h5py
import tempfile
import numpy as np

from pyscf.pbc import gto, pwscf
from pyscf import lib

import unittest


class KnownValues(unittest.TestCase):
    def _run_test(self, pseudo, atom, a, e_tot0, e_corr0):
        nk = 1
        ke_cutoff = 30
        cell = gto.Cell(
            atom=atom,
            a=a,
            spin=2, # triplet
            basis="gth-szv",
            pseudo=pseudo,
            ke_cutoff=ke_cutoff,
            mesh=[19, 19, 19],
        )
        cell.build()
        cell.verbose = 6

        # kpts
        kmesh = [nk]*3
        kpts = cell.make_kpts(kmesh)
        nkpts = len(kpts)

        # tempfile
        swapfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        chkfile = swapfile.name
        swapfile = None

        # krhf
        pwmf = pwscf.KUHF(cell, kpts)
        pwmf.nvir = 4 # request 4 virtual states
        pwmf.chkfile = chkfile
        pwmf.kernel()
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
        pwmp = pwscf.KUMP2(pwmf)
        pwmp.kernel()
        assert(abs(pwmp.e_corr - e_corr0) < 1.e-6)

    def test_gth(self):
        pseudo = "gth-pade"
        atom = "C 0 0 0"
        a = np.eye(3) * 4   # atom in a cubic box
        e_tot0 = -5.39796638192271
        e_corr0 = -0.00682323936825284
        self._run_test(pseudo, atom, a, e_tot0, e_corr0)
    
    def test_ccecp(self):
        pseudo = "ccecp"
        atom = "C 0 0 0"
        a = np.eye(3) * 4   # atom in a cubic box
        e_tot0 = -5.35343662020727
        e_corr0 = -0.00670287547309327
        self._run_test(pseudo, atom, a, e_tot0, e_corr0)


if __name__ == "__main__":
    unittest.main()
