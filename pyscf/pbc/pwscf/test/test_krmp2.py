""" First do RHF and RMP2 calcs in a Gaussian basis, then re-evaluate the RHF
and RMP2 energies using the PW code (for the fixed orbitals obtained from the
Gaussian-based calculations). The energies obtained from the two approaches
should agree.
"""


import h5py
import tempfile
import numpy as np

from pyscf.pbc import gto, scf, pwscf, mp
from pyscf.pbc.pwscf import khf, pw_helper
from pyscf.pbc.pwscf import kmp2
from pyscf import lib
import pyscf.lib.parameters as param

import unittest


class KnownValues(unittest.TestCase):
    def test_krmp2(self):
        kmesh = [2,1,1]
        ke_cutoff = 100
        pseudo = "gth-pade"
        exxdiv = "ewald"
        atom = "H 0 0 0; H 0.9 0 0"
        a = np.eye(3) * 3

        # cell
        cell = gto.Cell(
            atom=atom,
            a=a,
            basis="gth-szv",
            pseudo=pseudo,
            ke_cutoff=ke_cutoff
        )
        cell.build()
        cell.verbose = 0
        kpts = cell.make_kpts(kmesh)

        # GTO
        gmf = scf.KRHF(cell, kpts)
        gmf.exxdiv = exxdiv
        gmf.kernel()
        gmp = mp.KMP2(gmf)
        gmp.kernel()

        # PW
        pmf = pw_helper.gtomf2pwmf(gmf)
        pmp = kmp2.PWKRMP2(pmf)
        pmp.kernel()
        assert(abs(gmp.e_corr - pmp.e_corr) < 1.e-6)


if __name__ == "__main__":
    unittest.main()
