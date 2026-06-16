""" First do UHF and UMP2 calcs in a Gaussian basis, then re-evaluate the UHF
and UMP2 energies using the PW code (for the fixed orbitals obtained from the
Gaussian-based calculations). The energies obtained from the two approaches
should agree.
"""


import h5py
import tempfile
import numpy as np

from pyscf.pbc import gto, scf, pwscf, mp
from pyscf.pbc.mp.kump2 import KUMP2
from pyscf.pbc.pwscf import kuhf, pw_helper
from pyscf import lib
from pyscf.pbc.pwscf import kmp2, kump2
import pyscf.lib.parameters as param

import unittest


class KnownValues(unittest.TestCase):
    def _check_kump2(self, kmesh):
        ke_cutoff = 50
        pseudo = "gth-pade"
        exxdiv = "ewald"
        atom = "C 0 0 0"
        a = np.eye(3) * 4

        # cell
        cell = gto.Cell(
            atom=atom,
            a=a,
            basis="gth-szv",
            pseudo=pseudo,
            ke_cutoff=ke_cutoff,
            spin=2
        )
        cell.build()
        cell.verbose = 0
        kpts = cell.make_kpts(kmesh)
        nkpts = len(kpts)

        # GTO
        if (np.array(kmesh) == 1).all():
            gmf = scf.UHF(cell)
            gmf.exxdiv = exxdiv
            gmf.kernel()
            gmp = mp.UMP2(gmf)
            gmp.kernel()
        else:
            # TODO make a test comparing to GTO
            # after KUMP2 is implemented in PySCF
            raise NotImplementedError
            gmf = scf.KUHF(cell, kpts)
            gmf.exxdiv = exxdiv
            gmf.kernel()
            gmp = KUMP2(gmf)
            gmp.kernel()

        # PW
        pmf = pw_helper.gtomf2pwmf(gmf)
        pmp = kump2.PWKUMP2(pmf)
        pmp.kernel()
        assert(abs(gmp.e_corr - pmp.e_corr) < 1.e-6)

    def test_kump2(self):
        self._check_kump2([1, 1, 1])

    def test_kump2_vs_krmp2(self):
        kmesh = [2, 1, 1]
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

        # restricted PW
        gmf = scf.KRHF(cell, kpts)
        gmf.exxdiv = exxdiv
        gmf.kernel()
        pmf = pw_helper.gtomf2pwmf(gmf)
        pmp = kmp2.PWKRMP2(pmf)
        pmp.kernel()

        # unrestricted PW
        gmf = scf.KUHF(cell, kpts)
        gmf.exxdiv = exxdiv
        gmf.kernel()
        pmf = pw_helper.gtomf2pwmf(gmf)
        upmp = kump2.PWKUMP2(pmf)
        upmp.kernel()
        assert(abs(upmp.e_corr - pmp.e_corr) < 1.e-6)


if __name__ == "__main__":
    unittest.main()
