""" First do RHF and RCCSD calcs in a Gaussian basis, then re-evaluate the RHF
and RCCSD energies using the PW code (for the fixed orbitals obtained from the
Gaussian-based calculations). The energies obtained from the two approaches
should agree.
"""


import h5py
import tempfile
import numpy as np
from numpy.testing import assert_allclose

from pyscf.pbc import gto, scf, pwscf, cc
from pyscf.pbc.pwscf import khf, pw_helper
from pyscf import lib
import pyscf.lib.parameters as param

import unittest


class KnownValues(unittest.TestCase):
    def _run_test(self, atom, a, basis, pseudo, ke_cutoff, kmesh, exxdiv,
                  test_scf=True):
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
        nkpts = len(kpts)

        # GTO
        gmf = scf.KRHF(cell, kpts)
        gmf.exxdiv = exxdiv
        gmf.kernel()
        gcc = cc.KCCSD(gmf)
        gcc.kernel()

        # PW
        pmf = pw_helper.gtomf2pwmf(gmf)
        pmf.build()
        pmf.update_pp(pmf.mo_coeff)
        pmf.update_k(pmf.mo_coeff, pmf.mo_occ)
        pcc = pwscf.PWKRCCSD(pmf)
        pcc.kernel()
        assert(abs(gcc.e_corr - pcc.e_corr) < 1.e-6)

        if test_scf:
            pwmf = pwscf.KRHF(cell, kpts, ecut_wf=20)
            # need some virtual orbitals to converge davidson
            pwmf.nvir = 4
            pwmf.nvir_extra = 3
            pwmf.conv_tol = 1e-8
            pwmf.init_guess = "scf"
            # pwmf.damp_type = "simple"
            pwmf.damp_factor = 0.0
            pwmf.kernel()
            pwcc = pwscf.PWKRCCSD(pwmf)
            pwcc.kernel()
            # Just to make sure the code stays consistent
            assert_allclose(pwcc.e_corr, -0.03234330656841895, atol=1.e-4, rtol=0)

    def test_krccsd(self):
        ke_cutoff = 50
        basis = "gth-szv"
        pseudo = "gth-pade"
        exxdiv = "ewald"
        atom = "Li 0 0 0; Li 1.75 1.75 1.75"
        a = np.eye(3) * 3.5

        # same occ per kpt
        kmesh = [2,1,1]
        self._run_test(atom, a, basis, pseudo, ke_cutoff, kmesh, exxdiv)
        # diff occ per kpt (i.e., needs padding)
        kmesh = [2,2,1]
        self._run_test(atom, a, basis, pseudo, ke_cutoff, kmesh, exxdiv,
                       test_scf=False)


if __name__ == "__main__":
    unittest.main()
