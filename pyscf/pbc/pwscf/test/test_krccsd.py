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
                  test_scf=True, keep_exxdiv=False):
        # cell
        cell = gto.Cell(
            atom=atom,
            a=a,
            basis="gth-szv",
            pseudo=pseudo,
            ke_cutoff=ke_cutoff,
            verbose=0,
        )
        cell.build()
        kpts = cell.make_kpts(kmesh)

        # GTO
        gmf = scf.KRHF(cell, kpts)
        gmf.exxdiv = exxdiv
        gmf.kernel()
        assert gmf.converged
        gcc = cc.KCCSD(gmf)
        gcc.keep_exxdiv = keep_exxdiv
        gcc.kernel()
        assert gcc.converged
        etrip = gcc.ccsd_t()
        ips_ref = gcc.ipccsd()

        # PW
        pmf = pw_helper.gtomf2pwmf(gmf)
        pmf.build()
        pmf.update_pp(pmf.mo_coeff)
        pmf.update_k(pmf.mo_coeff, pmf.mo_occ)
        assert pmf.converged
        pcc = pwscf.PWKRCCSD(pmf)
        pcc.keep_exxdiv = keep_exxdiv
        pcc.kernel()
        assert pcc.converged
        etrip_test = pcc.ccsd_t()
        # Check CCSD(T)
        assert_allclose(gcc.e_corr, pcc.e_corr, atol=1.e-6, rtol=0)
        assert_allclose(etrip, etrip_test, atol=1.e-5, rtol=0)

        pcc.ecut_eri = 25
        pcc.kernel()
        assert pcc.converged
        # Check EOM-CCSD
        etrip_test = pcc.ccsd_t()
        ips_test = pcc.ipccsd()
        assert_allclose(gcc.e_corr, pcc.e_corr, atol=1.e-4, rtol=0)
        assert_allclose(etrip, etrip_test, atol=1.e-5, rtol=0)
        assert_allclose(ips_ref[0], ips_test[0], atol=3.e-5, rtol=0)
        assert_allclose(ips_ref[1], ips_test[1], atol=1.e-3, rtol=0)

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
            assert pwmf.converged
            pwcc = pwscf.PWKRCCSD(pwmf)
            pwcc.kernel()
            assert pwcc.converged
            # Just to make sure the code stays consistent
            assert_allclose(pwcc.e_corr, -0.03234330656841895, atol=1.e-4, rtol=0)
            pwcc.ecut_eri = 15
            pwcc.kernel()
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
        self._run_test(atom, a, basis, pseudo, ke_cutoff, kmesh, "ewald")
        kmesh = [3,1,1]
        self._run_test(atom, a, basis, pseudo, ke_cutoff, kmesh, "ewald",
                       test_scf=False, keep_exxdiv=True)
        self._run_test(atom, a, basis, pseudo, ke_cutoff, kmesh, None,
                       test_scf=False, keep_exxdiv=True)
        self._run_test(atom, a, basis, pseudo, ke_cutoff, kmesh, None,
                       test_scf=False)

        # diff occ per kpt (i.e., needs padding)
        kmesh = [2,2,1]
        self._run_test(atom, a, basis, pseudo, ke_cutoff, kmesh, "ewald",
                       test_scf=False)


if __name__ == "__main__":
    unittest.main()
