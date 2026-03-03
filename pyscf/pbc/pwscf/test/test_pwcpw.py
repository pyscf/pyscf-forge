""" Check PW occ + CPW vir for MP2
CPW stands for "contracted PW", which refers to a PW expansion vector with
*fixed* coefficient. This example generates such CPWs from the ccecp-cc-pvdz
basis set.
"""


import h5py
import tempfile
import numpy as np
from numpy.testing import assert_allclose

from pyscf.pbc import gto, pwscf
from pyscf import lib

import unittest


class KnownValues(unittest.TestCase):
    def test_pwcpw(self):

        kmesh = [2,1,1]
        ke_cutoff = 30
        basis_cpw = "ccecp-cc-pvdz"
        pseudo = "gth-pade"
        atom = "C 0 0 0; C 0.89169994 0.89169994 0.89169994"
        a = np.asarray(
            [[0.       , 1.78339987, 1.78339987],
            [1.78339987, 0.        , 1.78339987],
            [1.78339987, 1.78339987, 0.        ]])

        # cell
        cell = gto.Cell(
            atom=atom,
            a=a,
            basis="gth-szv",
            pseudo=pseudo,
            ke_cutoff=ke_cutoff,
            mesh=[10, 10, 10],
        )
        cell.build()
        cell.verbose = 0

        # kpts
        kpts = cell.make_kpts(kmesh)
        nkpts = len(kpts)

        # HF
        mf = pwscf.KRHF(cell, kpts)
        mf.kernel()
        # For pyscf<=2.10
        # assert_allclose(mf.e_tot, -10.6754924867542, atol=1.e-6, rtol=0)
        assert_allclose(mf.e_tot, -10.6755254339103, atol=1.e-6, rtol=0)

        # MP2
        moe_ks, mocc_ks = mf.get_cpw_virtual(basis_cpw)
        mf.dump_moe(moe_ks, mocc_ks)
        mmp = pwscf.KMP2(mf)
        mmp.kernel()
        # For pyscf<=2.10
        # emp_ref = -0.215895180360867
        emp_ref = -0.21590657102232175
        assert_allclose(mmp.e_corr, emp_ref, atol=1.e-6, rtol=0)

        # HF with a plane-wave cutoff
        mf = pwscf.KRHF(cell, kpts, ecut_wf=20)
        mf.kernel()

        # MP2
        moe_ks, mocc_ks = mf.get_cpw_virtual(basis_cpw)
        mf.dump_moe(moe_ks, mocc_ks)
        mmp = pwscf.KMP2(mf)
        mmp.kernel()
        # higher threshold because we use different basis
        assert_allclose(mmp.e_corr, emp_ref, atol=1.e-3, rtol=0)


if __name__ == "__main__":
    unittest.main()
