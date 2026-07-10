import pytest
from deriv_numerical import DipoleDerivGenerator, NumericDiff
from pyscf.dh import DFDH
from pyscf import gto, scf
import numpy as np


# pytestmark = pytest.mark.xfail(reason="polar code numerically inaccurate (~2% error vs numerical diff)")


def _mol():
    return gto.Mole(atom="N 0. 0. 0.; H .9 0. 0.; H 0. 1. 0.; H 0. 0. 1.1", basis="cc-pVDZ", verbose=0).build()


def _mol_to_dipole(mol, xc):
    def fx(component, interval):
        mf = DFDH(mol, xc)
        def get_hcore(mol=mol):
            return scf.rhf.get_hcore(mol) - interval * mol.intor("int1e_r")[component]
        mf.mf_s.get_hcore = mf.mf_n.get_hcore = get_hcore
        return mf.run().dipole()
    return fx


class TestDFPolar:
    def test_mp2(self):
        mol = _mol()
        nde = - NumericDiff(DipoleDerivGenerator(_mol_to_dipole(mol, "MP2"))).derivative
        de = DFDH(mol, "MP2").polar_method().run().pol_tot
        assert np.allclose(nde, de, atol=1e-5, rtol=1e-4)

    def test_xyg3(self):
        mol = _mol()
        nde = - NumericDiff(DipoleDerivGenerator(_mol_to_dipole(mol, "XYG3"))).derivative
        de = DFDH(mol, "XYG3").polar_method().run().pol_tot
        assert np.allclose(nde, de, atol=1e-5, rtol=1e-4)

    def test_b2plyp(self):
        mol = _mol()
        nde = - NumericDiff(DipoleDerivGenerator(_mol_to_dipole(mol, "B2PLYP"))).derivative
        de = DFDH(mol, "B2PLYP").polar_method().run().pol_tot
        assert np.allclose(nde, de, atol=1e-5, rtol=1e-4)

    def test_xygj_os(self):
        mol = _mol()
        nde = - NumericDiff(DipoleDerivGenerator(_mol_to_dipole(mol, "XYGJ-OS"))).derivative
        de = DFDH(mol, "XYGJ-OS").polar_method().run().pol_tot
        assert np.allclose(nde, de, atol=1e-5, rtol=1e-4)
