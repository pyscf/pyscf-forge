import pytest
from deriv_numerical import DipoleDerivGenerator, NumericDiff
from pyscf.dh import DFDH
from pyscf import gto, scf
import numpy as np


def _mol():
    return gto.Mole(atom="C 0. 0. 0.; H .9 0. 0.; H 0. 1. 0.; H 0. 0. 1.1", basis="cc-pVDZ", spin=1, verbose=0).build()


def _mol_to_dipole(mol, xc):
    def fx(component, interval):
        mf = DFDH(mol, xc)
        def get_hcore(mol=mol):
            return scf.rhf.get_hcore(mol) - interval * mol.intor("int1e_r")[component]
        mf.mf_s.get_hcore = mf.mf_n.get_hcore = get_hcore
        return mf.run().polar_method().dipole()
    return fx


class TestUDFPolar:

    def test_b2plyp_slow(self):
        mol = _mol()
        nde = - NumericDiff(DipoleDerivGenerator(_mol_to_dipole(mol, "B2PLYP"))).derivative
        de = DFDH(mol, "B2PLYP").polar_method().run().pol_tot
        assert np.allclose(nde, de, atol=1e-5, rtol=1e-4)

    def test_mp2(self):
        REF = np.array([[ 8.495484511042, -0.096973444459, -0.143329946215, -0.096972818643,
         9.77976889886 , -0.268521758608, -0.143330817361, -0.268523421283,
         11.302525957644]]).reshape((3, 3))
        de = DFDH(_mol(), "MP2").polar_method().run().pol_tot
        assert np.allclose(REF, de, atol=1e-5, rtol=1e-4)

    def test_xyg3(self):
        REF = np.array([[ 8.491681792953, -0.091021200692, -0.143730291653, -0.091021200212,
         9.791669873535, -0.278553934662, -0.143730291646, -0.278553934681,
         11.353024483936]]).reshape((3, 3))
        de = DFDH(_mol(), "XYG3").polar_method().run().pol_tot
        assert np.allclose(REF, de, atol=1e-5, rtol=1e-4)

    def test_xygj_os(self):
        REF = np.array([[ 8.527886902706, -0.094788933065, -0.145766099944, -0.09478893212 ,
         9.830231416889, -0.279058059955, -0.145766099992, -0.279058059949,
         11.386861078971]]).reshape((3, 3))
        de = DFDH(_mol(), "XYGJ-OS").polar_method().run().pol_tot
        assert np.allclose(REF, de, atol=1e-5, rtol=1e-4)
