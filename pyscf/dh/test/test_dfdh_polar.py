import pytest
from deriv_numerical import DipoleDerivGenerator, NumericDiff
from pyscf.dh import DFDH
from pyscf import gto, scf
import numpy as np


def _mol():
    return gto.Mole(atom="N 0. 0. 0.; H .9 0. 0.; H 0. 1. 0.; H 0. 0. 1.1", basis="cc-pVDZ", verbose=0).build()


def _mol_to_dipole(mol, xc):
    def fx(component, interval):
        mf = DFDH(mol, xc)
        def get_hcore(mol=mol):
            return scf.rhf.get_hcore(mol) - interval * mol.intor("int1e_r")[component]
        mf.mf_s.get_hcore = mf.mf_n.get_hcore = get_hcore
        return mf.run().polar_method().dipole()
    return fx


class TestDFPolar:

    def test_b2plyp(self):
        mol = _mol()
        nde = - NumericDiff(DipoleDerivGenerator(_mol_to_dipole(mol, "B2PLYP"))).derivative
        de = DFDH(mol, "B2PLYP").polar_method().run().pol_tot
        assert np.allclose(nde, de, atol=1e-5, rtol=1e-4)

    def test_mp2(self):
        REF = np.array([[ 7.287332131227, -0.143671547054, -0.203498074527, -0.143671618067,
         8.724054795277, -0.334215302667, -0.203498168983, -0.33421529318 ,
         10.345461647677]]).reshape((3, 3))
        de = DFDH(_mol(), "MP2").polar_method().run().pol_tot
        assert np.allclose(REF, de, atol=1e-5, rtol=1e-4)

    def test_xyg3(self):
        REF = np.array([[ 7.267950702094, -0.150453822162, -0.218852668583, -0.150453822202,
         8.751947502917, -0.366999912293, -0.218852668702, -0.366999912383,
         10.473371520993]]).reshape((3, 3))
        de = DFDH(_mol(), "XYG3").polar_method().run().pol_tot
        assert np.allclose(REF, de, atol=1e-5, rtol=1e-4)

    def test_xygj_os(self):
        REF = np.array([[ 7.279946203272, -0.148991221551, -0.215161986695, -0.148991221593,
         8.771801652821, -0.361506774213, -0.215161986816, -0.361506774303,
         10.495615774576]]).reshape((3, 3))
        de = DFDH(_mol(), "XYGJ-OS").polar_method().run().pol_tot
        assert np.allclose(REF, de, atol=1e-5, rtol=1e-4)
