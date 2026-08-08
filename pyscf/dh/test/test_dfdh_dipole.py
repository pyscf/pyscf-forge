import pytest
from deriv_numerical import DipoleDerivGenerator, NumericDiff
from pyscf.dh import DFDH
from pyscf import gto, scf
import numpy as np


def _mol():
    return gto.Mole(atom="N 0. 0. 0.; H .9 0. 0.; H 0. 1. 0.; H 0. 0. 1.1", basis="cc-pVDZ", verbose=0).build()


def _mol_to_eng(mol, xc):
    def fx(component, interval):
        mf = DFDH(mol, xc)
        def get_hcore(mol=mol):
            return scf.rhf.get_hcore(mol) - interval * mol.intor("int1e_r")[component]
        mf.mf_s.get_hcore = mf.mf_n.get_hcore = get_hcore
        mf.run()
        return mf.e_tot
    return fx


class TestDFDipole:

    def test_b2plyp(self):
        mol = _mol()
        dip_nuc = np.einsum("At, A-> t", mol.atom_coords(), mol.atom_charges())
        nde = NumericDiff(DipoleDerivGenerator(_mol_to_eng(mol, "B2PLYP"))).derivative + dip_nuc
        de = DFDH(mol, "B2PLYP").run().polar_method().dipole()
        assert np.allclose(nde, de, atol=1e-5, rtol=1e-4)

    def test_mp2(self):
        REF = np.array([0.513968430874, 0.494934113883, 0.469052040835])
        de = DFDH(_mol(), "MP2").run().polar_method().dipole()
        assert np.allclose(REF, de, atol=1e-5, rtol=1e-4)

    def test_xyg3(self):
        REF = np.array([0.516715680877, 0.497164080305, 0.467699209941])
        de = DFDH(_mol(), "XYG3").run().polar_method().dipole()
        assert np.allclose(REF, de, atol=1e-5, rtol=1e-4)

    def test_xygj_os(self):
        REF = np.array([0.519030495499, 0.498740432488, 0.468021122351])
        de = DFDH(_mol(), "XYGJ-OS").run().polar_method().dipole()
        assert np.allclose(REF, de, atol=1e-5, rtol=1e-4)
