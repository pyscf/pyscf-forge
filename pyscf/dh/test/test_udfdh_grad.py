import pytest
from deriv_numerical import NucCoordDerivGenerator, NumericDiff
from pyscf.dh import DFDH
from pyscf import gto
import numpy as np


def _mol():
    return gto.Mole(atom="C 0. 0. 0.; H .9 0. 0.; H 0. 1. 0.; H 0. 0. 1.1", basis="cc-pVDZ", spin=1, verbose=0).build()


def _mol_to_eng(xc):
    def fx(mol):
        return DFDH(mol, xc).run().e_tot
    return fx



class TestUDFGrad:
    def test_mp2(self):
        mol = _mol()
        nde = NumericDiff(NucCoordDerivGenerator(mol, _mol_to_eng("MP2"))).derivative.reshape(-1, 3)
        de = DFDH(mol, "MP2").nuc_grad_method().run().grad_tot
        assert np.allclose(nde, de, atol=1e-5, rtol=1e-4)

    def test_xyg3(self):
        mol = _mol()
        nde = NumericDiff(NucCoordDerivGenerator(mol, _mol_to_eng("XYG3"))).derivative.reshape(-1, 3)
        de = DFDH(mol, "XYG3").nuc_grad_method().run().grad_tot
        assert np.allclose(nde, de, atol=1e-5, rtol=1e-4)

    def test_xygj_os(self):
        mol = _mol()
        nde = NumericDiff(NucCoordDerivGenerator(mol, _mol_to_eng("XYGJ-OS"))).derivative.reshape(-1, 3)
        de = DFDH(mol, "XYGJ-OS").nuc_grad_method().run().grad_tot
        assert np.allclose(nde, de, atol=1e-5, rtol=1e-4)

    @pytest.mark.xfail(reason="known ~2.5e-5 analytical vs numerical discrepancy")
    def test_b2plyp(self):
        mol = _mol()
        nde = NumericDiff(NucCoordDerivGenerator(mol, _mol_to_eng("B2PLYP"))).derivative.reshape(-1, 3)
        de = DFDH(mol, "B2PLYP").nuc_grad_method().run().grad_tot
        assert np.allclose(nde, de, atol=1e-5, rtol=1e-4)
