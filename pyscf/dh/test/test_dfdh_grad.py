import pytest
from deriv_numerical import NucCoordDerivGenerator, NumericDiff
from pyscf.dh import DFDH
from pyscf import gto
import numpy as np


def _mol():
    return gto.Mole(atom="N 0. 0. 0.; H .9 0. 0.; H 0. 1. 0.; H 0. 0. 1.1", basis="cc-pVDZ", verbose=0).build()


def _mol_to_eng(xc):
    def fx(mol):
        return DFDH(mol, xc).run().e_tot
    return fx


class TestDFGrad:

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

    def test_b2plyp(self):
        REF = np.array([[ 0.13023341603 , -0.004235200797, -0.072766686093, -0.170875783491,
         0.025976874261,  0.021095470322,  0.023380525371, -0.036308773283,
         0.016020178926,  0.017261572487,  0.014565159726,  0.035648771535]]).reshape((4, 3))
        de = DFDH(_mol(), "B2PLYP").nuc_grad_method().run().grad_tot
        assert np.allclose(REF, de, atol=1e-5, rtol=1e-4)

    def test_xygj_os(self):
        REF = np.array([[ 0.130065980247, -0.00557420808 , -0.074528598118, -0.171547491911,
         0.026500152151,  0.021547427293,  0.023850842385, -0.035807666449,
         0.01636781729 ,  0.017630592194,  0.014880537863,  0.036612219285]]).reshape((4, 3))
        de = DFDH(_mol(), "XYGJ-OS").nuc_grad_method().run().grad_tot
        assert np.allclose(REF, de, atol=1e-5, rtol=1e-4)

    def test_xyg3_tuple(self):
        REF = np.array([[ 0.120274613071, -0.015728504853, -0.085326169707, -0.16311102246 ,
         0.0272324229  ,  0.022397854712,  0.024509841245, -0.027148203586,
         0.017206874359,  0.018326380786,  0.015643331864,  0.045720381028]]).reshape((4, 3))
        xc = ["HF", "0.8033*HF - 0.0140*LDA + 0.2107*B88, 0.6789*LYP", 0.3211, 1, 1]
        de = DFDH(_mol(), xc).nuc_grad_method().run().grad_tot
        assert np.allclose(REF, de, atol=1e-5, rtol=1e-4)
