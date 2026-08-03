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

    def test_mp2_slow(self):
        mol = _mol()
        nde = NumericDiff(NucCoordDerivGenerator(mol, _mol_to_eng("MP2"))).derivative.reshape(-1, 3)
        de = DFDH(mol, "MP2").nuc_grad_method().run().grad_tot
        assert np.allclose(nde, de, atol=1e-5, rtol=1e-4)

    def test_xyg3(self):
        REF = np.array([[ 0.168850247065,  0.012824010859, -0.073998331122, -0.250420876946,
         0.049562935594,  0.045175984426,  0.044607461767, -0.096879719501,
         0.037939756021,  0.036963283405,  0.034491570053, -0.009119030705]]).reshape((4, 3))
        de = DFDH(_mol(), "XYG3").nuc_grad_method().run().grad_tot
        assert np.allclose(REF, de, atol=1e-5, rtol=1e-4)

    def test_b2plyp(self):
        REF = np.array([[ 0.171914408372,  0.017028295453, -0.069116908089, -0.251905263547,
         0.048609984888,  0.044287992881,  0.043751025003, -0.099473086866,
         0.037212462784,  0.036238921271,  0.03383197114 , -0.012386427154]]).reshape((4, 3))
        de = DFDH(_mol(), "B2PLYP").nuc_grad_method().run().grad_tot
        assert np.allclose(REF, de, atol=1e-5, rtol=1e-4)

    def test_xygj_os(self):
        REF = np.array([[ 0.17338323122 ,  0.017083197023, -0.069959873818, -0.254205690044,
         0.049118103116,  0.044752230437,  0.044206922718, -0.100366282311,
         0.037579438915,  0.036616273841,  0.034163878184, -0.012373575431]]).reshape((4, 3))
        de = DFDH(_mol(), "XYGJ-OS").nuc_grad_method().run().grad_tot
        assert np.allclose(REF, de, atol=1e-5, rtol=1e-4)
