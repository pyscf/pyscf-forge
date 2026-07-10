import unittest
from pyscf import dh, gto


class TestRMP2LikeDH(unittest.TestCase):

    def test_B2GPPLYP_D3BJ(self):
        # reference: MRCC 2022-03-18
        REF_ETOT = -76.378332323817
        mol = gto.Mole(atom="H; O 1 2.0; H 2 2.0 1 104.2458898548", basis="cc-pVTZ", unit="AU", verbose=0).build()
        mf = dh.DFDH(mol, xc="B2GPPLYP-D3BJ", mp2_backend="dfmp2", frozen=1).run()
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=4)

    def test_XYG3(self):
        # reference: MRCC 2022-03-18
        REF_ETOT = -76.400701189007
        mol = gto.Mole(atom="O; H 1 2; H 1 2 2 104.2458898548", unit="AU", basis="cc-pVTZ").build()
        mf = dh.DFDH(mol, xc="XYG3", mp2_backend="dfmp2", frozen=1).run()
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=4)


