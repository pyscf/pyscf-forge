import unittest
from pyscf import gto, dh


class TestLowMem(unittest.TestCase):
    def test_low_mem_df(self):
        mol = gto.Mole(atom="Ne", basis="cc-pVTZ").build()
        mf = dh.DFDH(mol, xc="XYG3").run()
        REF_ETOT = mf.e_tot
        mol = gto.Mole(atom="Ne", basis="cc-pVTZ", max_memory=0.0000001).build()
        mf = dh.DFDH(mol, xc="XYG3").run()
        self.assertAlmostEqual(mf.e_tot, REF_ETOT)
