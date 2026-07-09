import unittest
from pyscf import dh, gto


class TestUMP2LikeDH(unittest.TestCase):

    @unittest.skip("frozen core not yet supported")
    def test_B2GPPLYP_D3BJ(self):
        # reference: MRCC 2022-03-18
        REF_ETOT = -55.841686701788
        mol = gto.Mole(atom="H; N 1 2.0; H 2 2.0 1 104.2458898548",
                       basis="cc-pVTZ", unit="AU", spin=1, verbose=0).build()
        mf = dh.DFDH(mol, xc="B2GPPLYP-D3BJ", frozen="FreezeNobleGasCore").run()
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)

    @unittest.skip("frozen core not yet supported")
    def test_XYG3(self):
        # reference: MRCC 2022-03-18
        REF_ETOT = -55.863759071398
        mol = gto.Mole(atom="N; H 1 2; H 1 2 2 104.2458898548", spin=1, unit="AU", basis="cc-pVTZ").build()
        mf = dh.DFDH(mol, xc="XYG3", frozen="FreezeNobleGasCore").run()
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)

    @unittest.skip("frozen core not yet supported")
    def test_RS_PBE_P86(self):
        # reference: MRCC 2022-03-18
        REF_ETOT = -5.576733720503658E+01
        mol = gto.Mole(
            atom="""
                N     0.00000000    0.00000000   -0.12502304
                H     0.00000000    1.43266384    0.99210317
                H     0.00000000   -1.43266384    0.99210317""",
            basis="aug-cc-pVDZ", spin=1, unit="AU").build()
        mf = dh.DFDH(mol, xc="RS-PBE-P86", frozen="FreezeNobleGasCore").run()
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)

    @unittest.skip("frozen core not yet supported")
    def test_RS_PBE_PBE(self):
        # reference: MRCC 2022-03-18
        REF_ETOT = -55.753154450773
        mol = gto.Mole(
            atom="""
                N     0.00000000    0.00000000   -0.12502304
                H     0.00000000    1.43266384    0.99210317
                H     0.00000000   -1.43266384    0.99210317""",
            basis="aug-cc-pVDZ", spin=1, unit="AU").build()
        mf = dh.DFDH(mol, xc="RS-PBE-PBE", frozen="FreezeNobleGasCore").run()
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)

    @unittest.skip("frozen core not yet supported")
    def test_RS_B88_LYP(self):
        # reference: MRCC 2022-03-18
        REF_ETOT = -55.775322590985
        mol = gto.Mole(
            atom="""
                N     0.00000000    0.00000000   -0.12502304
                H     0.00000000    1.43266384    0.99210317
                H     0.00000000   -1.43266384    0.99210317""",
            basis="aug-cc-pVDZ", spin=1, unit="AU").build()
        mf = dh.DFDH(mol, xc="RS-B88-LYP", frozen="FreezeNobleGasCore").run()
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)

    @unittest.skip("frozen core not yet supported")
    def test_RS_PW91_PW91(self):
        # reference: MRCC 2022-03-18
        REF_ETOT = -55.773193557166
        mol = gto.Mole(
            atom="""
                N     0.00000000    0.00000000   -0.12502304
                H     0.00000000    1.43266384    0.99210317
                H     0.00000000   -1.43266384    0.99210317""",
            basis="aug-cc-pVDZ", spin=1, unit="AU").build()
        mf = dh.DFDH(mol, xc="RS-PW91-PW91", frozen="FreezeNobleGasCore").run()
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)
