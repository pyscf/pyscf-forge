import unittest
from pyscf import dh, gto


class TestUDFDH(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mol = gto.Mole(atom="C 0. 0. 0.; H .9 0. 0.; H 0. 1. 0.; H 0. 0. 1.1", basis="cc-pVDZ", spin=1, verbose=0).build()

    def _ref(self, xc):
        return dh.DFDH(self.mol, xc=xc, mp2_backend="ajz").run().e_tot

    def _check(self, xc, backend, ref, places):
        mf = dh.DFDH(self.mol, xc=xc, mp2_backend=backend).run()
        self.assertAlmostEqual(mf.e_tot, ref, places=places,
            msg=f"{xc} {backend}: {mf.e_tot:.10f} != {ref:.10f}")

    def test_mp2_backend(self):
        tests = [
            ("MP2", 8, 4),
            ("B2PLYP", 8, 5),
            ("SCS-MP2", 8, 4),
        ]
        for xc, places_native, places_dfmp2 in tests:
            ref = self._ref(xc)
            self._check(xc, "dfmp2_native", ref, places_native)
            self._check(xc, "dfmp2", ref, places_dfmp2)

    def test_B2GPPLYP_D3BJ(self):
        # reference: MRCC 2022-03-18
        REF_ETOT = -55.841686701788
        mol = gto.Mole(atom="H; N 1 2.0; H 2 2.0 1 104.2458898548",
                       basis="cc-pVTZ", unit="AU", spin=1, verbose=0).build()
        mf = dh.DFDH(mol, xc="B2GPPLYP-D3BJ", mp2_backend="dfmp2", frozen=1).run()
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=4)

    def test_XYG3(self):
        # reference: MRCC 2022-03-18
        REF_ETOT = -55.863759071398
        mol = gto.Mole(atom="N; H 1 2; H 1 2 2 104.2458898548", spin=1, unit="AU", basis="cc-pVTZ").build()
        mf = dh.DFDH(mol, xc="XYG3", mp2_backend="dfmp2", frozen=1).run()
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=4)
