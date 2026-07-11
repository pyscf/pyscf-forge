import unittest
from pyscf import dh, gto


class TestRDFDH(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="cc-pVDZ", verbose=0).build()

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
            ("XYGJ-OS", 8, 4),
            ("SCS-MP2", 8, 4),
        ]
        for xc, places_native, places_dfmp2 in tests:
            ref = self._ref(xc)
            self._check(xc, "dfmp2_native", ref, places_native)
            self._check(xc, "dfmp2", ref, places_dfmp2)

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

    def test_dsdpbep86_d3bj(self):
        """Cross-validation against REST (Rust Electronic-Structure Toolkit)."""
        from pyscf.dh.xccode import parse_xc_dh
        (xc, xc_n, c_os, c_ss), xc_add = parse_xc_dh("DSD-PBEP86-D3BJ")
        self.assertEqual(xc, "0.69*HF + 0.31*PBE, 0.44*P86")
        self.assertAlmostEqual(c_os, 0.52, places=6)
        self.assertAlmostEqual(c_ss, 0.22, places=6)
        self.assertEqual(xc_add["D3"]["version"], "d3bj")

        REF_SCF = -56.3349732703
        REF_ETOT = -56.4305347110
        mol = gto.Mole(atom="""
            N  -2.1988391019  1.8973746268  0.0000000000
            H  -1.1788391019  1.8973746268  0.0000000000
            H  -2.5388353987  1.0925460144 -0.5263586446
            H  -2.5388400276  2.7556271745 -0.4338224694
        """, basis="cc-pVDZ", unit="Angstrom", verbose=0).build()
        mf = dh.DFDH(mol, xc="DSD-PBEP86-D3BJ", auxbasis_jk="def2-universal-jkfit", auxbasis_ri="def2-universal-jkfit").run()
        self.assertAlmostEqual(mf.mf_s.e_tot, REF_SCF, places=3)
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=6)
