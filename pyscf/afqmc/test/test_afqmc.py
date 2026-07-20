#!/usr/bin/env python

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

HAS_JAX = (
    importlib.util.find_spec("jax") is not None
    and importlib.util.find_spec("jaxlib") is not None
)

if HAS_JAX:
    from pyscf import afqmc, cc, gto, scf


@unittest.skipUnless(HAS_JAX, "requires jax and jaxlib")
class KnownValues(unittest.TestCase):
    RHF_REF = -75.75594174131398
    RHF_ERR_REF = 0.01213379336719581
    CISD_REF = -75.72869718476204
    CISD_ERR_REF = 0.0002352938315467452
    UHF_REF = -55.43066756011652
    UHF_ERR_REF = 0.00761980459817991
    UCISD_REF = -55.41533781603285
    UCISD_ERR_REF = 0.0001071700818560977
    RHF_WALKER_KINDS = ("restricted", "unrestricted")

    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(
            atom="""
            O        0.0000000000      0.0000000000      0.0000000000
            H        0.9562300000      0.0000000000      0.0000000000
            H       -0.2353791634      0.9268076728      0.0000000000
            """,
            basis="sto-6g",
            verbose=0,
        )
        cls.mf = scf.RHF(cls.mol)
        cls.mf.kernel()
        cls.mycc = cc.CCSD(cls.mf)
        cls.mycc.kernel()

        cls.mol_open = gto.M(
            atom="""
            N        0.0000000000      0.0000000000      0.0000000000
            H        1.0225900000      0.0000000000      0.0000000000
            H       -0.2281193615      0.9968208791      0.0000000000
            """,
            basis="sto-6g",
            spin=1,
            verbose=0,
        )
        cls.mf_open = scf.UHF(cls.mol_open).newton()
        cls.mf_open.kernel()
        cls.myucc = cc.UCCSD(cls.mf_open)
        cls.myucc.kernel()

    def _make_rhf_calc(self, walker_kind="restricted"):
        calc = afqmc.AFQMC(self.mf)
        calc.walker_kind = walker_kind
        calc.n_walkers = 5
        calc.n_eql_blocks = 4
        calc.n_blocks = 20
        calc.seed = 1234
        calc.chol_cut = 1e-6
        calc.mixed_precision = False
        return calc

    def _make_cisd_calc(self):
        calc = afqmc.AFQMC(self.mycc)
        calc.walker_kind = "restricted"
        calc.n_walkers = 5
        calc.n_eql_blocks = 4
        calc.n_blocks = 20
        calc.seed = 1234
        calc.chol_cut = 1e-6
        calc.mixed_precision = False
        return calc

    def _make_uhf_calc(self):
        calc = afqmc.AFQMC(self.mf_open)
        calc.walker_kind = "unrestricted"
        calc.n_walkers = 5
        calc.n_eql_blocks = 4
        calc.n_blocks = 20
        calc.seed = 1234
        calc.chol_cut = 1e-6
        calc.mixed_precision = False
        return calc

    def _make_ucisd_calc(self):
        calc = afqmc.AFQMC(self.myucc)
        calc.walker_kind = "unrestricted"
        calc.n_walkers = 5
        calc.n_eql_blocks = 4
        calc.n_blocks = 20
        calc.seed = 1234
        calc.chol_cut = 1e-6
        calc.mixed_precision = False
        return calc

    def _assert_matches_trot(self, mean, err, e_ref, err_ref):
        self.assertTrue(np.isfinite(mean))
        self.assertTrue(np.isnan(err) or np.isfinite(err))
        self.assertTrue(np.isclose(mean, e_ref), (mean, e_ref, mean - e_ref))
        self.assertTrue(np.isclose(err, err_ref), (err, err_ref, err - err_ref))

    def test_rhf_kernel(self):
        for walker_kind in self.RHF_WALKER_KINDS:
            with self.subTest(walker_kind=walker_kind):
                mean, err = self._make_rhf_calc(walker_kind=walker_kind).kernel()
                self._assert_matches_trot(mean, err, self.RHF_REF, self.RHF_ERR_REF)

    def test_cisd_kernel(self):
        mean, err = self._make_cisd_calc().kernel()
        self._assert_matches_trot(mean, err, self.CISD_REF, self.CISD_ERR_REF)

    def test_uhf_kernel(self):
        mean, err = self._make_uhf_calc().kernel()
        self._assert_matches_trot(mean, err, self.UHF_REF, self.UHF_ERR_REF)

    def test_ucisd_kernel(self):
        mean, err = self._make_ucisd_calc().kernel()
        self._assert_matches_trot(mean, err, self.UCISD_REF, self.UCISD_ERR_REF)

    def test_from_staged_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "h2o_afqmc.h5"
            source = afqmc.AFQMC(self.mycc)
            source.walker_kind = "restricted"
            source.chol_cut = 1e-6
            source.save_staged(path)

            loaded = afqmc.AFQMC.from_staged(
                path,
                n_eql_blocks=4,
                n_blocks=20,
                seed=1234,
                n_walkers=5,
            )
            loaded.walker_kind = "restricted"
            loaded.mixed_precision = False
            mean, err = loaded.kernel()

            self.assertTrue(path.exists())
            self.assertEqual(loaded.staged.trial.kind.lower(), "cisd")
            self._assert_matches_trot(mean, err, self.CISD_REF, self.CISD_ERR_REF)


if __name__ == "__main__":
    print("Tests for AFQMC phaseless driver")
    unittest.main()
