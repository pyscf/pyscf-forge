#!/usr/bin/env python

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np


os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

HAS_JAX = (
    importlib.util.find_spec("jax") is not None and importlib.util.find_spec("jaxlib") is not None
)

if HAS_JAX:
    from pyscf import afqmc, cc, gto, scf


@unittest.skipUnless(HAS_JAX, "requires jax and jaxlib")
class KnownValues(unittest.TestCase):
    RHF_REF = -1.1288555895527692
    CISD_REF = -1.1372839078633414

    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(
            atom="H 0 0 0; H 0 0 0.74",
            basis="sto-3g",
            verbose=0,
        )
        cls.mf = scf.RHF(cls.mol).run()
        cls.mycc = cc.CCSD(cls.mf)
        cls.mycc.kernel()

    def _make_rhf_calc(self):
        calc = afqmc.AFQMC(self.mf)
        calc.n_walkers = 4
        calc.n_eql_blocks = 2
        calc.n_blocks = 8
        calc.seed = 7
        calc.mixed_precision = False
        return calc

    def _make_cisd_calc(self):
        calc = afqmc.AFQMC(self.mycc)
        calc.n_walkers = 4
        calc.n_eql_blocks = 2
        calc.n_blocks = 8
        calc.seed = 11
        calc.mixed_precision = False
        return calc

    def test_rhf_kernel_smoke(self):
        mean, err = self._make_rhf_calc().kernel()
        self.assertTrue(np.isfinite(mean))
        self.assertTrue(np.isnan(err) or np.isfinite(err))
        self.assertAlmostEqual(mean, self.RHF_REF, places=6)

    def test_cisd_kernel_smoke(self):
        mean, err = self._make_cisd_calc().kernel()
        self.assertTrue(np.isfinite(mean))
        self.assertTrue(np.isnan(err) or np.isfinite(err))
        self.assertAlmostEqual(mean, self.CISD_REF, places=6)

    def test_from_staged_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "h2_afqmc.h5"
            source = afqmc.AFQMC(self.mycc)
            source.save_staged(path)

            loaded = afqmc.AFQMC.from_staged(
                path,
                n_eql_blocks=2,
                n_blocks=8,
                seed=11,
                n_walkers=4,
            )
            loaded.mixed_precision = False
            mean, err = loaded.kernel()

            self.assertTrue(path.exists())
            self.assertEqual(loaded.staged.trial.kind.lower(), "cisd")
            self.assertTrue(np.isfinite(mean))
            self.assertTrue(np.isnan(err) or np.isfinite(err))
            self.assertAlmostEqual(mean, self.CISD_REF, places=6)


if __name__ == "__main__":
    print("Tests for AFQMC phaseless driver")
    unittest.main()
