#!/usr/bin/env python

import importlib.util
import os
import unittest

import numpy as np


os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

HAS_JAX = (
    importlib.util.find_spec("jax") is not None and importlib.util.find_spec("jaxlib") is not None
)

if HAS_JAX:
    from pyscf import afqmc, gto, scf


@unittest.skipUnless(HAS_JAX, "requires jax and jaxlib")
class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(
            atom="H 0 0 0; H 0 0 0.74",
            basis="sto-3g",
            verbose=0,
        )
        cls.mf = scf.RHF(cls.mol).run()

    def test_free_projection_smoke(self):
        calc = afqmc.AFQMCFP(self.mf)
        calc.n_walkers = 4
        calc.n_blocks = 2
        calc.n_prop_steps = 2
        calc.n_traj = 2
        calc.seed = 5
        calc.ene0 = self.mf.e_tot

        mean, err = calc.kernel()
        mean = np.asarray(mean)
        err = np.asarray(err)

        self.assertEqual(mean.shape, (calc.n_blocks + 1,))
        self.assertEqual(err.shape, (calc.n_blocks + 1,))
        self.assertTrue(np.all(np.isfinite(mean.real)))
        self.assertTrue(np.all(np.isfinite(mean.imag)))
        self.assertTrue(np.all(np.isfinite(err)))
        self.assertAlmostEqual(float(mean[0].real), float(self.mf.e_tot), places=12)
        self.assertAlmostEqual(float(err[0]), 0.0, places=12)


if __name__ == "__main__":
    print("Tests for AFQMC free projection driver")
    unittest.main()
