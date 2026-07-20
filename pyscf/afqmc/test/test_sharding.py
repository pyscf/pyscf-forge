#!/usr/bin/env python

import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HELPER = Path(__file__).with_name("_sharding_helper.py")
HAS_JAX = (
    importlib.util.find_spec("jax") is not None and importlib.util.find_spec("jaxlib") is not None
)


@unittest.skipUnless(HAS_JAX, "requires jax and jaxlib")
class KnownValues(unittest.TestCase):
    def test_data_sharding_matches_unsharded_block(self):
        env = os.environ.copy()
        env["PYSCF_EXT_PATH"] = str(ROOT)
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [sys.executable, str(HELPER)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        self.assertIn("ok", output)


if __name__ == "__main__":
    print("Tests for AFQMC sharding")
    unittest.main()
