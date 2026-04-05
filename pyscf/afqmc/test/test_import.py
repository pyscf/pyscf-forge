#!/usr/bin/env python

import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HELPER = Path(__file__).with_name("_import_boundary_helper.py")


class KnownValues(unittest.TestCase):
    def test_missing_jax_error_is_friendly(self):
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
        self.assertIn("Mole", output)
        self.assertIn("LNO", output)
        if sys.version_info[:2] < (3, 10):
            self.assertIn("Python 3.10 or newer", output)
            self.assertNotIn("requires a separate JAX installation", output)
        else:
            self.assertIn("requires a separate JAX installation", output)
            self.assertIn("pip install -U jax", output)
            self.assertIn('pip install -U "jax[cuda12]"', output)
            self.assertIn('pip install -U "jax[cuda13]"', output)


if __name__ == "__main__":
    print("Tests for optional JAX import boundary")
    unittest.main()
