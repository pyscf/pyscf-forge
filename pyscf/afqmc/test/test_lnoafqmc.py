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
    from pyscf import gto, scf
    from pyscf.afqmc.afqmc import AfqmcLnoFrag, run_afqmc_lno_helper
    from pyscf.afqmc.staging import load as load_staged
    from pyscf.afqmc.lnoafqmc import LNOAFQMC, prep_local_orbitals


@unittest.skipUnless(HAS_JAX, "requires jax and jaxlib")
class KnownValues(unittest.TestCase):
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
        cls.mf = scf.RHF(cls.mol).density_fit()
        cls.mf.kernel()

    def test_prep_local_orbitals_pm(self):
        lo_coeff, frag_lolist = prep_local_orbitals(self.mf)
        nocc = int(np.count_nonzero(self.mf.mo_occ))

        self.assertEqual(lo_coeff.shape[1], nocc)
        self.assertEqual(len(frag_lolist), nocc)
        self.assertEqual(frag_lolist, [[i] for i in range(nocc)])

    def test_prep_local_orbitals_rejects_unknown_method(self):
        with self.assertRaisesRegex(ValueError, "not supported"):
            prep_local_orbitals(self.mf, localization_method="iao")

    def test_calc_rhf_matches_trot_helper_refs(self):
        refs = [
            (-0.06714345, 0.01170026, 0),
            (-0.04946941, 0.00817172, 1),
        ]

        for e_ref, err_ref, norb_frozen in refs:
            with self.subTest(norb_frozen=norb_frozen):
                mo_coeff = self.mf.mo_coeff
                frozen_orbitals = np.array([i for i in range(norb_frozen)], dtype=np.int64)
                norb_act = mo_coeff.shape[1] - norb_frozen
                nactocc = self.mol.nelectron // 2 - norb_frozen
                prjlo = np.array([[0] for _ in range(nactocc - 1)] + [[1]])

                elcorr_afqmc, err_afqmc = run_afqmc_lno_helper(
                    self.mf,
                    mo_coeff=mo_coeff,
                    norb_act=norb_act,
                    nelec_act=nactocc * 2,
                    frozen_orbitals=frozen_orbitals,
                    n_walkers=5,
                    nblocks=20,
                    seed=1234,
                    chol_cut=1e-5,
                    target_error=0,
                    dt=0.01,
                    prjlo=prjlo,
                    n_eql=4,
                )

                self.assertAlmostEqual(float(elcorr_afqmc), e_ref, places=6)
                self.assertAlmostEqual(float(err_afqmc), err_ref, places=6)

    def test_lno_stage_roundtrips_array_frozen_orbitals(self):
        frozen_orbitals = np.array([0], dtype=np.int64)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "lno_stage.h5"

            af = AfqmcLnoFrag(self.mf, frozen_orbitals=frozen_orbitals)
            af.save_staged(path)

            staged = load_staged(path)
            af_loaded = AfqmcLnoFrag.from_staged(path)

            self.assertTrue(np.array_equal(staged.ham.frozen, frozen_orbitals))
            self.assertTrue(np.array_equal(staged.trial.frozen, frozen_orbitals))
            self.assertTrue(np.array_equal(staged.meta["frozen"], frozen_orbitals))
            self.assertIsNotNone(af_loaded.frozen_orbitals)
            self.assertTrue(np.array_equal(af_loaded.frozen_orbitals, frozen_orbitals))


if __name__ == "__main__":
    print("Tests for AFQMC LNO adapter")
    unittest.main()
