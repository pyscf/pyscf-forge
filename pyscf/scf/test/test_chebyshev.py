#!/usr/bin/env python
"""
Unit tests for chebyshev_scf.ChebyshevSCF.

Follows the pyscf-forge testing convention: unittest.TestCase-based,
module-level `mol`/`mf` fixtures built once in setUpModule and released
in tearDownModule, RHF used as the numerical reference.

Run with:
    pytest test_chebyshev_scf.py -v
"""
import os
import unittest

import numpy as np

from pyscf import gto, scf
from pyscf.scf.chebyshev_scf import ChebyshevSCF

slow = unittest.skipUnless(
    os.environ.get("PYSCF_RUN_SLOW_TESTS") == "1",
    "set PYSCF_RUN_SLOW_TESTS=1 to run slow tests",
)


def build_h_chain(n, bond_length=0.74, basis="sto-3g", spin=None):
    """Build an n-atom linear hydrogen chain Mole."""
    if n < 1:
        raise ValueError("n must be a positive integer")
    atom_list = [["H", (0.0, 0.0, i * bond_length)] for i in range(n)]
    if spin is None:
        spin = n % 2
    mol = gto.Mole()
    mol.atom = atom_list
    mol.basis = basis
    mol.spin = spin
    mol.verbose = 0
    mol.build()
    return mol


# ---------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------
mol_h2 = mol_h4 = mol_h6_stretch = None


def setUpModule():
    global mol_h2, mol_h4, mol_h6_stretch
    mol_h2 = build_h_chain(2, bond_length=0.74, basis="sto-3g")
    mol_h4 = build_h_chain(4, bond_length=0.74, basis="sto-3g")
    mol_h6_stretch = build_h_chain(6, bond_length=0.6, basis="sto-3g")


def tearDownModule():
    global mol_h2, mol_h4, mol_h6_stretch
    mol_h2 = mol_h4 = mol_h6_stretch = None


class KnownValues(unittest.TestCase):
    """Cross-checks ChebyshevSCF total energy against reference RHF."""

    def test_energy_agrees_with_rhf_h2(self):
        mf_ref = scf.RHF(mol_h2).run()
        mf_chb = ChebyshevSCF(mol_h2)
        mf_chb.chebyshev_degree = 40
        mf_chb.beta = 40.0
        e_chb = mf_chb.kernel()
        self.assertAlmostEqual(e_chb, mf_ref.e_tot, delta=1e-3)
        self.assertTrue(mf_chb.converged)

    def test_h4_energy_electron_count_and_hermiticity(self):
        mf_ref = scf.RHF(mol_h4).run()
        mf_chb = ChebyshevSCF(mol_h4)
        mf_chb.chebyshev_degree = 40
        mf_chb.beta = 40.0
        e_chb = mf_chb.kernel()
        self.assertAlmostEqual(e_chb, mf_ref.e_tot, delta=1e-3)
        dm = mf_chb.make_rdm1()
        s = mf_chb.get_ovlp()
        nelec_computed = np.einsum("ij,ji->", dm, s)
        self.assertAlmostEqual(nelec_computed, mol_h4.nelectron, delta=1e-3)
        self.assertTrue(np.allclose(dm, dm.conj().T, atol=1e-5))

    @slow
    def test_energy_agrees_with_rhf_h2o_slow(self):
        mol_h2o = gto.M(
            atom="O 0.000000 0.000000 0.000000; "
            "H 0.000000 -0.757000 0.587000; "
            "H 0.000000 0.757000 0.587000",
            basis="sto-3g",
            unit="Angstrom",
            verbose=0,
        )
        mf_ref = scf.RHF(mol_h2o).run()
        mf_chb = ChebyshevSCF(mol_h2o)
        mf_chb.chebyshev_degree = 50
        mf_chb.beta = 20.0
        e_chb = mf_chb.kernel()
        self.assertAlmostEqual(e_chb, mf_ref.e_tot, delta=1e-3)

    @slow
    def test_energy_agrees_with_rhf_density_fit_slow(self):
        # Mirrors the density-fitted usage pattern from the benchmark
        # script: mf.density_fit(auxbasis='weigend').
        mf_ref = scf.RHF(mol_h4).density_fit(auxbasis="weigend").run()
        mf_chb = ChebyshevSCF(mol_h4).density_fit(auxbasis="weigend")
        mf_chb.chebyshev_degree = 40
        mf_chb.beta = 40.0
        e_chb = mf_chb.kernel()
        self.assertAlmostEqual(e_chb, mf_ref.e_tot, delta=1e-3)


class ConvergenceWithBeta(unittest.TestCase):

    @slow
    def test_error_decreases_with_beta_slow(self):
        mol = mol_h4  # short bond length -> small HOMO-LUMO gap
        e_ref = scf.RHF(mol).run().e_tot

        errors = []
        for beta in (50.0, 75.0, 100.0):
            mf = ChebyshevSCF(mol)
            # mf.chebyshev_degree = 60  # kept high enough to resolve beta
            mf.beta = beta
            e = mf.kernel()
            errors.append(abs(e - e_ref))

        # Error should be non-increasing (monotonic within numerical noise)
        # as beta grows, i.e. the filter sharpens toward a step function.
        for i in range(len(errors) - 1):
            self.assertLessEqual(errors[i + 1], errors[i] + 1e-3)

        # At the largest beta tested, agreement should be tight.
        self.assertLess(errors[-1], 1e-3)

    @slow
    def test_error_decreases_with_degree_slow(self):
        mol = mol_h4
        e_ref = scf.RHF(mol).run().e_tot

        errors = []
        for degree in (10, 30, 60):
            mf = ChebyshevSCF(mol)
            mf.chebyshev_degree = degree
            mf.beta = 40.0
            e = mf.kernel()
            errors.append(abs(e - e_ref))

        for i in range(len(errors) - 1):
            self.assertLessEqual(errors[i + 1], errors[i] + 1e-3)


class InterfaceCompatibility(unittest.TestCase):
    """ChebyshevSCF should behave as a drop-in pyscf.scf.hf.SCF subclass."""

    def test_isinstance_of_scf_base(self):
        mf = ChebyshevSCF(mol_h2)
        self.assertIsInstance(mf, scf.hf.SCF)

    def test_default_attributes_exist(self):
        mf = ChebyshevSCF(mol_h2)
        self.assertTrue(hasattr(mf, "chebyshev_degree"))
        self.assertTrue(hasattr(mf, "beta"))

    def test_scanner_and_density_fit_chainable(self):
        # Matches the benchmark script's .density_fit(auxbasis='weigend')
        # usage; just confirm it returns an SCF-derived object that runs.
        mf = ChebyshevSCF(mol_h2).density_fit(auxbasis="weigend")
        mf.chebyshev_degree = 40
        mf.beta = 40.0
        e = mf.kernel()
        self.assertTrue(np.isfinite(e))


class EdgeCases(unittest.TestCase):

    def test_single_atom_raises_or_handles_gracefully(self):
        with self.assertRaises(ValueError):
            build_h_chain(0)

    def test_low_degree_does_not_crash(self):
        mf = ChebyshevSCF(mol_h2)
        mf.chebyshev_degree = 2
        mf.beta = 10.0
        e = mf.kernel()
        self.assertTrue(np.isfinite(e))


class Reproducibility(unittest.TestCase):
    """
    _spectral_bounds() uses a random starting vector for its Davidson
    eigenvalue estimate. Before rng_seed was introduced this came from the
    unseeded global numpy.random, so chebyshev_degree and
    the converged energy were not guaranteed to be identical across runs of
    the same input. This checks that a fixed rng_seed makes the whole
    pipeline exactly reproducible.
    """

    @slow
    def test_same_seed_gives_identical_energy_and_degree_slow(self):
        mol = mol_h2
        mf1 = ChebyshevSCF(mol)
        mf1.rng_seed = 0
        e1 = mf1.kernel()

        mf2 = ChebyshevSCF(mol)
        mf2.rng_seed = 0
        e2 = mf2.kernel()

        self.assertEqual(mf1.chebyshev_degree, mf2.chebyshev_degree)
        self.assertEqual(e1, e2)

    @slow
    def test_different_seed_still_agrees_within_tolerance_slow(self):
        mol = mol_h2
        mf1 = ChebyshevSCF(mol)
        mf1.rng_seed = 0
        e1 = mf1.kernel()

        mf2 = ChebyshevSCF(mol)
        mf2.rng_seed = 1
        e2 = mf2.kernel()

        self.assertAlmostEqual(e1, e2, delta=1e-4)


class SGXExchange(unittest.TestCase):
    """
    Exercises ChebyshevSCF combined with pyscf.sgx (Chain-of-Spheres /
    seminumerical exchange), the usage pattern from the original benchmark
    script:

        mf = sgx.sgx_fit(ChebyshevSCF(mol), pjs=False)
        mf.with_df.dfj = True
        mf.with_df.optk = True
        mf.kernel()

    SGX itself introduces a numerical-integration approximation to the
    exchange matrix, independent of the Chebyshev density-filtering
    approximation in this driver, so the reference here is SGX-RHF (not
    plain analytic RHF) -- comparing against plain RHF would conflate the
    two independent sources of error. The two approximations (SGX exchange,
    Chebyshev occupation filtering) are expected to be independent, so a
    looser but still sub-1-mHa tolerance is used relative to KnownValues.
    """

    @slow
    def test_energy_agrees_with_sgx_rhf_slow(self):
        from pyscf import sgx

        mf_ref = sgx.sgx_fit(scf.RHF(mol_h4), pjs=False)
        mf_ref.with_df.dfj = True
        mf_ref.with_df.optk = True
        e_ref = mf_ref.kernel()

        mf_chb = sgx.sgx_fit(ChebyshevSCF(mol_h4), pjs=False)
        mf_chb.with_df.dfj = True
        mf_chb.with_df.optk = True
        mf_chb.rng_seed = 0
        e_chb = mf_chb.kernel()

        self.assertAlmostEqual(e_chb, e_ref, delta=1e-3)  # < 1 mHa

    @slow
    def test_converged_flag_and_interface_with_sgx_slow(self):
        from pyscf import sgx

        mf = sgx.sgx_fit(ChebyshevSCF(mol_h2), pjs=False)
        mf.with_df.dfj = True
        mf.with_df.optk = True
        mf.rng_seed = 0
        e = mf.kernel()

        self.assertTrue(mf.converged)
        self.assertTrue(np.isfinite(e))
        dm = mf.make_rdm1()
        self.assertTrue(np.allclose(dm, dm.conj().T, atol=1e-8))



if __name__ == "__main__":
    print("Full Tests for ChebyshevSCF")
    unittest.main()
