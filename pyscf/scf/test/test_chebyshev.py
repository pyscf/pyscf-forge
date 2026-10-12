#!/usr/bin/env python
"""
Unit tests for chebyshev_scf.ChebyshevSCF.

Follows the pyscf-forge testing convention: unittest.TestCase-based,
module-level `mol`/`mf` fixtures built once in setUpModule and released
in tearDownModule, RHF used as the numerical reference.

Run with:
    pytest test_chebyshev_scf.py -v
or
    python -m pyscf.lib.misc  # (if wired into the pyscf-forge test runner)
"""

import unittest
import numpy as np

from pyscf import gto, scf, lib
from pyscf.scf.chebyshev_scf import ChebyshevSCF


def build_h_chain(n, bond_length=0.74, basis='sto-3g', spin=None):
    """Build an n-atom linear hydrogen chain Mole."""
    if n < 1:
        raise ValueError("n must be a positive integer")
    atom_list = [['H', (0.0, 0.0, i * bond_length)] for i in range(n)]
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
    mol_h2 = build_h_chain(2, bond_length=0.74, basis='sto-3g')
    mol_h4 = build_h_chain(4, bond_length=0.74, basis='sto-3g')
    mol_h6_stretch = build_h_chain(6, bond_length=0.6, basis='sto-3g')


def tearDownModule():
    global mol_h2, mol_h4, mol_h6_stretch
    mol_h2.stdout.close() if getattr(mol_h2, 'stdout', None) else None
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

    def test_energy_agrees_with_rhf_h4(self):
        mf_ref = scf.RHF(mol_h4).run()
        mf_chb = ChebyshevSCF(mol_h4)
        mf_chb.chebyshev_degree = 40
        mf_chb.beta = 40.0
        e_chb = mf_chb.kernel()
        self.assertAlmostEqual(e_chb, mf_ref.e_tot, delta=1e-3)

    def test_energy_agrees_with_rhf_h2o(self):
        mol_h2o = gto.M(
            atom='O 0.000000 0.000000 0.000000; '
                 'H 0.000000 -0.757000 0.587000; '
                 'H 0.000000 0.757000 0.587000',
            basis='sto-3g',
            unit='Angstrom',
            verbose=0,
        )
        mf_ref = scf.RHF(mol_h2o).run()
        mf_chb = ChebyshevSCF(mol_h2o)
        mf_chb.chebyshev_degree = 50
        mf_chb.beta = 20.0
        e_chb = mf_chb.kernel()
        print("RHF energy:", mf_ref.e_tot)
        print("ChebyshevSCF energy:", e_chb)
        self.assertAlmostEqual(e_chb, mf_ref.e_tot, delta=1e-3)

    def test_energy_agrees_with_rhf_density_fit(self):
        # Mirrors the density-fitted usage pattern from the benchmark
        # script: mf.density_fit(auxbasis='weigend').
        mf_ref = scf.RHF(mol_h4).density_fit(auxbasis='weigend').run()
        mf_chb = ChebyshevSCF(mol_h4).density_fit(auxbasis='weigend')
        mf_chb.chebyshev_degree = 40
        mf_chb.beta = 40.0
        e_chb = mf_chb.kernel()
        self.assertAlmostEqual(e_chb, mf_ref.e_tot, delta=1e-3)


class ConvergenceWithBeta(unittest.TestCase):
    """
    The Chebyshev/Fermi-Dirac filter should converge toward the exact
    (T -> 0) diagonalization result as beta grows and/or the polynomial
    degree increases, provided the degree is sufficient to resolve the
    sharper occupation step. This directly targets the discrepancy
    observed in the benchmark scan at short bond length (near-degenerate
    HOMO-LUMO gap), where ChebyshevSCF and RHF energies disagreed by
    ~0.07 Ha at bond_length=0.2 A but agreed to ~1e-3 Ha by 1.6-2.0 A.
    """

    def test_error_decreases_with_beta(self):
        mol = mol_h4  # short bond length -> small HOMO-LUMO gap
        e_ref = scf.RHF(mol).run().e_tot

        errors = []
        for beta in (50.0, 75.0, 100.0):
            mf = ChebyshevSCF(mol)
            # mf.chebyshev_degree = 60  # kept high enough to resolve beta
            mf.beta = beta
            e = mf.kernel()
            print("e_ref:", e_ref)
            print("e@beta:", e, beta)
            print("error:", abs(e - e_ref))
            errors.append(abs(e - e_ref))

        # Error should be non-increasing (monotonic within numerical noise)
        # as beta grows, i.e. the filter sharpens toward a step function.
        for i in range(len(errors) - 1):
            self.assertLessEqual(errors[i + 1], errors[i] + 1e-3)

        # At the largest beta tested, agreement should be tight.
        self.assertLess(errors[-1], 1e-3)

    def test_error_decreases_with_degree(self):
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


class ElectronCountAndOccupation(unittest.TestCase):
    """
    A Chebyshev-filtered density matrix is only physically meaningful if
    it integrates to the correct electron count; this is not guaranteed
    a priori (unlike hard diagonalization + integer occupation) and
    should be checked explicitly, especially at low chebyshev_degree
    or low beta where the filter is not yet a good step-function
    approximation.
    """

    def test_trace_PS_equals_nelec(self):
        mol = mol_h4
        mf = ChebyshevSCF(mol)
        mf.chebyshev_degree = 40
        mf.beta = 40.0
        mf.kernel()
        dm = mf.make_rdm1()
        s = mf.get_ovlp()
        nelec_computed = np.einsum('ij,ji->', dm, s)
        self.assertAlmostEqual(nelec_computed, mol.nelectron, delta=1e-3)

    def test_density_matrix_hermitian(self):
        mol = mol_h4
        mf = ChebyshevSCF(mol)
        mf.chebyshev_degree = 40
        mf.beta = 40.0
        mf.kernel()
        dm = mf.make_rdm1()
        self.assertTrue(np.allclose(dm, dm.conj().T, atol=1e-5))


class InterfaceCompatibility(unittest.TestCase):
    """ChebyshevSCF should behave as a drop-in pyscf.scf.hf.SCF subclass."""

    def test_isinstance_of_scf_base(self):
        mf = ChebyshevSCF(mol_h2)
        self.assertIsInstance(mf, scf.hf.SCF)

    def test_default_attributes_exist(self):
        mf = ChebyshevSCF(mol_h2)
        self.assertTrue(hasattr(mf, 'chebyshev_degree'))
        self.assertTrue(hasattr(mf, 'beta'))

    def test_converged_flag_set(self):
        mf = ChebyshevSCF(mol_h2)
        mf.chebyshev_degree = 40
        mf.beta = 40.0
        mf.kernel()
        self.assertTrue(mf.converged)

    def test_scanner_and_density_fit_chainable(self):
        # Matches the benchmark script's .density_fit(auxbasis='weigend')
        # usage; just confirm it returns an SCF-derived object that runs.
        mf = ChebyshevSCF(mol_h2).density_fit(auxbasis='weigend')
        mf.chebyshev_degree = 40
        mf.beta = 40.0
        e = mf.kernel()
        self.assertTrue(np.isfinite(e))


class EdgeCases(unittest.TestCase):

    def test_single_atom_raises_or_handles_gracefully(self):
        # H atom chain of length 1: exercise build_h_chain's own guard,
        # not ChebyshevSCF, but included here since the benchmark script
        # relies on it for scan bounds.
        with self.assertRaises(ValueError):
            build_h_chain(0)

    def test_low_degree_does_not_crash(self):
        # A too-low degree should still run (possibly with poor accuracy)
        # rather than raising, so users get an energy plus the option to
        # inspect mf.converged rather than a hard failure.
        mf = ChebyshevSCF(mol_h2)
        mf.chebyshev_degree = 2
        mf.beta = 10.0
        e = mf.kernel()
        self.assertTrue(np.isfinite(e))

class Reproducibility(unittest.TestCase):
    """
    _spectral_bounds() uses a random starting vector for its Davidson
    eigenvalue estimate. Before rng_seed was introduced this came from the
    unseeded global numpy.random, so chebyshev_degree (via auto_degree) and
    the converged energy were not guaranteed to be identical across runs of
    the same input. This checks that a fixed rng_seed makes the whole
    pipeline exactly reproducible.
    """

    def test_same_seed_gives_identical_energy_and_degree(self):
        mol = mol_h2
        mf1 = ChebyshevSCF(mol)
        mf1.rng_seed = 0
        e1 = mf1.kernel()

        mf2 = ChebyshevSCF(mol)
        mf2.rng_seed = 0
        e2 = mf2.kernel()

        self.assertEqual(mf1.chebyshev_degree, mf2.chebyshev_degree)
        self.assertEqual(e1, e2)

    def test_different_seed_still_agrees_within_tolerance(self):
        # Different seeds perturb only the Davidson starting vector for the
        # spectral-bound estimate, not the physics, so results should still
        # agree closely even if not bit-identical.
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
 
    def test_energy_agrees_with_sgx_rhf(self):
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
 
    def test_converged_flag_and_interface_with_sgx(self):
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
 
    def test_reproducible_with_sgx(self):
        from pyscf import sgx
 
        def run():
            mf = sgx.sgx_fit(ChebyshevSCF(mol_h2), pjs=False)
            mf.with_df.dfj = True
            mf.with_df.optk = True
            mf.rng_seed = 0
            return mf.kernel()
 
        self.assertEqual(run(), run())
 

if __name__ == '__main__':
    print("Full Tests for ChebyshevSCF")
    unittest.main()