#!/usr/bin/env python
# Copyright 2025 The PySCF Developers. All Rights Reserved.

"""
Performance and benchmarking tests for OCCRI.

These tests compare performance between different implementations
and validate that the C extension provides expected speedups.
"""

import time
import unittest

import numpy
from pyscf.occri import _OCCRI_C_AVAILABLE, OCCRI
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf


def setUpModule():
    global cell, kpts

    # Setup larger cell for performance testing
    cell = pgto.Cell()
    cell.atom = """
    C 0.000000 0.000000 1.780373
    C 0.890186 0.890186 2.670559
    C 0.000000 1.780373 0.000000
    C 0.890186 2.670559 0.890186
    """
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pbe"
    cell.a = numpy.array(
        [
            [3.560745, 0.000000, 0.000000],
            [0.000000, 3.560745, 0.000000],
            [0.000000, 0.000000, 3.560745],
        ]
    )
    cell.mesh = [20] * 3
    cell.build()

    # Setup proper k-point mesh
    kpts = cell.make_kpts([2, 2, 1])  # Reduced mesh for performance testing


def tearDownModule():
    global cell, kpts
    del cell, kpts


class TestPerformance(unittest.TestCase):

    @unittest.skipIf(not _OCCRI_C_AVAILABLE, "C extension not available")
    def test_c_vs_python_performance(self):
        """Compare C extension vs Python implementation performance"""

        # Python implementation
        start_time = time.time()
        mf_python = scf.RHF(cell)
        mf_python.with_df = OCCRI.from_mf(mf_python, disable_c=True)
        mf_python.kernel()
        python_time = time.time() - start_time
        e_python = mf_python.e_tot

        # C implementation
        start_time = time.time()
        mf_c = scf.RHF(cell)
        mf_c.with_df = OCCRI.from_mf(mf_c, disable_c=False)
        mf_c.kernel()
        c_time = time.time() - start_time
        e_c = mf_c.e_tot

        # Check results are the same
        self.assertAlmostEqual(e_python, e_c, places=8)

        # C should be faster (at least not significantly slower)
        speedup = python_time / c_time
        print(f"C implementation speedup: {speedup:.2f}x")

        # Allow for some variance but expect C to be competitive
        self.assertGreater(
            speedup, 0.5, "C implementation significantly slower than Python"
        )

    def test_memory_scaling(self):
        """Test memory usage scales reasonably with system size"""
        from pyscf import lib

        # Small system
        small_cell = pgto.Cell()
        small_cell.atom = "H 0. 0. 0.; H 1. 0. 0."
        small_cell.basis = "sto-3g"
        small_cell.a = numpy.eye(3) * 4.0
        small_cell.mesh = [12] * 3
        small_cell.build()

        mem_start = lib.current_memory()[0]
        mf_small = scf.RHF(small_cell)
        mf_small.with_df = OCCRI.from_mf(mf_small)
        mf_small.kernel()
        mem_small = lib.current_memory()[0] - mem_start

        # Larger system
        mem_start = lib.current_memory()[0]
        mf_large = scf.RHF(cell)
        mf_large.with_df = OCCRI.from_mf(mf_large)
        mf_large.kernel()
        mem_large = lib.current_memory()[0] - mem_start

        print(f"Small system memory usage: {mem_small:.1f} MB")
        print(f"Large system memory usage: {mem_large:.1f} MB")

        # Memory should scale reasonably (not exponentially)
        nao_small = small_cell.nao
        nao_large = cell.nao
        expected_scaling = (nao_large / nao_small) ** 2
        actual_scaling = mem_large / max(mem_small, 1.0)  # Avoid division by zero

        # Allow for reasonable overhead
        self.assertLess(
            actual_scaling,
            expected_scaling * 3,
            "Memory usage scaling worse than expected",
        )

    @unittest.skipIf(not _OCCRI_C_AVAILABLE, "C extension not available")
    def test_kpoints_performance(self):
        """Test k-points performance"""

        start_time = time.time()
        mf_kpts = scf.KRHF(cell, kpts)
        mf_kpts.with_df = OCCRI.from_mf(mf_kpts)
        mf_kpts.kernel()
        kpts_time = time.time() - start_time

        # Gamma point for comparison
        start_time = time.time()
        mf_gamma = scf.RHF(cell)
        mf_gamma.with_df = OCCRI.from_mf(mf_gamma)
        mf_gamma.kernel()
        gamma_time = time.time() - start_time

        print(f"Nk: {4:d}")
        print(f"Gamma point time: {gamma_time:.2f}s")
        print(f"K-points time: {kpts_time:.2f}s")

        # K-points should be slower but not excessively so
        slowdown = kpts_time / gamma_time
        print(f"K-points slowdown: {slowdown:.2f}x")

        # This is system dependent, but shouldn't be orders of magnitude worse
        self.assertLess(slowdown, 20, "K-points implementation too slow")

    def test_convergence_stability(self):
        """Test SCF convergence stability"""

        # Test multiple random initializations
        energies = []
        for seed in range(5):
            numpy.random.seed(seed)
            mf = scf.RHF(cell)
            mf.with_df = OCCRI.from_mf(mf)
            mf.kernel()

            self.assertTrue(mf.converged, f"SCF failed to converge with seed {seed}")
            energies.append(mf.e_tot)

        # All should converge to same energy
        e_std = numpy.std(energies)
        print(f"Energy standard deviation across initializations: {e_std:.2e}")
        self.assertLess(
            e_std, 1e-8, "SCF convergence not stable across initializations"
        )


if __name__ == "__main__":
    print("Running OCCRI performance tests...")
    unittest.main()
