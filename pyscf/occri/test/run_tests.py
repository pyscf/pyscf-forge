#!/usr/bin/env python
"""
Test runner for OCCRI test suite.

Usage:
    python run_tests.py              # Run all tests (OCCRI + ISDFX)
    python run_tests.py --quick      # Run only quick functional tests (OCCRI + ISDFX)
    python run_tests.py --perf       # Run only performance tests (expensive k-points, energy comparisons)
"""

import argparse
import sys
import unittest


def main():
    parser = argparse.ArgumentParser(description="Run OCCRI tests")
    parser.add_argument(
        "--quick", action="store_true", help="Run only quick functional tests"
    )
    parser.add_argument(
        "--perf", action="store_true", help="Run only performance tests"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Setup verbosity
    verbosity = 2 if args.verbose else 1

    if args.quick:
        # Run only main functional tests
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        try:
            from test_occri import TestOCCRI
            suite.addTests(loader.loadTestsFromTestCase(TestOCCRI))
            print("Loaded OCCRI functional tests")
        except ImportError as e:
            print(f"Could not load OCCRI tests: {e}")
            
        # Load ISDFX functional tests
        try:
            from test_isdfx import (TestUtilityFunctions, TestCholeskyDecomposition, 
                                   TestISDFX, TestPivotSelection, TestFittingFunctions,
                                   TestTHCPotential, TestExchangeMatrixEvaluation,
                                   TestISdfxIntegration)
            
            suite.addTests(loader.loadTestsFromTestCase(TestUtilityFunctions))
            suite.addTests(loader.loadTestsFromTestCase(TestCholeskyDecomposition))
            suite.addTests(loader.loadTestsFromTestCase(TestISDFX))
            suite.addTests(loader.loadTestsFromTestCase(TestPivotSelection))
            suite.addTests(loader.loadTestsFromTestCase(TestFittingFunctions))
            suite.addTests(loader.loadTestsFromTestCase(TestTHCPotential))
            suite.addTests(loader.loadTestsFromTestCase(TestExchangeMatrixEvaluation))
            suite.addTests(loader.loadTestsFromTestCase(TestISdfxIntegration))
            print("Loaded ISDFX functional tests")
        except ImportError as e:
            print(f"Could not load ISDFX tests: {e}")

        print("Running quick functional tests...")

    elif args.perf:
        # Run only performance tests
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        # Set flag for expensive tests
        sys.modules[__name__].RUN_PERFORMANCE_TESTS = True
        
        try:
            from test_performance import TestPerformance
            suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
            print("Loaded OCCRI performance tests")
        except ImportError as e:
            print(f"Could not load OCCRI performance tests: {e}")
        
        # Load ISDFX energy tests with expensive k-point tests
        try:
            from test_isdfx_energy import TestISdfxEnergyComparison
            suite.addTests(loader.loadTestsFromTestCase(TestISdfxEnergyComparison))
            print("Loaded ISDFX energy performance tests")
        except ImportError as e:
            print(f"Could not load ISDFX energy tests: {e}")
            
        # Load ISDFX k-point tests (expensive)
        try:
            from test_isdfx import TestISdfxKpoints
            suite.addTests(loader.loadTestsFromTestCase(TestISdfxKpoints))
            print("Loaded ISDFX k-point performance tests")
        except ImportError as e:
            print(f"Could not load ISDFX k-point tests: {e}")
            
        print("Running OCCRI performance tests...")

    else:
        # Run all tests
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        # Set flag for expensive tests (needed for some ISDFX energy tests)
        sys.modules[__name__].RUN_PERFORMANCE_TESTS = True

        # Load functional tests
        try:
            from test_occri import TestOCCRI

            suite.addTests(loader.loadTestsFromTestCase(TestOCCRI))
            print("Loaded functional tests")
        except ImportError as e:
            print(f"Could not load functional tests: {e}")

        # Load performance tests
        try:
            from test_performance import TestPerformance
            suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
            print("Loaded performance tests")
        except ImportError as e:
            print(f"Could not load performance tests: {e}")

        # Load ISDFX functional tests  
        try:
            from test_isdfx import (TestUtilityFunctions, TestCholeskyDecomposition, 
                                   TestISDFX, TestPivotSelection, TestFittingFunctions,
                                   TestTHCPotential, TestExchangeMatrixEvaluation,
                                   TestISdfxIntegration)
            
            suite.addTests(loader.loadTestsFromTestCase(TestUtilityFunctions))
            suite.addTests(loader.loadTestsFromTestCase(TestCholeskyDecomposition))
            suite.addTests(loader.loadTestsFromTestCase(TestISDFX))
            suite.addTests(loader.loadTestsFromTestCase(TestPivotSelection))
            suite.addTests(loader.loadTestsFromTestCase(TestFittingFunctions))
            suite.addTests(loader.loadTestsFromTestCase(TestTHCPotential))
            suite.addTests(loader.loadTestsFromTestCase(TestExchangeMatrixEvaluation))
            suite.addTests(loader.loadTestsFromTestCase(TestISdfxIntegration))
            print("Loaded ISDFX functional tests")
        except ImportError as e:
            print(f"Could not load ISDFX tests: {e}")

        # Load ISDFX energy tests (standard tests only)
        try:
            from test_isdfx_energy import TestISdfxEnergyComparison
            suite.addTests(loader.loadTestsFromTestCase(TestISdfxEnergyComparison))
            print("Loaded ISDFX energy tests")
        except ImportError as e:
            print(f"Could not load ISDFX energy tests: {e}")

        print("Running all OCCRI tests...")

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
