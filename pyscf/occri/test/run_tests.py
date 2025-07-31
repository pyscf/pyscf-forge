#!/usr/bin/env python
"""
Test runner for OCCRI test suite.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py --quick      # Run only quick tests
    python run_tests.py --perf       # Run only performance tests
"""

import sys
import unittest
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run OCCRI tests')
    parser.add_argument('--quick', action='store_true', 
                        help='Run only quick functional tests')
    parser.add_argument('--perf', action='store_true',
                        help='Run only performance tests') 
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    args = parser.parse_args()
    
    # Setup verbosity
    verbosity = 2 if args.verbose else 1
    
    if args.quick:
        # Run only main functional tests
        from test_occri import TestOCCRI
        suite = unittest.TestLoader().loadTestsFromTestCase(TestOCCRI)
        print("Running quick OCCRI tests...")
        
    elif args.perf:
        # Run only performance tests
        from test_performance import TestPerformance
        suite = unittest.TestLoader().loadTestsFromTestCase(TestPerformance)
        print("Running OCCRI performance tests...")
        
    else:
        # Run all tests
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
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
            
        print("Running all OCCRI tests...")
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)

if __name__ == '__main__':
    main()