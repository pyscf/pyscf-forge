"""
Test suite for multigrid OCCRI implementation

Tests for multigrid components including grid hierarchy,
interpolation operators, solvers, and k-point exchange
matrix evaluation.
"""

import unittest
import numpy
from pyscf.pbc import gto, scf
