#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: Bhavnesh Jangid

import unittest

import numpy as np

from pyscf import gto, mcscf, scf, siso
from pyscf.siso import addons


class DiatomicSystem(unittest.TestCase):
    """Exercise the main SOC-analysis functions on a small OH doublet."""

    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(
            atom='O 0 0 0; H 0 0 0.97', basis='sto-3g', spin=1,
            verbose=0)
        mf = scf.ROHF(cls.mol).run()
        cls.modelspace = [(1, 2)]
        cls.mc = siso.sacasscf_solver(
            mcscf.CASSCF(mf, 2, 1), cls.modelspace).run()
        cls.mysiso = siso.SISO(cls.mc, ham='BP', amf=True).run()
        cls.data = addons.generate_siso_data(
            cls.mol, cls.mc, mysiso=cls.mysiso)
        cls.analysis = addons.soc_analysis(cls.mysiso, cls.data)

    def test_main_analysis_functions(self):
        _, l_values, spin_values, s2_values = (
            self.analysis.compute_L_values())
        np.testing.assert_allclose(l_values, [0.0], atol=1e-8)
        np.testing.assert_allclose(spin_values, [0.5], atol=1e-12)
        np.testing.assert_allclose(s2_values, [0.75], atol=1e-12)

        j_matrix, j_values = self.analysis.compute_J_values()
        np.testing.assert_allclose(j_values, [0.5, 0.5], atol=1e-8)
        np.testing.assert_allclose(j_matrix, 0.75 * np.eye(2), atol=1e-8)

        _, lambda_values = (
            self.analysis.compute_L_values_for_diatomics(axis='z'))
        _, omega_values = self.analysis.compute_omega_values(axis='z')
        np.testing.assert_allclose(lambda_values, [0.0, 0.0], atol=1e-8)
        np.testing.assert_allclose(omega_values, [0.5, 0.5], atol=1e-8)

        coefficients, root_weights = self.analysis.soc_state_analysis(
            state=(0, 1))
        np.testing.assert_allclose(
            coefficients.conj().T @ coefficients, np.eye(2), atol=1e-10)
        np.testing.assert_allclose(root_weights, np.ones((1, 2)), atol=1e-10)


if __name__ == '__main__':
    unittest.main()
