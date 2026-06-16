# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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

import unittest
import numpy as np
from pyscf.tdscf._krylov_tools import krylov_solver 


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Set random seed for reproducibility
        np.random.seed(0)
        self.A_size = 3000
        self.n_vec = 5
        self.n_states = 3
        self.scaling = 4

        # Generate symmetric matrix A
        A = np.random.rand(self.A_size, self.A_size) * 0.01
        A = A + A.T
        np.fill_diagonal(A, (np.random.rand(self.A_size) + 2) * self.scaling)
        self.A = A
        self.hdiag = np.diag(A)

        # Generate random right-hand side and omega shift
        self.rhs = np.random.rand(self.n_vec, self.A_size) * self.scaling
        self.omega_shift = (np.random.rand(self.n_vec) + 2) * self.scaling / 2

        @staticmethod
        def matrix_vector_product(x):
            return x.dot(self.A)

        # self.A_single = self.A.astype(np.float32)
        @staticmethod
        def matrix_vector_product_single(x):
            return np.asarray(x.dot(self.A),dtype=np.float32)


        self.matrix_vector_product = matrix_vector_product
        self.matrix_vector_product_single = matrix_vector_product_single

        # Reference eigenvalues and eigenvectors
        ref_eigvals, ref_eigvecs = np.linalg.eigh(self.A)
        self.ref_eigenvalues = np.asarray(ref_eigvals)[:self.n_states]
        self.ref_eigenvectors = ref_eigvecs[:,:self.n_states].T
        # Reference solutions for linear system
        self.ref_solution_vectors = np.linalg.solve(self.A, self.rhs.T).T
        # Reference solutions for shifted linear system
        self.ref_solution_vectors_shifted = np.zeros_like(self.ref_solution_vectors)
        for i in range(self.n_vec):
            shifted_A = self.A - self.omega_shift[i] * np.eye(self.A_size)
            self.ref_solution_vectors_shifted[i,:] = np.linalg.solve(shifted_A, self.rhs[i])

        self.places_double = 5
        self.places_single = 3


    def test_krylov_eigenvalue(self):
        """Test Krylov solver for eigenvalue problem"""
        _, eigenvalues, eigenvectors = krylov_solver(
            matrix_vector_product=self.matrix_vector_product,
            hdiag=self.hdiag,
            problem_type='eigenvalue',
            n_states=self.n_states,
            conv_tol=1e-6,
            max_iter=35,
            gram_schmidt=False,
            verbose=4,
            single=False
        )

        _, eigenvalues_single, eigenvectors_single = krylov_solver(
            matrix_vector_product=self.matrix_vector_product_single,
            hdiag=self.hdiag,
            problem_type='eigenvalue',
            n_states=self.n_states,
            conv_tol=1e-5,
            max_iter=40,
            gram_schmidt=True,
            verbose=4,
            single=True
        )

        print('eigenvectors.shape', eigenvectors.shape)
        # Compare eigenvalues
        # double precison
        self.assertAlmostEqual(
            float(np.linalg.norm(eigenvalues - self.ref_eigenvalues)), 0, places=self.places_double ,
            msg="Double precision Eigenvalues do not match reference within tolerance"
        )

        self.assertAlmostEqual(
            float(np.linalg.norm(np.abs(eigenvectors) - np.abs(self.ref_eigenvectors))), .0, places=4,
            msg="Double precision: Eigenvectors do not match reference within tolerance"
        )

        # single precision
        self.assertAlmostEqual(
            float(np.linalg.norm(eigenvalues_single - np.asarray(self.ref_eigenvalues, dtype=np.float32))), .0, places=self.places_single ,
            msg="Single precision Eigenvalues do not match reference within tolerance"
        )

        self.assertAlmostEqual(
            float(np.linalg.norm(np.abs(eigenvectors_single) - np.abs(np.asarray(self.ref_eigenvectors, dtype=np.float32)))), .0, places=self.places_single - 2,
            msg="Single precision Eigenvectors do not match reference within tolerance"
        )


    def test_krylov_linear(self):
        """Test Krylov solver for linear system"""
        _, solution_vectors = krylov_solver(
            matrix_vector_product=self.matrix_vector_product,
            hdiag=self.hdiag,
            problem_type='linear',
            rhs=self.rhs,
            conv_tol=1e-8,
            max_iter=35,
            gram_schmidt=True,
            verbose=4,
            single=False
        )


        _, solution_vectors_single = krylov_solver(
            matrix_vector_product=self.matrix_vector_product_single,
            hdiag=self.hdiag,
            problem_type='linear',
            rhs=self.rhs,
            conv_tol=1e-5,
            max_iter=35,
            gram_schmidt=True,
            verbose=4,
            single=True
        )


        # Compare solutions
        self.assertAlmostEqual(
            float(np.linalg.norm(solution_vectors - self.ref_solution_vectors)), .0, places=self.places_double,
            msg="Double precision Linear system solutions do not match reference within tolerance"
        )

        self.assertAlmostEqual(
            float(np.linalg.norm(solution_vectors_single - self.ref_solution_vectors)), .0, places=self.places_single,
            msg="Single precision Linear system solutions do not match reference within tolerance"
        )


    def test_krylov_shifted_linear(self):
        """Test Krylov solver for shifted linear system"""
        _, solution_vectors_shifted = krylov_solver(
            matrix_vector_product=self.matrix_vector_product,
            hdiag=self.hdiag,
            problem_type='shifted_linear',
            rhs=self.rhs,
            omega_shift=self.omega_shift,
            conv_tol=1e-8,
            max_iter=35,
            gram_schmidt=True,
            verbose=4,
            single=False
        )


        _, solution_vectors_shifted_single = krylov_solver(
            matrix_vector_product=self.matrix_vector_product_single,
            hdiag=self.hdiag,
            problem_type='shifted_linear',
            rhs=self.rhs,
            omega_shift=self.omega_shift,
            conv_tol=1e-5,
            max_iter=35,
            gram_schmidt=True,
            verbose=4,
            single=True
        )


        # Compare solutions
        self.assertAlmostEqual(
            float(np.linalg.norm(solution_vectors_shifted - self.ref_solution_vectors_shifted)), .0, places=self.places_double,
            msg="Double precision Shifted linear system solutions do not match reference within tolerance"
        )

        self.assertAlmostEqual(
            float(np.linalg.norm(solution_vectors_shifted_single - self.ref_solution_vectors_shifted)), .0, places=self.places_single ,
            msg="Single precision Shifted linear system solutions do not match reference within tolerance"
        )


if __name__ == "__main__":
    print("Running tests for Krylov solver with eigenvalue, linear, and shifted linear problems")
    unittest.main()