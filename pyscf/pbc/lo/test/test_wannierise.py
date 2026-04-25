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

import numpy
import pytest
from scipy.linalg import expm

from pyscf.pbc import gto, scf
from pyscf.pbc.lo.bvectors import find_bvectors
from pyscf.pbc.lo.overlap import compute_mmn
from pyscf.pbc.lo.projection import compute_amn, lowdin_orthonormalize
from pyscf.pbc.lo.spread import spread_decomposition
from pyscf.pbc.lo.wannierise import (
    wannierise, rotate_mmn, _spread_gradient, _step_unitary,
)


@pytest.fixture(scope='module')
def kmf_he():
    cell = gto.Cell()
    cell.unit = 'Bohr'
    cell.atom = 'He 0 0 0'
    cell.a = numpy.eye(3) * 5.0
    cell.basis = 'sto-3g'
    cell.verbose = 0
    cell.build()
    kpts = cell.make_kpts([2, 2, 2])
    mf = scf.KRHF(cell, kpts=kpts)
    mf.conv_tol = 1e-10
    mf.kernel()
    return mf


def _random_unitary_stack(rng, nk, n):
    U = numpy.empty((nk, n, n), dtype=numpy.complex128)
    for ki in range(nk):
        A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        A = A - A.conj().T           # anti-Hermitian
        U[ki] = expm(A)
    return U


@pytest.fixture(scope='module')
def kmf_h2():
    '''H2 dimer per cell, 2 bands, tight enough to be a full-basis test.'''
    cell = gto.Cell()
    cell.unit = 'Bohr'
    cell.atom = 'H 0 0 0; H 0 0 1.5'
    cell.a = numpy.eye(3) * 8.0
    cell.basis = 'sto-3g'
    cell.verbose = 0
    cell.build()
    kpts = cell.make_kpts([2, 2, 2])
    mf = scf.KRHF(cell, kpts=kpts)
    mf.conv_tol = 1e-10
    mf.kernel()
    return mf


def test_spread_identity_on_unitary_m(kmf_he):
    '''Omega_I + Omega_OD + Omega_D = sum_n spread_n (MV paper identity).'''
    bv = find_bvectors(kmf_he.cell, kmf_he.kpts)
    M = compute_mmn(kmf_he, bv)                              # (nk, n_nn, 1, 1) here
    centers, spreads, oi, ood, od = spread_decomposition(M, bv)
    numpy.testing.assert_allclose(oi + ood + od, spreads.sum(), atol=1e-10)


def test_rotate_mmn_preserves_shape_and_unitarity(kmf_he):
    bv = find_bvectors(kmf_he.cell, kmf_he.kpts)
    _bvecs, _weights, kpb_idx, _kpb_g = bv
    M = compute_mmn(kmf_he, bv)
    nk, n_nn, n_bands, _ = M.shape
    # Identity rotation gives back M.
    U = numpy.tile(numpy.eye(n_bands, dtype=numpy.complex128), (nk, 1, 1))
    M_rot = rotate_mmn(M, U, kpb_idx)
    numpy.testing.assert_allclose(M_rot, M, atol=1e-14)


def test_rotate_mmn_matches_explicit_loop():
    rng = numpy.random.default_rng(0)
    nk, n_nn, n_wann = 3, 4, 2
    M = rng.normal(size=(nk, n_nn, n_wann, n_wann)) + \
        1j * rng.normal(size=(nk, n_nn, n_wann, n_wann))
    U = _random_unitary_stack(rng, nk, n_wann)
    # Arbitrary k+b connectivity (use cyclic permutation).
    kpb_idx = numpy.array([[(k + b) % nk for b in range(n_nn)] for k in range(nk)])

    M_rot = rotate_mmn(M, U, kpb_idx)
    for ki in range(nk):
        for bi in range(n_nn):
            kj = kpb_idx[ki, bi]
            expected = U[ki].conj().T @ M[ki, bi] @ U[kj]
            numpy.testing.assert_allclose(M_rot[ki, bi], expected, atol=1e-12)


def test_step_preserves_unitarity():
    rng = numpy.random.default_rng(1)
    nk, n = 4, 3
    U = _random_unitary_stack(rng, nk, n)
    # Anti-Hermitian step.
    W = numpy.empty((nk, n, n), dtype=numpy.complex128)
    for ki in range(nk):
        A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        W[ki] = 0.1 * (A - A.conj().T)
    U_new = _step_unitary(U, W)
    for ki in range(nk):
        gram = U_new[ki].conj().T @ U_new[ki]
        numpy.testing.assert_allclose(gram, numpy.eye(n), atol=1e-12)


def test_gradient_is_anti_hermitian(kmf_he):
    bv = find_bvectors(kmf_he.cell, kmf_he.kpts)
    _bvecs, _weights, kpb_idx, _kpb_g = bv
    M = compute_mmn(kmf_he, bv)
    nk, n_nn, n_wann, _ = M.shape
    # Identity U is fine; spreads/centers still well-defined.
    U = numpy.tile(numpy.eye(n_wann, dtype=numpy.complex128), (nk, 1, 1))
    M_rot = rotate_mmn(M, U, kpb_idx)
    centers, *_ = spread_decomposition(M_rot, bv)
    G = _spread_gradient(M_rot, centers, bv)
    for ki in range(nk):
        numpy.testing.assert_allclose(
            G[ki], -G[ki].conj().T, atol=1e-10,
            err_msg=f'gradient not anti-Hermitian at k={ki}')


def test_monotonic_descent_multi_wannier(kmf_h2):
    '''Full pipeline on H2 crystal (2 bands -> 2 MLWFs): the final spread
    identity holds and U_final is unitary.'''
    bv = find_bvectors(kmf_h2.cell, kmf_h2.kpts)
    M = compute_mmn(kmf_h2, bv)
    proj = [
        {'center': (0.0, 0.0, 0.0), 'l': 0, 'm': 0, 'zeta': 1.0},
        {'center': (0.0, 0.0, 1.5), 'l': 0, 'm': 0, 'zeta': 1.0},
    ]
    A = compute_amn(kmf_h2, proj)
    U_init = lowdin_orthonormalize(A)

    U, _centers, spreads, oi, od, ood, _conv = wannierise(
        M, U_init, bv, max_iter=60, conv_tol=1e-12)

    # Unitarity preserved at every k.
    n_wann = U.shape[2]
    for ki in range(U.shape[0]):
        gram = U[ki].conj().T @ U[ki]
        numpy.testing.assert_allclose(gram, numpy.eye(n_wann), atol=1e-10)
    # Spread identity holds on the final state.
    numpy.testing.assert_allclose(oi + od + ood, spreads.sum(), atol=1e-9)


def test_pipeline_on_he_crystal(kmf_he):
    '''End-to-end Tier A on He: single-band trivial case.

    With one band, U is 1x1 and wannierise only adjusts an overall phase.
    We verify the spread identity holds and U remains unitary.
    '''
    bv = find_bvectors(kmf_he.cell, kmf_he.kpts)
    M = compute_mmn(kmf_he, bv)
    proj = [{'center': (0.0, 0.0, 0.0), 'l': 0, 'm': 0, 'zeta': 1.5}]
    A = compute_amn(kmf_he, proj)
    U_init = lowdin_orthonormalize(A)

    U, _centers, spreads, oi, od, ood, _conv = wannierise(
        M, U_init, bv, max_iter=30, conv_tol=1e-12)
    numpy.testing.assert_allclose(oi + od + ood, spreads.sum(), atol=1e-10)
    for ki in range(len(kmf_he.kpts)):
        gram = U[ki].conj().T @ U[ki]
        numpy.testing.assert_allclose(gram, numpy.eye(1), atol=1e-10)


def test_wannierise_preserves_omega_i_on_rectangular_U(kmf_h2):
    '''With n_bands > n_wann, wannierise optimizes only the within-subspace
    gauge (Omega_D + Omega_OD). Omega_I is determined by the chosen subspace
    and must stay constant across iterations.
    '''
    from pyscf.pbc.lo.spread import spread_decomposition
    bv = find_bvectors(kmf_h2.cell, kmf_h2.kpts)
    _bvecs, _weights, kpb_idx, _kpb_g = bv
    M = compute_mmn(kmf_h2, bv)
    # Use a single s projection at H2 midpoint -> n_wann=1 selected from 2 bands.
    proj = [{'center': (0.0, 0.0, 0.75), 'l': 0, 'm': 0, 'zeta': 1.0}]
    A = compute_amn(kmf_h2, proj)
    U_init = lowdin_orthonormalize(A)               # shape (nk, 2, 1), isometric

    # Omega_I at the initial (Lowdin) subspace.
    M_rot0 = rotate_mmn(M, U_init, kpb_idx)
    _, _, oi_init, _, _ = spread_decomposition(M_rot0, bv)

    _U, _c, _s, oi, _d, _od, _conv = wannierise(
        M, U_init, bv, max_iter=30, conv_tol=1e-12)
    numpy.testing.assert_allclose(oi, oi_init, atol=1e-10)


def test_wannierise_rejects_n_bands_less_than_n_wann():
    nk, n_nn, n_bands, n_wann = 2, 6, 1, 2
    M = numpy.zeros((nk, n_nn, n_bands, n_bands), dtype=numpy.complex128)
    U = numpy.zeros((nk, n_bands, n_wann), dtype=numpy.complex128)
    bv = (numpy.zeros((n_nn, 3)),
          numpy.zeros(n_nn),
          numpy.zeros((nk, n_nn), dtype=int),
          numpy.zeros((nk, n_nn, 3), dtype=int))
    with pytest.raises(ValueError, match='less than'):
        wannierise(M, U, bv)
