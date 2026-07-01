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

from pyscf.pbc import gto, scf
from pyscf.pbc.lo.mlwf.bvectors import find_bvectors
from pyscf.pbc.lo.mlwf.overlap import compute_mmn
from pyscf.pbc.lo.mlwf.projection import compute_amn, lowdin_orthonormalize
from pyscf.pbc.lo.mlwf.disentangle import disentangle
from pyscf.pbc.lo.mlwf.wannierise import wannierise


@pytest.fixture(scope='module')
def kmf_h2():
    '''H2 dimer per cell (2 sto-3g bands / k).'''
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


def test_disentangle_trivial_square_case(kmf_h2):
    '''When n_bands == n_wann, disentangle reduces to Lowdin(A).'''
    bv = find_bvectors(kmf_h2.cell, kmf_h2.kpts)
    M = compute_mmn(kmf_h2, bv)
    proj = [
        {'center': (0.0, 0.0, 0.0), 'l': 0, 'm': 0, 'zeta': 1.0},
        {'center': (0.0, 0.0, 1.5), 'l': 0, 'm': 0, 'zeta': 1.0},
    ]
    A = compute_amn(kmf_h2, proj)
    U_dis, _oi, converged = disentangle(M, A, bv)
    numpy.testing.assert_allclose(U_dis, lowdin_orthonormalize(A), atol=1e-12)
    assert converged


def test_disentangle_entangled_isometric(kmf_h2):
    '''1-of-2 subspace selection: U_dis is isometric per k.'''
    bv = find_bvectors(kmf_h2.cell, kmf_h2.kpts)
    M = compute_mmn(kmf_h2, bv)
    proj = [{'center': (0.0, 0.0, 0.75), 'l': 0, 'm': 0, 'zeta': 1.0}]
    A = compute_amn(kmf_h2, proj)

    U_dis, _oi, _conv = disentangle(M, A, bv, max_iter=200, conv_tol=1e-12)
    nk = M.shape[0]
    for ki in range(nk):
        gram = U_dis[ki].conj().T @ U_dis[ki]
        numpy.testing.assert_allclose(gram, numpy.eye(1), atol=1e-10)


def test_disentangle_then_wannierise_pipeline(kmf_h2):
    '''Compose disentangle -> wannierise; Omega_I is locked by the subspace.'''
    bv = find_bvectors(kmf_h2.cell, kmf_h2.kpts)
    M = compute_mmn(kmf_h2, bv)
    proj = [{'center': (0.0, 0.0, 0.75), 'l': 0, 'm': 0, 'zeta': 1.0}]
    A = compute_amn(kmf_h2, proj)

    U_dis, oi_dis, _ = disentangle(M, A, bv, max_iter=200, conv_tol=1e-12)
    # wannierise accepts rectangular U_dis (n_bands > n_wann).
    U, _c, spreads, oi, od, ood, _ = wannierise(
        M, U_dis, bv, max_iter=60, conv_tol=1e-12)

    # Omega_I is locked by the disentanglement step and should stay constant.
    numpy.testing.assert_allclose(oi, oi_dis, atol=1e-9)
    # Final U is isometric.
    nk, nb, nw = U.shape
    for ki in range(nk):
        gram = U[ki].conj().T @ U[ki]
        numpy.testing.assert_allclose(gram, numpy.eye(nw), atol=1e-10)
    # Spread identity holds.
    numpy.testing.assert_allclose(oi + od + ood, spreads.sum(), atol=1e-9)


def _dummy_bvectors(n_nn, nk):
    return (numpy.zeros((n_nn, 3)),
            numpy.zeros(n_nn),
            numpy.zeros((nk, n_nn), dtype=int),
            numpy.zeros((nk, n_nn, 3), dtype=int))


def test_disentangle_rejects_incompatible_shapes():
    nk, n_nn, n_bands, n_wann = 2, 6, 3, 2
    M = numpy.zeros((nk, n_nn, n_bands, n_bands), dtype=numpy.complex128)
    A_bad = numpy.zeros((nk, n_bands + 1, n_wann), dtype=numpy.complex128)
    with pytest.raises(ValueError, match='inconsistent'):
        disentangle(M, A_bad, _dummy_bvectors(n_nn, nk))


def test_disentangle_rejects_n_bands_less_than_n_wann():
    nk, n_nn, n_bands, n_wann = 2, 6, 1, 2
    M = numpy.zeros((nk, n_nn, n_bands, n_bands), dtype=numpy.complex128)
    A = numpy.zeros((nk, n_bands, n_wann), dtype=numpy.complex128)
    with pytest.raises(ValueError, match='n_bands'):
        disentangle(M, A, _dummy_bvectors(n_nn, nk))
