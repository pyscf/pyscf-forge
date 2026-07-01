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

from pyscf.pbc import gto
from pyscf.pbc.lo.mlwf.bvectors import find_bvectors


def _cubic_cell(a_bohr=4.0):
    cell = gto.Cell()
    cell.unit = 'Bohr'
    cell.atom = 'H 0 0 0'
    cell.a = numpy.eye(3) * a_bohr
    cell.basis = 'sto-3g'
    cell.verbose = 0
    cell.build()
    return cell


def test_rejects_single_kpoint():
    cell = _cubic_cell()
    with pytest.raises(ValueError, match='at least 2 k-points'):
        find_bvectors(cell, numpy.zeros((1, 3)))


def test_cubic_4x4x4_one_shell():
    '''Cubic mesh: 1 shell, 6 vectors, equal weights, |b| = 2pi/(N a).'''
    a = 4.0
    N = 4
    cell = _cubic_cell(a)
    kpts = cell.make_kpts([N, N, N])
    bvecs, weights, _kpb_idx, _kpb_g = find_bvectors(cell, kpts)

    assert bvecs.shape == (6, 3)
    assert weights.shape == (6,)

    expected_norm = 2 * numpy.pi / (N * a)
    numpy.testing.assert_allclose(
        numpy.linalg.norm(bvecs, axis=1), expected_norm, atol=1e-10)

    # cubic symmetry -> all weights equal, value is 1 / (2 |b|^2)
    expected_w = 1.0 / (2.0 * expected_norm ** 2)
    numpy.testing.assert_allclose(weights, expected_w, atol=1e-10)


def test_b1_completeness_cubic():
    cell = _cubic_cell()
    kpts = cell.make_kpts([4, 4, 4])
    bvecs, weights, _kpb_idx, _kpb_g = find_bvectors(cell, kpts)
    bbt = numpy.einsum('b,bi,bj->ij', weights, bvecs, bvecs)
    numpy.testing.assert_allclose(bbt, numpy.eye(3), atol=1e-10)


def test_b1_completeness_anisotropic_mesh():
    '''2x2x4 mesh on cubic lattice requires two shells.'''
    cell = _cubic_cell()
    kpts = cell.make_kpts([2, 2, 4])
    bvecs, weights, _kpb_idx, _kpb_g = find_bvectors(cell, kpts)
    bbt = numpy.einsum('b,bi,bj->ij', weights, bvecs, bvecs)
    numpy.testing.assert_allclose(bbt, numpy.eye(3), atol=1e-10)


def test_connectivity_cubic():
    '''k + b must equal kpts[kpb_idx] + kpb_g @ recip for every (k, b).'''
    cell = _cubic_cell()
    kpts = cell.make_kpts([4, 4, 4])
    bvecs, _weights, kpb_idx, kpb_g = find_bvectors(cell, kpts)
    recip = cell.reciprocal_vectors()

    nk = len(kpts)
    nb = len(bvecs)
    for i in range(nk):
        for b in range(nb):
            kpb_expected = kpts[i] + bvecs[b]
            kpb_actual = kpts[kpb_idx[i, b]] + kpb_g[i, b] @ recip
            numpy.testing.assert_allclose(
                kpb_expected, kpb_actual, atol=1e-10,
                err_msg=f'k={i}, b={b}')


def test_explicit_kmesh_matches_inferred():
    cell = _cubic_cell()
    kpts = cell.make_kpts([3, 3, 3])
    inf_b, inf_w, inf_idx, inf_g = find_bvectors(cell, kpts)
    exp_b, exp_w, exp_idx, exp_g = find_bvectors(cell, kpts, kmesh=(3, 3, 3))
    numpy.testing.assert_allclose(inf_b, exp_b)
    numpy.testing.assert_allclose(inf_w, exp_w)
    numpy.testing.assert_array_equal(inf_idx, exp_idx)
    numpy.testing.assert_array_equal(inf_g, exp_g)


def test_kmesh_product_mismatch_raises():
    cell = _cubic_cell()
    kpts = cell.make_kpts([4, 4, 4])
    with pytest.raises(ValueError, match='product'):
        find_bvectors(cell, kpts, kmesh=(2, 2, 2))
