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

from pyscf import dft
from pyscf.pbc import gto, scf
from pyscf.pbc.dft import numint as pbc_numint
from pyscf.pbc.lo.mlwf.projection import (
    compute_amn,
    lowdin_orthonormalize,
    _evaluate_trial,
    _real_sph_harm,
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


def _amn_direct_reference(kmf, proj_guess, grids):
    '''Independent direct-integral implementation.

    Orders contractions differently from compute_amn (full nao-band matmul on
    the AO axis before integrating over space, versus AO-to-MO then integrate).
    Matching this within numerical noise confirms the pipeline consistency.
    '''
    cell = kmf.cell
    kpts = numpy.asarray(kmf.kpts)
    nk = len(kpts)
    coords = grids.coords
    weights = grids.weights

    n_wann = len(proj_guess)
    n_bands = kmf.mo_coeff[0].shape[1]

    gs = numpy.stack([_evaluate_trial(coords, p) for p in proj_guess])

    ao_kpts = pbc_numint.eval_ao_kpts(cell, coords, kpts=kpts)

    A = numpy.empty((nk, n_bands, n_wann), dtype=numpy.complex128)
    for ki in range(nk):
        # <phi_nu(k) | g_n> evaluated on the grid: (nao, n_wann)
        phi_g = numpy.einsum('i,iv,ni->vn',
                              weights, ao_kpts[ki].conj(), gs,
                              optimize=True)
        C = kmf.mo_coeff[ki]                             # (nao, n_bands)
        A[ki] = C.conj().T @ phi_g                        # (n_bands, n_wann)
    return A


def test_real_sph_harm_unit_sphere_normalization():
    '''|Y_lm|^2 integrated over the unit sphere equals 1 for each supported (l, m).'''
    rng = numpy.random.default_rng(0)
    pts = rng.normal(size=(20000, 3))
    pts /= numpy.linalg.norm(pts, axis=1, keepdims=True)
    r = numpy.linalg.norm(pts, axis=1)
    for (l, m) in [(0, 0), (1, -1), (1, 0), (1, 1),
                   (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)]:
        Y = _real_sph_harm(pts, r, l, m)
        mean_sq = numpy.mean(Y ** 2) * 4.0 * numpy.pi
        assert abs(mean_sq - 1.0) < 0.05, f'(l={l}, m={m}) normalization off: {mean_sq}'


def test_amn_shape(kmf_he):
    proj = [{'center': (0.0, 0.0, 0.0), 'l': 0, 'm': 0, 'zeta': 1.0}]
    A = compute_amn(kmf_he, proj)
    assert A.shape == (len(kmf_he.kpts), kmf_he.mo_coeff[0].shape[1], 1)
    assert A.dtype == numpy.complex128


def test_amn_matches_direct_reference_s(kmf_he):
    '''compute_amn matches an independently-ordered direct integral.'''
    proj_guess = [
        {'center': (0.0, 0.0, 0.0), 'l': 0, 'm': 0, 'zeta': 1.5},
    ]
    grids = dft.gen_grid.Grids(kmf_he.cell)
    grids.level = 3
    grids.build()
    A_ours = compute_amn(kmf_he, proj_guess, grids=grids)
    A_ref = _amn_direct_reference(kmf_he, proj_guess, grids)
    numpy.testing.assert_allclose(A_ours, A_ref, atol=1e-12, rtol=0)


def test_amn_matches_direct_reference_p_and_d(kmf_he):
    '''Independent reference match across the full (l, m) set in Tier A.'''
    proj_guess = [
        {'center': (0.5, 0.5, 0.5), 'l': 1, 'm': 0, 'zeta': 1.2},
        {'center': (0.5, 0.5, 0.5), 'l': 1, 'm': 1, 'zeta': 1.2},
        {'center': (0.5, 0.5, 0.5), 'l': 1, 'm': -1, 'zeta': 1.2},
        {'center': (1.0, 0.0, 0.0), 'l': 2, 'm': 0, 'zeta': 1.8},
        {'center': (1.0, 0.0, 0.0), 'l': 2, 'm': -2, 'zeta': 1.8},
    ]
    grids = dft.gen_grid.Grids(kmf_he.cell)
    grids.level = 3
    grids.build()
    A_ours = compute_amn(kmf_he, proj_guess, grids=grids)
    A_ref = _amn_direct_reference(kmf_he, proj_guess, grids)
    numpy.testing.assert_allclose(A_ours, A_ref, atol=1e-12, rtol=0)


def test_amn_hybrid_equals_linear_combination(kmf_he):
    '''An sp hybrid via components equals (1/sqrt 2)(s + px) projection.'''
    center = (0.0, 0.0, 0.0)
    zeta = 1.5
    inv_sqrt2 = 1.0 / numpy.sqrt(2.0)
    hybrid = [{'center': center, 'zeta': zeta,
               'components': [(inv_sqrt2, 0, 0), (inv_sqrt2, 1, 1)]}]
    singles = [
        {'center': center, 'l': 0, 'm': 0, 'zeta': zeta},
        {'center': center, 'l': 1, 'm': 1, 'zeta': zeta},
    ]
    A_hybrid = compute_amn(kmf_he, hybrid)
    A_singles = compute_amn(kmf_he, singles)
    A_expected = inv_sqrt2 * (A_singles[..., 0] + A_singles[..., 1])
    numpy.testing.assert_allclose(A_hybrid[..., 0], A_expected, atol=1e-10)


def test_lowdin_orthonormalizes_per_k(kmf_he):
    proj_guess = [
        {'center': (0.0, 0.0, 0.0), 'l': 0, 'm': 0, 'zeta': 1.5},
    ]
    A = compute_amn(kmf_he, proj_guess)
    U = lowdin_orthonormalize(A)
    nk, nb, nw = U.shape
    for ki in range(nk):
        gram = U[ki].conj().T @ U[ki]
        numpy.testing.assert_allclose(gram, numpy.eye(nw), atol=1e-10)


def test_lowdin_rejects_rank_deficient_shape():
    A = numpy.zeros((2, 1, 2), dtype=numpy.complex128)
    with pytest.raises(ValueError, match='n_bands >= n_wann'):
        lowdin_orthonormalize(A)


def test_rejects_empty_proj_guess(kmf_he):
    with pytest.raises(ValueError, match='empty'):
        compute_amn(kmf_he, [])
