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
from pyscf.pbc.lo import kernel


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


@pytest.fixture(scope='module')
def kmf_h2():
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


def test_kernel_isolated_single_band(kmf_he):
    '''He / sto-3g: n_bands = n_wann = 1, trivial isolated-band case.'''
    proj = [{'center': (0.0, 0.0, 0.0), 'l': 0, 'm': 0, 'zeta': 1.5}]
    U, centers, spreads, oi, od, ood, converged = kernel(
        kmf_he, proj, conv_tol=1e-12)
    assert U.shape == (len(kmf_he.kpts), 1, 1)
    assert centers.shape == (1, 3)
    assert spreads.shape == (1,)
    for ki in range(U.shape[0]):
        gram = U[ki].conj().T @ U[ki]
        numpy.testing.assert_allclose(gram, numpy.eye(1), atol=1e-10)
    numpy.testing.assert_allclose(oi + od + ood, spreads.sum(), atol=1e-9)
    assert converged


def test_kernel_isolated_two_bands(kmf_h2):
    '''H2 sto-3g: n_bands = n_wann = 2, full-basis isolated-band case.'''
    proj = [
        {'center': (0.0, 0.0, 0.0), 'l': 0, 'm': 0, 'zeta': 1.0},
        {'center': (0.0, 0.0, 1.5), 'l': 0, 'm': 0, 'zeta': 1.0},
    ]
    U, centers, spreads, oi, od, ood, _ = kernel(
        kmf_h2, proj, max_iter=80, conv_tol=1e-12)
    assert U.shape == (len(kmf_h2.kpts), 2, 2)
    assert centers.shape == (2, 3)
    assert spreads.shape == (2,)
    for ki in range(U.shape[0]):
        gram = U[ki].conj().T @ U[ki]
        numpy.testing.assert_allclose(gram, numpy.eye(2), atol=1e-10)
    numpy.testing.assert_allclose(oi + od + ood, spreads.sum(), atol=1e-9)


def test_kernel_entangled_with_dis_win(kmf_h2):
    '''H2 sto-3g with dis_win covering both bands -> disentangle 2 -> 1.'''
    proj = [{'center': (0.0, 0.0, 0.75), 'l': 0, 'm': 0, 'zeta': 1.0}]
    # Generous outer window that includes both H2 MOs at every k.
    U, _c, _s, _oi, _od, _ood, _ = kernel(
        kmf_h2, proj, dis_win=(-5.0, 5.0),
        dis_max_iter=300, conv_tol=1e-12)
    assert U.shape == (len(kmf_h2.kpts), 2, 1)
    for ki in range(U.shape[0]):
        gram = U[ki].conj().T @ U[ki]
        numpy.testing.assert_allclose(gram, numpy.eye(1), atol=1e-10)


def test_kernel_requires_matching_n_bands_without_dis_win(kmf_h2):
    '''Passing fewer projections than bands without dis_win is an error.'''
    proj = [{'center': (0.0, 0.0, 0.75), 'l': 0, 'm': 0, 'zeta': 1.0}]
    with pytest.raises(ValueError, match='dis_win=None'):
        kernel(kmf_h2, proj)


def test_kernel_rejects_empty_proj(kmf_he):
    with pytest.raises(ValueError, match='empty'):
        kernel(kmf_he, [])


def test_kernel_rejects_single_kpoint():
    '''Single-kpoint kmf is rejected up front.'''
    cell = gto.Cell()
    cell.unit = 'Bohr'
    cell.atom = 'He 0 0 0'
    cell.a = numpy.eye(3) * 5.0
    cell.basis = 'sto-3g'
    cell.verbose = 0
    cell.build()
    mf = scf.KRHF(cell, kpts=numpy.zeros((1, 3)))
    mf.conv_tol = 1e-10
    mf.kernel()
    proj = [{'center': (0.0, 0.0, 0.0), 'l': 0, 'm': 0, 'zeta': 1.5}]
    with pytest.raises(ValueError, match='at least 2 k-points'):
        kernel(mf, proj)


def test_kernel_dis_froz_not_implemented(kmf_h2):
    proj = [{'center': (0.0, 0.0, 0.75), 'l': 0, 'm': 0, 'zeta': 1.0}]
    with pytest.raises(NotImplementedError, match='dis_froz'):
        kernel(kmf_h2, proj, dis_win=(-5.0, 5.0), dis_froz=(-2.0, 2.0))


def test_kernel_empty_dis_win(kmf_h2):
    proj = [{'center': (0.0, 0.0, 0.75), 'l': 0, 'm': 0, 'zeta': 1.0}]
    with pytest.raises(ValueError, match='empty'):
        kernel(kmf_h2, proj, dis_win=(2.0, 1.0))


def test_kernel_dis_win_without_enough_bands(kmf_he):
    '''dis_win that yields fewer bands than n_wann is rejected.'''
    # He/sto-3g has a single band; asking for 2 Wannier functions fails.
    proj = [
        {'center': (0.0, 0.0, 0.0), 'l': 0, 'm': 0, 'zeta': 1.5},
        {'center': (0.0, 0.0, 0.0), 'l': 0, 'm': 0, 'zeta': 1.5},
    ]
    with pytest.raises(ValueError, match='selects only'):
        kernel(kmf_he, proj, dis_win=(-5.0, 5.0))
