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

'''
B-vector shells for finite-difference k-derivatives on a uniform mesh.

We need a set of nearest-neighbor offsets {b} whose weights {w_b} satisfy

    sum_b w_b b_alpha b_beta = delta_{alpha, beta}    (MV "B1")

Shells are added one at a time by distance and the weights solved by
least squares until the residual drops below tolerance. Appendix B of
Marzari & Vanderbilt, Phys. Rev. B 56, 12847 (1997).
'''

import numpy


def find_bvectors(cell, kpts, *, kmesh=None,
                  kmesh_tol=1e-6, b1_tol=1e-10, max_shells=12, search_range=3):
    '''Find the b-vector shells that satisfy B1 for this k-mesh.

    Args:
        cell : pbc.gto.Cell
        kpts : (nk, 3) ndarray
            Cartesian k-points (1/Bohr). Must come from a uniform MP grid.

    Kwargs:
        kmesh : (3,) or None
            MP grid size. Inferred from kpts when None.
        kmesh_tol : float
            Used for two related things: grouping b-vectors into shells of
            equal |b|, and matching k+b back to the k-point list.
        b1_tol : float
            How close the B1 residual has to get before we accept the shell set.
        max_shells, search_range : int
            Safety limits. search_range is the half-width of the integer
            lattice of candidate b-vectors (in units of the k-mesh spacing).

    Returns:
        bvecs    : (n_nn, 3) Cartesian b-vectors, 1/Bohr.
        weights  : (n_nn,) per-b finite-difference weights, Bohr**2.
        kpb_idx  : (nk, n_nn) int. The k-point index that k+b wraps to.
        kpb_g    : (nk, n_nn, 3) int. The reciprocal-lattice shift G such that
                   kpts[k] + bvecs[b] = kpts[kpb_idx[k, b]] + G @ recip.
    '''
    kpts = numpy.asarray(kpts, dtype=float)
    if kpts.ndim != 2 or kpts.shape[1] != 3:
        raise ValueError(f'kpts must have shape (nk, 3); got {kpts.shape}')
    if kpts.shape[0] < 2:
        raise ValueError('find_bvectors requires at least 2 k-points')

    if kmesh is None:
        kmesh = _infer_kmesh(cell, kpts, kmesh_tol)
    else:
        kmesh = numpy.asarray(kmesh, dtype=int)
        if kmesh.shape != (3,):
            raise ValueError(f'kmesh must be a length-3 sequence; got shape {kmesh.shape}')
    if int(numpy.prod(kmesh)) != kpts.shape[0]:
        raise ValueError(
            f'kmesh={tuple(int(m) for m in kmesh)} has product '
            f'{int(numpy.prod(kmesh))} but {kpts.shape[0]} k-points were given')

    recip = cell.reciprocal_vectors()
    shells = _candidate_shells(recip, kmesh, kmesh_tol, max_shells, search_range)

    bvecs, weights = _select_shells(shells, b1_tol)
    kpb_idx, kpb_g = _build_connectivity(recip, kpts, bvecs, kmesh_tol)
    return bvecs, weights, kpb_idx, kpb_g


def _infer_kmesh(cell, kpts, tol):
    '''Read back the MP grid from a list of k-points.'''
    frac = cell.get_scaled_kpts(kpts)
    # Wrap into [0, 1). The +tol nudge keeps tiny negative values
    # (round-off below zero) from wrapping to ~1.
    frac = frac - numpy.floor(frac + tol)
    mp = numpy.zeros(3, dtype=int)
    for alpha in range(3):
        vals = numpy.sort(frac[:, alpha])
        count = 1
        for i in range(1, len(vals)):
            if vals[i] - vals[i - 1] > tol:
                count += 1
        mp[alpha] = count
    if int(numpy.prod(mp)) != kpts.shape[0]:
        raise ValueError(
            f'Cannot infer uniform MP grid: guessed {tuple(int(m) for m in mp)} but '
            f'have {kpts.shape[0]} k-points. Pass kmesh= explicitly.')
    return mp


def _candidate_shells(recip, kmesh, tol, max_shells, search_range):
    '''Enumerate candidate b-vectors, grouped into shells of equal |b|.'''
    db = recip / kmesh[:, None]  # spacing of the k-point sublattice in 1/Bohr
    grid = numpy.arange(-search_range, search_range + 1)
    ijk = numpy.array(numpy.meshgrid(grid, grid, grid, indexing='ij')).reshape(3, -1).T
    ijk = ijk[numpy.any(ijk != 0, axis=1)]
    bvecs = ijk @ db
    norms = numpy.linalg.norm(bvecs, axis=1)

    order = numpy.argsort(norms)
    bvecs = bvecs[order]
    norms = norms[order]

    shells = []
    i = 0
    n = len(norms)
    while i < n and len(shells) < max_shells:
        j = i
        while j < n and norms[j] - norms[i] < tol:
            j += 1
        shells.append(bvecs[i:j].copy())
        i = j
    return shells


def _select_shells(shells, tol):
    '''Add shells one at a time and solve for weights until B1 holds.'''
    target = numpy.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    for n in range(1, len(shells) + 1):
        A = numpy.zeros((n, 6))
        for s in range(n):
            shell = shells[s]
            A[s, 0] = (shell[:, 0] ** 2).sum()
            A[s, 1] = (shell[:, 1] ** 2).sum()
            A[s, 2] = (shell[:, 2] ** 2).sum()
            A[s, 3] = (shell[:, 0] * shell[:, 1]).sum()
            A[s, 4] = (shell[:, 0] * shell[:, 2]).sum()
            A[s, 5] = (shell[:, 1] * shell[:, 2]).sum()
        w, *_ = numpy.linalg.lstsq(A.T, target, rcond=None)
        if numpy.max(numpy.abs(A.T @ w - target)) < tol:
            bvecs = numpy.concatenate(shells[:n], axis=0)
            weights = numpy.concatenate([
                numpy.full(len(shells[s]), w[s]) for s in range(n)
            ])
            return bvecs, weights
    raise RuntimeError(
        f'B1 completeness relation not satisfied within {len(shells)} shells. '
        f'Increase max_shells or search_range.')


def _build_connectivity(recip, kpts, bvecs, tol):
    '''For each (k, b), find which k-point k+b wraps to and the G shift.'''
    inv_recip = numpy.linalg.inv(recip)
    frac_kpts = kpts @ inv_recip
    frac_kpts = frac_kpts - numpy.floor(frac_kpts + tol)
    frac_b = bvecs @ inv_recip

    nk = len(kpts)
    nb = len(bvecs)
    kpb_idx = numpy.zeros((nk, nb), dtype=int)
    kpb_g = numpy.zeros((nk, nb, 3), dtype=int)

    for i in range(nk):
        for b in range(nb):
            diff = (frac_kpts[i] + frac_b[b])[None, :] - frac_kpts
            g = numpy.round(diff)
            residual = numpy.linalg.norm(diff - g, axis=1)
            j = int(numpy.argmin(residual))
            if residual[j] > tol:
                raise ValueError(
                    f'k+b not matched to any k-point for k={i}, b={b} '
                    f'(min residual {residual[j]:.3e} > tol {tol:.3e})')
            kpb_idx[i, b] = j
            kpb_g[i, b] = g[j]
    return kpb_idx, kpb_g
