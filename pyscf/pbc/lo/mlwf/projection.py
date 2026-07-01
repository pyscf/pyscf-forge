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
Initial projection overlap A_mn^k = <psi_mk | g_n>  (MV Eq. 62 / SMV Eq. 22).

Trial orbitals have the form
    g_n(r) = R_zeta(|r - r_n|) * sum_i coeff_i Y_{l_i, m_i}(hat{r - r_n})
with a hydrogenic 1s radial R_zeta(r) = 2 zeta^(3/2) exp(-zeta r). That
matches Wannier90's r=1 default in the .win "projections" block, which
makes it easy to reproduce a wannier90 setup here.

The integral is evaluated directly on PySCF's primary-cell DFT grid:
    A[k, m, n] = sum_i w_i * conj(psi_mk(r_i)) * g_n(r_i)
with psi_mk from the Bloch AOs. Accuracy is limited by the grid cutoff
at the cell boundary, so the trial orbital should fit inside the cell
(zeta large enough).
'''

import numpy

from pyscf import dft


def compute_amn(kmf, proj_guess, *, spin=0, band_indices=None,
                grid_level=3, grids=None):
    '''Project each Bloch state onto each trial orbital.

    Args:
        kmf : pbc.scf.KRHF / KUHF / KRKS / KUKS (converged).
        proj_guess : list[dict]. One entry per Wannier function:
                {'center': (x, y, z) in Bohr,
                 'l': 0, 1, or 2,
                 'm': -l..l,
                 'zeta': float}
            For hybrids, drop 'l'/'m' and use
                {'center': ..., 'zeta': ...,
                 'components': [(coeff, l_i, m_i), ...]}

    Kwargs:
        spin : 0 or 1. Picks the spin channel for KUHF/KUKS.
        band_indices : which bands to keep (uniform across k). None = all.
        grid_level : DFT grid level (0-9). Ignored if `grids` is passed.
        grids : an already-built pyscf.dft.gen_grid.Grids. Useful to share
            a grid across multiple compute_amn calls.

    Returns:
        A : (nk, n_bands, n_wann) complex128, A[k, m, n] = <psi_mk | g_n>.
    '''
    from pyscf.pbc.dft import numint as pbc_numint
    from pyscf.pbc.lo.mlwf.overlap import _mo_for_spin

    cell = kmf.cell
    kpts = numpy.asarray(kmf.kpts)
    nk = kpts.shape[0]

    mo_coeff = _mo_for_spin(kmf, spin, nk)

    if band_indices is None:
        band_slice = slice(None)
        n_bands = numpy.asarray(mo_coeff[0]).shape[1]
    else:
        band_slice = numpy.asarray(band_indices)
        n_bands = band_slice.size

    n_wann = len(proj_guess)
    if n_wann == 0:
        raise ValueError('proj_guess is empty')

    if grids is None:
        grids = dft.gen_grid.Grids(cell)
        grids.level = grid_level
        grids.build()
    coords = grids.coords
    weights = grids.weights

    # Each row wg[n] is g_n evaluated on the grid, pre-multiplied by the
    # integration weights so the later einsum is a plain dot product.
    gs = numpy.stack([_evaluate_trial(coords, p) for p in proj_guess])
    wg = weights[None, :] * gs

    ao_kpts = pbc_numint.eval_ao_kpts(cell, coords, kpts=kpts)  # list of (ngrid, nao)

    A = numpy.empty((nk, n_bands, n_wann), dtype=numpy.complex128)
    for ki in range(nk):
        C = numpy.asarray(mo_coeff[ki])[:, band_slice]         # (nao, n_bands)
        psi_k = ao_kpts[ki] @ C                                 # (ngrid, n_bands)
        A[ki] = numpy.einsum('ni,im->mn', wg, psi_k.conj(), optimize=True)
    return A


def lowdin_orthonormalize(A):
    '''Per-k Lowdin/SVD orthonormalization, giving U(k) with U(k)^H U(k) = I.

    Uses the SVD A = V Sigma W^H and drops the singular values: U = V W^H.
    Works for rectangular A as long as n_bands >= n_wann.
    '''
    A = numpy.asarray(A)
    if A.ndim != 3:
        raise ValueError(f'A must be 3D (nk, n_bands, n_wann); got shape {A.shape}')
    if A.shape[1] < A.shape[2]:
        raise ValueError(
            f'A has {A.shape[1]} bands < {A.shape[2]} Wannier functions; '
            f'orthonormalization requires n_bands >= n_wann')
    nk = A.shape[0]
    U = numpy.empty_like(A)
    for ki in range(nk):
        V, _s, Wh = numpy.linalg.svd(A[ki], full_matrices=False)
        U[ki] = V @ Wh
    return U


def _evaluate_trial(coords, proj):
    '''Evaluate one trial orbital on a grid. Returns a (ngrid,) array.'''
    center = numpy.asarray(proj['center'], dtype=float)
    zeta = float(proj['zeta'])
    dr = coords - center[None, :]
    r = numpy.linalg.norm(dr, axis=1)
    R = 2.0 * zeta ** 1.5 * numpy.exp(-zeta * r)

    if 'components' in proj:
        Y = numpy.zeros_like(r)
        for coeff, li, mi in proj['components']:
            Y += float(coeff) * _real_sph_harm(dr, r, li, mi)
    else:
        Y = _real_sph_harm(dr, r, int(proj['l']), int(proj['m']))

    return R * Y


def _real_sph_harm(dr, r, l, m):
    '''Real Y_lm on a grid. r[i] must equal |dr[i]|.

    At r=0 we return the l=0 constant for s and zero for l>=1, which is
    the physical limit and avoids the 0/0 in the Cartesian forms below.
    '''
    x, y, z = dr[:, 0], dr[:, 1], dr[:, 2]
    safe_r = numpy.where(r > 1e-30, r, 1.0)

    if l == 0:
        return numpy.full_like(r, 1.0 / numpy.sqrt(4.0 * numpy.pi))

    if l == 1:
        prefac = numpy.sqrt(3.0 / (4.0 * numpy.pi))
        cart = {-1: y, 0: z, 1: x}[m]
        Y = prefac * cart / safe_r
    elif l == 2:
        r2 = x * x + y * y + z * z
        safe_r2 = safe_r ** 2
        if m == -2:
            Y = numpy.sqrt(15.0 / (4.0 * numpy.pi)) * x * y / safe_r2
        elif m == -1:
            Y = numpy.sqrt(15.0 / (4.0 * numpy.pi)) * y * z / safe_r2
        elif m == 0:
            Y = numpy.sqrt(5.0 / (16.0 * numpy.pi)) * (3.0 * z * z - r2) / safe_r2
        elif m == 1:
            Y = numpy.sqrt(15.0 / (4.0 * numpy.pi)) * x * z / safe_r2
        elif m == 2:
            Y = numpy.sqrt(15.0 / (16.0 * numpy.pi)) * (x * x - y * y) / safe_r2
        else:
            raise ValueError(f'Invalid m={m} for l=2')
    else:
        raise NotImplementedError(f'l={l} not supported; Tier A covers l in (0, 1, 2)')

    return numpy.where(r > 1e-30, Y, 0.0)
