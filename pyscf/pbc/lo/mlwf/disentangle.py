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
SMV disentanglement: pick the best n_wann-dim subspace at each k.

When there are more bands available than Wannier functions we want, this
iteratively picks the n_wann-dim subspace that minimizes Omega_I, the
gauge-invariant part of the spread. The core update is SMV Eq. (14):
diagonalize Z(k) = sum_b w_b M(k, b) P(k+b) M(k, b)^H and keep the top
eigenvectors. (Souza, Marzari & Vanderbilt, PRB 65, 035109 (2001).)
'''

import numpy

from pyscf.pbc.lo.mlwf.projection import lowdin_orthonormalize
from pyscf.pbc.lo.mlwf.spread import spread_decomposition
from pyscf.pbc.lo.mlwf.wannierise import rotate_mmn


def disentangle(M_raw, A, bvectors, *,
                max_iter=200, conv_tol=1e-10):
    '''Pick the n_wann-dim subspace that minimizes Omega_I.

    Args:
        M_raw : (nk, n_nn, n_bands, n_bands) Bloch overlaps in the band basis.
            If you're applying an outer energy window, pre-filter M_raw and A
            to the kept bands. Tier A also assumes the kept band indices are
            the same at every k (no bands crossing the window boundary).
        A : (nk, n_bands, n_wann) starting projection overlaps from
            compute_amn(), filtered to the same band subset as M_raw.
        bvectors : 4-tuple from find_bvectors().

    Kwargs:
        max_iter : iteration cap.
        conv_tol : threshold on |dOmega_I| in Bohr**2.

    Returns (U_dis, omega_i, converged). When n_bands == n_wann there's
    no subspace choice to make; we skip the iteration and just return the
    Lowdin orthonormalization of A.
    '''
    _bvecs, weights, kpb_idx, _kpb_g = bvectors
    nk, n_nn, n_bands, _ = M_raw.shape
    _, _, n_wann = A.shape
    if A.shape[:2] != (nk, n_bands):
        raise ValueError(
            f'A shape {A.shape} is inconsistent with M_raw shape {M_raw.shape}')
    if n_bands < n_wann:
        raise ValueError(f'n_bands={n_bands} < n_wann={n_wann}')

    U = lowdin_orthonormalize(A)

    # No subspace to pick — short-circuit.
    if n_bands == n_wann:
        M_rot = rotate_mmn(M_raw, U, kpb_idx)
        _, _, oi, _, _ = spread_decomposition(M_rot, bvectors)
        return U, oi, True

    oi_prev = numpy.inf
    converged = False

    for it in range(max_iter):
        M_rot = rotate_mmn(M_raw, U, kpb_idx)
        _, _, oi, _, _ = spread_decomposition(M_rot, bvectors)

        if it > 0 and abs(oi - oi_prev) < conv_tol:
            converged = True
            break
        oi_prev = oi

        # P(k) = U(k) U(k)^H is the projector onto the current subspace.
        P = numpy.einsum('kiw,kjw->kij', U, U.conj())

        # SMV update: at each k, diagonalize Z(k) and keep the n_wann
        # eigenvectors with the largest eigenvalues. We symmetrize Z by
        # hand so eigh sees a clean Hermitian input.
        U_new = numpy.empty_like(U)
        for ki in range(nk):
            Z = numpy.zeros((n_bands, n_bands), dtype=numpy.complex128)
            for bi in range(n_nn):
                kj = int(kpb_idx[ki, bi])
                Mkb = M_raw[ki, bi]
                Z += weights[bi] * Mkb @ P[kj] @ Mkb.conj().T
            Z = 0.5 * (Z + Z.conj().T)
            _eigvals, eigvecs = numpy.linalg.eigh(Z)
            U_new[ki] = eigvecs[:, -n_wann:]   # eigh sorts ascending; take the top

        U = U_new

    # Recompute the spread on the final U so the return value is consistent.
    M_rot = rotate_mmn(M_raw, U, kpb_idx)
    _, _, oi, _, _ = spread_decomposition(M_rot, bvectors)

    return U, oi, converged
