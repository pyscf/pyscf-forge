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
MLWF steepest-descent loop on the unitary gauge (MV 1997).

For isolated bands (n_bands == n_wann) this minimizes the full spread Omega.
For the disentangled case (n_bands > n_wann), Omega_I is fixed by the
subspace already picked by disentangle(), and we only optimize Omega_D +
Omega_OD within that subspace.
'''

import numpy
from scipy.linalg import expm

from pyscf.pbc.lo.spread import spread_decomposition


def wannierise(M_raw, U_init, bvectors, *,
               max_iter=200, conv_tol=1e-10, alpha=None):
    '''Steepest descent on the gauge unitaries U(k).

    Args:
        M_raw : (nk, n_nn, n_bands, n_bands) Bloch overlaps in the band basis.
        U_init : (nk, n_bands, n_wann) starting point. Usually Lowdin(A) or
            the U_dis from disentangle().
        bvectors : 4-tuple from find_bvectors().

    Kwargs:
        max_iter, conv_tol : iteration limit, and the |dOmega| threshold.
        alpha : step size. Defaults to MV's natural scale 1 / (4 sum w_b),
            which is the right order of magnitude in the small-step limit.

    Returns (U, centers, spreads, omega_i, omega_d, omega_od, converged).
    '''
    _bvecs, weights, kpb_idx, _kpb_g = bvectors
    nk, n_nn, n_bands, _ = M_raw.shape
    _, _, n_wann = U_init.shape
    if n_bands < n_wann:
        raise ValueError(f'n_bands={n_bands} is less than n_wann={n_wann}')

    if alpha is None:
        alpha = 1.0 / (4.0 * float(numpy.sum(weights)))

    U = U_init.astype(numpy.complex128, copy=True)
    omega_prev = numpy.inf
    converged = False

    for it in range(max_iter):
        M_rot = rotate_mmn(M_raw, U, kpb_idx)
        centers, spreads, oi, ood, od = spread_decomposition(M_rot, bvectors)
        omega = oi + ood + od

        if it > 0 and abs(omega - omega_prev) < conv_tol:
            converged = True
            break
        omega_prev = omega

        G = _spread_gradient(M_rot, centers, bvectors)
        U = _step_unitary(U, -alpha * G)

    # Recompute spreads/centers from the last U so the return is consistent.
    M_rot = rotate_mmn(M_raw, U, kpb_idx)
    centers, spreads, oi, ood, od = spread_decomposition(M_rot, bvectors)

    return U, centers, spreads, oi, od, ood, converged


def rotate_mmn(M_raw, U, kpb_idx):
    '''Apply the gauge: M_rot[k, b] = U(k)^H @ M_raw[k, b] @ U(k+b).

    Works for both square U (isolated bands) and rectangular U (disentangled
    case). The output is always (nk, n_nn, n_wann, n_wann).
    '''
    nk, n_nn = M_raw.shape[:2]
    n_wann = U.shape[2]
    M_rot = numpy.empty((nk, n_nn, n_wann, n_wann), dtype=numpy.complex128)
    for ki in range(nk):
        Uk_H = U[ki].conj().T
        for bi in range(n_nn):
            kj = int(kpb_idx[ki, bi])
            M_rot[ki, bi] = Uk_H @ M_raw[ki, bi] @ U[kj]
    return M_rot


def _spread_gradient(M, centers, bvectors):
    '''Anti-Hermitian gradient G(k) = dOmega_tilde/dW(k) — MV Eq 57.

    The result is anti-Hermitian by construction (A_R is anti-Hermitian
    and S_T is too, since dividing by 2i flips Hermitian to anti-Hermitian),
    so exp(-alpha * G) stays unitary at each step.
    '''
    bvecs, weights, _kpb_idx, _kpb_g = bvectors
    nk, n_nn, n_wann, _ = M.shape

    M_diag = numpy.diagonal(M, axis1=2, axis2=3)                   # (nk, n_nn, n_wann)
    log_im = numpy.angle(M_diag)
    b_dot_r = numpy.einsum('bx,nx->bn', bvecs, centers)            # (n_nn, n_wann)
    q = log_im + b_dot_r[None, :, :]                                # (nk, n_nn, n_wann)

    # R_mn = M_mn * conj(M_nn). Broadcasts the diagonal across the m axis.
    R = M * M_diag.conj()[:, :, None, :]

    # T_mn = (M_mn / M_nn) * q_n. Guard the divide; on the converged manifold
    # |M_nn| is well away from zero, but a tiny floor keeps things sane.
    M_diag_safe = numpy.where(numpy.abs(M_diag) > 1e-30, M_diag, 1.0)
    T = M / M_diag_safe[:, :, None, :] * q[:, :, None, :]

    R_H = R.conj().transpose(0, 1, 3, 2)
    T_H = T.conj().transpose(0, 1, 3, 2)
    A_R = 0.5 * (R - R_H)
    S_T = (T + T_H) / (2j)

    return 4.0 * numpy.einsum('b,kbmn->kmn', weights, A_R - S_T)


def _step_unitary(U, W):
    '''Apply U_new[k] = U[k] @ expm(W[k]). W is anti-Hermitian, so expm
    is unitary and U_new stays on the Stiefel manifold.'''
    nk = U.shape[0]
    U_new = numpy.empty_like(U)
    for ki in range(nk):
        U_new[ki] = U[ki] @ expm(W[ki])
    return U_new
