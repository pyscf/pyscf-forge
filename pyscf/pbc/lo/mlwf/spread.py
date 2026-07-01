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
Spread functional Omega = Omega_I + Omega_OD + Omega_D.

MV Eqs. (31) and (34)-(36). Only Omega_D and Omega_OD are sensitive to
the unitary gauge; Omega_I depends only on the subspace (important for
disentanglement).
'''

import numpy


def spread_decomposition(M, bvectors):
    '''Centers, per-WF spreads, and the Omega_I/Omega_OD/Omega_D split.

    M is the Wannier-gauge overlap U(k)^H M_raw U(k+b). bvectors comes
    from find_bvectors; only bvecs and weights are used here.

    Returns (centers, spreads, omega_i, omega_od, omega_d). Centers are
    in Bohr, everything else in Bohr^2. As a sanity check on the output,
    omega_i + omega_od + omega_d should equal spreads.sum().
    '''
    bvecs, weights, _kpb_idx, _kpb_g = bvectors
    nk, n_nn, n_wann, _ = M.shape

    M_diag = numpy.diagonal(M, axis1=2, axis2=3)                # (nk, n_nn, n_wann)
    log_im = numpy.angle(M_diag)                                 # Im ln M_nn
    abs_sq_diag = (M_diag * M_diag.conj()).real                  # |M_nn|^2

    # Centers (MV Eq 31): a weighted average of b * Im ln M_nn.
    centers = -numpy.einsum('b,bx,kbn->nx', weights, bvecs, log_im) / nk

    # <r^2>_n per the small-b expansion in MV.
    r2_n = numpy.einsum(
        'b,kbn->n', weights, (1.0 - abs_sq_diag) + log_im ** 2) / nk
    spreads = r2_n - numpy.sum(centers ** 2, axis=1)

    # Omega_I (Eq 34): the piece that doesn't depend on the gauge.
    abs_sq_full = (M * M.conj()).real
    sum_full_kb = abs_sq_full.sum(axis=(2, 3))                   # (nk, n_nn)
    omega_i = (numpy.sum(weights) * n_wann
               - numpy.einsum('b,kb->', weights, sum_full_kb) / nk)

    # Omega_OD (Eq 35): off-diagonal part of |M|^2.
    sum_diag_kb = abs_sq_diag.sum(axis=2)
    omega_od = numpy.einsum(
        'b,kb->', weights, sum_full_kb - sum_diag_kb) / nk

    # Omega_D (Eq 36): how much the diagonal phases drift from b . <r>_n.
    b_dot_r = numpy.einsum('bx,nx->bn', bvecs, centers)
    q = log_im + b_dot_r[None, :, :]
    omega_d = numpy.einsum('b,kbn->', weights, q ** 2) / nk

    return centers, spreads, omega_i, omega_od, omega_d
