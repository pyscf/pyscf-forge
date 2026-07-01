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
Bloch overlap matrix M_mn^(k, b) = <u_mk | u_n,k+b>.

MV Eq. (25). This is the same formula pbc.tools.pywannier90 feeds to
wannier90's .mmn, so the output drops into that pipeline unchanged.
'''

import numpy

from pyscf.pbc.df import ft_ao


def compute_mmn(kmf, bvectors, *, spin=0, band_indices=None):
    '''Bloch overlap M[k, b, m, n] = <u_mk | u_n,k+b>.

    Args:
        kmf : pbc.scf.KRHF / KUHF / KRKS / KUKS (converged).
        bvectors : 4-tuple from find_bvectors. Only kpb_idx and kpb_g are
            actually used here.

    Kwargs:
        spin : 0 or 1. Picks the spin channel for KUHF/KUKS; ignored for
            restricted.
        band_indices : which bands to keep (uniform across k). None = all.

    Returns:
        M : (nk, n_nn, n_bands, n_bands) complex128.
    '''
    bvecs, _weights, kpb_idx, kpb_g = bvectors

    cell = kmf.cell
    kpts = numpy.asarray(kmf.kpts)
    nk = kpts.shape[0]
    n_nn = bvecs.shape[0]

    mo_coeff = _mo_for_spin(kmf, spin, nk)
    if band_indices is None:
        band_slice = slice(None)
        n_bands = numpy.asarray(mo_coeff[0]).shape[1]
    else:
        band_slice = numpy.asarray(band_indices)
        n_bands = band_slice.size

    recip = cell.reciprocal_vectors()
    M = numpy.empty((nk, n_nn, n_bands, n_bands), dtype=numpy.complex128)
    zero_q = numpy.zeros(3)

    for ki in range(nk):
        k1 = kpts[ki]
        Cm = numpy.asarray(mo_coeff[ki])[:, band_slice]
        for bi in range(n_nn):
            kj = int(kpb_idx[ki, bi])
            G = kpb_g[ki, bi]
            k2 = kpts[kj] + G @ recip  # k+b, unfolded out of the BZ

            s_ao = ft_ao.ft_aopair(cell, k1 - k2, kpti_kptj=[k2, k1], q=zero_q)[0]

            Cn = numpy.asarray(mo_coeff[kj])[:, band_slice]
            M[ki, bi] = numpy.einsum('nu,vm,uv->nm',
                                      Cn.T.conj(), Cm, s_ao).conj()
    return M


def _mo_for_spin(kmf, spin, nk):
    return _spin_select(kmf.mo_coeff, spin, nk, 'mo_coeff')


def _mo_energy_for_spin(kmf, spin, nk):
    return _spin_select(kmf.mo_energy, spin, nk, 'mo_energy')


def _spin_select(obj, spin, nk, what):
    # RHF shapes: (nk, ...) array or nk-long list. UHF shapes: length-2
    # outer (one per spin). We try array layouts first, then fall back to
    # the list-of-per-k form used in older pyscf versions.
    if hasattr(obj, 'ndim'):
        if obj.ndim in (2, 3):                    # (nk, nao[, nmo])
            return obj
        if obj.ndim in (3, 4) and obj.shape[0] == 2:
            return obj[spin]
    if len(obj) == 2 and hasattr(obj[0], '__len__') and len(obj[0]) == nk:
        return obj[spin]
    if len(obj) == nk:
        return obj
    raise ValueError(
        f'Could not determine spin structure of kmf.{what}: expected RHF-shape '
        f'(nk entries) or UHF-shape (2 x nk entries).')
