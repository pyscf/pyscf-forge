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
Top-level MLWF driver.

The pipeline is: initial projections A -> (optional) disentanglement to
pick a subspace -> steepest-descent Wannierization on the unitary gauge.
Called as a plain function on a converged k-point SCF object.
'''

import numpy


def kernel(kmf, proj_guess, *, spin=0,
           dis_win=None, dis_froz=None,
           max_iter=100, dis_max_iter=200,
           conv_tol=1e-10, verbose=None):
    '''Build maximally localized Wannier functions from a converged k-point SCF.

    Args:
        kmf : pbc.scf.KRHF / KUHF / KRKS / KUKS (converged, with at least 2 k-points).
        proj_guess : list[dict], one entry per Wannier function. n_wann is the
            length of this list. Each entry is
                {'center': (x, y, z) in Bohr,
                 'l': 0/1/2, 'm': in [-l, l], 'zeta': float}
            For hybrids, swap 'l'/'m' for
                'components': [(coeff, l, m), ...].

    Kwargs:
        spin : 0 or 1, picks the spin channel for KUHF/KUKS. Run twice if
            you need both spins.
        dis_win : (e_min, e_max) in Hartree. If None, no disentanglement
            and we require n_bands == n_wann. Otherwise the outer window
            picks a contiguous, k-uniform band subset (no bands crossing
            the window boundary at different k).
        dis_froz : (e_min, e_max). Not implemented in Tier A.
        max_iter, dis_max_iter, conv_tol : iteration controls. conv_tol is
            on the total spread change between iterations, in Bohr**2.
        verbose : currently unused. Reserved for future logger hookup.

    Returns (mo_coeff, centers, spreads, omega_i, omega_d, omega_od, converged).
    The converged flag is the AND of the disentangle and wannierise
    convergence flags. mo_coeff is the localized coefficient array for the
    selected spin channel, with shape (nk, nao, n_wann). Centers are in Bohr
    and spreads are in Bohr**2.
    '''
    from pyscf.pbc.lo.mlwf.bvectors import find_bvectors
    from pyscf.pbc.lo.mlwf.overlap import compute_mmn, _mo_for_spin
    from pyscf.pbc.lo.mlwf.projection import compute_amn
    from pyscf.pbc.lo.mlwf.disentangle import disentangle
    from pyscf.pbc.lo.mlwf.wannierise import wannierise

    if dis_froz is not None:
        raise NotImplementedError(
            'dis_froz (frozen inner window) is not implemented in Tier A')

    kpts = numpy.asarray(kmf.kpts)
    if kpts.ndim != 2 or kpts.shape[1] != 3:
        raise ValueError(f'kmf.kpts must have shape (nk, 3); got {kpts.shape}')
    if kpts.shape[0] < 2:
        raise ValueError('kernel requires at least 2 k-points')
    if not proj_guess:
        raise ValueError('proj_guess is empty')

    n_wann = len(proj_guess)
    nk = kpts.shape[0]

    mo_coeff = _mo_for_spin(kmf, spin, nk)
    n_bands_full = numpy.asarray(mo_coeff[0]).shape[1]

    if dis_win is None:
        if n_bands_full != n_wann:
            raise ValueError(
                f'dis_win=None requires n_bands == n_wann; got n_bands='
                f'{n_bands_full}, n_wann={n_wann}. Provide dis_win=(e_min, e_max) '
                f'to disentangle from a larger band window.')
        band_indices = None
    else:
        band_indices = _resolve_outer_window(kmf, spin, nk, dis_win, n_wann)

    bv = find_bvectors(kmf.cell, kpts)
    M_raw = compute_mmn(kmf, bv, spin=spin, band_indices=band_indices)
    A = compute_amn(kmf, proj_guess, spin=spin, band_indices=band_indices)

    U_dis, _omega_i_dis, dis_converged = disentangle(
        M_raw, A, bv, max_iter=dis_max_iter, conv_tol=conv_tol)
    U, centers, spreads, omega_i, omega_d, omega_od, wan_converged = wannierise(
        M_raw, U_dis, bv, max_iter=max_iter, conv_tol=conv_tol)

    converged = bool(dis_converged and wan_converged)
    mo_coeff_loc = _rotate_mo_coeff(mo_coeff, U, band_indices)
    return mo_coeff_loc, centers, spreads, omega_i, omega_d, omega_od, converged


def _rotate_mo_coeff(mo_coeff, U, band_indices):
    '''Rotate Bloch MO coefficients into the localized Wannier gauge.'''
    if band_indices is None:
        band_slice = slice(None)
    else:
        band_slice = numpy.asarray(band_indices)

    mo_coeff_loc = []
    for ki in range(U.shape[0]):
        coeff = numpy.asarray(mo_coeff[ki])[:, band_slice]
        mo_coeff_loc.append(coeff @ U[ki])
    return numpy.asarray(mo_coeff_loc)


def _resolve_outer_window(kmf, spin, nk, dis_win, n_wann):
    '''Translate dis_win into a single shared band-index array.

    Raises NotImplementedError if the band set inside the window differs
    from one k to another — that's the entangled-metallic case we don't
    handle in Tier A (it would need ragged per-k arrays everywhere).
    '''
    from pyscf.pbc.lo.mlwf.overlap import _mo_energy_for_spin
    e_min, e_max = dis_win
    if e_max <= e_min:
        raise ValueError(f'dis_win is empty: e_min={e_min} >= e_max={e_max}')

    mo_energy = _mo_energy_for_spin(kmf, spin, nk)
    indices_per_k = []
    for ki in range(nk):
        e = numpy.asarray(mo_energy[ki])
        mask = (e >= e_min) & (e <= e_max)
        indices_per_k.append(numpy.where(mask)[0])

    for ki in range(1, nk):
        if not numpy.array_equal(indices_per_k[0], indices_per_k[ki]):
            raise NotImplementedError(
                f'dis_win selects different band indices across k-points '
                f'(k=0: {indices_per_k[0].tolist()}, k={ki}: '
                f'{indices_per_k[ki].tolist()}). Tier A requires a uniform '
                f'band subset; this usually means picking a window that lies '
                f'fully inside a gap at every k.')

    band_indices = indices_per_k[0]
    if band_indices.size < n_wann:
        raise ValueError(
            f'dis_win selects only {band_indices.size} bands but '
            f'{n_wann} Wannier functions requested')
    return band_indices
