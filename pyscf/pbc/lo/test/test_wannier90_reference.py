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
End-to-end validation of pyscf.pbc.lo against a compiled wannier90.x.

Workflow:
    1. Write a .win file for the PySCF system.
    2. Run wannier90.x -pp to get its b-vector shell choice (.nnkp).
    3. Parse .nnkp into the (bvecs, weights, kpb_idx, kpb_g) tuple.
    4. Build M_mn, A_mn, and band energies with pyscf.pbc.lo and write them in
       wannier90 format (.mmn, .amn, .eig).
    5. Run wannier90.x to localize.
    6. Parse final spreads/centers from .wout.
    7. Run pyscf.pbc.lo's wannierise on the SAME (M, A, bvectors) inputs.
    8. Compare spreads and centers.
'''

import os
import re
import subprocess
import tempfile

import numpy
import pytest

from pyscf.pbc import gto, scf
from pyscf.pbc.lo.overlap import compute_mmn
from pyscf.pbc.lo.projection import compute_amn
from pyscf.pbc.lo.disentangle import disentangle
from pyscf.pbc.lo.wannierise import wannierise


BOHR_TO_ANG = 0.52917721067  # CODATA; wannier90's default value
HARTREE_TO_EV = 27.21138602

W90_EXE = os.environ.get('W90_EXE')


# ---------------------------------------------------------------- fixtures


@pytest.fixture(scope='module')
def w90_exe():
    if W90_EXE is None:
        pytest.skip('Set W90_EXE=/path/to/wannier90.x to enable the '
                    'wannier90 cross-validation tests.')
    if not os.path.exists(W90_EXE):
        pytest.skip(f'wannier90.x not found at {W90_EXE}')
    return W90_EXE


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


# --------------------------------------------------------- write .win file


def _write_win(path, cell, kpts_frac, mp_grid, proj_guess,
               n_wann, n_bands, num_iter=400, conv_tol=1e-12,
               dis_num_iter=0, dis_win=None):
    '''Minimal wannier90 .win for isolated- or disentangled-band runs in Bohr.'''
    a = cell.lattice_vectors()  # Bohr (cell.unit='Bohr')
    symbols = [cell.atom_symbol(i) for i in range(cell.natm)]
    coords = cell.atom_coords()  # Bohr

    inv_a = numpy.linalg.inv(a)

    with open(path, 'w') as f:
        f.write(f'num_wann  = {n_wann}\n')
        f.write(f'num_bands = {n_bands}\n')
        f.write(f'num_iter  = {num_iter}\n')
        f.write(f'conv_tol  = {conv_tol:.3e}\n')
        f.write('conv_window = 5\n')
        f.write('search_shells = 36\n')
        f.write(f'dis_num_iter = {dis_num_iter}\n')
        if dis_win is not None:
            f.write(f'dis_win_min = {dis_win[0]:.6f}\n')
            f.write(f'dis_win_max = {dis_win[1]:.6f}\n')
            f.write(f'dis_conv_tol = {conv_tol:.3e}\n')

        f.write('\nbegin unit_cell_cart\nbohr\n')
        for row in a:
            f.write(f'{row[0]:.10f} {row[1]:.10f} {row[2]:.10f}\n')
        f.write('end unit_cell_cart\n')

        f.write('\nbegin atoms_cart\nbohr\n')
        for sym, xyz in zip(symbols, coords):
            f.write(f'{sym} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}\n')
        f.write('end atoms_cart\n')

        f.write('\nbegin projections\n')
        for proj in proj_guess:
            frac = inv_a.T @ numpy.asarray(proj['center'], dtype=float)
            if 'components' in proj:
                raise NotImplementedError(
                    'Hybrid projections not serialized to .win here.')
            l, m = int(proj['l']), int(proj['m'])
            label = _LM_LABEL[(l, m)]
            zeta = float(proj['zeta'])
            f.write(f'f={frac[0]:.10f},{frac[1]:.10f},{frac[2]:.10f}:'
                    f'{label}:r=1:zona={zeta}\n')
        f.write('end projections\n')

        f.write(f'\nmp_grid : {mp_grid[0]} {mp_grid[1]} {mp_grid[2]}\n')
        f.write('\nbegin kpoints\n')
        for k in kpts_frac:
            f.write(f'{k[0]:.10f} {k[1]:.10f} {k[2]:.10f}\n')
        f.write('end kpoints\n')


_LM_LABEL = {
    (0, 0): 's',
    (1, 0): 'pz', (1, 1): 'px', (1, -1): 'py',
    (2, 0): 'dz2', (2, 1): 'dxz', (2, -1): 'dyz',
    (2, 2): 'dx2-y2', (2, -2): 'dxy',
}


# -------------------------------------------------------------- .nnkp parse


def _parse_nnkp(path, kpts_cart, recip):
    '''Parse .nnkp and produce (bvectors_tuple, raw_per_k_info).

    Wannier90 lists the nnkpts in a k-dependent order (at k=0 the first
    neighbor may be +b; at another k the first neighbor of the same magnitude
    may be -b). Our bvectors tuple uses a single shared bvecs[bi] array across
    k, so we canonicalize to the k=0 ordering and return per-k permutation
    info (`raw_kpb_idx`, `raw_kpb_g`, and the permutation) so the caller can
    still write `.mmn` in wannier90's native per-k order.
    '''
    with open(path) as f:
        text = f.read()
    match = re.search(r'begin nnkpts\s*\n\s*(\d+)\s*\n(.*?)\nend nnkpts',
                      text, re.DOTALL)
    if not match:
        raise RuntimeError(f'Could not find nnkpts block in {path}')
    n_nn = int(match.group(1))
    rows = match.group(2).strip().split('\n')
    nk = len(kpts_cart)
    assert len(rows) == nk * n_nn

    raw_kpb_idx = numpy.empty((nk, n_nn), dtype=int)
    raw_kpb_g = numpy.empty((nk, n_nn, 3), dtype=int)
    raw_bvecs = numpy.empty((nk, n_nn, 3), dtype=float)

    for row_i, line in enumerate(rows):
        parts = line.split()
        ki = int(parts[0]) - 1
        kj = int(parts[1]) - 1
        G = numpy.array([int(x) for x in parts[2:5]])
        bi = row_i % n_nn
        assert row_i // n_nn == ki
        raw_kpb_idx[ki, bi] = kj
        raw_kpb_g[ki, bi] = G
        raw_bvecs[ki, bi] = kpts_cart[kj] + G @ recip - kpts_cart[ki]

    # Canonical ordering is whatever wannier90 picked at k=0.
    canonical = raw_bvecs[0].copy()

    # Permutation: perm[ki, bi_canon] -> bi_w90 at ki such that
    # raw_bvecs[ki, bi_w90] == canonical[bi_canon].
    perm = numpy.full((nk, n_nn), -1, dtype=int)
    for ki in range(nk):
        for bi_canon in range(n_nn):
            diffs = numpy.linalg.norm(
                raw_bvecs[ki] - canonical[bi_canon], axis=1)
            matches = numpy.where(diffs < 1e-8)[0]
            if len(matches) == 0:
                raise RuntimeError(
                    f'At k={ki}, canonical b-vector '
                    f'{canonical[bi_canon]} not found in wannier90 nnkpts')
            perm[ki, bi_canon] = matches[0]

    # Reorder raw_* to canonical.
    canon_kpb_idx = numpy.empty((nk, n_nn), dtype=int)
    canon_kpb_g = numpy.empty((nk, n_nn, 3), dtype=int)
    for ki in range(nk):
        canon_kpb_idx[ki] = raw_kpb_idx[ki, perm[ki]]
        canon_kpb_g[ki] = raw_kpb_g[ki, perm[ki]]

    bv = (canonical, _b1_weights(canonical), canon_kpb_idx, canon_kpb_g)
    return bv, raw_kpb_idx, raw_kpb_g, perm


def _b1_weights(bvecs):
    '''Solve sum_b w_b b_a b_b = delta_{a,b} for the given b-vector set.

    Groups b-vectors into equidistant shells, then least-squares.
    '''
    tol = 1e-6
    norms = numpy.linalg.norm(bvecs, axis=1)
    order = numpy.argsort(norms)
    sorted_bvecs = bvecs[order]
    sorted_norms = norms[order]

    shell_starts = [0]
    for i in range(1, len(sorted_norms)):
        if sorted_norms[i] - sorted_norms[i - 1] > tol:
            shell_starts.append(i)
    shell_starts.append(len(sorted_norms))
    n_shells = len(shell_starts) - 1

    A = numpy.zeros((n_shells, 6))
    for s in range(n_shells):
        sh = sorted_bvecs[shell_starts[s]:shell_starts[s + 1]]
        A[s, 0] = (sh[:, 0] ** 2).sum()
        A[s, 1] = (sh[:, 1] ** 2).sum()
        A[s, 2] = (sh[:, 2] ** 2).sum()
        A[s, 3] = (sh[:, 0] * sh[:, 1]).sum()
        A[s, 4] = (sh[:, 0] * sh[:, 2]).sum()
        A[s, 5] = (sh[:, 1] * sh[:, 2]).sum()
    target = numpy.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    w_shells, *_ = numpy.linalg.lstsq(A.T, target, rcond=None)

    weights_sorted = numpy.empty(len(bvecs))
    for s in range(n_shells):
        weights_sorted[shell_starts[s]:shell_starts[s + 1]] = w_shells[s]
    weights = numpy.empty_like(weights_sorted)
    weights[order] = weights_sorted
    return weights


# ----------------------------------------------------- .mmn, .amn, .eig writers


def _write_mmn(path, M_canonical, raw_kpb_idx, raw_kpb_g, perm):
    '''Write .mmn iterating in wannier90's native per-k b-vector order.

    M_canonical is indexed in our canonical (k=0) bi ordering. perm[ki, bi_canon]
    maps canonical bi to wannier90's per-k bi_w90. We invert per-k to iterate
    bi_w90 and pick up the corresponding canonical M block.
    '''
    nk, n_nn, n_bands, _ = M_canonical.shape
    inv_perm = numpy.empty_like(perm)
    for ki in range(nk):
        inv_perm[ki, perm[ki]] = numpy.arange(n_nn)

    with open(path, 'w') as f:
        f.write('Generated by pyscf.pbc.lo test harness\n')
        f.write(f'    {n_bands}    {nk}    {n_nn}\n')
        for ki in range(nk):
            for bi_w90 in range(n_nn):
                bi_canon = int(inv_perm[ki, bi_w90])
                kj = int(raw_kpb_idx[ki, bi_w90]) + 1
                G = raw_kpb_g[ki, bi_w90]
                f.write(f'    {ki + 1}  {kj}  {G[0]}  {G[1]}  {G[2]}\n')
                block = M_canonical[ki, bi_canon]
                for m in range(n_bands):
                    for n in range(n_bands):
                        v = block[m, n]
                        f.write(f'    {v.real:22.18e}  {v.imag:22.18e}\n')


def _write_amn(path, A):
    '''Write .amn. A shape: (nk, n_bands, n_wann), A[k, m, n] = <psi_mk | g_n>.'''
    nk, n_bands, n_wann = A.shape
    with open(path, 'w') as f:
        f.write('Generated by pyscf.pbc.lo test harness\n')
        f.write(f'    {n_bands}    {nk}    {n_wann}\n')
        for ki in range(nk):
            for n in range(n_wann):
                for m in range(n_bands):
                    v = A[ki, m, n]
                    f.write(f'    {m + 1}    {n + 1}    {ki + 1}    '
                            f'{v.real:22.18e}    {v.imag:22.18e}\n')


def _write_eig(path, energies_ev):
    '''Write .eig. energies_ev shape (nk, n_bands) in eV.'''
    nk, n_bands = energies_ev.shape
    with open(path, 'w') as f:
        for ki in range(nk):
            for m in range(n_bands):
                f.write(f'    {m + 1}    {ki + 1}    {energies_ev[ki, m]:22.18e}\n')


# --------------------------------------------------------------- .wout parse


_FINAL_STATE_RE = re.compile(
    r'WF centre and spread\s+(\d+)\s*\(\s*'
    r'(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*\)\s+'
    r'(-?\d+\.\d+)')


def _parse_wout(path):
    '''Extract the final Wannier centers and spreads from the .wout.

    Returns (centers_ang, spreads_ang2, total_ang2) in wannier90's native units.
    '''
    with open(path) as f:
        text = f.read()
    # Use the last "Final State" block before the summary.
    match = re.search(r'Final State(.*?)Sum of centres and spreads',
                      text, re.DOTALL)
    if not match:
        raise RuntimeError(f'No Final State block in {path}')
    block = match.group(1)
    centers, spreads = [], []
    for m in _FINAL_STATE_RE.finditer(block):
        centers.append([float(m.group(2)), float(m.group(3)), float(m.group(4))])
        spreads.append(float(m.group(5)))
    total = sum(spreads)
    return numpy.array(centers), numpy.array(spreads), total


# ----------------------------------------------------------------- the test


def test_kernel_matches_wannier90_h2(kmf_h2, w90_exe, tmp_path):
    cell = kmf_h2.cell
    kpts_cart = numpy.asarray(kmf_h2.kpts)
    kpts_frac = cell.get_scaled_kpts(kpts_cart)
    recip = cell.reciprocal_vectors()

    proj_guess = [
        {'center': (0.0, 0.0, 0.0), 'l': 0, 'm': 0, 'zeta': 1.0},
        {'center': (0.0, 0.0, 1.5), 'l': 0, 'm': 0, 'zeta': 1.0},
    ]
    n_wann = n_bands = 2

    seedname = 'h2'
    win_path = tmp_path / f'{seedname}.win'
    _write_win(win_path, cell, kpts_frac, mp_grid=(2, 2, 2),
               proj_guess=proj_guess, n_wann=n_wann, n_bands=n_bands)

    # Step: wannier90.x -pp -> .nnkp
    subprocess.run([w90_exe, '-pp', seedname], cwd=tmp_path,
                   check=True, capture_output=True)

    bv, raw_kpb_idx, raw_kpb_g, perm = _parse_nnkp(
        tmp_path / f'{seedname}.nnkp', kpts_cart, recip)

    # Step: build M, A on exactly those b-vectors.
    M = compute_mmn(kmf_h2, bv)
    A = compute_amn(kmf_h2, proj_guess)

    energies_ev = numpy.asarray(
        [numpy.asarray(kmf_h2.mo_energy[ki]) * HARTREE_TO_EV
         for ki in range(len(kpts_cart))])

    _write_mmn(tmp_path / f'{seedname}.mmn', M, raw_kpb_idx, raw_kpb_g, perm)
    _write_amn(tmp_path / f'{seedname}.amn', A)
    _write_eig(tmp_path / f'{seedname}.eig', energies_ev)

    # Step: wannier90.x -> localize
    result = subprocess.run([w90_exe, seedname], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, \
        f'wannier90.x failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}'

    w90_centers, w90_spreads, w90_total = _parse_wout(tmp_path / f'{seedname}.wout')

    # Step: our pipeline on the same (M, A, bv) inputs.
    U_dis, _oi_dis, _ = disentangle(M, A, bv, max_iter=400, conv_tol=1e-12)
    _U, centers, spreads, _oi, _od, _ood, _ = wannierise(
        M, U_dis, bv, max_iter=400, conv_tol=1e-12)

    our_total = float(spreads.sum())
    w90_total_bohr2 = w90_total / (BOHR_TO_ANG ** 2)
    assert abs(our_total - w90_total_bohr2) < 1e-4, \
        (f'Total spread mismatch: pyscf {our_total:.6f} Bohr^2, '
         f'wannier90 {w90_total_bohr2:.6f} Bohr^2 '
         f'({w90_total:.6f} Ang^2)')

    # Centers: compare the set, since wannier90 may order differently.
    our_centers_ang = centers * BOHR_TO_ANG
    _assert_centers_match(w90_centers, our_centers_ang,
                          tolerance=1e-3, box_a=cell.a * BOHR_TO_ANG)


def test_kernel_matches_wannier90_h2_disentangled(kmf_h2, w90_exe, tmp_path):
    '''Same H2 system, but disentangle 2 bands -> 1 Wannier function.'''
    cell = kmf_h2.cell
    kpts_cart = numpy.asarray(kmf_h2.kpts)
    kpts_frac = cell.get_scaled_kpts(kpts_cart)
    recip = cell.reciprocal_vectors()

    proj_guess = [
        {'center': (0.0, 0.0, 0.75), 'l': 0, 'm': 0, 'zeta': 1.0},
    ]
    n_wann, n_bands = 1, 2

    # Pick a window covering both H2 bands at every k.
    e_all = numpy.concatenate(
        [numpy.asarray(kmf_h2.mo_energy[k]) * HARTREE_TO_EV
         for k in range(len(kpts_cart))])
    dis_win = (float(e_all.min()) - 1.0, float(e_all.max()) + 1.0)

    seedname = 'h2d'
    _write_win(tmp_path / f'{seedname}.win', cell, kpts_frac, (2, 2, 2),
               proj_guess, n_wann=n_wann, n_bands=n_bands,
               dis_num_iter=400, dis_win=dis_win)

    subprocess.run([w90_exe, '-pp', seedname], cwd=tmp_path,
                   check=True, capture_output=True)
    bv, raw_kpb_idx, raw_kpb_g, perm = _parse_nnkp(
        tmp_path / f'{seedname}.nnkp', kpts_cart, recip)

    M = compute_mmn(kmf_h2, bv)
    A = compute_amn(kmf_h2, proj_guess)

    energies_ev = numpy.asarray(
        [numpy.asarray(kmf_h2.mo_energy[ki]) * HARTREE_TO_EV
         for ki in range(len(kpts_cart))])

    _write_mmn(tmp_path / f'{seedname}.mmn', M, raw_kpb_idx, raw_kpb_g, perm)
    _write_amn(tmp_path / f'{seedname}.amn', A)
    _write_eig(tmp_path / f'{seedname}.eig', energies_ev)

    result = subprocess.run([w90_exe, seedname], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, \
        f'wannier90.x failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}'

    w90_centers, w90_spreads, w90_total = _parse_wout(tmp_path / f'{seedname}.wout')

    U_dis, _oi_dis, _ = disentangle(M, A, bv, max_iter=400, conv_tol=1e-12)
    _U, centers, spreads, _oi, _od, _ood, _ = wannierise(
        M, U_dis, bv, max_iter=400, conv_tol=1e-12)

    our_total = float(spreads.sum())
    w90_total_bohr2 = w90_total / (BOHR_TO_ANG ** 2)
    assert abs(our_total - w90_total_bohr2) < 1e-3, \
        (f'Total spread mismatch: pyscf {our_total:.6f} Bohr^2, '
         f'wannier90 {w90_total_bohr2:.6f} Bohr^2 '
         f'({w90_total:.6f} Ang^2)')

    our_centers_ang = centers * BOHR_TO_ANG
    _assert_centers_match(w90_centers, our_centers_ang,
                          tolerance=5e-3, box_a=cell.a * BOHR_TO_ANG)


def _assert_centers_match(a, b, tolerance, box_a):
    '''Compare two sets of Wannier centers modulo permutation and lattice wrap.

    Centers are defined modulo a unit-cell translation; match each `a[i]` to
    some `b[j]` (with `j` distinct per `i`) up to the given tolerance.
    '''
    a = numpy.asarray(a)
    b = numpy.asarray(b)
    assert a.shape == b.shape
    inv = numpy.linalg.inv(box_a.T)
    used = set()
    for i, ai in enumerate(a):
        best_j, best_d = -1, numpy.inf
        for j, bj in enumerate(b):
            if j in used:
                continue
            diff_cart = ai - bj
            frac = inv @ diff_cart
            frac -= numpy.round(frac)
            d = numpy.linalg.norm(box_a.T @ frac)
            if d < best_d:
                best_d = d
                best_j = j
        assert best_d < tolerance, \
            (f'No match for w90 center {ai} (best residual {best_d:.3e}, '
             f'tolerance {tolerance}, pyscf centers: {b})')
        used.add(best_j)
