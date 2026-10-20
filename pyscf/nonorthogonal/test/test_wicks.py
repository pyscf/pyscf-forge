#!/usr/bin/env python
# Copyright 2026 The PySCF Developers. All Rights Reserved.
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
#

import unittest
import itertools
import numpy
import scipy.linalg
from pyscf import gto, scf, ao2mo, fci
from pyscf import nonorthogonal
from pyscf.nonorthogonal import Determinant, WickEngine, _wicks


def setUpModule():
    global mol, mf, s_half, h1, eri
    mol = gto.M(atom='Be 0 0 0; H .3 .2 -1.3; H -.1 .4 1.235', basis='sto-3g',
                verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    # Lowdin-orthonormal AO basis of the brute-force reference
    s = mol.intor('int1e_ovlp')
    s_half = scipy.linalg.sqrtm(s).real
    x = numpy.linalg.inv(s_half)
    h1 = x.T.dot(mf.get_hcore()).dot(x)
    eri = ao2mo.restore(1, ao2mo.full(mol, x), mol.nao)


def tearDownModule():
    global mol, mf, s_half, h1, eri
    del mol, mf, s_half, h1, eri


def _spin_vector(c_occ):
    '''FCI string coefficients det(U[I,:]) of one spin'''
    norb = c_occ.shape[0]
    nel = c_occ.shape[1]
    strs = fci.cistring.make_strings(range(norb), nel)
    v = numpy.empty(len(strs), dtype=c_occ.dtype)
    for k, st in enumerate(strs):
        orbs = [p for p in range(norb) if (int(st) >> p) & 1]
        v[k] = numpy.linalg.det(c_occ[orbs])
    return v


def _fci_vector(refs, det):
    '''FCI vector of det in the Lowdin-orthonormal AO basis'''
    (ca, cb), (occa, occb) = refs[det.parent]
    vecs = []
    for c, occ, holes, parts in ((ca, occa, det.alpha_holes, det.alpha_particles),
                                 (cb, occb, det.beta_holes, det.beta_particles)):
        orbs = sorted((set(numpy.where(occ)[0]) - set(holes)) | set(parts))
        vecs.append(_spin_vector(s_half.dot(c[:, orbs])))
    return numpy.einsum('i,j->ij', *vecs)


def _brute_hs(refs, dets):
    '''Hamiltonian and overlap matrices by FCI-space expansion'''
    norb = mol.nao
    nelec = mol.nelec
    h2 = fci.direct_spin1.absorb_h1e(h1, eri, norb, nelec, .5)
    vecs = [_fci_vector(refs, d) for d in dets]
    hvecs = []
    for v in vecs:
        hv = fci.direct_spin1.contract_2e(h2, v.real.copy(), norb, nelec)
        if numpy.iscomplexobj(v):
            hv = hv + 1j * fci.direct_spin1.contract_2e(h2, v.imag.copy(), norb, nelec)
        hvecs.append(hv)
    nd = len(dets)
    h = numpy.empty((nd, nd), dtype=complex)
    s = numpy.empty((nd, nd), dtype=complex)
    for i, j in itertools.product(range(nd), repeat=2):
        s[i, j] = numpy.vdot(vecs[i], vecs[j])
        h[i, j] = numpy.vdot(vecs[i], hvecs[j]) + mol.energy_nuc() * s[i, j]
    return h, s


def _rotate(c, occ, scale, seed, cplx=False, mix=True):
    '''Random unitary rotation of the MOs.  mix=False rotates only within the
    occupied and within the virtual MOs of occ.'''
    rng = numpy.random.RandomState(seed)
    n = c.shape[1]
    a = rng.normal(size=(n, n))
    if cplx:
        a = a + 1j * rng.normal(size=(n, n))
    a = (a - a.conj().T) * scale
    if not mix:
        a[numpy.ix_(occ, ~occ)] = 0
        a[numpy.ix_(~occ, occ)] = 0
    return c.dot(scipy.linalg.expm(a))


def _rotate_pairs(c, nocc, sigmas, phase=0):
    '''Rotate (HOMO-j, LUMO+j) so the occupied overlap with c has singular
    values sigmas'''
    c = c.astype(complex if phase else float)
    for j, sigma in enumerate(sigmas):
        o, v = nocc - 1 - j, nocc + j
        co, cv = c[:, o].copy(), c[:, v].copy()
        theta = numpy.arccos(sigma)
        e = numpy.exp(1j * phase) if phase else 1
        c[:, o] = co * numpy.cos(theta) + cv * numpy.sin(theta) * e
        c[:, v] = cv * numpy.cos(theta) - co * numpy.sin(theta) * numpy.conj(e)
    return c


def _swap(occ, holes, parts):
    occ = occ.copy()
    occ[list(holes)] = False
    occ[list(parts)] = True
    return occ


def _dets(refs, nper, maxrank, seed):
    '''The references and random excitations of each'''
    rng = numpy.random.RandomState(seed)
    dets = []
    for p, (c, (occa, occb)) in enumerate(refs):
        dets.append(Determinant(p))
        for k in range(nper):
            ra = rng.randint(0, maxrank + 1)
            rb = rng.randint(0, maxrank + 1 - ra)
            oa, va = numpy.where(occa)[0], numpy.where(~occa)[0]
            ob, vb = numpy.where(occb)[0], numpy.where(~occb)[0]
            ra = min(ra, len(oa), len(va))
            rb = min(rb, len(ob), len(vb))
            dets.append(Determinant(p, rng.choice(oa, ra, replace=False),
                                    rng.choice(va, ra, replace=False),
                                    rng.choice(ob, rb, replace=False),
                                    rng.choice(vb, rb, replace=False)))
    return dets


def _engine(refs, **kwargs):
    return WickEngine(mol, [(numpy.asarray(c), numpy.asarray(occ).astype(float))
                            for c, occ in refs], **kwargs)


def _references(cplx=False):
    '''RHF, a generic reference, and references with m_a = 1 and
    (m_a, m_b) = (2, 1) zero-overlap pairs against the RHF determinant, with
    occupied orbitals not first in the MO order'''
    c = mf.mo_coeff.astype(complex if cplx else float)
    occ = mf.mo_occ > 0
    nocc = occ.sum()
    o1 = _swap(occ, [nocc-1], [nocc])
    o2 = _swap(occ, [nocc-2, nocc-1], [nocc, nocc+1])
    return [((c, c), (occ, occ)),
            ((_rotate(c, occ, .1, 1, cplx), _rotate(c, occ, .12, 2, cplx)), (occ, occ)),
            ((_rotate(c, o1, .3, 3, cplx, False), c), (o1, occ)),
            ((_rotate(c, o2, .3, 4, cplx, False), _rotate(c, o1, .3, 5, cplx, False)),
             (o2, o1))]


class KnownValues(unittest.TestCase):
    def check(self, refs, dets, tol=1e-9):
        wicks = _engine(refs)
        h_ref, s_ref = _brute_hs(refs, dets)
        h, s = wicks.build_matrices(dets)
        self.assertAlmostEqual(abs(h - h_ref).max(), 0, delta=tol)
        self.assertAlmostEqual(abs(s - s_ref).max(), 0, delta=tol)
        pairs = numpy.array(list(itertools.product(range(len(dets)), repeat=2)))
        h, s = wicks.matrix_elements(dets, pairs, general=True)
        self.assertAlmostEqual(abs(h - h_ref.ravel()).max(), 0, delta=tol)
        self.assertAlmostEqual(abs(s - s_ref.ravel()).max(), 0, delta=tol)
        s = wicks.overlaps(dets, pairs)
        self.assertAlmostEqual(abs(s - s_ref.ravel()).max(), 0, delta=tol)
        return wicks

    def test_real(self):
        refs = _references()
        wicks = self.check(refs, _dets(refs, 8, 3, 5))
        m = wicks._env[0][:16, :2].reshape(4, 4, 2)
        self.assertEqual(m[0, 2].tolist(), [1, 0])
        self.assertEqual(m[0, 3].tolist(), [2, 1])

    def test_unsorted_determinants(self):
        refs = _references()
        dets = _dets(refs, 4, 2, 10)
        wicks = _engine(refs)
        h, s = wicks.build_matrices(dets)
        perm = numpy.random.RandomState(11).permutation(len(dets))
        h1, s1 = wicks.build_matrices([dets[k] for k in perm])
        self.assertAlmostEqual(abs(h1 - h[perm][:, perm]).max(), 0, 12)
        self.assertAlmostEqual(abs(s1 - s[perm][:, perm]).max(), 0, 12)

    def test_complex(self):
        refs = _references(cplx=True)
        self.check(refs, _dets(refs, 6, 3, 6))

    def test_quadruples(self):
        refs = _references()[:2]
        self.check(refs, _dets(refs, 10, 4, 7))

    def test_scalar_kernels(self):
        refs = _references()
        try:
            _wicks.libwick.NOWICKset_simd(0)
            self.check(refs, _dets(refs, 6, 3, 8))
        finally:
            _wicks.libwick.NOWICKset_simd(-1)

    def test_near_zero_pairs(self):
        # Near-zero singular values are evaluated at the corners of their
        # 1/S_k terms
        c = mf.mo_coeff
        occ = mf.mo_occ > 0
        nocc = occ.sum()
        for sa, sb, phase, ncorner in (([1e-3], [], 0, 2),
                                       ([0, 1e-3], [], 0, 2),
                                       ([1e-3, 2e-3], [2e-3], 0, 8),
                                       ([1e-3], [2e-3], .7, 4)):
            ca = _rotate_pairs(c, nocc, sa, phase)
            cb = _rotate_pairs(c, nocc, sb, phase)
            c0 = c.astype(ca.dtype)
            refs = [((c0, c0), (occ, occ)), ((ca, cb), (occ, occ))]
            wicks = self.check(refs, _dets(refs, 12, 3, 9))
            self.assertEqual(wicks._env[0][1, _wicks.NCORNER], ncorner)

    def test_matrix_element(self):
        refs = _references()
        wicks = _engine(refs)
        d0 = Determinant(0)
        d1 = Determinant(2, alpha_holes=[3], alpha_particles=[6])
        h_ref, s_ref = _brute_hs(refs, [d0, d1])
        h, s = wicks.matrix_element(d0, d1)
        self.assertAlmostEqual(h, h_ref[0, 1].real, 9)
        self.assertAlmostEqual(s, s_ref[0, 1].real, 9)
        self.assertAlmostEqual(wicks.overlap(d1, d0), s_ref[1, 0].real, 9)
        # The RHF determinant reproduces the SCF energy
        self.assertAlmostEqual(wicks.matrix_element(d0, d0)[0], mf.e_tot, 9)
        wicks.with_nuc = False
        self.assertAlmostEqual(wicks.matrix_element(d0, d0)[0],
                               mf.e_tot - mol.energy_nuc(), 9)

    def test_scf_references(self):
        mf1 = scf.UHF(mol)
        dm = mf1.get_init_guess()
        dm[0][:2, :2] *= 1.2
        mf1.kernel(dm)
        wicks = WickEngine(mol, [mf, mf1])
        h, s = wicks.build_matrices([Determinant(0), Determinant(1)])
        self.assertAlmostEqual(h[1, 1], mf1.e_tot, 9)
        self.assertAlmostEqual(s[1, 1], 1, 12)
        self.assertAlmostEqual(h[0, 1], h[1, 0], 12)

    def test_invalid_determinants(self):
        refs = _references()
        wicks = _engine(refs)
        self.assertRaises(ValueError, Determinant, 0, [0], [])
        self.assertRaises(ValueError, Determinant, 0, [0, 0], [4, 5])
        self.assertRaises(ValueError, wicks.overlap, Determinant(0), Determinant(9))
        self.assertRaises(ValueError, wicks.overlap, Determinant(0),
                          Determinant(0, alpha_holes=[5], alpha_particles=[6]))
        self.assertRaises(ValueError, nonorthogonal.WickEngine, mol, [])


if __name__ == "__main__":
    print("Full Tests for nonorthogonal Wick matrix elements")
    unittest.main()
