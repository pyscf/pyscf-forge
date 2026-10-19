#!/usr/bin/env python
#
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

'''
Matrix elements between arbitrary excitations of nonorthogonal reference
determinants with the extended nonorthogonal Wick theorem.

A Determinant is an excitation of one reference: occupied orbitals of the
reference given in ``*_holes`` are replaced by virtual orbitals in
``*_particles`` (indices of the reference's MO coefficients).  The
determinant is the Slater determinant of its occupied MOs in ascending MO
index order, alpha orbitals before beta orbitals, which is the convention of
pyscf.fci.

Once the intermediates of every ordered reference pair are built, the cost of
a matrix element depends only on the excitation ranks and the numbers of
zero-overlap orbital pairs, not on the basis size.

References:
    H. G. A. Burton, J. Chem. Phys. 154, 144109 (2021)
    H. G. A. Burton, J. Chem. Phys. 157, 204109 (2022)
'''

import numpy
from pyscf import lib
from pyscf import scf
from pyscf.lib import logger
from pyscf.nonorthogonal import _wicks

__all__ = ['Determinant', 'WickEngine']


class Determinant:
    '''Excitation of the reference determinant ``parent``.

    Args:
        parent : int
            Index of the reference in WickEngine.references.
        alpha_holes, alpha_particles, beta_holes, beta_particles : lists of int
            MO indices of the parent reference.  Each spin must have as many
            holes as particles.  The order within a list does not matter.
    '''
    def __init__(self, parent, alpha_holes=(), alpha_particles=(),
                 beta_holes=(), beta_particles=()):
        self.parent = int(parent)
        self.alpha_holes = tuple(sorted(int(i) for i in alpha_holes))
        self.alpha_particles = tuple(sorted(int(i) for i in alpha_particles))
        self.beta_holes = tuple(sorted(int(i) for i in beta_holes))
        self.beta_particles = tuple(sorted(int(i) for i in beta_particles))
        if (len(self.alpha_holes) != len(self.alpha_particles) or
                len(self.beta_holes) != len(self.beta_particles)):
            raise ValueError('Each spin needs as many holes as particles')
        if (len(set(self.alpha_holes)) != len(self.alpha_holes) or
                len(set(self.alpha_particles)) != len(self.alpha_particles) or
                len(set(self.beta_holes)) != len(self.beta_holes) or
                len(set(self.beta_particles)) != len(self.beta_particles)):
            raise ValueError('Repeated orbital in an excitation')

    @property
    def rank(self):
        return len(self.alpha_holes), len(self.beta_holes)

    def __repr__(self):
        return ('Determinant(%d, alpha %s->%s, beta %s->%s)' %
                (self.parent, self.alpha_holes, self.alpha_particles,
                 self.beta_holes, self.beta_particles))


def _reference_orbitals(ref):
    '''(ca, cb, occa, occb) of an SCF object or a (mo_coeff, mo_occ) pair'''
    if isinstance(ref, scf.hf.SCF):
        mo_coeff, mo_occ = ref.mo_coeff, ref.mo_occ
    else:
        mo_coeff, mo_occ = ref
    mo_coeff = numpy.asarray(mo_coeff)
    mo_occ = numpy.asarray(mo_occ)
    if mo_coeff.ndim == 3:
        ca, cb = mo_coeff
        occa = mo_occ[0] > 0
        occb = mo_occ[1] > 0
    else:
        ca = cb = mo_coeff
        occa = mo_occ > 0
        occb = mo_occ > 1
    return ca, cb, occa, occb


def _parity(seq):
    '''Parity of the permutation sorting seq: +1 or -1'''
    seq = list(seq)
    odd = 0
    for i in range(len(seq)):
        for j in range(i+1, len(seq)):
            if seq[i] > seq[j]:
                odd ^= 1
    return -1 if odd else 1


def _excitation_phase(nocc, holes, parts):
    '''Fermionic sign of removing holes (descending) and adding particles
    (ascending) on the occupied-first orbital string {0..nocc-1}.'''
    occ = set(range(nocc))
    odd = 0
    for i in sorted(holes, reverse=True):
        odd ^= sum(1 for k in occ if k < i) & 1
        occ.discard(i)
    for a in sorted(parts):
        odd ^= sum(1 for k in occ if k < a) & 1
        occ.add(a)
    return (-1 if odd else 1), occ


class WickEngine(lib.StreamObject):
    '''Extended nonorthogonal Wick theorem matrix elements.

    Attributes:
        tol : float
            Singular values of the occupied orbital overlap below tol are
            treated as zero-overlap orbital pairs.
        with_nuc : bool
            Whether hamiltonian elements include E_nuc * S.

    Examples:

    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.5', basis='cc-pvdz')
    >>> mf0 = scf.RHF(mol).run()
    >>> mf1 = scf.UHF(mol).run()
    >>> wicks = nonorthogonal.WickEngine(mol, [mf0, mf1])
    >>> d0 = nonorthogonal.Determinant(0)
    >>> d1 = nonorthogonal.Determinant(1, alpha_holes=[0], alpha_particles=[2])
    >>> h, s = wicks.matrix_element(d0, d1)
    >>> H, S = wicks.build_matrices([d0, d1])
    '''

    tol = 1e-10
    with_nuc = True

    _keys = {'mol', 'references', 'tol', 'with_nuc', 'nmo', 'nelec'}

    def __init__(self, mol, references, tol=None, hcore=None, eri=None, ovlp=None):
        self.mol = mol
        self.verbose = mol.verbose
        self.stdout = mol.stdout
        self.references = list(references)
        if tol is not None:
            self.tol = tol
        self._hcore = hcore
        self._eri = eri
        self._ovlp = ovlp
        self._env = None
        self._refs = None

        orbs = [_reference_orbitals(r) for r in self.references]
        if not orbs:
            raise ValueError('At least one reference is required')
        nmo = orbs[0][0].shape[1]
        nelec = (int(orbs[0][2].sum()), int(orbs[0][3].sum()))
        for ca, cb, occa, occb in orbs:
            if ca.shape[1] != nmo or cb.shape[1] != nmo:
                raise ValueError('References must have the same number of MOs')
            if (occa.sum(), occb.sum()) != nelec:
                raise ValueError('References must have the same (nalpha, nbeta)')
        self.nmo = nmo
        self.nelec = nelec
        self._orbs = orbs

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('******** %s ********', self.__class__)
        log.info('number of references = %d', len(self.references))
        log.info('nmo = %d, nelec = %s', self.nmo, self.nelec)
        log.info('tol = %g', self.tol)
        return self

    def build(self):
        '''Construct the intermediates of every ordered reference pair'''
        log = logger.new_logger(self)
        cput0 = (logger.process_clock(), logger.perf_counter())
        mol = self.mol
        s = self._ovlp if self._ovlp is not None else mol.intor_symmetric('int1e_ovlp')
        h = self._hcore if self._hcore is not None else scf.hf.get_hcore(mol)
        eri = self._eri
        if eri is None:
            eri = mol.intor('int2e', aosym='s1')
        nao = s.shape[0]
        eri = numpy.asarray(eri).reshape(nao, nao, nao, nao)

        refs = []
        self._occpos = []
        dtype = numpy.double
        for ca, cb, occa, occb in self._orbs:
            # occupied orbitals first, keeping the relative MO order
            orda = numpy.argsort(~occa, kind='stable')
            ordb = numpy.argsort(~occb, kind='stable')
            posa = numpy.argsort(orda)
            posb = numpy.argsort(ordb)
            refs.append((ca[:, orda], cb[:, ordb], int(occa.sum()), int(occb.sum())))
            self._occpos.append((orda, ordb, posa, posb, occa, occb))
            if numpy.iscomplexobj(ca) or numpy.iscomplexobj(cb):
                dtype = numpy.complex128
        refs = [(numpy.asarray(ca, dtype=dtype), numpy.asarray(cb, dtype=dtype), na, nb)
                for ca, cb, na, nb in refs]
        self._env = _wicks.build(s, h, eri, refs, self.tol, dtype)
        log.timer('Wick intermediates', *cput0)
        pairinfo = self._env[0]
        nref = len(self.references)
        log.debug('zero-overlap pairs (m_a, m_b):\n%s',
                  pairinfo[:, :2].reshape(nref, nref, 2))
        log.debug('intermediate slab: %.1f MB', self._env[1].nbytes / 1e6)
        return self

    def _records(self, dets):
        '''Pack determinants into the int32 records of nowick.h'''
        if self._env is None:
            self.build()
        dets = list(dets)
        kmax = max([1] + [max(d.rank) for d in dets])
        rec = numpy.zeros((len(dets), _wicks.DHEAD + 4 * kmax), dtype=numpy.int32)
        for k, d in enumerate(dets):
            if not 0 <= d.parent < len(self.references):
                raise ValueError('%s: no such reference' % d)
            orda, ordb, posa, posb, occa, occb = self._occpos[d.parent]
            sign = 1
            osign = 1
            rec[k, 0] = d.parent
            rec[k, 1:3] = d.rank
            for spin, (holes, parts, order, pos, occ) in enumerate(
                    ((d.alpha_holes, d.alpha_particles, orda, posa, occa),
                     (d.beta_holes, d.beta_particles, ordb, posb, occb))):
                if not all(occ[i] for i in holes) or any(occ[a] for a in parts):
                    raise ValueError('%s: holes must be occupied and particles '
                                     'virtual in the parent' % d)
                nocc = int(occ.sum())
                hp = sorted(pos[i] for i in holes)
                pp = sorted(pos[a] for a in parts)
                phase, occset = _excitation_phase(nocc, hp, pp)
                # sign between occupied-first and original MO order
                parity = _parity(order[p] for p in sorted(occset))
                sign *= phase * parity
                osign *= parity
                r = len(hp)
                base = _wicks.DHEAD + 2 * kmax * spin
                rec[k, base:base+r] = hp
                rec[k, base+kmax:base+kmax+r] = pp
            rec[k, 3] = sign
            rec[k, 4] = osign
        return rec, kmax

    @property
    def energy_nuc(self):
        return self.mol.energy_nuc() if self.with_nuc else 0.

    def overlap(self, bra, ket):
        '''<bra|ket>'''
        rec, kmax = self._records([bra, ket])
        return _wicks.overlap_pairs(self._env, rec, kmax, [[0, 1]])[0]

    def matrix_element(self, bra, ket):
        '''(<bra|H|ket>, <bra|ket>)'''
        rec, kmax = self._records([bra, ket])
        h, s = _wicks.hamiltonian_pairs(self._env, rec, kmax, [[0, 1]], self.energy_nuc)
        return h[0], s[0]

    def matrix_elements(self, dets, pairs, general=False):
        '''Hamiltonian and overlap elements for (bra, ket) index pairs of dets'''
        rec, kmax = self._records(dets)
        return _wicks.hamiltonian_pairs(self._env, rec, kmax, pairs, self.energy_nuc,
                                        general=general)

    def overlaps(self, dets, pairs):
        '''Overlaps for (bra, ket) index pairs of dets'''
        rec, kmax = self._records(dets)
        return _wicks.overlap_pairs(self._env, rec, kmax, pairs)

    def build_matrices(self, dets):
        '''Dense Hamiltonian and overlap matrices over dets'''
        rec, kmax = self._records(dets)
        order = numpy.argsort(rec[:, 0], kind='stable')
        h, s = _wicks.hamiltonian_matrix(self._env, rec[order], kmax, self.energy_nuc)
        inv = numpy.argsort(order)
        return h[inv[:, None], inv], s[inv[:, None], inv]
