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
Nonorthogonal configuration interaction (NOCI) over excitations of several
reference determinants.

The NOCI space holds every reference and its excitations up to a given
level.  Matrix elements between excitations of the same reference use the
Slater-Condon rules, and those between different references the extended
nonorthogonal Wick theorem (see pyscf.nonorthogonal.wicks).  The dense
generalised eigenvalue problem is solved in the eigenvectors of the overlap
matrix that are not linearly dependent.
'''

import itertools
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.nonorthogonal.wicks import Determinant, WickEngine

__all__ = ['excitations', 'kernel', 'NOCI', 'NOCISD', 'NOCISDT']


def excitations(wicks, parent, level, ranks=None):
    '''Determinants of a reference excited by up to level electrons.

    Args:
        wicks : WickEngine
            Engine holding the references.
        parent : int
            Index of the reference.
        level : int
            Largest excitation rank, alpha plus beta.

    Kwargs:
        ranks : sequence of int
            Excitation ranks to include.  Default is 0, 1, ..., level.

    Returns:
        A list of Determinant, ordered by rank and, within a rank, from all
        alpha to all beta excitations.
    '''
    occa, occb = wicks.occupations(parent)
    oa, va = numpy.where(occa)[0], numpy.where(~occa)[0]
    ob, vb = numpy.where(occb)[0], numpy.where(~occb)[0]
    if ranks is None:
        ranks = range(level + 1)
    dets = []
    for r in ranks:
        if not 0 <= r <= level:
            raise ValueError('Excitation rank %d outside 0..%d' % (r, level))
        for ra in range(r, -1, -1):
            rb = r - ra
            for ha, pa, hb, pb in itertools.product(
                    itertools.combinations(oa, ra), itertools.combinations(va, ra),
                    itertools.combinations(ob, rb), itertools.combinations(vb, rb)):
                dets.append(Determinant(parent, ha, pa, hb, pb))
    return dets


def kernel(mynoci, dets, nroots=1, lindep=1e-10):
    '''Solve H c = E S c over dets.

    Returns:
        e : the lowest nroots energies
        c : coefficients of the determinants, one column per root
        nindep : number of linearly independent states
    '''
    log = logger.new_logger(mynoci)
    cput0 = (logger.process_clock(), logger.perf_counter())
    h, s = mynoci.wicks.build_matrices(dets)
    cput1 = log.timer('NOCI matrices', *cput0)
    e, c, seig = lib.linalg_helper.safe_eigh(h, s, lindep)
    nindep = numpy.count_nonzero(seig >= lindep)
    log.info('%d determinants, %d linearly independent', len(dets), nindep)
    log.timer('NOCI diagonalisation', *cput1)
    if nindep < nroots:
        log.warn('Only %d linearly independent states for nroots = %d',
                 nindep, nroots)
    nroots = min(nroots, nindep)
    return e[:nroots], c[:, :nroots], nindep


class NOCI(lib.StreamObject):
    '''Nonorthogonal configuration interaction over excitations of several
    reference determinants.

    Attributes:
        verbose : int
            Print level.  Default value equals to :class:`Mole.verbose`.
        level : int
            Excitation level of every reference.  0 is reference NOCI, 2 is
            NOCISD.  Default is 0.
        nroots : int
            Number of states to solve.  Default is 1.
        lindep : float
            Eigenvectors of the overlap matrix with eigenvalues below lindep
            are removed.  Default is 1e-10.
        tol : float
            Singular values of the occupied orbital overlap below tol are
            treated as zero-overlap orbital pairs.  Default is 1e-10.

    Saved results:
        e_tot : float or array
            NOCI energy, or energies when nroots > 1.
        ci : ndarray
            Coefficients of the determinants in dets, a column per root when
            nroots > 1.
        dets : list of Determinant
            The NOCI space.
        nindep : int
            Number of linearly independent states of the NOCI space.
        wicks : WickEngine
            Matrix elements between the determinants.

    Examples:

    >>> mol = gto.M(atom='H 0 0 0; H 0 0 2.0', basis='cc-pvdz')
    >>> mf0 = scf.RHF(mol).run()
    >>> mf1 = scf.UHF(mol).run(...)
    >>> mynoci = nonorthogonal.NOCISD(mol, [mf0, mf1]).run()
    >>> mynoci.e_tot
    '''

    level = 0
    nroots = 1
    lindep = 1e-10
    tol = 1e-10

    _keys = {'mol', 'references', 'level', 'nroots', 'lindep', 'tol',
             'wicks', 'dets', 'nindep', 'e_tot', 'ci'}

    def __init__(self, mol, references, level=None):
        '''
        Args:
            mol : Mole
            references : list
                SCF objects or (mo_coeff, mo_occ) pairs with the same number
                of MOs and the same numbers of alpha and beta electrons.
        '''
        self.mol = mol
        self.verbose = mol.verbose
        self.stdout = mol.stdout
        self.references = list(references)
        if level is not None:
            self.level = level
        self.wicks = None
        self.dets = None
        self.nindep = None
        self.e_tot = None
        self.ci = None

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('number of references = %d', len(self.references))
        log.info('excitation level = %d', self.level)
        log.info('nroots = %d', self.nroots)
        log.info('lindep = %g', self.lindep)
        log.info('tol = %g', self.tol)
        return self

    def build(self, h1e=None, eri=None, ovlp=None, ecore=None):
        '''Construct the Wick intermediates of every reference pair for a
        Hamiltonian in the AO basis.  See WickEngine.build for the arguments;
        those not given are taken from the references (get_hamiltonian).'''
        self.wicks = WickEngine(self.mol, self.references, tol=self.tol)
        self.wicks.verbose = self.verbose
        self.wicks.stdout = self.stdout
        self.wicks.build(h1e, eri, ovlp, ecore)
        return self

    def make_space(self):
        '''Every reference and its excitations up to self.level'''
        if self.wicks is None:
            self.build()
        dets = []
        for parent in range(len(self.references)):
            dets.extend(excitations(self.wicks, parent, self.level))
        return dets

    def kernel(self, h1e=None, eri=None, ovlp=None, ecore=None, dets=None):
        '''
        Kwargs:
            h1e, eri, ovlp, ecore :
                Hamiltonian in the AO basis: core Hamiltonian, two-electron
                integrals (ij|kl), AO overlap and constant energy.  Those not
                given are taken from the first reference that is an SCF object
                (get_hcore, get_ovlp, energy_nuc and its incore integrals), or
                computed from mol.
            dets : list of Determinant
                The NOCI space.  Default is make_space().

        Returns:
            e_tot, ci
        '''
        self.dump_flags()
        self.build(h1e, eri, ovlp, ecore)
        if dets is None:
            dets = self.make_space()
        self.dets = list(dets)
        logger.info(self, 'NOCI space of %d determinants', len(self.dets))
        e, c, self.nindep = kernel(self, self.dets, self.nroots, self.lindep)
        if self.nroots == 1:
            self.e_tot, self.ci = e[0], c[:, 0]
        else:
            self.e_tot, self.ci = e, c
        self._finalize()
        return self.e_tot, self.ci

    def _finalize(self):
        if self.nroots == 1:
            logger.note(self, 'NOCI(level=%d) E = %.15g', self.level, self.e_tot)
        else:
            for k, e in enumerate(self.e_tot):
                logger.note(self, 'NOCI(level=%d) root %d  E = %.15g',
                            self.level, k, e)
        return self


def NOCISD(mol, references):
    '''NOCI over the singles and doubles of every reference'''
    return NOCI(mol, references, level=2)


def NOCISDT(mol, references):
    '''NOCI over the singles, doubles and triples of every reference'''
    return NOCI(mol, references, level=3)
