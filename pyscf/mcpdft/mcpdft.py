#!/usr/bin/env python
# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
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
import numpy as np
from copy import deepcopy
from pyscf import ao2mo, fci, mcscf, lib, __config__
from pyscf.lib import logger
from pyscf.dft import gen_grid
from pyscf.mcscf import mc1step
from pyscf.mcscf.addons import StateAverageMCSCFSolver, state_average_mix
from pyscf.mcscf.addons import state_average_mix_, StateAverageMixFCISolver
from pyscf.mcscf.df import _DFCASSCF, _DFCAS
from pyscf.mcpdft import pdft_veff, pdft_feff
from pyscf.mcpdft.otfnal import transfnal, get_transfnal
from pyscf.mcpdft import _dms

def energy_tot(mc, mo_coeff=None, ci=None, ot=None, state=0, verbose=None):
    '''Calculate MC-PDFT total energy

    Args:
        mc : an instance of CASSCF or CASCI class
            Note: this function does not currently run the CASSCF or
            CASCI calculation itself prior to calculating the MC-PDFT
            energy. Call mc.kernel () before passing to this function!

    Kwargs:
        mo_coeff : ndarray of shape (nao, nmo)
            Molecular orbital coefficients
        ci : ndarray or list of length (nroots)
            CI vector or vectors.
        ot : an instance of on-top functional class - see otfnal.py
        state : int
            If mc describes a state-averaged calculation, select the
            state (0-indexed).
        verbose : int
            Verbosity of logger output; defaults to mc.verbose

    Returns:
        e_tot : float
            Total MC-PDFT energy including nuclear repulsion energy
        E_ot : float
            On-top (cf. exchange-correlation) energy
    '''
    if ot is None: ot = mc.otfnal
    ot.reset(mol=mc.mol)  # scanner mode safety
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    if verbose is None: verbose = mc.verbose
    t0 = (logger.process_clock(), logger.perf_counter())

    # Allow MC-PDFT to be subclassed, and also allow this function to be
    # called without mc being an instance of MC-PDFT class

    casdm1s = mc.make_one_casdm1s(ci, state=state)
    casdm2 = mc.make_one_casdm2(ci, state=state)
    t0 = logger.timer(ot, 'rdms', *t0)

    if callable(getattr(mc, 'energy_mcwfn', None)):
        e_mcwfn = mc.energy_mcwfn(ot=ot, mo_coeff=mo_coeff, casdm1s=casdm1s,
                                  casdm2=casdm2, verbose=verbose)
    else:
        e_mcwfn = energy_mcwfn(ot=ot, mo_coeff=mo_coeff, casdm1s=casdm1s,
                               casdm2=casdm2, verbose=verbose)
    t0 = logger.timer(ot, 'MC wfn energy', *t0)

    if callable(getattr(mc, 'energy_dft', None)):
        e_dft = mc.energy_dft(ot=ot, mo_coeff=mo_coeff, casdm1s=casdm1s,
                              casdm2=casdm2)
    else:
        e_dft = energy_dft(ot=ot, mo_coeff=mo_coeff, casdm1s=casdm1s,
                           casdm2=casdm2)
    t0 = logger.timer(ot, 'E_ot', *t0)

    e_tot = e_mcwfn + e_dft
    return e_tot, e_dft


# Consistency with PySCF convention
kernel = energy_tot  # backwards compatibility


def energy_elec(mc, *args, **kwargs):
    e_tot, E_ot = energy_tot(mc, *args, **kwargs)
    e_elec = e_tot - mc._scf.energy_nuc()
    return e_elec, E_ot


def energy_mcwfn(mc, mo_coeff=None, ci=None, ot=None, state=0, casdm1s=None,
                 casdm2=None, verbose=None):
    '''Compute the parts of the MC-PDFT energy arising from the wave
    function

    Args:
        mc : an instance of CASSCF or CASCI class
            Note: this function does not currently run the CASSCF or
            CASCI calculation itself prior to calculating the MC-PDFT
            energy. Call mc.kernel () before passing to this function!

    Kwargs:
        mo_coeff : ndarray of shape (nao, nmo)
            contains molecular orbital coefficients
        ci : list or ndarray
            contains ci vectors
        ot : an instance of on-top functional class - see otfnal.py
        state : int
            If mc describes a state-averaged calculation, select the
            state (0-indexed).
        casdm1s : ndarray or compatible of shape (2,ncas,ncas)
            Contains spin-separated active-space 1RDM
        casdm2 : ndarray or compatible of shape [ncas,]*4
            Contains spin-summed active-space 2RDM

    Returns:
        e_mcwfn : float
            Energy from the multiconfigurational wave function:
            nuclear repulsion + 1e + coulomb
    '''

    if ot is None: ot = mc.otfnal
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    if verbose is None: verbose = mc.verbose
    if casdm1s is None: casdm1s = mc.make_one_casdm1s(ci=ci, state=state)
    if casdm2 is None: casdm2 = mc.make_one_casdm2(ci=ci, state=state)
    log = logger.new_logger(mc, verbose=verbose)
    nelecas = mc.nelecas
    dm1s = _dms.casdm1s_to_dm1s(mc, casdm1s, mo_coeff=mo_coeff)
    cascm2 = _dms.dm2_cumulant(casdm2, casdm1s)

    spin = abs(nelecas[0] - nelecas[1])
    omega, alpha, hyb = ot._numint.rsh_and_hybrid_coeff(ot.otxc, spin=spin)
    if abs(omega) > 1e-11:
        raise NotImplementedError("range-separated on-top functionals")
    hyb_x, hyb_c = hyb

    Vnn = mc._scf.energy_nuc()
    h = mc._scf.get_hcore()
    dm1 = dm1s[0] + dm1s[1]
    if log.verbose >= logger.DEBUG or abs(hyb_x) > 1e-10:
        vj, vk = mc._scf.get_jk(dm=dm1s)
        vj = vj[0] + vj[1]
    else:
        vj = mc._scf.get_j(dm=dm1)
    Te_Vne = np.tensordot(h, dm1)
    # (vj_a + vj_b) * (dm_a + dm_b)
    E_j = np.tensordot(vj, dm1) / 2
    # (vk_a * dm_a) + (vk_b * dm_b) Mind the difference!
    if log.verbose >= logger.DEBUG or abs(hyb_x) > 1e-10:
        E_x = -(np.tensordot(vk[0], dm1s[0]) + np.tensordot(vk[1], dm1s[1]))
        E_x /= 2.0
    else:
        E_x = 0
    log.debug('CAS energy decomposition:')
    log.debug('Vnn = %s', Vnn)
    log.debug('Te + Vne = %s', Te_Vne)
    log.debug('E_j = %s', E_j)
    log.debug('E_x = %s', E_x)
    E_c = 0
    if log.verbose >= logger.DEBUG or abs(hyb_c) > 1e-10:
        # g_pqrs * l_pqrs / 2
        # if log.verbose >= logger.DEBUG:
        aeri = ao2mo.restore(1, mc.get_h2eff(mo_coeff), mc.ncas)
        E_c = np.tensordot(aeri, cascm2, axes=4) / 2
        log.debug('E_c = %s', E_c)
    if abs(hyb_x) > 1e-10 or abs(hyb_c) > 1e-10:
        log.debug(('Adding %s * %s CAS exchange, %s * %s CAS correlation to '
                   'E_ot'), hyb_x, E_x, hyb_c, E_c)
    e_mcwfn = Vnn + Te_Vne + E_j + (hyb_x * E_x) + (hyb_c * E_c)
    return e_mcwfn


def energy_dft(mc, mo_coeff=None, ci=None, ot=None, state=0, casdm1s=None,
               casdm2=None, max_memory=None, hermi=1):
    ''' Wrap to ot.energy_ot for subclassing. '''
    if ot is None: ot = mc.otfnal
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    if casdm1s is None: casdm1s = mc.make_one_casdm1s(ci, state=state)
    if casdm2 is None: casdm2 = mc.make_one_casdm2(ci, state=state)
    if max_memory is None: max_memory = mc.max_memory
    return ot.energy_ot(casdm1s, casdm2, mo_coeff, mc.ncore,
                        max_memory=max_memory, hermi=hermi)


def get_energy_decomposition(mc, mo_coeff=None, ci=None, ot=None, otxc=None,
                             grids_level=None, grids_attr=None,
                             split_x_c=None, verbose=None):
    '''Compute a decomposition of the MC-PDFT energy into nuclear
    potential (E0), one-electron (E1), Coulomb (E2c), exchange (EOTx),
    correlation (EOTc) terms, and additionally the nonclassical part
    (E2nc) of the MC-SCF energy:

    E(MC-SCF) = E0 + E1 + E2c + Enc
    E(MC-PDFT) = E0 + E1 + E2c + EOTx + EOTc

    Only compatible with pure translated or fully-translated fnals. If
    mc.fcisolver.nroots > 1, lists are returned for everything except
    the nuclear potential energy.

    Args:
        mc : an instance of CASSCF or CASCI class

    Kwargs:
        mo_coeff : ndarray
            Contains MO coefficients
        ci : ndarray or list of length nroots
            Contains CI vectors
        ot : an instance of (translated) on-top density fnal class
        otxc : string
            identity of translated functional; overrides ot
        grids_level : integer
            level preset for DFT quadrature grids
        grids_attr : dictionary
            general attributes for DFT quadrature grids
        split_x_c : logical
            whether to split the exchange and correlation parts of the
            ot functional into two separate contributions


    Returns:
        e_nuc : float
            E0 = sum_A>B ZA*ZB/rAB
        e_1e : float or list of length nroots
            E1 = <T+sum_A ZA/rA>
        e_coul : float or list of length nroots
            E2c = 1/2 int rho(1)rho(2)/r12 d1d2
        e_otxc : float or list of length nroots
            EOTxc = translated functional energy
            if split_x_c == True, this is instead
            EOTx = exchange part of translated functional energy
        e_otc : float or list of length nroots
            only returned if split_x_c == True
            EOTc = correlation part of translated functional
        e_ncwfn : float or list of length nroots
            E2ncc = <H> - E0 - E1 - E2c
    '''
    if verbose is None: verbose = mc.verbose
    log = logger.new_logger(mc, verbose)
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    if grids_attr is None: grids_attr = {}
    if grids_level is not None: grids_attr['level'] = grids_level
    if len(grids_attr) or (otxc is not None):
        old_ot = ot if (ot is not None) else mc.otfnal
        old_grids = old_ot.grids
        # TODO: general compatibility with arbitrary (non-translated) fnals
        if otxc is None: otxc = old_ot.otxc
        new_ot = get_transfnal(mc.mol, otxc)
        new_ot.grids.__dict__.update(old_grids.__dict__)
        new_ot.grids.__dict__.update(**grids_attr)
        ot = new_ot
    if ot is None: ot = mc.otfnal
    if split_x_c is None:
        split_x_c = True
        log.warn(
            'Currently, split_x_c in get_energy_decomposition defaults to '
            'True.\nThis default will change to False in the near future.'
        )

    hyb_x, hyb_c = ot._numint.hybrid_coeff(ot.otxc)
    if hyb_x > 1e-10 or hyb_c > 1e-10:
        raise NotImplementedError("Decomp for hybrid PDFT fnals")
    if not isinstance(ot, transfnal):
        raise NotImplementedError("Decomp for non-translated PDFT fnals")

    if split_x_c:
        ot = list(ot.split_x_c())
    else:
        ot = [ot, ]
    e_nuc = mc._scf.energy_nuc()
    h = mc.get_hcore()
    nroots = getattr(mc.fcisolver, 'nroots', 1)
    if nroots > 1:
        e_1e = []
        e_coul = []
        e_otxc = []
        e_ncwfn = []
        nelec_root = [mc.nelecas, ] * nroots
        if isinstance(mc.fcisolver, StateAverageMixFCISolver):
            nelec_root = []
            for s in mc.fcisolver.fcisolvers:
                ne_root_s = mc.fcisolver._get_nelec(s, mc.nelecas)
                nelec_root.extend([ne_root_s, ] * s.nroots)
        for ci_i, nelec in zip(ci, nelec_root):
            row = _get_e_decomp(mc, ot, mo_coeff, ci_i, e_nuc, h, nelec)
            e_1e.append(row[0])
            e_coul.append(row[1])
            e_otxc.append(row[2])
            e_ncwfn.append(row[3])
        e_otxc = [[e[i] for e in e_otxc] for i in range(len(e_otxc[0]))]
    else:
        e_1e, e_coul, e_otxc, e_ncwfn = _get_e_decomp(
            mc, ot, mo_coeff, ci, e_nuc, h, mc.nelecas
        )
    if split_x_c:
        e_otx, e_otc = e_otxc
        return e_nuc, e_1e, e_coul, e_otx, e_otc, e_ncwfn
    else:
        return e_nuc, e_1e, e_coul, e_otxc[0], e_ncwfn


def _get_e_decomp(mc, ot, mo_coeff, ci, e_nuc, h, nelecas):
    ncore, ncas = mc.ncore, mc.ncas
    _rdms = mcscf.CASCI(mc._scf, ncas, nelecas)
    _rdms.fcisolver = fci.solver(mc._scf.mol, singlet=False, symm=False)
    _rdms.mo_coeff = mo_coeff
    _rdms.ci = ci
    _casdms = _rdms.fcisolver
    h1, h0 = _rdms.h1e_for_cas()
    h2 = ao2mo.restore(1, _rdms.get_h2eff(), ncas)
    dm1s = np.stack(_rdms.make_rdm1s(), axis=0)
    dm1 = dm1s[0] + dm1s[1]
    j = _rdms._scf.get_j(dm=dm1)
    e_1e = np.dot(h.ravel(), dm1.ravel())
    e_coul = np.dot(j.ravel(), dm1.ravel()) / 2
    adm1, adm2 = _casdms.make_rdm12(_rdms.ci, ncas, nelecas)
    e_mcscf = h0 + np.dot(h1.ravel(), adm1.ravel()) + (
            np.dot(h2.ravel(), adm2.ravel()) * 0.5)
    adm1s = np.stack(_casdms.make_rdm1s(ci, ncas, nelecas), axis=0)
    adm2 = _casdms.make_rdm12(_rdms.ci, ncas, nelecas)[1]
    e_otxc = [fnal.energy_ot(adm1s, adm2, mo_coeff, ncore,
                             max_memory=mc.max_memory)
              for fnal in ot]
    e_ncwfn = e_mcscf - e_nuc - e_1e - e_coul
    return e_1e, e_coul, e_otxc, e_ncwfn


class _mcscf_env:
    '''Prevent MC-SCF step of MC-PDFT from overwriting redefined
    quantities e_states and e_tot '''

    def __init__(self, mc):
        self.mc = mc
        self.e_tot = deepcopy(self.mc.e_tot)
        self.e_states = deepcopy(getattr(self.mc, 'e_states', None))

    def __enter__(self):
        self.mc._in_mcscf_env = True

    def __exit__(self, type, value, traceback):
        self.mc.e_tot = self.e_tot
        if getattr(self.mc, 'e_states', None) is not None:
            self.mc.e_mcscf = np.array(self.mc.e_states)
        if self.e_states is not None:
            try:
                self.mc.e_states = self.e_states
            except AttributeError as e:
                self.mc.fcisolver.e_states = self.e_states
                assert (self.mc.e_states is self.e_states), str(e)
            # TODO: redesign this. MC-SCF e_states is stapled to
            # fcisolver.e_states, but I don't want MS-PDFT to be
            # because that makes no sense
        self.mc._in_mcscf_env = False


class _PDFT:
    # Metaclass parent; unusable on its own
    '''MC-PDFT child class. All total energy quantities (e_tot,
    e_states) are MC-PDFT total energies:

    E = Vnn + h_pq*D_pq + g_pqrs*D_pq*D_rs/2 + E_ot
      = T + Vnn + Vne + E_Coul + E_ot
      = E_classical + E_ot

    Extra attributes:
        otfnal : instance of :class:`otfnal`
            The on-top energy functional class
        otxc : string
            Synonym for `otfnal.otxc`
        grids : instance of :class:`Grids`
            Synonym for `otfnal.grids`

    Additional saved results:
        e_mcscf : float or list of length nroots
            MC-SCF total energy or energies:
            Vnn + h_pq*D_pq + g_pqrs*d_pqrs/2
        e_ot : float or list of length nroots
            On-top nonclassical term in the MC-PDFT energy
    '''

    _mc_class = None

    def __init__(self, scf, ncas, nelecas, my_ot=None, grids_level=None,
                 grids_attr=None, **kwargs):
        # Keep the same initialization pattern for backwards-compatibility.
        # Use a separate intializer for the ot functional
        if grids_attr is None: grids_attr = {}
        _mc_class_no_df = self._mc_class
        if issubclass (_mc_class_no_df, _DFCAS):
            _mc_class_no_df = lib.drop_class (_mc_class_no_df, _DFCAS)
        _mc_class_no_df.__init__(self, scf, ncas, nelecas)
        if issubclass (self._mc_class, _DFCAS):
            self._mc_class.__init__(self, self, scf.with_df)
        keys = set(('e_ot', 'e_mcscf', 'get_pdft_veff', 'get_pdft_feff', 'e_states', 'otfnal',
                    'grids', 'max_cycle_fp', 'conv_tol_ci_fp', 'mcscf_kernel'))
        self.max_cycle_fp = getattr(__config__, 'mcscf_mcpdft_max_cycle_fp',
                                    50)
        self.conv_tol_ci_fp = getattr(__config__,
                                      'mcscf_mcpdft_conv_tol_ci_fp', 1e-8)
        self.mcscf_kernel = self._mc_class.kernel
        self._in_mcscf_env = False
        self._keys = set(self.__dict__.keys()).union(keys)
        if grids_level is not None:
            grids_attr['level'] = grids_level
        if my_ot is not None:
            self._init_ot_grids(my_ot, grids_attr=grids_attr)

    def _init_ot_grids(self, my_ot, grids_attr=None):
        if grids_attr is None: grids_attr = {}
        old_grids = getattr(self, 'grids', None)
        if isinstance(my_ot, (str, np.string_)):
            self.otfnal = get_transfnal(self.mol, my_ot)
        else:
            self.otfnal = my_ot
        if isinstance(old_grids, gen_grid.Grids):
            self.otfnal.grids = old_grids
        # self.grids = self.otfnal.grids
        self.grids.__dict__.update(grids_attr)
        for key in grids_attr:
            assert (getattr(self.grids, key, None) == getattr(
                self.otfnal.grids, key, None))
        # Make sure verbose and stdout don't accidentally change
        # (i.e., in scanner mode)
        self.otfnal.verbose = self.verbose
        self.otfnal.stdout = self.stdout

    @property
    def grids(self):
        return self.otfnal.grids

    @grids.setter
    def grids(self, x):
        self.otfnal.grids = x
        return self.otfnal.grids

    def optimize_mcscf_(self, mo_coeff=None, ci0=None, **kwargs):
        '''Optimize the MC-SCF wave function underlying an MC-PDFT calculation.
        Has the same calling signature as the parent kernel method. '''
        with _mcscf_env(self):
            self.e_mcscf, self.e_cas, self.ci, self.mo_coeff, self.mo_energy = \
                self._mc_class.kernel(self, mo_coeff, ci0=ci0, **kwargs)
        return self.e_mcscf, self.e_cas, self.ci, self.mo_coeff, self.mo_energy

    def compute_pdft_energy_(self, mo_coeff=None, ci=None, ot=None, otxc=None,
                             grids_level=None, grids_attr=None, **kwargs):
        '''Compute the MC-PDFT energy(ies) (and update stored data)
        with the MC-SCF wave function fixed. '''
        if mo_coeff is not None: self.mo_coeff = mo_coeff
        if ci is not None: self.ci = ci
        if ot is not None: self.otfnal = ot
        if otxc is not None: self.otxc = otxc
        if grids_attr is None: grids_attr = {}
        if grids_level is not None: grids_attr['level'] = grids_level
        if len(grids_attr): self.grids.__dict__.update(**grids_attr)
        nroots = getattr(self.fcisolver, 'nroots', 1)
        epdft = [self.energy_tot(mo_coeff=self.mo_coeff, ci=self.ci, state=ix,
                                 logger_tag='MC-PDFT state {}'.format(ix))
                 for ix in range(nroots)]
        self.e_ot = [e_ot for e_tot, e_ot in epdft]
        if isinstance(self, StateAverageMCSCFSolver):
            e_states = [e_tot for e_tot, e_ot in epdft]
            try:
                self.e_states = e_states
            except AttributeError as e:
                self.fcisolver.e_states = e_states
                assert (self.e_states is e_states), str(e)
            # TODO: redesign this. MC-SCF e_states is stapled to
            # fcisolver.e_states, but I don't want MS-PDFT to be
            # because that makes no sense
            self.e_tot = np.dot(e_states, self.weights)
            e_states = self.e_states
        elif nroots > 1:  # nroots>1 CASCI
            self.e_tot = [e_tot for e_tot, e_ot in epdft]
            e_states = self.e_tot
        else:  # nroots==1 not StateAverage class
            self.e_tot, self.e_ot = epdft[0]
            e_states = [self.e_tot]
        return self.e_tot, self.e_ot, e_states

    def kernel(self, mo_coeff=None, ci0=None, otxc=None, grids_attr=None,
               grids_level=None, **kwargs):
        self.optimize_mcscf_(mo_coeff=mo_coeff, ci0=ci0, **kwargs)
        self.compute_pdft_energy_(otxc=otxc, grids_attr=grids_attr,
                                  grids_level=grids_level, **kwargs)
        # TODO: edit StateAverageMCSCF._finalize in pyscf.mcscf.addons
        # to use the proper name of the class rather than "CASCI", so
        # that I can meaningfully play with "finalize" here
        return (self.e_tot, self.e_ot, self.e_mcscf, self.e_cas, self.ci,
                self.mo_coeff, self.mo_energy)

    def dump_flags(self, verbose=None):
        self._mc_class.dump_flags(self, verbose=verbose)
        log = logger.new_logger(self, verbose)
        log.info('on-top pair density exchange-correlation functional: %s',
                 self.otfnal.otxc)

    def get_pdft_veff(self, mo=None, ci=None, state=0, casdm1s=None,
                      casdm2=None, incl_coul=False, paaa_only=False, aaaa_only=False,
                      jk_pc=False, drop_mcwfn=False, incl_energy=False):
        '''Get the 1- and 2-body MC-PDFT effective potentials for a set
        of mos and ci vectors

        Kwargs:
            mo : ndarray of shape (nao,nmo)
                A full set of molecular orbital coefficients. Taken from
                self if not provided
            ci : list or ndarray
                CI vectors. Taken from self if not provided
            state : integer
                Indexes a specific state in state-averaged calculations.
            casdm1s : ndarray of shape (2,ncas,ncas)
                Spin-separated 1-RDM in the active space
            casdm2 : ndarray of shape (ncas,ncas,ncas,ncas)
                Spin-summed 2-RDM in the active space
            incl_coul : logical
                If true, includes the Coulomb repulsion energy in the
                1-body effective potential.
            paaa_only : logical
                If true, only the paaa 2-body effective potential
                elements are evaluated; the rest of ppaa are filled with
                zeros.
            aaaa_only : logical
                If true, only the aaaa 2-body effective potential
                elements are evaluated; the rest of ppaa are filled with
                zeros.
            jk_pc : logical
                If true, calculate the ppii=pipi 2-body effective
                potential in veff2.j_pc and veff2.k_pc. Otherwise these
                arrays are filled with zeroes.
            drop_mcwfn : logical
                If true, drops the normal CASSCF wave function contribution
                (ie the ``Hartree exchange-correlation'') from the response
            incl_energy : logical
                If true, includes the on-top potential energy as a 3rd return argument

        Returns:
            veff1 : ndarray of shape (nao, nao)
                1-body effective potential in the AO basis
                May include classical Coulomb potential term (see
                incl_coul kwarg)
            veff2 : pyscf.mcscf.mc_ao2mo._ERIS instance
                Relevant 2-body effective potential in the MO basis
            E_ot : float
                On-top energy. Only included if incl_energy is true.
        '''
        t0 = (logger.process_clock(), logger.perf_counter())
        if mo is None: mo = self.mo_coeff
        if ci is None: ci = self.ci
        if casdm1s is None: casdm1s = self.make_one_casdm1s(ci, state=state)
        if casdm2 is None: casdm2 = self.make_one_casdm2(ci, state=state)
        ncore, ncas = self.ncore, self.ncas
        dm1s = _dms.casdm1s_to_dm1s(self, casdm1s, mo_coeff=mo,
                                    ncore=ncore, ncas=ncas)
        cascm2 = _dms.dm2_cumulant(casdm2, casdm1s)

        E_ot, pdft_veff1, pdft_veff2 = pdft_veff.kernel(self.otfnal, dm1s,
                                                        cascm2, mo, ncore, ncas, max_memory=self.max_memory,
                                                        paaa_only=paaa_only, aaaa_only=aaaa_only, jk_pc=jk_pc,
                                                        drop_mcwfn=drop_mcwfn)

        if incl_coul:
            pdft_veff1 += self._scf.get_j(self.mol, dm1s[0] + dm1s[1])

        logger.timer(self, 'get_pdft_veff', *t0)

        if incl_energy:
            return pdft_veff1, pdft_veff2, E_ot

        else:
            return pdft_veff1, pdft_veff2

    def get_pdft_feff(self, mo=None, ci=None, state=0, casdm1s=None,
                      casdm2=None, c_dm1s=None, c_cascm2=None,
                      paaa_only=False, aaaa_only=False, jk_pc=False, incl_coul=False, delta=False):
        """casdm1s and casdm2 are the values that are put into the kernel
        whereas the c_dm1s and c_cascm2 are the densities which multiply the
        kernel function (ie the contraction in terms of normal 1 and 2-rdm
        quantities.)

        incl_coul includes the coulomb interaction with the contracting density!
        delta actually sets contracted density to contracted_density - density (like delta in lpdft grads)
        """
        t0 = (logger.process_clock(), logger.perf_counter())
        if mo is None: mo = self.mo_coeff
        if ci is None: ci = self.ci
        if casdm1s is None: casdm1s = self.make_one_casdm1s(ci, state=state)
        if casdm2 is None: casdm2 = self.make_one_casdm2(ci, state=state)
        ncore, ncas = self.ncore, self.ncas

        dm1s = _dms.casdm1s_to_dm1s(self, casdm1s, mo_coeff=mo, ncore=ncore,
                                    ncas=ncas)
        cascm2 = _dms.dm2_cumulant(casdm2, casdm1s)

        if c_dm1s is None:
            c_dm1s = dm1s

        if c_cascm2 is None:
            c_cascm2 = cascm2

        pdft_feff1, pdft_feff2 = pdft_feff.kernel(self.otfnal, dm1s, cascm2,
                                                  c_dm1s, c_cascm2, mo, ncore,
                                                  ncas,
                                                  max_memory=self.max_memory,
                                                  paaa_only=paaa_only,
                                                  aaaa_only=aaaa_only,
                                                  jk_pc=jk_pc, delta=delta)

        if incl_coul:
            if delta:
                c_dm1s -= dm1s

            pdft_feff1 += self._scf.get_j(self.mol, c_dm1s[0] + c_dm1s[1])

        logger.timer(self, 'get_pdft_feff', *t0)
        return pdft_feff1, pdft_feff2

    def _state_average_nuc_grad_method(self, state=None):
        if not isinstance(self, mc1step.CASSCF):
            raise NotImplementedError("CASCI-based PDFT nuclear gradients")
        elif getattr(self, 'frozen', None) is not None:
            raise NotImplementedError("PDFT nuclear gradients with frozen orbitals")
        elif isinstance(self, _DFCASSCF):
            from pyscf.df.grad.mcpdft import Gradients
        else:
            from pyscf.grad.mcpdft import Gradients
        return Gradients(self, state=state)

    def nuc_grad_method(self):
        return self._state_average_nuc_grad_method(state=None)

    def dip_moment(self, unit='Debye', origin='Coord_Center', state=0):
        if not isinstance(self, mc1step.CASSCF):
            raise NotImplementedError("CASCI-based PDFT dipole moments")
        elif getattr(self, 'frozen', None) is not None:
            raise NotImplementedError("PDFT dipole moments with frozen orbitals")
        elif isinstance(self, _DFCASSCF):
            raise NotImplementedError("PDFT dipole moments with density-fitting ERIs")
        # Monkeypatch for double prop folders
        # TODO: more elegant solution
        import os
        mypath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        myproppath = os.path.join(mypath, 'prop')
        # suppress irrelevant warnings when 'properties' ext mod installed
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Module.*is under testing")
            from pyscf import prop
        prop.__path__.append(myproppath)
        prop.__path__ = list(set(prop.__path__))
        from pyscf.prop.dip_moment.mcpdft import ElectricDipole
        dip_obj = ElectricDipole(self)
        mol_dipole = dip_obj.kernel(state=state, unit=unit, origin=origin)
        return mol_dipole

    def get_energy_decomposition(self, mo_coeff=None, ci=None, ot=None,
                                 otxc=None, grids_level=None,
                                 grids_attr=None, split_x_c=None,
                                 verbose=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ci is None: ci = self.ci
        if verbose is None: verbose = self.verbose
        return get_energy_decomposition(
            self, mo_coeff=mo_coeff, ci=ci, ot=ot, otxc=otxc,
            grids_level=grids_level, grids_attr=grids_attr,
            split_x_c=split_x_c, verbose=verbose
        )

    def state_average_mix(self, fcisolvers=None, weights=(0.5, 0.5)):
        return state_average_mix(self, fcisolvers, weights)

    def state_average_mix_(self, fcisolvers=None, weights=(0.5, 0.5)):
        state_average_mix_(self, fcisolvers, weights)
        return self

    def multi_state_mix(self, fcisolvers=None, weights=(0.5, 0.5), method='CMS'):
        if method.upper() == "LIN":
            from pyscf.mcpdft.lpdft import linear_multi_state_mix
            return linear_multi_state_mix(self, fcisolvers=fcisolvers, weights=weights)

        else:
            raise NotImplementedError(f"StateAverageMix not available for {method}")

    def multi_state(self, weights=(0.5, 0.5), method='CMS'):
        if method.upper() == "LIN":
            from pyscf.mcpdft.lpdft import linear_multi_state
            return linear_multi_state(self, weights=weights)

        else:
            from pyscf.mcpdft.mspdft import multi_state
            return multi_state(self, weights=weights,
                               diabatization=method)

    def state_interaction(self, weights=(0.5, 0.5), diabatization='CMS'):
        logger.warn(self, ('"state_interaction" for multi-state PDFT is '
                           'deprecated. Use multi_state instead. In the '
                           'future this will raise an error.'))
        return self.multi_state(weights=weights, diabatization=diabatization)

    @property
    def otxc(self):
        return self.otfnal.otxc

    @otxc.setter
    def otxc(self, x):
        self._init_ot_grids(x)

    make_one_casdm1s = _dms.make_one_casdm1s
    make_one_casdm2 = _dms.make_one_casdm2
    energy_mcwfn = energy_mcwfn
    energy_dft = energy_dft

    def energy_tot(self, mo_coeff=None, ci=None, ot=None, state=0,
                   verbose=None, otxc=None, grids_level=None, grids_attr=None,
                   logger_tag='MC-PDFT'):
        ''' Compute the MC-PDFT energy of a single state '''
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ci is None: ci = self.ci
        if grids_attr is None: grids_attr = {}
        if grids_level is not None: grids_attr['level'] = grids_level
        if len(grids_attr) or (otxc is not None):
            old_ot = ot if (ot is not None) else self.otfnal
            old_grids = old_ot.grids
            # TODO: general compatibility with arbitrary (non-translated) fnals
            if otxc is None: otxc = old_ot.otxc
            new_ot = get_transfnal(self.mol, otxc)
            new_ot.grids.__dict__.update(old_grids.__dict__)
            new_ot.grids.__dict__.update(**grids_attr)
            ot = new_ot
        elif ot is None:
            ot = self.otfnal
        e_tot, e_ot = energy_tot(self, mo_coeff=mo_coeff, ot=ot, ci=ci,
                                 state=state, verbose=verbose)
        logger.note(self, '%s E = %s, Eot(%s) = %s', logger_tag,
                    e_tot, ot.otxc, e_ot)
        return e_tot, e_ot


def get_mcpdft_child_class(mc, ot, **kwargs):
    # Inheritance magic
    mc_doc = (mc.__class__.__doc__ or
              'No docstring for MC-SCF parent method')

    class PDFT(_PDFT, mc.__class__):
        __doc__ = mc_doc + '\n\n' + _PDFT.__doc__
        _mc_class = mc.__class__

    pdft = PDFT(mc._scf, mc.ncas, mc.nelecas, my_ot=ot, **kwargs)
    _keys = pdft._keys.copy()
    pdft.__dict__.update(mc.__dict__)
    pdft._keys = pdft._keys.union(_keys)
    return pdft
