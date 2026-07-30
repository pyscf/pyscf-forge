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
# Authors: Minseok Oh <msjeff2001@snu.ac.kr>
#          Byungjoo Kim <creeperkim28@snu.ac.kr>
# Edited by: Seunghoon Lee <seunghoonlee@snu.ac.kr>

'''
Grouped-Bath Pair-Density Functional Theory (GBPDFT)

This module is the GBCI-reference counterpart of `pyscf.mcpdft.mcpdft`.
It evaluates translated on-top pair-density functional energies using GBCI
reference states.

References:
[1] Orbital-relaxed bath theory for charge-transfer processes in
    transition-metal complexes
    Minseok Oh, Jiseong Park, Byungjoo Kim, Hyeok Lim and Seunghoon Lee
    Phys. Chem. Chem. Phys. 2026
'''

import numpy as np
from pyscf import __config__
from pyscf.lib import logger
from pyscf.mcpdft import mcpdft

from pyscf.gbci import rdm as gbci_rdm
from pyscf.gbci import otpd


def energy_mcwfn(mc, mo_coeff=None, ci=None, ot=None, dm1s=None,
                 e_gbci=None, verbose=None):
    """Compute the hybrid-adjusted wave-function part of GBPDFT."""
    if ot is None:
        ot = mc.otfnal
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    if ci is None:
        ci = mc.ci
    if dm1s is None:
        dm1s = mc.make_rdm1s(ci, mo_coeff=mo_coeff)
    if e_gbci is None:
        e_gbci = mc.e_gbci
    if verbose is None:
        verbose = mc.verbose

    log = logger.new_logger(mc, verbose=verbose)
    hyb_x, hyb_c = ot._numint.rsh_and_hybrid_coeff(ot.otxc, mc.mol.spin)[2]
    if abs(hyb_x - hyb_c) > 1e-10:
        log.warn("exchange and correlation hybridization differ")
        log.warn(
            "may lead to unphysical results, see "
            "https://github.com/pyscf/pyscf-forge/issues/128")

    hcore = mc._scf.get_hcore()
    dm1 = dm1s[0] + dm1s[1]
    if log.verbose >= logger.DEBUG or abs(hyb_x) > 1e-10:
        vj, vk = mc._scf.get_jk(dm=dm1s)
        vj = vj[0] + vj[1]
    else:
        vj = mc._scf.get_j(dm=dm1)
        vk = None

    e_classical = (
        mc._scf.energy_nuc()
        + np.tensordot(hcore, dm1)
        + 0.5 * np.tensordot(vj, dm1)
    )
    e_x = 0.0
    if vk is not None:
        e_x = -0.5 * (
            np.tensordot(vk[0], dm1s[0]) + np.tensordot(vk[1], dm1s[1]))

    e_c = 0.0
    if abs(hyb_c) > 1e-10:
        e_c = e_gbci - e_classical - e_x
    return e_classical + hyb_x * e_x + hyb_c * e_c


def energy_dft(mc, mo_coeff=None, ci=None, ot=None, dm1s=None,
               otpd_data=None, ot_root_cache=None):
    """Compute GBPDFT on-top energy by the one-step pair-density route."""
    if ot is None:
        ot = mc.otfnal
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    if ci is None:
        ci = mc.ci
    if dm1s is None:
        dm1s = mc.make_rdm1s(ci, mo_coeff=mo_coeff)
    if otpd_data is None:
        otpd_data = mc.make_otpd_intermediates(mo_coeff=mo_coeff)
    if ot_root_cache is None:
        ot_root_cache = otpd.make_root_intermediates(ci, otpd_data)
    return otpd.energy_ot(
        ot, dm1s, mo_coeff, mc.ncore, mc.ncas, ci=ci, data=otpd_data,
        ot_root_cache=ot_root_cache, max_memory=mc.max_memory)


class _BPDFT:
    """GBPDFT mixin parallel to MC-PDFT's _PDFT class."""

    _mc_class = None

    def __init__(self, scf, ncas, nelecas, my_ot=None, grids_level=None,
                 grids_attr=None, **kwargs):
        if grids_attr is None:
            grids_attr = {}
        self._mc_class.__init__(self, scf, ncas, nelecas, **kwargs)
        self.e_ot = None
        self.e_gbci = None
        self.e_states = None
        self.otfnal = None
        self.chkfile = self._scf.chkfile
        self.max_cycle_fp = getattr(__config__, 'gbpdft_max_cycle_fp', 50)
        self._in_gbci_env = False
        if grids_level is not None:
            grids_attr['level'] = grids_level
        if my_ot is not None:
            self._init_ot_grids(my_ot, grids_attr=grids_attr)

    _init_ot_grids = mcpdft._PDFT._init_ot_grids

    @property
    def grids(self):
        return self.otfnal.grids

    @grids.setter
    def grids(self, value):
        self.otfnal.grids = value
        return self.otfnal.grids

    @property
    def otxc(self):
        return self.otfnal.otxc

    @otxc.setter
    def otxc(self, value):
        self._init_ot_grids(value)

    def optimize_gbci_(self, mo_coeff=None, ci0=None, **kwargs):
        """Run the underlying GBCI solver and keep the GBCI energy."""
        self.e_gbci, self.e_cas, self.ci = self._mc_class.kernel(
            self, mo_coeff=mo_coeff, ci0=ci0, **kwargs)
        return self.e_gbci, self.e_cas, self.ci

    def _select_root(self, values, root):
        arr = np.asarray(values, dtype=object)
        if arr.ndim == 0:
            if root != 0:
                raise IndexError("root index out of range")
            return values
        return values[root]

    def make_otpd_intermediates(self, mo_coeff=None, intermediates=None):
        """Build CI-independent intermediates for the on-top pair density."""
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if intermediates is None:
            intermediates = self.get_gbci_intermediates(mo_coeff)
        return otpd.make_otpd_intermediates(
            self._scf.get_ovlp(self.mol), mo_coeff, self.ncas, self.nelecas,
            self.ncore, intermediates["dmet_core_list"],
            intermediates["conf_info_list"], intermediates["ov_list"])

    def compute_pdft_energy_(self, mo_coeff=None, ci=None, ot=None, otxc=None,
                             grids_level=None, grids_attr=None, **kwargs):
        """Compute GBPDFT energies with the GBCI wave function fixed."""
        if mo_coeff is not None:
            self.mo_coeff = mo_coeff
        if ci is not None:
            self.ci = ci
        if ot is not None:
            self.otfnal = ot
        if otxc is not None:
            self.otxc = otxc
        if grids_attr is None:
            grids_attr = {}
        if grids_level is not None:
            grids_attr['level'] = grids_level
        if grids_attr:
            self.grids.__dict__.update(grids_attr)

        intermediates = self.get_gbci_intermediates(self.mo_coeff)
        otpd_data = self.make_otpd_intermediates(
            self.mo_coeff, intermediates=intermediates)

        nroots = getattr(self.fcisolver, 'nroots', 1)
        ci_list = self.ci if nroots > 1 else [self.ci]
        e_tot_list = []
        e_ot_list = []

        for root, ci_root in enumerate(ci_list):
            dm1s = np.asarray(gbci_rdm.make_rdm1s_precomputed(
                ci_root, otpd_data, mo_coeff=self.mo_coeff))
            root_cache = otpd.make_root_intermediates(ci_root, otpd_data)
            e_ot = energy_dft(
                self, self.mo_coeff, ci_root, self.otfnal, dm1s,
                otpd_data, root_cache)
            e_gbci_root = self._select_root(self.e_gbci, root)
            e_wfn = energy_mcwfn(
                self, self.mo_coeff, ci_root, self.otfnal, dm1s,
                e_gbci_root)
            e_tot_list.append(e_wfn + e_ot)
            e_ot_list.append(e_ot)

        self.e_ot = np.asarray(e_ot_list)
        self.e_states = np.asarray(e_tot_list)
        if nroots == 1:
            self.e_tot = self.e_states[0]
            self.e_ot = self.e_ot[0]
        else:
            self.e_tot = self.e_states
        return self.e_tot, self.e_ot, self.e_states

    def kernel(self, mo_coeff=None, ci0=None, otxc=None, grids_attr=None,
               grids_level=None, **kwargs):
        self.optimize_gbci_(mo_coeff=mo_coeff, ci0=ci0, **kwargs)
        self.compute_pdft_energy_(
            otxc=otxc, grids_attr=grids_attr, grids_level=grids_level)
        return (self.e_tot, self.e_ot, self.e_gbci, self.e_cas, self.ci,
                self.mo_coeff, self.mo_energy)

    energy_mcwfn = energy_mcwfn
    energy_dft = energy_dft

    def multi_state(self, weights=(0.5, 0.5), method="XMS"):
        """Build a multi-state GBPDFT object."""
        from pyscf.gbci import msgbpdft
        return msgbpdft.multi_state(self, weights=weights,
                                    diabatization=method)


def _attach_gbpdft_class(mc, ot, **kwargs):
    """Create a GBPDFT child object from a GBCI-like parent."""
    class GBPDFT(_BPDFT, mc.__class__):
        __doc__ = (mc.__class__.__doc__ or '') + '\n\n' + _BPDFT.__doc__
        _mc_class = mc.__class__

    pdft = GBPDFT(mc._scf, mc.ncas, mc.nelecas, my_ot=ot, **kwargs)
    pdft.__dict__.update(mc.__dict__)
    pdft._init_ot_grids(ot, grids_attr=kwargs.get('grids_attr', None))
    return pdft


def get_gbpdft_child_class(mc, ot, **kwargs):
    """Attach GBPDFT to an existing GBCI object."""
    return _attach_gbpdft_class(mc, ot, **kwargs)


def _is_gbci_solver(obj):
    """Return whether an object exposes the GBCI reference interface."""
    return (hasattr(obj, 'fcisolver')
            and hasattr(obj, 'optimize_mo')
            and hasattr(obj, 'get_gbci_intermediates'))


def _GBPDFT(gbci_class, mc_or_mf, ot, ncas=None, nelecas=None, ncore=None,
             group_a=None, **kwargs):
    """Build GBPDFT on top of a GBCI reference class."""
    if _is_gbci_solver(mc_or_mf):
        return get_gbpdft_child_class(mc_or_mf, ot, **kwargs)
    if ncas is None or nelecas is None:
        raise ValueError(
            "ncas and nelecas are required when building GBPDFT from an SCF "
            "object")
    mc = gbci_class(mc_or_mf, ncas, nelecas, ncore=ncore, group_a=group_a)
    return get_gbpdft_child_class(mc, ot, **kwargs)


def GBCI(mc_or_mf, ot, ncas=None, nelecas=None, ncore=None, group_a=None,
         **kwargs):
    """Build GBPDFT on top of a GBCI reference."""
    from pyscf import gbci
    return _GBPDFT(
        gbci.GBCI, mc_or_mf, ot, ncas=ncas, nelecas=nelecas, ncore=ncore,
        group_a=group_a, **kwargs)


def gbci(*args, **kwargs):
    """Alias for :func:`GBCI`."""
    return GBCI(*args, **kwargs)
