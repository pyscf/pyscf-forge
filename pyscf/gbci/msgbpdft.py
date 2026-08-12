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
# Author: Minseok Oh <msjeff2001@snu.ac.kr>
# Edited by: Seunghoon Lee <seunghoonlee@snu.ac.kr>

'''
Multi-State Grouped-Bath Pair-Density Functional Theory (MS-GBPDFT)

This module is the GBCI-reference counterpart of `pyscf.mcpdft.mspdft`.
It builds the effective Hamiltonian in a GBCI model space and combines the
GBCI reference coupling with GBPDFT diagonal energies for multi-state
state-interaction calculations.

References:
[1] Orbital-relaxed bath theory for charge-transfer processes in
    transition-metal complexes
    Minseok Oh, Jiseong Park, Byungjoo Kim, Hyeok Lim and Seunghoon Lee
    Phys. Chem. Chem. Phys. 2026
[2] Multi-state pair-density functional theory
    J. J. Bao, C. Zhou, Z. Varga, S. Kanchanakungwankul, L. Gagliardi
    and D. G. Truhlar
    Faraday Discuss. 2020, 224, 348-372
'''

import numpy as np

from pyscf import lib
from pyscf.fci import cistring
from pyscf.mcpdft import mspdft
from pyscf.gbci import direct_gbci


def _as_ci_list(ci):
    if isinstance(ci, (list, tuple)):
        return list(ci)
    arr = np.asarray(ci)
    if arr.ndim >= 3:
        return [arr[i] for i in range(arr.shape[0])]
    return [ci]


def _contract_hamiltonian(solver, erieff, civec, ncas, nelecas,
                          conf_info_list, ov_list, ecore_list, link_index,
                          ts, t_nonzero):
    if hasattr(solver, "base") and hasattr(solver, "ss_penalty"):
        old_contract = getattr(solver, "_contract_h_base", None)
        had_contract = hasattr(solver, "_contract_h_base")
        solver._contract_h_base = solver.base.contract_h
        try:
            return solver.contract_h(
                erieff, civec, ncas, nelecas, conf_info_list, ov_list,
                ecore_list, link_index, ts, t_nonzero)
        finally:
            if had_contract:
                solver._contract_h_base = old_contract
            else:
                del solver._contract_h_base
    return solver.contract_h(
        erieff, civec, ncas, nelecas, conf_info_list, ov_list,
        ecore_list, link_index, ts, t_nonzero)


def make_heff_gbci(mc, mo_coeff=None, ci=None):
    """Build the GBCI Hamiltonian matrix in the provided CI basis."""
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    if ci is None:
        ci = mc.ci
    ci_list = _as_ci_list(ci)

    intermediates = mc.get_gbci_intermediates(mo_coeff)
    dmet_act_list = mc.get_active_dm(mo_coeff)
    h1e, ecore_list = mc.get_h1cas(
        dmet_act_list, intermediates["mo_list"],
        intermediates["dmet_core_list"])
    eri = mc.get_h2eff(mo_coeff)
    erieff = mc.fcisolver.absorb_h1e(h1e, eri, mc.ncas, mc.nelecas, 0.5)

    link_indexa = cistring.gen_linkstr_index(range(mc.ncas), mc.nelecas[0])
    link_indexb = cistring.gen_linkstr_index(range(mc.ncas), mc.nelecas[1])
    link_index = (link_indexa, link_indexb)
    na = cistring.num_strings(mc.ncas, mc.nelecas[0])
    nb = cistring.num_strings(mc.ncas, mc.nelecas[1])
    ts = direct_gbci.gen_excitations(mc.ncas, mc.nelecas, na, nb, link_index)
    t_nonzero = direct_gbci.gen_nonzero_excitations(*ts)

    hci = []
    for civec in ci_list:
        hc = _contract_hamiltonian(
            mc.fcisolver, erieff, np.asarray(civec), mc.ncas, mc.nelecas,
            intermediates["conf_info_list"], intermediates["ov_list"],
            ecore_list, link_index, ts, t_nonzero)
        hci.append(np.asarray(hc).reshape(np.asarray(civec).shape))

    nroots = len(ci_list)
    dtype = np.result_type(*(np.asarray(c) for c in ci_list), *hci)
    heff = np.empty((nroots, nroots), dtype=dtype)
    for i, ci_bra in enumerate(ci_list):
        for j, hc_ket in enumerate(hci):
            heff[i, j] = np.vdot(ci_bra, hc_ket)
    return 0.5 * (heff + heff.conj().T)


def get_diabfns(obj):
    """Return the objective and state-rotation functions for MS-GBPDFT."""
    if obj.upper() == "XMS":
        from pyscf.gbci.xmsgbpdft import safock_energy, solve_safock
        return safock_energy, solve_safock
    raise RuntimeError("Only XMS-GBPDFT is currently implemented")


class _MSGBPDFT(mspdft._MSPDFT):
    """Multi-state GBPDFT mixin for GBCI-based references."""

    def __init__(self, mc, diabatizer, diabatize, diabatization, weights):
        super().__init__(mc, diabatizer, diabatize, diabatization)
        self.weights = np.asarray(weights, dtype=np.double)
        self.heff_mcscf = None
        self.hdiag_pdft = None
        self.diabatic_e_ot = None
        self._in_mcscf_env = False
        self._keys = set(getattr(self, "_keys", ())).union((
            "e_gbci", "heff_gbci", "si_gbci", "diabatic_e_ot"))

    @property
    def e_mcscf(self):
        if self.e_gbci is None:
            return None
        return np.asarray(self.e_gbci)

    @e_mcscf.setter
    def e_mcscf(self, value):
        self.e_gbci = value

    @property
    def heff_gbci(self):
        return self.heff_mcscf

    @heff_gbci.setter
    def heff_gbci(self, value):
        self.heff_mcscf = value

    @property
    def si_gbci(self):
        return self.si_mcscf

    @si_gbci.setter
    def si_gbci(self, value):
        self.si_mcscf = value

    def get_ci_adiabats(self, ci=None, uci="MSGBPDFT"):
        if isinstance(uci, (str, np.bytes_)):
            key = uci.upper()
            uci = {
                "GBCI": "MCSCF",
                "MSGBPDFT": "MSPDFT",
            }.get(key, key)
        if ci is not None:
            ci = _as_ci_list(ci)
        return super().get_ci_adiabats(ci=ci, uci=uci)

    get_ci_basis = get_ci_adiabats

    def optimize_mcscf_(self, mo_coeff=None, ci0=None, **kwargs):
        return self.optimize_gbci_(mo_coeff=mo_coeff, ci0=ci0, **kwargs)

    def compute_pdft_energy_(self, *args, **kwargs):
        old_e_gbci = self.e_gbci
        if getattr(self, "heff_mcscf", None) is not None:
            self.e_gbci = np.asarray(self.heff_mcscf.diagonal()).real
        try:
            results = super().compute_pdft_energy_(*args, **kwargs)
        finally:
            self.e_gbci = old_e_gbci
        self.diabatic_e_ot = results[1]
        return results

    def nuc_grad_method(self):
        raise NotImplementedError("MS-GBPDFT nuclear gradients")

    def nac_method(self):
        raise NotImplementedError("MS-GBPDFT nonadiabatic couplings")

    def dip_moment(self, *args, **kwargs):
        raise NotImplementedError("MS-GBPDFT dipole moments")

    def trans_moment(self, *args, **kwargs):
        raise NotImplementedError("MS-GBPDFT transition dipole moments")

    make_heff_mcscf = make_heff_gbci
    make_heff_gbci = make_heff_gbci

    def _log_diabats(self):
        log = lib.logger.new_logger(self, self.verbose)
        if log.verbose < lib.logger.NOTE:
            return
        log.note("%s diabatic states:", self.__class__.__name__)
        for i, (e_pdft, e_gbci) in enumerate(zip(
                self.hdiag_pdft, self.heff_gbci.diagonal().real)):
            log.note("  State %d  EPDFT = %.15g  EGBCI = %.15g",
                     i, e_pdft, e_gbci)

    def _log_adiabats(self):
        log = lib.logger.new_logger(self, self.verbose)
        if log.verbose < lib.logger.NOTE:
            return
        log.note("%s adiabatic states:", self.__class__.__name__)
        for i, energy in enumerate(self.e_states):
            log.note("  State %d weight %g  EMSGBPDFT = %.15g",
                     i, self.weights[i], energy)


def multi_state(mc, weights=(0.5, 0.5), diabatization="XMS"):
    """Build a multi-state GBPDFT object."""
    if isinstance(mc, _MSGBPDFT):
        raise RuntimeError("already a multi-state GBPDFT solver")
    weights = np.asarray(weights, dtype=np.double)
    if weights.ndim != 1 or len(weights) < 2:
        raise ValueError("MS-GBPDFT requires at least two state weights")
    if abs(np.sum(weights) - 1.0) > 1e-8:
        raise ValueError("MS-GBPDFT weights must sum to 1")

    mc.fcisolver.nroots = len(weights)
    mc.fcisolver.weights = weights
    mc.weights = weights
    diabatizer, diabatize = get_diabfns(diabatization)
    mcbase_class = mc.__class__

    class MSGBPDFT(_MSGBPDFT, mcbase_class):
        pass

    MSGBPDFT.__name__ = diabatization.upper() + mcbase_class.__name__
    return MSGBPDFT(mc, diabatizer, diabatize, diabatization, weights)
