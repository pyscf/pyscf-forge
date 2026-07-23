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
from scipy import linalg

from pyscf import lib
from pyscf.fci import cistring
from pyscf.gbci import direct_gbci


def _as_ci_list(ci):
    if isinstance(ci, (list, tuple)):
        return list(ci)
    arr = np.asarray(ci)
    if arr.ndim >= 3:
        return [arr[i] for i in range(arr.shape[0])]
    return [ci]


def _as_real_vector(values):
    return np.asarray(values, dtype=float).reshape(-1)


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


def make_heff_gbci(mc, mo_coeff=None, ci=None, debug=False):
    """Build the GBCI Hamiltonian matrix in the provided CI basis."""
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    if ci is None:
        ci = mc.ci
    ci_list = _as_ci_list(ci)

    intermediates = mc.get_gbci_intermediates(mo_coeff, debug=debug)
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


class _MSGBPDFT:
    """Multi-state GBPDFT mixin for GBCI-based references."""

    def __init__(self, mc, diabatizer, diabatize, diabatization, weights):
        self.__dict__.update(mc.__dict__)
        self._diabatizer = diabatizer
        self._diabatize = diabatize
        self.diabatization = diabatization
        self.weights = np.asarray(weights, dtype=float)
        self._e_states = None
        self.si_gbci = None
        self.si_pdft = None
        self.heff_gbci = None
        self.hdiag_pdft = None
        self.diabatic_e_ot = None

    @property
    def e_states(self):
        return self._e_states

    @e_states.setter
    def e_states(self, value):
        self._e_states = value

    @property
    def si(self):
        return self.si_pdft

    @si.setter
    def si(self, value):
        self.si_pdft = value

    def _eig_si(self, heff):
        return linalg.eigh(heff)

    def get_heff_offdiag(self):
        heff_offdiag = self.heff_gbci.copy()
        heff_offdiag[np.diag_indices_from(heff_offdiag)] = 0.0
        return heff_offdiag

    def get_heff_pdft(self):
        heff_pdft = self.heff_gbci.copy()
        heff_pdft[np.diag_indices_from(heff_pdft)] = self.hdiag_pdft
        return 0.5 * (heff_pdft + heff_pdft.conj().T)

    def get_ci_adiabats(self, ci=None, uci="MSGBPDFT"):
        si_dict = {"GBCI": self.si_gbci, "MSGBPDFT": self.si_pdft}
        if isinstance(uci, (str, np.bytes_)):
            key = uci.upper()
            if key not in si_dict:
                raise RuntimeError("valid uci : 'GBCI', 'MSGBPDFT', or ndarray")
            uci = si_dict[key]
        if ci is None:
            ci = self.ci
        return list(np.tensordot(uci.T, np.asarray(_as_ci_list(ci)), axes=1))

    get_ci_basis = get_ci_adiabats

    def diabatize(self, ci=None, ci0=None, **kwargs):
        if ci is None:
            ci = self.ci
        ci_list = _as_ci_list(ci)
        if ci0 is not None:
            ci0_list = _as_ci_list(ci0)
            ovlp = np.tensordot(
                np.asarray(ci_list).conj(), np.asarray(ci0_list),
                axes=((1, 2), (1, 2)))
            u, _, vh = linalg.svd(ovlp)
            ci_list = self.get_ci_basis(ci=ci_list, uci=np.dot(u, vh))
        return self._diabatize(self, ci=ci_list, **kwargs)

    def diabatizer(self, mo_coeff=None, ci=None, **kwargs):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if ci is None:
            ci = self.ci
        return self._diabatizer(self, mo_coeff=mo_coeff, ci=ci, **kwargs)

    def _compute_diabatic_pdft_diag(self, otxc=None, grids_level=None,
                                    grids_attr=None, debug=False):
        old_e_gbci = self.e_gbci
        self.e_gbci = np.asarray(self.heff_gbci.diagonal()).real
        try:
            _, e_ot, hdiag = self.compute_pdft_energy_(
                otxc=otxc, grids_level=grids_level, grids_attr=grids_attr,
                debug=debug)
        finally:
            self.e_gbci = old_e_gbci
        self.diabatic_e_ot = e_ot
        return np.asarray(hdiag, dtype=float)

    def kernel(self, mo_coeff=None, ci0=None, otxc=None, grids_level=None,
               grids_attr=None, debug=False, **kwargs):
        reset = getattr(self.otfnal, "reset", None)
        if callable(reset):
            reset(mol=self.mol)
        if ci0 is None and isinstance(getattr(self, "ci", None), list):
            ci0 = [c.copy() for c in self.ci]

        self.optimize_gbci_(mo_coeff=mo_coeff, ci0=ci0, debug=debug)
        diab_conv, self.ci = self.diabatize(
            ci=self.ci, ci0=ci0, debug=debug)
        self.converged = bool(getattr(self, "converged", True) and diab_conv)

        self.heff_gbci = self.make_heff_gbci(debug=debug)
        e_gbci, si_gbci = self._eig_si(self.heff_gbci)
        ref_e = _as_real_vector(self.e_gbci)
        if len(ref_e) == len(e_gbci):
            err = linalg.norm(np.sort(ref_e) - np.sort(e_gbci))
            if err > 1e-7:
                lib.logger.warn(
                    self, "XMS-GBPDFT heff_gbci eigenvalues differ from "
                    "GBCI root energies by %.3g", err)
        self.e_gbci = e_gbci
        self.si_gbci = si_gbci

        self.hdiag_pdft = self._compute_diabatic_pdft_diag(
            otxc=otxc, grids_level=grids_level, grids_attr=grids_attr,
            debug=debug)
        self.e_states, self.si_pdft = self._eig_si(self.get_heff_pdft())
        self.e_tot = np.dot(self.e_states, self.weights)
        self.e_ot = self.diabatic_e_ot
        self._log_diabats()
        self._log_adiabats()
        return (self.e_tot, self.e_ot, self.e_gbci, self.e_cas, self.ci,
                self.mo_coeff, getattr(self, "mo_energy", None))

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
    weights = np.asarray(weights, dtype=float)
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
