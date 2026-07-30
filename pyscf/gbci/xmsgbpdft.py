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
Extended Multi-State GBPDFT (XMS-GBPDFT)

This module is the GBCI-reference counterpart of `pyscf.mcpdft.xmspdft`.
It constructs the state-averaged Fock operator and the intermediate-state
rotation used by the XMS-GBPDFT variant of multi-state GBPDFT.

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


def _as_ci_list(ci):
    if isinstance(ci, (list, tuple)):
        return list(ci)
    arr = np.asarray(ci)
    if arr.ndim >= 3:
        return [arr[i] for i in range(arr.shape[0])]
    return [ci]


def _get_rdm1s_data(mc, mo_coeff=None, data=None):
    if data is not None:
        return data
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    intermediates = mc.get_gbci_intermediates(mo_coeff)
    return mc.precompute_rdm1s(
        mo_coeff=mo_coeff,
        dmet_core_list=intermediates["dmet_core_list"],
        conf_info_list=intermediates["conf_info_list"],
        ov_list=intermediates["ov_list"])


def make_weighted_rdm1s(mc, mo_coeff=None, ci=None, weights=None, data=None):
    """Build the state-averaged spin-separated AO 1-RDM."""
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    if ci is None:
        ci = mc.ci
    ci_list = _as_ci_list(ci)
    if weights is None:
        weights = getattr(mc, "weights", None)
    if weights is None:
        weights = np.ones(len(ci_list)) / len(ci_list)
    weights = np.asarray(weights, dtype=np.double)
    if len(weights) != len(ci_list):
        raise ValueError("weights and ci must have the same number of roots")

    data = _get_rdm1s_data(mc, mo_coeff=mo_coeff, data=data)
    dtype = np.result_type(mo_coeff, *(np.asarray(c) for c in ci_list))
    dm1s_sa = np.zeros((2, mo_coeff.shape[0], mo_coeff.shape[0]), dtype=dtype)
    for weight, ci_root in zip(weights, ci_list):
        dm1s_sa += weight * np.asarray(mc.make_rdm1s(
            ci_root, mo_coeff=mo_coeff, data=data))
    return dm1s_sa


def make_safock_ao(mc, mo_coeff=None, ci=None, weights=None, data=None):
    """Build the spin-free state-averaged AO Fock matrix."""
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    dm1s_sa = make_weighted_rdm1s(
        mc, mo_coeff=mo_coeff, ci=ci, weights=weights, data=data)
    dm_sa = dm1s_sa[0] + dm1s_sa[1]
    hcore = mc._scf.get_hcore(mc.mol)
    vj, vk = mc._scf.get_jk(dm=dm_sa)
    return hcore + vj - 0.5 * vk


def make_fock_gbci(mc, mo_coeff=None, ci=None, weights=None, data=None):
    """Compute the XMS state-averaged Fock matrix in root space.

    The matrix elements are ``<I|F^SA|J>``.  For GBCI the transition density is
    evaluated in the full AO space so bath/core transition-density changes are
    included in the intermediate-state rotation.
    """
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    if ci is None:
        ci = mc.ci
    ci_list = _as_ci_list(ci)
    if weights is None:
        weights = getattr(mc, "weights", None)
    if weights is None:
        weights = np.ones(len(ci_list)) / len(ci_list)

    data = _get_rdm1s_data(mc, mo_coeff=mo_coeff, data=data)
    fock_ao = make_safock_ao(
        mc, mo_coeff=mo_coeff, ci=ci_list, weights=weights, data=data)

    nroots = len(ci_list)
    dtype = np.result_type(fock_ao, *(np.asarray(c) for c in ci_list))
    safock = np.empty((nroots, nroots), dtype=dtype)
    for i, ci_bra in enumerate(ci_list):
        for j, ci_ket in enumerate(ci_list):
            tdm1 = mc.trans_rdm1(
                ci_bra, ci_ket, mo_coeff=mo_coeff, data=data)
            safock[i, j] = lib.einsum("pq,pq->", fock_ao, tdm1)
    return 0.5 * (safock + safock.conj().T)


def diagonalize_safock(mc, mo_coeff=None, ci=None, data=None):
    """Diagonalize the XMS state-averaged Fock matrix."""
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    if ci is None:
        ci = mc.ci
    fock = make_fock_gbci(
        mc, mo_coeff=mo_coeff, ci=ci, data=data)
    return linalg.eigh(fock)


def safock_energy(mc, mo_coeff=None, ci=None, **kwargs):
    """Return the XMS SA-Fock objective value and zero first derivative."""
    if ci is None:
        ci = mc.ci
    nroots = len(_as_ci_list(ci))
    dsa_fock = np.zeros(nroots * (nroots - 1) // 2)
    e_states, _ = diagonalize_safock(
        mc, mo_coeff=mo_coeff, ci=ci)
    return np.dot(e_states, mc.weights), dsa_fock, None


def solve_safock(mc, mo_coeff=None, ci=None, **kwargs):
    """Rotate CI roots into the XMS intermediate-state basis."""
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    if ci is None:
        ci = mc.ci
    ci_list = _as_ci_list(ci)
    data = kwargs.get("data", None)
    _, si_pdft = diagonalize_safock(
        mc, mo_coeff=mo_coeff, ci=ci_list, data=data)
    ci_rot = np.tensordot(si_pdft.T, np.asarray(ci_list), axes=1)
    return True, list(ci_rot)
