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
On-top pair-density contractions for GBPDFT

This module is the GBCI-reference counterpart of `pyscf.mcpdft.otpd`.
It evaluates the on-top pair density on numerical grids for grouped-bath
core and active-space contributions.

References:
[1] Orbital-relaxed bath theory for charge-transfer processes in
    transition-metal complexes
    Minseok Oh, Jiseong Park, Byungjoo Kim, Hyeok Lim and Seunghoon Lee
    Phys. Chem. Chem. Phys. 2026
[2] Multiconfiguration Pair-Density Functional Theory
    Giovanni Li Manni, Rebecca K. Carlson, Sijie Luo, Dongxia Ma,
    Jeppe Olsen, Donald G. Truhlar and Laura Gagliardi
    J. Chem. Theory Comput. 2014, 10, 3669-3680
'''

import ctypes
import numpy as np
from pyscf import lib
from pyscf.mcpdft.otpd import _grid_ao2mo
from pyscf.gbci import rdm as gbci_rdm

libgbpdft = lib.load_library("libgbpdft")

_double_ptr = np.ctypeslib.ndpointer(dtype=np.float64, ndim=None,
                                     flags=("C_CONTIGUOUS", "ALIGNED"))
_int_ptr = np.ctypeslib.ndpointer(dtype=np.int32, ndim=None,
                                  flags=("C_CONTIGUOUS", "ALIGNED"))

libgbpdft.GBPDFTcontract_alpha_core.argtypes = [
    _double_ptr, _int_ptr, _double_ptr, _double_ptr, _int_ptr,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    _double_ptr,
]
libgbpdft.GBPDFTcontract_beta_core.argtypes = [
    _double_ptr, _int_ptr, _double_ptr, _double_ptr, _int_ptr,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    _double_ptr,
]
libgbpdft.GBPDFTcontract_active_pair.argtypes = [
    _double_ptr, _int_ptr, _double_ptr, _double_ptr, _double_ptr,
    _int_ptr, ctypes.c_int, _int_ptr, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, _double_ptr,
]
libgbpdft.GBPDFTcontract_core_pair.argtypes = [
    _double_ptr, _double_ptr, _double_ptr, ctypes.c_int,
    _double_ptr, ctypes.c_int, _double_ptr, _double_ptr, _double_ptr,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, _double_ptr,
]


def _as_real_contiguous(arr, name, dtype=np.float64):
    """Return a C-contiguous real array for the compiled contraction kernels."""
    if np.iscomplexobj(arr):
        raise NotImplementedError(
            "C GBPDFT kernels currently support real-valued arrays only")
    return np.ascontiguousarray(arr, dtype=dtype)


def make_otpd_intermediates(S, mo_coeff, ncas, nelecas, ncore,
                            dmet_core_list, conf_info_list, ov_list):
    """Build CI-independent intermediates for the GBPDFT pair density."""
    data = gbci_rdm.precompute_rdm1s_data(
        mo_coeff, ncas, nelecas, ncore, dmet_core_list,
        conf_info_list, ov_list)
    data = dict(data)

    _, nmo = mo_coeff.shape
    conf_info_list = data["conf_info_list"]
    ov_list = data["ov_list"]
    dtype = data["dtype"]
    ngroup = ov_list.shape[0]
    dcore_mo = np.empty((ngroup, ngroup, nmo, nmo), dtype=dtype)
    Smo = S @ mo_coeff
    CdagS = mo_coeff.conj().T @ S
    for p1 in range(ngroup):
        for p2 in range(ngroup):
            dcore_mo[p1, p2] = CdagS @ dmet_core_list[p1, p2] @ Smo

    kpair = ngroup * ngroup
    Dpair = np.empty((nmo, nmo, kpair), dtype=dtype)
    for p1 in range(ngroup):
        for p2 in range(ngroup):
            Dpair[:, :, p1 * ngroup + p2] = dcore_mo[p1, p2]

    plist = conf_info_list.reshape(-1)
    group_p, inv_p = np.unique(plist, return_inverse=True)
    dgroup = dcore_mo[group_p, group_p]

    data.update({
        "S": S,
        "ngroup": ngroup,
        "Kpair": kpair,
        "dcore_mo": dcore_mo,
        "Dpair": Dpair,
        "plist": plist,
        "group_p": group_p,
        "inv_p": inv_p,
        "dgroup_T": np.transpose(dgroup, (1, 2, 0)).copy(),
    })
    return data


def _contract_alpha_core(ci, conf_info_list, ov_list, t1a, t1a_nz,
                         ncas, ngroup):
    """Build the alpha active-core contraction tensor with the C backend."""
    na, nb = ci.shape
    kpair = ngroup * ngroup
    ci = _as_real_contiguous(ci, "ci")
    conf_info_list = np.ascontiguousarray(conf_info_list, dtype=np.int32)
    ov_list = _as_real_contiguous(ov_list, "ov_list")
    t1a = _as_real_contiguous(t1a, "t1a")
    t1a_nz = np.ascontiguousarray(t1a_nz, dtype=np.int32)
    Ka = np.empty((ncas, ncas, kpair), dtype=np.float64)
    libgbpdft.GBPDFTcontract_alpha_core(
        ci, conf_info_list, ov_list, t1a, t1a_nz, t1a_nz.shape[0],
        ncas, na, nb, ngroup, Ka)
    return Ka


def _contract_beta_core(ci, conf_info_list, ov_list, t1b, t1b_nz,
                        ncas, ngroup):
    """Build the beta active-core contraction tensor with the C backend."""
    na, nb = ci.shape
    kpair = ngroup * ngroup
    ci = _as_real_contiguous(ci, "ci")
    conf_info_list = np.ascontiguousarray(conf_info_list, dtype=np.int32)
    ov_list = _as_real_contiguous(ov_list, "ov_list")
    t1b = _as_real_contiguous(t1b, "t1b")
    t1b_nz = np.ascontiguousarray(t1b_nz, dtype=np.int32)
    Kb = np.empty((ncas, ncas, kpair), dtype=np.float64)
    libgbpdft.GBPDFTcontract_beta_core(
        ci, conf_info_list, ov_list, t1b, t1b_nz, t1b_nz.shape[0],
        ncas, na, nb, ngroup, Kb)
    return Kb


def _contract_active_pair(ci, conf_info_list, ov_list, t1a, t1b,
                          t1a_nz, t1b_nz, ncas):
    """Build the active-active opposite-spin tensor with the C backend."""
    na, nb = ci.shape
    ngroup = ov_list.shape[0]
    ci = _as_real_contiguous(ci, "ci")
    conf_info_list = np.ascontiguousarray(conf_info_list, dtype=np.int32)
    ov_list = _as_real_contiguous(ov_list, "ov_list")
    t1a = _as_real_contiguous(t1a, "t1a")
    t1b = _as_real_contiguous(t1b, "t1b")
    t1a_nz = np.ascontiguousarray(t1a_nz, dtype=np.int32)
    t1b_nz = np.ascontiguousarray(t1b_nz, dtype=np.int32)
    out = np.empty((ncas, ncas, ncas, ncas), dtype=np.float64)
    libgbpdft.GBPDFTcontract_active_pair(
        ci, conf_info_list, ov_list, t1a, t1b,
        t1a_nz, t1a_nz.shape[0], t1b_nz, t1b_nz.shape[0],
        ncas, na, nb, ngroup, out)
    return out


def _contract_core_pair(M, Ma, Dgroup_pack, Dpair_pack, Ka, Kb, wgroup):
    """Accumulate core-core and active-core pair-density terms on a grid."""
    ngrids = M.shape[0]
    nmo = M.shape[1]
    ncas = Ma.shape[1]
    ndiag = Dgroup_pack.shape[0]
    kpair = Dpair_pack.shape[0]
    M = _as_real_contiguous(M, "M")
    Ma = _as_real_contiguous(Ma, "Ma")
    Dgroup_pack = _as_real_contiguous(Dgroup_pack, "Dgroup_pack")
    Dpair_pack = _as_real_contiguous(Dpair_pack, "Dpair_pack")
    Ka = _as_real_contiguous(Ka, "Ka")
    Kb = _as_real_contiguous(Kb, "Kb")
    wgroup = _as_real_contiguous(wgroup, "wgroup")
    out = np.empty(ngrids, dtype=np.float64)
    libgbpdft.GBPDFTcontract_core_pair(
        M, Ma, Dgroup_pack, ndiag, Dpair_pack, kpair, Ka, Kb, wgroup,
        ngrids, nmo, ncas, out)
    return out


def make_root_intermediates(ci, data):
    """Build root-dependent intermediates for the GBPDFT pair density."""
    ci = np.asarray(ci)
    if ci.ndim == 1:
        ci = ci.reshape(data["na"], data["nb"])
    dtype = np.result_type(ci, data["dtype"])
    ci = np.asarray(ci, dtype=dtype)

    conf_info_list = data["conf_info_list"]
    ov_list = data["ov_list"]
    ncas = data["ncas"]
    ngroup = data["ngroup"]
    t1a = data["t1a"]
    t1b = data["t1b"]
    t1a_nz = data["t1a_nz"]
    t1b_nz = data["t1b_nz"]

    w_diag = np.abs(ci.reshape(-1))**2
    wgroup = np.bincount(
        data["inv_p"], weights=w_diag, minlength=len(data["group_p"])
    ).astype(dtype, copy=False)

    ci_contig = np.ascontiguousarray(ci)
    conf_info_contig = np.ascontiguousarray(conf_info_list)
    ov_contig = np.ascontiguousarray(ov_list)
    t1a_contig = np.ascontiguousarray(t1a)
    t1b_contig = np.ascontiguousarray(t1b)
    t1a_nz_contig = np.ascontiguousarray(t1a_nz)
    t1b_nz_contig = np.ascontiguousarray(t1b_nz)

    Gamma_ab_ac = _contract_active_pair(
        ci_contig, conf_info_contig, ov_contig, t1a_contig, t1b_contig,
        t1a_nz_contig, t1b_nz_contig, ncas)
    Ka = _contract_alpha_core(
        ci_contig, conf_info_contig, ov_contig, t1a_contig,
        t1a_nz_contig, ncas, ngroup)
    Kb = _contract_beta_core(
        ci_contig, conf_info_contig, ov_contig, t1b_contig,
        t1b_nz_contig, ncas, ngroup)

    return {
        "wgroup": wgroup,
        "dgroup_T": data["dgroup_T"],
        "Dpair": data["Dpair"],
        "Dgroup_pack": np.ascontiguousarray(
            data["dgroup_T"].transpose(2, 0, 1)),
        "Dpair_pack": np.ascontiguousarray(data["Dpair"].transpose(2, 0, 1)),
        "Gamma_ab_ac": Gamma_ab_ac,
        "Ka": Ka,
        "Kb": Kb,
    }


def _contract_grid_mats(grid_a, grid_b, packed_mats):
    """Contract grid orbital values with a stack of transition matrices."""
    ngrids, norb = grid_a.shape
    nvec = packed_mats.shape[0]
    out = np.empty((ngrids, nvec),
                   dtype=np.result_type(grid_a, grid_b, packed_mats))

    grid_a = np.ascontiguousarray(grid_a)
    grid_b = np.ascontiguousarray(grid_b)
    for ivec in range(nvec):
        tmp = grid_a @ packed_mats[ivec]
        out[:, ivec] = np.einsum('gi,gi->g', tmp, grid_b, optimize=True)
    return out


def get_ontop_pair_density(ot, ao, ci, data, mo=None, deriv=0,
                           non0tab=None, ot_root_cache=None):
    """Build the on-top pair density directly on a grid block."""
    if deriv != 0:
        raise NotImplementedError("GBPDFT currently supports Pi_deriv=0 only")
    if ot_root_cache is None:
        ot_root_cache = make_root_intermediates(ci, data)
    if mo is None:
        mo = data["mo_coeff"]

    ao_shape = False
    if ao.ndim == 2:
        ao_shape = True
        ao = ao.reshape(1, ao.shape[0], ao.shape[1])

    mo_grid = _grid_ao2mo(ot.mol, ao, mo, non0tab=non0tab)
    M = mo_grid[0]
    Ma = M[:, data["ncore"]:data["ncore"] + data["ncas"]]

    Pi = np.zeros((1, M.shape[0]), dtype=np.result_type(M, data["dtype"]))
    Gamma = ot_root_cache["Gamma_ab_ac"]
    if Gamma.size:
        tmp = lib.einsum("gc,gd,abcd->gab", Ma, Ma, Gamma, optimize=True)
        Pi[0] += lib.einsum("ga,gb,gab->g", Ma, Ma, tmp, optimize=True)

    Pi[0] += _contract_core_pair(
        np.ascontiguousarray(M), np.ascontiguousarray(Ma),
        ot_root_cache["Dgroup_pack"], ot_root_cache["Dpair_pack"],
        ot_root_cache["Ka"], ot_root_cache["Kb"], ot_root_cache["wgroup"])

    if ao_shape:
        ao = ao.reshape(ao.shape[1], ao.shape[2])
    return Pi


def energy_ot(ot, dm1s, mo_coeff, ncore, ncas, ci=None, data=None,
              ot_root_cache=None, max_memory=1000, hermi=1):
    """Compute the GBPDFT on-top energy without materializing full 2-RDM."""
    if ci is None or data is None:
        raise ValueError("ci and precomputed GBPDFT data are required")

    ni = ot._numint
    if ot.xctype == 'HF':
        return 0.0
    if ot.Pi_deriv != 0:
        raise NotImplementedError("GBPDFT currently supports Pi_deriv=0 only")

    nao = mo_coeff.shape[0]
    make_rho = tuple(
        ni._gen_rho_evaluator(ot.mol, dm1s[i], hermi) for i in range(2))

    E_ot = 0.0
    for ao, mask, weight, _ in ni.block_loop(
            ot.mol, ot.grids, nao, ot.dens_deriv, max_memory):
        rho = np.asarray([m[0](0, ao, mask, ot.xctype) for m in make_rho])
        Pi = get_ontop_pair_density(
            ot, ao, ci, data, mo_coeff, ot.Pi_deriv, mask,
            ot_root_cache=ot_root_cache)
        if rho.ndim == 2:
            rho = np.expand_dims(rho, 1)
        if Pi.ndim == 1:
            Pi = np.expand_dims(Pi, 0)
        E_ot += ot.eval_ot(rho, Pi, dderiv=0, weights=weight)[0].dot(weight)
    return E_ot
