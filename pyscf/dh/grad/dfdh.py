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
#
# Authors:
#          Zhenyu Zhu <ajz34@outlook.com>
#          Shirong Wang <srwang20@fudan.edu.cn>
#

from __future__ import annotations
from pyscf.dh.resp import RDHRespMixin
from pyscf.dh.resp import _get_Y_mo
from pyscf.dh.dh import DHBase
from pyscf.dh.dhutil import calc_batch_size, gen_batch, gen_shl_batch, timing, as_scanner_grad, available_memory
from pyscf import gto, lib, df
from pyscf.df.grad.rhf import _int3c_wrapper as int3c_wrapper
import numpy as np

einsum = lib.einsum


def kernel(mf_dh: Gradients, **kwargs):
    dump_t_ijab = mf_dh.with_t_ijab

    mf_dh.base.build()
    if mf_dh.base.mo_coeff is None:
        mf_dh.base.run_scf(**kwargs)
        mf_dh.__dict__.update(mf_dh.base.__dict__)
    H_1_ao = get_H_1_ao(mf_dh.mol)
    if mf_dh.D.ndim == 2:
        H_1_mo = mf_dh.mo_coeff.T @ H_1_ao @ mf_dh.mo_coeff
    else:
        H_1_mo = np.array([einsum("up, Auv, vq -> Apq", mf_dh.mo_coeff[σ], H_1_ao, mf_dh.mo_coeff[σ]) for σ in range(2)])
    mf_dh.H_1_ao = H_1_ao
    mf_dh.H_1_mo = H_1_mo

    S_1_ao = get_S_1_ao(mf_dh.mol)
    if mf_dh.D.ndim == 2:
        S_1_mo = mf_dh.mo_coeff.T @ S_1_ao @ mf_dh.mo_coeff
    else:
        S_1_mo = np.array([einsum("up, Auv, vq -> Apq", mf_dh.mo_coeff[σ], S_1_ao, mf_dh.mo_coeff[σ]) for σ in range(2)])
    mf_dh._S_1_ao = S_1_ao

    Y_mo = _get_Y_mo(mf_dh.df_jk, mf_dh.df_ri, mf_dh.mo_coeff,
                          mf_dh.base.eval_pt2, mf_dh._incore_Y_mo,
                          max_memory=mf_dh.max_memory)
    eri = mf_dh.get_eri_cpks(Y_mo)
    mf_dh.tensors.consume(Y_mo).consume(eri)

    mf_dh.prepare_xc_kernel()
    pt2_res, eng_bi = mf_dh._prepare_pt2(dump_t_ijab=dump_t_ijab)
    mf_dh.tensors.consume(pt2_res)
    if mf_dh.base.eng_tot is None and eng_bi is not None:
        DHBase.kernel(mf_dh.base, eng_bi=(None,) + tuple(eng_bi))
    mf_dh.prepare_lagrangian(gen_W=True)
    mf_dh.prepare_D_r()

    mf_dh.grad_jk = mf_dh.get_gradient_jk()
    D_r = mf_dh.tensors.load("D_r")
    mf_dh.grad_gga = _get_gradient_gga(mf_dh, D_r)
    mf_dh.grad_pt2 = mf_dh.get_gradient_pt2()

    grad_enfunc = _get_gradient_enfunc(mf_dh, S_1_mo, H_1_ao)
    grad_contrib = mf_dh.mf_s.Gradients().grad_nuc()
    grad_contrib += grad_enfunc
    grad_contrib = mf_dh._add_dispersion_gradient(grad_contrib)
    mf_dh.grad_enfunc = grad_contrib

    mf_dh.grad_tot = mf_dh.de = mf_dh.grad_jk + mf_dh.grad_gga + mf_dh.grad_pt2 + mf_dh.grad_enfunc
    return mf_dh.grad_tot

def _get_gradient_enfunc(mf, S_1_mo, H_1_ao):
    xc_n, D, so, eo, Co, mf_n = mf.xc_n, mf.D, mf.so, mf.eo, mf.Co, mf.mf_n
    natm = H_1_ao.shape[0] // 3
    grad_contrib = np.zeros(natm * 3)
    if D.ndim == 2:
        grad_contrib += einsum("Auv, uv -> A", H_1_ao, D)
        if xc_n is None:
            grad_contrib -= 2 * np.einsum("Ai, i -> A", S_1_mo[:, so, so].diagonal(0, -1, -2), eo)
        else:
            nc_F_0_ij = einsum("ui, uv, vj -> ij", Co, mf_n.get_fock(dm=D), Co)
            grad_contrib -= 2 * einsum("Aij, ij -> A", S_1_mo[:, so, so], nc_F_0_ij)
    else:
        grad_contrib += np.einsum("Auv, suv -> A", H_1_ao, D, optimize=True)
        if xc_n is None:
            for σ in range(2):
                grad_contrib -= np.einsum("Ai, i -> A", S_1_mo[σ][:, so[σ], so[σ]].diagonal(0, -1, -2), eo[σ])
        else:
            F_0_ao_n = mf_n.get_fock(dm=D)
            nc_F_0_ij = [(Co[σ].T @ F_0_ao_n[σ] @ Co[σ]) for σ in range(2)]
            for σ in range(2):
                grad_contrib -= einsum("Aij, ij -> A", S_1_mo[σ][:, so[σ], so[σ]], nc_F_0_ij[σ])
    grad_contrib = grad_contrib.reshape(natm, 3)
    return grad_contrib


@timing
def get_H_1_ao(mol):
    from pyscf.grad.rhf import get_hcore
    natm, nao = mol.natm, mol.nao
    h1 = get_hcore(mol)
    H_1_ao = np.zeros((natm, 3, nao, nao))
    for A, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        with mol.with_rinv_at_nucleus(A):
            vrinv = mol.intor("int1e_iprinv", comp=3)
            vrinv *= -mol.atom_charge(A)
        vrinv[:, p0:p1] += h1[:, p0:p1]
        H_1_ao[A] = vrinv + vrinv.transpose(0, 2, 1)
    H_1_ao = H_1_ao.reshape(natm * 3, nao, nao)
    return H_1_ao


@timing
def get_S_1_ao(mol: gto.Mole):
    natm, nao = mol.natm, mol.nao
    int1e_ipovlp = mol.intor("int1e_ipovlp")
    S_1_ao = np.zeros((natm, 3, nao, nao))
    for A, (_, _, A0, A1) in enumerate(mol.aoslice_by_atom()):
        sA = slice(A0, A1)
        S_1_ao[A, :, sA, :] = - int1e_ipovlp[:, sA, :]
    S_1_ao += S_1_ao.swapaxes(-1, -2)
    S_1_ao = S_1_ao.reshape(natm * 3, nao, nao)
    return S_1_ao


def generator_L_1(aux):
    # derivative of cholesky lower triangular 2c2e integral
    # this involves direct inverse of 2c2e integral, so their should be no auxiliary basis dependency
    # L here does not refer to PT2 lagrangian
    L = np.linalg.cholesky(aux.intor("int2c2e"))
    L_inv = np.linalg.inv(L)
    l = np.zeros_like(L)
    for i in range(l.shape[0]):
        l[i, :i] = 1
        l[i, i] = 1 / 2
    int2c2e_1 = aux.intor("int2c2e_ip1")

    def lambda_L_1(A):
        _, _, A0a, A1a = aux.aoslice_by_atom()[A]
        m = L_inv[:, A0a:A1a] @ int2c2e_1[:, A0a:A1a] @ L_inv.T
        m += m.swapaxes(-1, -2)
        L_1 = - L @ (l * m)
        return L_1
    return L_inv, lambda_L_1


@timing
def get_gradient_jk(dfobj: df.DF, C, D, D_r, Y_mo, cx, cx_n, max_memory=2000):
    max_memory = available_memory(max_memory)
    mol, aux = dfobj.mol, dfobj.auxmol
    natm, nao, nmo, nocc = mol.natm, mol.nao, C.shape[-1], mol.nelec[0]
    naux = Y_mo.shape[0]
    # this algorithm asserts naux = aux.nao, i.e. no linear dependency in auxiliary basis
    assert naux == aux.nao
    so = slice(0, nocc)

    D_r_symm = (D_r + D_r.T) / 2
    D_r_ao = C @ D_r_symm @ C.T
    D_mo = np.zeros((nmo, nmo))
    for i in range(nocc):
        D_mo[i, i] = 2

    Y_dot_D, Y_dot_D_r = np.zeros(naux), np.zeros(naux)
    for i in range(nocc):
        Y_dot_D += 2 * Y_mo[:, i, i]
    nbatch = calc_batch_size(nmo**2, max_memory)
    for saux in gen_batch(0, naux, nbatch):
        Y_dot_D_r[saux] = einsum("Ppq, pq -> P", Y_mo[saux], D_r_symm)

    Y_ip = np.asarray(Y_mo[:, so])

    L_inv, L_1_gen = generator_L_1(aux)
    int3c2e_ip1_gen = int3c_wrapper(mol, aux, "int3c2e_ip1", "s1")
    int3c2e_ip2_gen = int3c_wrapper(mol, aux, "int3c2e_ip2", "s1")
    C0, C1 = C[:, so], cx * C @ D_r_symm + 0.5 * cx_n * C @ D_mo

    grad_contrib = np.zeros((natm, 3))
    for A in range(natm):
        shA0, shA1, _, _ = mol.aoslice_by_atom()[A]
        shA0a, shA1a, _, _ = aux.aoslice_by_atom()[A]

        Y_1_dot_D = np.zeros((3, naux))
        Y_1_dot_D_r = np.zeros((3, naux))
        Y_1_mo_D_r = np.zeros((3, naux, nocc, nmo))

        nbatch = calc_batch_size(3*(nao+nocc)*naux, max_memory, Y_1_mo_D_r.size + Y_ip.size)
        for shU0, shU1, U0, U1 in gen_shl_batch(mol, nbatch, shA0, shA1):
            su = slice(U0, U1)
            int3c2e_ip1 = int3c2e_ip1_gen((shU0, shU1, 0, mol.nbas, 0, aux.nbas))
            Y_1_mo_D_r -= einsum("tuvQ, PQ, ui, vp -> tPip", int3c2e_ip1, L_inv, C0[su], C1)
            Y_1_mo_D_r -= einsum("tuvQ, PQ, up, vi -> tPip", int3c2e_ip1, L_inv, C1[su], C0)
            Y_1_dot_D -= 2 * einsum("tuvQ, PQ, uv -> tP", int3c2e_ip1, L_inv, D[su])
            Y_1_dot_D_r -= 2 * einsum("tuvQ, PQ, uv -> tP", int3c2e_ip1, L_inv, D_r_ao[su])

        nbatch = calc_batch_size(3*nao*(nao+nocc), max_memory, Y_1_mo_D_r.size + Y_ip.size)
        for shP0, shP1, P0, P1 in gen_shl_batch(aux, nbatch, shA0a, shA1a):
            sp = slice(P0, P1)
            int3c2e_ip2 = int3c2e_ip2_gen((0, mol.nbas, 0, mol.nbas, shP0, shP1))
            Y_1_mo_D_r -= einsum("tuvQ, PQ, ui, vp -> tPip", int3c2e_ip2, L_inv[:, sp], C0, C1)
            Y_1_dot_D -= einsum("tuvQ, PQ, uv -> tP", int3c2e_ip2, L_inv[:, sp], D)
            Y_1_dot_D_r -= einsum("tuvQ, PQ, uv -> tP", int3c2e_ip2, L_inv[:, sp], D_r_ao)

        L_1 = L_1_gen(A)
        L_1_dot_inv = einsum("tRQ, PR -> tPQ", L_1, L_inv)
        Y_1_mo_D_r -= einsum("Qiq, qp, tPQ -> tPip", Y_ip, cx * D_r_symm + 0.5 * cx_n * D_mo, L_1_dot_inv)
        Y_1_dot_D -= einsum("Q, tPQ -> tP", Y_dot_D, L_1_dot_inv)
        Y_1_dot_D_r -= einsum("Q, tPQ -> tP", Y_dot_D_r, L_1_dot_inv)

        grad_contrib[A] = (
            + einsum("P, tP -> t", Y_dot_D, Y_1_dot_D_r)
            + einsum("P, tP -> t", Y_dot_D_r, Y_1_dot_D)
            + einsum("P, tP -> t", Y_dot_D, Y_1_dot_D)
            - 2 * einsum("Pip, tPip -> t", Y_ip, Y_1_mo_D_r))

    return grad_contrib


def _get_gradient_gga(mf, D_r):
    mf_s = mf.mf_s
    ni, grids, xc, xc_n, D = mf.ni, mf.grids, mf.xc, mf.xc_n, mf.D
    from pyscf import grad, hessian
    mol, C, mo_occ = mf_s.mol, mf_s.mo_coeff, mf_s.mo_occ
    natm = mol.natm
    grad_contrib = np.zeros((natm, 3))
    if ni._xc_type(xc_n if xc_n else xc) == "GGA":
        if D.ndim == 2:
            veff_1_gga = grad.rks.get_vxc(ni, mol, grids, xc_n if xc_n else xc, D)[1]
            for A, (_, _, A0, A1) in enumerate(mol.aoslice_by_atom()):
                grad_contrib[A] += 2 * einsum("tuv, uv -> t", veff_1_gga[:, A0:A1], D[A0:A1])
        else:
            veff_1_gga = grad.uks.get_vxc(ni, mol, grids, xc_n if xc_n else xc, D)[1]
            for A, (_, _, A0, A1) in enumerate(mol.aoslice_by_atom()):
                grad_contrib[A] += 2 * einsum("stuv, suv -> t", veff_1_gga[:, :, A0:A1], D[:, A0:A1])
    if ni._xc_type(xc) == "GGA" and D_r is not None:
        D_r_symm = (D_r + D_r.swapaxes(-1, -2)) / 2
        if D.ndim == 2:
            D_r_ao = einsum("up, pq, vq -> uv", C, D_r_symm, C)
            F_1_ao_dfa = np.array(hessian.rks._get_vxc_deriv1(mf_s.Hessian(), C, mo_occ, 2000))
            grad_contrib += einsum("uv, Atuv -> At", D_r_ao, F_1_ao_dfa)
        else:
            D_r_ao = einsum("sup, spq, svq -> suv", C, D_r_symm, C)
            F_1_ao_dfa = np.array(hessian.uks._get_vxc_deriv1(mf_s.Hessian(), C, mo_occ, 2000))
            grad_contrib += einsum("suv, sAtuv -> At", D_r_ao, F_1_ao_dfa)
    return grad_contrib


class GradientMixin(lib.StreamObject):
    """Shared gradient pipeline: D3/D4 dispersion gradient contributions."""

    def _add_dispersion_gradient(self, grad_contrib):
        mol = self.mol
        if "D3" in self.xc_add:
            from pyscf.dispersion.dftd3 import DFTD3Dispersion
            d3_info = self.xc_add["D3"]
            model = DFTD3Dispersion(mol, xc=d3_info["xc"], version=d3_info["version"])
            disp = model.get_dispersion(grad=True)
            grad_contrib += disp["gradient"]
        if "D4" in self.xc_add:
            from pyscf.dispersion.dftd4 import DFTD4Dispersion
            d4_info = self.xc_add["D4"]
            model = DFTD4Dispersion(mol, xc=d4_info["xc"], version=d4_info["version"])
            disp = model.get_dispersion(grad=True)
            grad_contrib += disp["gradient"]
        return grad_contrib


@timing
def _get_gradient_pt2(mf):
    tensors = mf.tensors
    C, e = mf.mo_coeff, mf.mo_energy
    mol, aux_ri = mf.mol, mf.df_ri.auxmol
    natm, nao, nmo, nocc, nvir, naux = mol.natm, mf.nao, mf.nmo, mf.nocc, mf.nvir, mf.df_ri.get_naoaux()
    assert naux == aux_ri.nao
    so, sv, sa = mf.so, mf.sv, mf.sa

    D_r = tensors.load("D_r")
    H_1_mo = mf.H_1_mo
    grad_corr = einsum("pq, Apq -> A", D_r, H_1_mo)
    if not mf.base.eval_pt2:
        return grad_corr.reshape(natm, 3)

    W_I = mf.W_I
    W_II = - einsum("pq, q -> pq", D_r, e)
    W_III = np.zeros((nmo, nmo))
    W_III[so, so] = - 0.5 * mf.Ax0_Core(so, so, sa, sa)(D_r)
    W = W_I + W_II + W_III
    W_ao = C @ W @ C.T
    S_1_ao = mf._S_1_ao
    grad_corr += einsum("uv, Auv -> A", W_ao, S_1_ao)

    L_inv, L_1_gen = generator_L_1(aux_ri)
    int3c2e_ip1_gen = int3c_wrapper(mol, aux_ri, "int3c2e_ip1", "s1")
    int3c2e_ip2_gen = int3c_wrapper(mol, aux_ri, "int3c2e_ip2", "s1")
    Y_ia_ri = np.asarray(tensors["Y_mo_ri"][:, so, sv])

    def lambda_Y_1_ia_ri(A):
        L_1_ri = L_1_gen(A)
        Y_1_ia_ri = np.zeros((3, naux, nocc, nvir))
        shA0, shA1, _, _ = mol.aoslice_by_atom()[A]
        shA0a, shA1a, _, _ = aux_ri.aoslice_by_atom()[A]

        nbatch = calc_batch_size(3*(nao+nocc)*naux, available_memory(mf.max_memory), Y_1_ia_ri.size)
        for shU0, shU1, U0, U1 in gen_shl_batch(mol, nbatch, shA0, shA1):
            su = slice(U0, U1)
            int3c2e_ip1 = int3c2e_ip1_gen((shU0, shU1, 0, mol.nbas, 0, aux_ri.nbas))
            Y_1_ia_ri -= einsum("tuvQ, PQ, ui, va -> tPia", int3c2e_ip1, L_inv, C[su, so], C[:, sv])
            Y_1_ia_ri -= einsum("tuvQ, PQ, ua, vi -> tPia", int3c2e_ip1, L_inv, C[su, sv], C[:, so])

        nbatch = calc_batch_size(3*nao*(nao+nocc), available_memory(mf.max_memory), Y_1_ia_ri.size)
        for shP0, shP1, P0, P1 in gen_shl_batch(aux_ri, nbatch, shA0a, shA1a):
            sp = slice(P0, P1)
            int3c2e_ip2 = int3c2e_ip2_gen((0, mol.nbas, 0, mol.nbas, shP0, shP1))
            Y_1_ia_ri -= einsum("tuvQ, PQ, ui, va -> tPia", int3c2e_ip2, L_inv[:, sp], C[:, so], C[:, sv])

        Y_1_ia_ri -= einsum("Qia, tRQ, PR -> tPia", Y_ia_ri, L_1_ri, L_inv)
        return Y_1_ia_ri

    G_ia_ri = tensors.load("G_ia_ri")
    for A in range(natm):
        grad_corr[3*A:3*A+3] += 4 * einsum("Pia, tPia -> t", G_ia_ri, lambda_Y_1_ia_ri(A))
    return grad_corr.reshape(natm, 3)


class Gradients(RDHRespMixin, GradientMixin):

    def __init__(self, method):
        self.__dict__.update(method.__dict__)
        self.base = method
        self.grad_jk = None
        self.grad_gga = None
        self.grad_pt2 = None
        self.grad_enfunc = None
        self.grad_tot = None
        self.de = None

    def get_gradient_jk(self):
        D_r = self.tensors.load("D_r")
        cx_n = self.cx_n if self.xc_n else self.cx
        return get_gradient_jk(self.df_jk, self.mo_coeff, self.D, D_r,
                               self.tensors["Y_mo_jk"], self.cx, cx_n, self.max_memory)

    def get_gradient_pt2(self):
        return _get_gradient_pt2(self)

    def base_method(self):
        return self.base

    kernel = kernel
    as_scanner = as_scanner_grad

