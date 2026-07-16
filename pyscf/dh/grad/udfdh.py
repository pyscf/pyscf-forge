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
# dh import
from pyscf.dh.resp import UDHRespMixin
from pyscf.dh.dhutil import calc_batch_size, gen_batch, gen_shl_batch, tot_size, timing, as_scanner_grad, available_memory
from pyscf.dh.grad.dfdh import generator_L_1, kernel, GradientMixin
# pyscf import
from pyscf import lib, df
from pyscf.df.grad.rhf import _int3c_wrapper as int3c_wrapper
# other import
import numpy as np
import itertools

einsum = lib.einsum
α, β = 0, 1
αα, αβ, ββ = 0, 1, 2


@timing
def get_gradient_jk(dfobj: df.DF, C, D, D_r, Y_mo, cx, cx_n, max_memory=2000):
    max_memory = available_memory(max_memory)
    mol, aux = dfobj.mol, dfobj.auxmol
    natm, nao, nmo, nocc = mol.natm, mol.nao, C.shape[-1], mol.nelec
    mocc = max(nocc)
    naux = Y_mo[0].shape[0]
    # this algorithm asserts naux = aux.nao, i.e. no linear dependency in auxiliary basis
    assert naux == aux.nao
    so = slice(0, nocc[α]), slice(0, nocc[β])

    D_r_symm = (D_r + D_r.swapaxes(-1, -2)) / 2
    D_r_ao = einsum("sup, spq, svq -> suv", C, D_r_symm, C)
    D_mo = np.zeros((2, nmo, nmo))
    for σ in (α, β):
        for i in range(nocc[σ]):
            D_mo[σ, i, i] = 1

    Y_dot_D, Y_dot_D_r = np.zeros((2, naux)), np.zeros((2, naux))
    nbatch = calc_batch_size(nmo**2, max_memory)
    for σ in (α, β):
        for i in range(nocc[σ]):
            Y_dot_D[σ] += Y_mo[σ][:, i, i]
        for saux in gen_batch(0, naux, nbatch):
            Y_dot_D_r[σ][saux] = einsum("Ppq, pq -> P", Y_mo[σ][saux], D_r_symm[σ])

    Y_ip = [np.asarray(Y_mo[σ][:, so[σ]]) for σ in (α, β)]
    L_inv, L_1_gen = generator_L_1(aux)
    int3c2e_ip1_gen = int3c_wrapper(mol, aux, "int3c2e_ip1", "s1")
    int3c2e_ip2_gen = int3c_wrapper(mol, aux, "int3c2e_ip2", "s1")
    C0 = [C[σ][:, so[σ]] for σ in (α, β)]
    D1 = [cx * D_r_symm[σ] + 0.5 * cx_n * D_mo[σ] for σ in (α, β)]
    C1 = [C[σ] @ D1[σ] for σ in (α, β)]

    grad_contrib = np.zeros((natm, 3))
    for A in range(natm):
        shA0, shA1, _, _ = mol.aoslice_by_atom()[A]
        shA0a, shA1a, _, _ = aux.aoslice_by_atom()[A]

        Y_1_mo_D_r = [np.zeros((3, naux, nocc[σ], nmo)) for σ in (α, β)]
        Y_1_dot_D, Y_1_dot_D_r = np.zeros((2, 3, naux)), np.zeros((2, 3, naux))

        pre_flop = tot_size(Y_1_mo_D_r, Y_ip, Y_1_dot_D, Y_1_dot_D_r)
        nbatch = calc_batch_size(3*(nao+mocc)*naux, max_memory, pre_flop)
        for shU0, shU1, U0, U1 in gen_shl_batch(mol, nbatch, shA0, shA1):
            su = slice(U0, U1)
            int3c2e_ip1 = int3c2e_ip1_gen((shU0, shU1, 0, mol.nbas, 0, aux.nbas))
            for σ in (α, β):
                Y_1_mo_D_r[σ] -= einsum("tuvQ, PQ, ui, vp -> tPip", int3c2e_ip1, L_inv, C0[σ][su], C1[σ])
                Y_1_mo_D_r[σ] -= einsum("tuvQ, PQ, up, vi -> tPip", int3c2e_ip1, L_inv, C1[σ][su], C0[σ])
                Y_1_dot_D[σ] -= 2 * einsum("tuvQ, PQ, uv -> tP", int3c2e_ip1, L_inv, D[σ][su])
                Y_1_dot_D_r[σ] -= 2 * einsum("tuvQ, PQ, uv -> tP", int3c2e_ip1, L_inv, D_r_ao[σ][su])

        nbatch = calc_batch_size(3*nao*(nao+mocc), max_memory, pre_flop)
        for shP0, shP1, P0, P1 in gen_shl_batch(aux, nbatch, shA0a, shA1a):
            sp = slice(P0, P1)
            int3c2e_ip2 = int3c2e_ip2_gen((0, mol.nbas, 0, mol.nbas, shP0, shP1))
            for σ in (α, β):
                Y_1_mo_D_r[σ] -= einsum("tuvQ, PQ, ui, vp -> tPip", int3c2e_ip2, L_inv[:, sp], C0[σ], C1[σ])
                Y_1_dot_D[σ] -= einsum("tuvQ, PQ, uv -> tP", int3c2e_ip2, L_inv[:, sp], D[σ])
                Y_1_dot_D_r[σ] -= einsum("tuvQ, PQ, uv -> tP", int3c2e_ip2, L_inv[:, sp], D_r_ao[σ])

        L_1 = L_1_gen(A)
        L_1_dot_inv = einsum("tRQ, PR -> tPQ", L_1, L_inv)
        for σ in (α, β):
            Y_1_mo_D_r[σ] -= einsum("Qiq, qp, tPQ -> tPip", Y_ip[σ], D1[σ], L_1_dot_inv)
            Y_1_dot_D[σ] -= einsum("Q, tPQ -> tP", Y_dot_D[σ], L_1_dot_inv)
            Y_1_dot_D_r[σ] -= einsum("Q, tPQ -> tP", Y_dot_D_r[σ], L_1_dot_inv)
            # RI-K contribution
            grad_contrib[A] += - 2 * einsum("Pip, tPip -> t", Y_ip[σ], Y_1_mo_D_r[σ])

        # RI-J contribution
        for σ, ς in itertools.product((α, β), (α, β)):
            grad_contrib[A] += (
                + einsum("P, tP -> t", Y_dot_D[σ], Y_1_dot_D_r[ς])
                + einsum("P, tP -> t", Y_dot_D_r[σ], Y_1_dot_D[ς])
                + einsum("P, tP -> t", Y_dot_D[σ], Y_1_dot_D[ς]))
    return grad_contrib


@timing
def _get_gradient_pt2(mf):
    tensors = mf.tensors
    C, e = mf.mo_coeff, mf.mo_energy
    mol, aux_ri = mf.mol, mf.df_ri.auxmol
    natm, nao, nocc, nvir, naux = mol.natm, mf.nao, mf.nocc, mf.nvir, mf.df_ri.get_naoaux()
    mocc = max(nocc)
    assert naux == aux_ri.nao
    so, sv, sa = mf.so, mf.sv, mf.sa

    D_r = tensors.load("D_r")
    H_1_mo = mf.H_1_mo
    grad_corr = einsum("spq, sApq -> A", D_r, H_1_mo)
    if not mf.base.eval_pt2:
        return grad_corr.reshape(natm, 3)

    W_I = mf.W_I
    W_II = - einsum("spq, sq -> spq", D_r, e)
    W_III_tmp = mf.Ax0_Core(so, so, sa, sa)(D_r)
    W = W_I + W_II
    for σ in (α, β):
        W[σ][so[σ], so[σ]] += - 0.5 * W_III_tmp[σ]
    W_ao = einsum("sup, spq, svq -> suv", C, W, C)
    S_1_ao = mf._S_1_ao
    grad_corr += np.einsum("suv, Auv -> A", W_ao, S_1_ao)
    grad_corr = grad_corr.reshape(natm, 3)

    L_inv, L_1_gen = generator_L_1(aux_ri)
    int3c2e_ip1_gen = int3c_wrapper(mol, aux_ri, "int3c2e_ip1", "s1")
    int3c2e_ip2_gen = int3c_wrapper(mol, aux_ri, "int3c2e_ip2", "s1")
    Y_ia_ri = [np.asarray(tensors["Y_mo_ri" + str(σ)][:, so[σ], sv[σ]]) for σ in (α, β)]
    G_ia_ri = [tensors.load("G_ia_ri" + str(σ)) for σ in (α, β)]

    for A in range(natm):
        L_1_ri = L_1_gen(A)
        Y_1_ia_ri = [np.zeros((3, naux, nocc[σ], nvir[σ])) for σ in (α, β)]
        shA0, shA1, _, _ = mol.aoslice_by_atom()[A]
        shA0a, shA1a, _, _ = aux_ri.aoslice_by_atom()[A]

        nbatch = calc_batch_size(3*(nao+mocc)*naux, available_memory(mf.max_memory), tot_size(Y_1_ia_ri))
        for shU0, shU1, U0, U1 in gen_shl_batch(mol, nbatch, shA0, shA1):
            su = slice(U0, U1)
            int3c2e_ip1 = int3c2e_ip1_gen((shU0, shU1, 0, mol.nbas, 0, aux_ri.nbas))
            for σ in (α, β):
                Y_1_ia_ri[σ] -= einsum("tuvQ, PQ, ui, va -> tPia", int3c2e_ip1, L_inv, C[σ][su, so[σ]], C[σ][:, sv[σ]])
                Y_1_ia_ri[σ] -= einsum("tuvQ, PQ, ua, vi -> tPia", int3c2e_ip1, L_inv, C[σ][su, sv[σ]], C[σ][:, so[σ]])

        nbatch = calc_batch_size(3*nao*(nao+mocc), available_memory(mf.max_memory), tot_size(Y_1_ia_ri))
        for shP0, shP1, P0, P1 in gen_shl_batch(aux_ri, nbatch, shA0a, shA1a):
            sp = slice(P0, P1)
            int3c2e_ip2 = int3c2e_ip2_gen((0, mol.nbas, 0, mol.nbas, shP0, shP1))
            for σ in (α, β):
                Y_1_ia_ri[σ] -= einsum("tuvQ, PQ, ui, va -> tPia", int3c2e_ip2, L_inv[:, sp], C[σ][:, so[σ]], C[σ][:, sv[σ]])

        for σ in (α, β):
            Y_1_ia_ri[σ] -= einsum("Qia, tRQ, PR -> tPia", Y_ia_ri[σ], L_1_ri, L_inv)
            grad_corr[A] += einsum("Pia, tPia -> t", G_ia_ri[σ], Y_1_ia_ri[σ])
    return grad_corr


class Gradients(UDHRespMixin, GradientMixin):

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
        Y_mo = [self.tensors["Y_mo_jk" + str(σ)] for σ in (α, β)]
        cx_n = self.cx_n if self.xc_n else self.cx
        return get_gradient_jk(self.df_jk, self.mo_coeff, self.D, D_r, Y_mo, self.cx, cx_n, self.max_memory)

    def get_gradient_pt2(self):
        return _get_gradient_pt2(self)

    kernel = kernel
    as_scanner = as_scanner_grad

    def base_method(self):
        return self.base

