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
from pyscf.dh.polar.rdfdh import kernel
from pyscf.dh.resp import UDHRespMixin
from pyscf.dh.dhutil import gen_batch, get_rho_from_dm_gga, tot_size, hermi_sum_last2dim
from pyscf.dh.xccode import xc_equal
# pyscf import
from pyscf import lib
from pyscf.lib.numpy_helper import ANTIHERMI
# other import
import numpy as np
import itertools

einsum = lib.einsum
α, β = 0, 1
αα, αβ, ββ = 0, 1, 2


def xc_kernel_full(f, nr, ng):
    s = set()
    l = lambda: itertools.product(*([range(2)] * nr + [range(3)] * ng))
    for t in l():
        s.add(tuple(sorted(t[:nr]) + sorted(t[nr:])))
    m = dict()
    for i, t in enumerate(sorted(s)):
        m[t] = i
    c = np.zeros([2] * nr + [3] * ng, dtype=int)
    for t in l():
        c[t] = m[tuple(sorted(t[:nr]) + sorted(t[nr:]))]
    return np.array([f.T[i] for i in c.flatten()]).reshape(c.shape + (f.shape[0],))


def _uks_gga_wv2_generator(fxc, kxc, weight):

    frr = xc_kernel_full(fxc[0], 2, 0)
    frg = xc_kernel_full(fxc[1], 1, 1)
    fgg = xc_kernel_full(fxc[2], 0, 2)
    frrr = xc_kernel_full(kxc[0], 3, 0)
    frrg = xc_kernel_full(kxc[1], 2, 1)
    frgg = xc_kernel_full(kxc[2], 1, 2)
    fggg = xc_kernel_full(kxc[3], 0, 3)

    ngrid = frr.shape[-1]
    z = np.zeros((ngrid,))

    pd1_fr = np.concatenate([frr, frg], axis=1)
    pd1_fg = np.concatenate([frg.transpose(1, 0, 2), fgg], axis=1)

    pd2_fr = np.zeros((2, 5, 5, ngrid))
    pd2_fr[:, :2, :2] = frrr
    pd2_fr[:, :2, 2:] = frrg
    pd2_fr[:, 2:, :2] = frrg.swapaxes(1, 2)
    pd2_fr[:, 2:, 2:] = frgg

    pd2_fg = np.zeros((3, 5, 5, ngrid))
    pd2_fg[:, :2, :2] = frrg.swapaxes(0, 2)
    pd2_fg[:, :2, 2:] = frgg.swapaxes(0, 1)
    pd2_fg[:, 2:, :2] = frgg.swapaxes(0, 2)
    pd2_fg[:, 2:, 2:] = fggg

    def _uks_gga_wv2_inner(rho0, rho1, rho2):
        rho0 = np.asarray(rho0)
        rho1 = np.asarray(rho1)
        rho2 = np.asarray(rho2)

        r1, r2 = rho1[:, 0], rho2[:, 0]
        n0, n1, n2 = rho0[:, 1:4], rho1[:, 1:4], rho2[:, 1:4]
        g01 = einsum("atg, btg -> abg", n0, n1)
        g02 = einsum("atg, btg -> abg", n0, n2)
        g12 = einsum("atg, btg -> abg", n1, n2)

        x1 = np.array([r1[0], r1[1], 2 * g01[0, 0], g01[0, 1] + g01[1, 0], 2 * g01[1, 1]])
        x2 = np.array([r2[0], r2[1], 2 * g02[0, 0], g02[0, 1] + g02[1, 0], 2 * g02[1, 1]])

        pd1_x1 = np.array([z, z, 2 * g12[0, 0], g12[0, 1] + g12[1, 0], 2 * g12[1, 1]])

        wv = np.zeros((2, 4, ngrid))
        wva, wvb = wv

        wva[0] += np.einsum("xyg, xg, yg -> g", pd2_fr[0], x1, x2)
        wvb[0] += np.einsum("xyg, xg, yg -> g", pd2_fr[1], x1, x2)
        wva[0] += np.einsum("xg, xg -> g", pd1_fr[0], pd1_x1)
        wvb[0] += np.einsum("xg, xg -> g", pd1_fr[1], pd1_x1)

        wva[1:] += np.einsum("xyg, xg, yg -> g", pd2_fg[0], x1, x2) * n0[0] * 2
        wva[1:] += np.einsum("xyg, xg, yg -> g", pd2_fg[1], x1, x2) * n0[1]
        wvb[1:] += np.einsum("xyg, xg, yg -> g", pd2_fg[1], x1, x2) * n0[0]
        wvb[1:] += np.einsum("xyg, xg, yg -> g", pd2_fg[2], x1, x2) * n0[1] * 2
        wva[1:] += np.einsum("xg, xg -> g", pd1_fg[0], pd1_x1) * n0[0] * 2
        wva[1:] += np.einsum("xg, xg -> g", pd1_fg[1], pd1_x1) * n0[1]
        wvb[1:] += np.einsum("xg, xg -> g", pd1_fg[1], pd1_x1) * n0[0]
        wvb[1:] += np.einsum("xg, xg -> g", pd1_fg[2], pd1_x1) * n0[1] * 2
        wva[1:] += np.einsum("xg, xg -> g", pd1_fg[0], x1) * n2[0] * 2
        wva[1:] += np.einsum("xg, xg -> g", pd1_fg[1], x1) * n2[1]
        wvb[1:] += np.einsum("xg, xg -> g", pd1_fg[1], x1) * n2[0]
        wvb[1:] += np.einsum("xg, xg -> g", pd1_fg[2], x1) * n2[1] * 2

        wva[1:] += einsum("xg, xg -> g", pd1_fg[0], x2) * n1[0] * 2
        wva[1:] += einsum("xg, xg -> g", pd1_fg[1], x2) * n1[1]
        wvb[1:] += einsum("xg, xg -> g", pd1_fg[1], x2) * n1[0]
        wvb[1:] += einsum("xg, xg -> g", pd1_fg[2], x2) * n1[1] * 2

        wva *= weight
        wva[0] *= .5  # v+v.T should be applied in the caller
        wvb *= weight
        wvb[0] *= .5  # v+v.T should be applied in the caller
        return wva, wvb
    return _uks_gga_wv2_inner


def _get_U_1(mf, H_1_mo):
    sv, so = mf.sv, mf.so
    H_1_ai = [H_1_mo[σ, :, sv[σ], so[σ]] for σ in (α, β)]
    U_1_ai = mf.solve_cpks(H_1_ai)
    U_1 = np.zeros_like(H_1_mo)
    for σ in (α, β):
        U_1[σ, :, sv[σ], so[σ]] = U_1_ai[σ]
        U_1[σ, :, so[σ], sv[σ]] = - U_1_ai[σ].swapaxes(-1, -2)
    return U_1


def _get_pdA_F_0_mo(mf, U_1, H_1_mo):
    so, sa = mf.so, mf.sa
    U_1_pi = [U_1[σ, :, :, so[σ]] for σ in (α, β)]

    pdA_F_0_mo = H_1_mo.copy()
    pdA_F_0_mo += einsum("sApq, sp -> sApq", U_1, mf.mo_energy)
    pdA_F_0_mo += einsum("sAqp, sq -> sApq", U_1, mf.mo_energy)
    pdA_F_0_mo += mf.Ax0_Core(sa, sa, sa, so)(U_1_pi)

    pdA_F_0_mo_n = None
    if mf.mf_n:
        F_0_ao_n = mf.mf_n.get_fock(dm=mf.D)
        F_0_mo_n = einsum("sup, suv, svq -> spq", mf.mo_coeff, F_0_ao_n, mf.mo_coeff)
        pdA_F_0_mo_n = np.array(H_1_mo)
        pdA_F_0_mo_n += einsum("sAmp, smq -> sApq", U_1, F_0_mo_n)
        pdA_F_0_mo_n += einsum("sAmq, spm -> sApq", U_1, F_0_mo_n)
        pdA_F_0_mo_n += mf.Ax0_Core(sa, sa, sa, so, xc=mf.xc_n)(U_1_pi)
    return pdA_F_0_mo, pdA_F_0_mo_n


def _get_pdA_Y_ia_ri(mf, U_1):
    Y_mo_ri = [mf.tensors["Y_mo_ri" + str(σ)] for σ in (α, β)]
    nocc, nvir, nmo, naux = mf.nocc, mf.nvir, mf.nmo, mf.df_ri.get_naoaux()
    mocc, mvir = max(nocc), max(nvir)
    so, sv = mf.so, mf.sv
    nprop = mf.nprop

    nbatch = mf.base.calc_batch_size(8 * nmo**2, U_1.size + nprop*naux*mocc*mvir)
    result = []
    for σ in (α, β):
        pdA_Y_ia_ri = np.zeros((nprop, naux, nocc[σ], nvir[σ]))
        for saux in gen_batch(0, naux, nbatch):
            pdA_Y_ia_ri[:, saux] = (
                + einsum("Ami, Pma -> APia", U_1[σ][:, :, so[σ]], Y_mo_ri[σ][saux, :, sv[σ]])
                + einsum("Ama, Pmi -> APia", U_1[σ][:, :, sv[σ]], Y_mo_ri[σ][saux, :, so[σ]]))
        result.append(pdA_Y_ia_ri)
    return result


def _get_polar_dms(mf, D_r, rho):
    U_1 = mf.U_1
    C = mf.mo_coeff
    so = mf.so
    mol, grids, xc = mf.mol, mf.grids, mf.xc
    ni = mf.ni
    dmU = np.array([C[σ] @ U_1[σ, :, :, so[σ]] @ C[σ, :, so[σ]].T for σ in (α, β)])
    dmU += dmU.swapaxes(-1, -2)
    dmR = np.array([C[σ] @ D_r[σ] @ C[σ].T for σ in (α, β)])
    dmR += dmR.swapaxes(-1, -2)
    dmX = np.concatenate([dmU, dmR[:, None]], axis=1)
    rhoX = get_rho_from_dm_gga(ni, mol, grids, dmX)
    _, _, _, kxc = ni.eval_xc(xc, rho, spin=1, deriv=3)
    return rhoX[:, :-1], rhoX[:, -1], kxc


def _get_Ax1_contrib(mf, rho, rhoU, rhoR, fxc, kxc):
    nprop = mf.nprop
    grids = mf.grids
    wv_generator = _uks_gga_wv2_generator(fxc, kxc, grids.weights)
    wv = np.zeros((2, nprop, 4, grids.weights.size))
    for i in range(nprop):
        wv[:, i] = np.asarray(wv_generator(rho, rhoU[:, i], rhoR), dtype=np.float64)
        wv[:, i, 0] *= 2
    return 0.5 * einsum("sArg, sBrg -> AB", rhoU, wv)


def _get_SCR3(mf, U_1, pdA_F_0_mo_n):
    tensors = mf.tensors
    so, sv = mf.so, mf.sv
    naux = mf.df_ri.get_naoaux()
    nocc, nvir = mf.nocc, mf.nvir
    nprop = mf.nprop
    SCR3 = [np.zeros((nprop, nvir[σ], nocc[σ])) for σ in (α, β)]
    if mf.xc_n:
        for σ in (α, β):
            SCR3[σ] += 2 * pdA_F_0_mo_n[σ][:, sv[σ], so[σ]]
    if not mf.base.eval_pt2:
        return SCR3
    G_ia_ri = [tensors.load("G_ia_ri" + str(σ)) for σ in (α, β)]
    pdA_G_ia_ri = [tensors.load("pdA_G_ia_ri" + str(σ)) for σ in (α, β)]
    Y_mo_ri = [tensors["Y_mo_ri" + str(σ)] for σ in (α, β)]
    nbatch = mf.base.calc_batch_size(10 * mf.nmo**2, tot_size(G_ia_ri, pdA_G_ia_ri))
    for σ in (α, β):
        for saux in gen_batch(0, naux, nbatch):
            G_blk = G_ia_ri[σ][saux]
            Y_blk = np.asarray(Y_mo_ri[σ][saux])
            pdA_G_blk = np.asarray(pdA_G_ia_ri[σ][:, saux])
            pdA_Y_blk = einsum("Ami, Pmj -> APij", U_1[σ][:, :, so[σ]], Y_blk[:, :, so[σ]])
            pdA_Y_blk += pdA_Y_blk.swapaxes(-1, -2)
            SCR3[σ] -= einsum("APja, Pij -> Aai", pdA_G_blk, Y_blk[:, so[σ], so[σ]])
            SCR3[σ] -= einsum("Pja, APij -> Aai", G_blk, pdA_Y_blk)
            pdA_Y_blk = einsum("Ama, Pmb -> APab", U_1[σ][:, :, sv[σ]], Y_blk[:, :, sv[σ]])
            pdA_Y_blk += pdA_Y_blk.swapaxes(-1, -2)
            SCR3[σ] += einsum("APib, Pab -> Aai", pdA_G_blk, Y_blk[:, sv[σ], sv[σ]])
            SCR3[σ] += einsum("Pib, APab -> Aai", G_blk, pdA_Y_blk)
    return SCR3


class Polar(UDHRespMixin):

    @property
    def nprop(self):
        return self.H_1_ao.shape[0]

    def __init__(self, method):
        self.__dict__.update(method.__dict__)
        self.base = method
        self.pol_scf = None
        self.pol_corr = None
        self.pol_tot = None
        self.de = None

    def get_U_1(self, H_1_mo):
        return _get_U_1(self, H_1_mo)

    def get_pdA_F_0_mo(self, U_1, H_1_mo):
        return _get_pdA_F_0_mo(self, U_1, H_1_mo)

    def get_pdA_Y_ia_ri(self, U_1):
        return _get_pdA_Y_ia_ri(self, U_1)

    def get_polar_dms(self):
        D_r = self.tensors.load("D_r")
        rho = self.tensors["rho"]
        rhoU, rhoR, kxc = _get_polar_dms(self, D_r, rho)
        self.tensors.create("rhoU", rhoU, incore=True)
        self.tensors.create("rhoR", rhoR, incore=True)
        self.tensors.create("kxc" + self.xc, kxc, incore=True)

    def get_Ax1_contrib(self):
        rho = self.tensors["rho"]
        rhoU = self.tensors.load("rhoU")
        rhoR = self.tensors.load("rhoR")
        fxc = self.tensors["fxc" + self.xc]
        kxc = self.tensors["kxc" + self.xc]
        return _get_Ax1_contrib(self, rho, rhoU, rhoR, fxc, kxc)

    def get_SCR3(self):
        return _get_SCR3(self, self.U_1, self.pdA_F_0_mo_n)

    def prepare_pt2_deriv(self):
        tensors = self.tensors
        c_os, c_ss = self.c_os, self.c_ss
        nocc, nvir, nmo, naux = self.nocc, self.nvir, self.nmo, self.df_ri.get_naoaux()
        mocc, mvir = max(nocc), max(nvir)
        so, sv = self.so, self.sv
        eo, ev = self.eo, self.ev
        nprop = self.nprop

        pdA_D_rdm1 = tensors.create("pdA_D_rdm1", shape=(2, nprop, nmo, nmo))
        if not self.base.eval_pt2:
            return self

        pdA_F_0_mo = self.pdA_F_0_mo
        Y_ia_ri = [tensors["Y_mo_ri" + str(σ)][:, so[σ], sv[σ]] for σ in (α, β)]
        pdA_Y_ia_ri = [tensors["pdA_Y_ia_ri" + str(σ)] for σ in (α, β)]

        pdA_G_ia_ri = [tensors.create("pdA_G_ia_ri" + str(σ), shape=(nprop, naux, nocc[σ], nvir[σ])) for σ in (α, β)]

        nbatch = self.base.calc_batch_size(8*mocc*mvir**2, tot_size(Y_ia_ri, pdA_Y_ia_ri, pdA_G_ia_ri, pdA_F_0_mo, pdA_D_rdm1))
        eval_ss = True if abs(c_ss) > 1e-7 else False
        for σς, σ, ς in (αα, α, α), (αβ, α, β), (ββ, β, β):
            if σς in (αα, ββ) and not eval_ss:
                continue
            D_jab = eo[ς][:, None, None] - ev[σ][None, :, None] - ev[ς][None, None, :]
            for sI in gen_batch(0, nocc[σ], nbatch):
                t_ijab = np.asarray(tensors["t_ijab" + str(σς)][sI])
                D_ijab = eo[σ][sI, None, None, None] + D_jab

                pdA_t_ijab = einsum("APia, Pjb -> Aijab", pdA_Y_ia_ri[σ][:, :, sI], Y_ia_ri[ς])
                pdA_t_ijab += einsum("APjb, Pia -> Aijab", pdA_Y_ia_ri[ς], Y_ia_ri[σ][:, sI])
                for sK in gen_batch(0, nocc[σ], nbatch):
                    t_kjab = t_ijab if sK == sI else tensors["t_ijab" + str(σς)][sK]
                    pdA_t_ijab -= einsum("Aki, kjab -> Aijab", pdA_F_0_mo[σ][:, sK, sI], t_kjab)
                pdA_t_ijab -= einsum("Akj, ikab -> Aijab", pdA_F_0_mo[ς][:, so[ς], so[ς]], t_ijab)
                pdA_t_ijab += einsum("Aca, ijcb -> Aijab", pdA_F_0_mo[σ][:, sv[σ], sv[σ]], t_ijab)
                pdA_t_ijab += einsum("Acb, ijac -> Aijab", pdA_F_0_mo[ς][:, sv[ς], sv[ς]], t_ijab)
                pdA_t_ijab /= D_ijab
                if σς in (αα, ββ):
                    # T_ijab = cc * 0.5 * c_ss * (t_ijab - t_ijab.swapaxes(-1, -2))
                    # pdA_T_ijab = cc * 0.5 * c_ss * (pdA_t_ijab - pdA_t_ijab.swapaxes(-1, -2))
                    T_ijab = 0.5 * c_ss * hermi_sum_last2dim(t_ijab, hermi=ANTIHERMI, inplace=False)
                    pdA_T_ijab = 0.5 * c_ss * hermi_sum_last2dim(pdA_t_ijab, hermi=ANTIHERMI, inplace=False)
                    pdA_D_rdm1[σ][:, so[σ], so[σ]] -= 2 * einsum("kiba, Akjba -> Aij", T_ijab, pdA_t_ijab)
                    pdA_D_rdm1[σ][:, sv[σ], sv[σ]] += 2 * einsum("ijac, Aijbc -> Aab", T_ijab, pdA_t_ijab)
                    pdA_G_ia_ri[σ][:, :, sI] += 4 * einsum("ijab, APjb -> APia", T_ijab, pdA_Y_ia_ri[σ])
                    pdA_G_ia_ri[σ][:, :, sI] += 4 * einsum("Aijab, Pjb -> APia", pdA_T_ijab, Y_ia_ri[σ])
                else:
                    T_ijab = c_os * t_ijab
                    pdA_T_ijab = c_os * pdA_t_ijab
                    for sJ in gen_batch(0, nocc[α], nbatch):
                        if sI == sJ:
                            T_jkab = T_ijab
                        else:
                            t_jkab = tensors["t_ijab" + str(αβ)][sJ]
                            T_jkab = c_os * t_jkab
                        pdA_D_rdm1[α][:, sI, sJ] -= einsum("jkba, Aikba -> Aij", T_jkab, pdA_t_ijab)
                    # pdA_D_rdm1[α][:, so[α], so[α]] -= einsum("ikab, Ajkab -> Aij", T_ijab, pdA_t_ijab)
                    pdA_D_rdm1[β][:, so[β], so[β]] -= einsum("kiba, Akjba -> Aij", T_ijab, pdA_t_ijab)
                    pdA_D_rdm1[α][:, sv[α], sv[α]] += einsum("ijac, Aijbc -> Aab", T_ijab, pdA_t_ijab)
                    pdA_D_rdm1[β][:, sv[β], sv[β]] += einsum("jica, Ajicb -> Aab", T_ijab, pdA_t_ijab)
                    pdA_G_ia_ri[α][:, :, sI] += 2 * einsum("ijab, APjb -> APia", T_ijab, pdA_Y_ia_ri[β])
                    pdA_G_ia_ri[α][:, :, sI] += 2 * einsum("Aijab, Pjb -> APia", pdA_T_ijab, Y_ia_ri[β])
                    pdA_G_ia_ri[β] += 2 * einsum("jiba, APjb -> APia", T_ijab, pdA_Y_ia_ri[α][:, :, sI])
                    pdA_G_ia_ri[β] += 2 * einsum("Ajiba, Pjb -> APia", pdA_T_ijab, Y_ia_ri[α][:, sI])
        pdA_D_rdm1[:] += pdA_D_rdm1.swapaxes(-1, -2)
        return self

    def prepare_polar(self):
        tensors = self.tensors
        so, sv, sa = self.so, self.sv, self.sa
        nprop = self.nprop

        H_1_mo = self.H_1_mo
        U_1 = self.U_1
        pdA_F_0_mo = self.pdA_F_0_mo
        D_r = tensors.load("D_r")
        pdA_D_rdm1 = tensors.load("pdA_D_rdm1")
        U_1_ai = [U_1[σ][:, sv[σ], so[σ]] for σ in (α, β)]

        SCR1 = np.asarray(self.Ax0_Core(sa, sa, sa, sa)(D_r))
        SCR2 = H_1_mo + np.asarray(self.Ax0_Core(sa, sa, sv, so)(U_1_ai))
        SCR3 = self.get_SCR3()

        pol_scf = np.zeros((nprop, nprop))
        pol_corr = np.zeros((nprop, nprop))

        for σ in (α, β):
            pol_scf -= 2 * einsum("Api, Bpi -> AB", H_1_mo[σ][:, :, so[σ]], U_1[σ][:, :, so[σ]])
            pol_corr -= einsum("Aai, Bma, mi -> AB", U_1[σ][:, sv[σ], so[σ]], U_1[σ][:, :, sv[σ]], SCR1[σ][:, so[σ]])
            pol_corr -= einsum("Aai, Bmi, ma -> AB", U_1[σ][:, sv[σ], so[σ]], U_1[σ][:, :, so[σ]], SCR1[σ][:, sv[σ]])
            pol_corr -= einsum("Apm, Bmq, pq -> AB", SCR2[σ], U_1[σ], D_r[σ])
            pol_corr -= einsum("Amq, Bmp, pq -> AB", SCR2[σ], U_1[σ], D_r[σ])
            pol_corr -= einsum("Apq, Bpq -> AB", SCR2[σ], pdA_D_rdm1[σ])
            pol_corr -= einsum("Bai, Aai -> AB", SCR3[σ], U_1[σ][:, sv[σ], so[σ]])
            pol_corr += einsum("Bki, Aai, ak -> AB", pdA_F_0_mo[σ][:, so[σ], so[σ]], U_1[σ][:, sv[σ], so[σ]], D_r[σ][sv[σ], so[σ]])
            pol_corr -= einsum("Bca, Aai, ci -> AB", pdA_F_0_mo[σ][:, sv[σ], sv[σ]], U_1[σ][:, sv[σ], so[σ]], D_r[σ][sv[σ], so[σ]])
        if not xc_equal(self.xc, "HF"):
            pol_corr -= self.Ax1_contrib

        self.pol_scf = pol_scf
        self.pol_corr = pol_corr
        self.de = self.pol_tot = pol_scf + pol_corr
        return self

    kernel = kernel

    def base_method(self):
        return self.base



