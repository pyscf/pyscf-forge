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
from pyscf.dh.resp import RDHRespMixin
from pyscf.dh.resp import _r_get_eri_cpks, _u_get_eri_cpks
from pyscf.dh.resp import _get_Y_mo, _prepare_pt2_r, _prepare_pt2_u
from pyscf.dh.dh import DHBase
from pyscf.dh.dhutil import gen_batch, get_rho_from_dm_gga, restricted_biorthogonalize, hermi_sum_last2dim
from pyscf.dh.xccode import xc_equal
from pyscf import lib
import numpy as np

einsum = lib.einsum


def kernel(mf: Polar):
    mf.base.run_scf()
    mf.__dict__.update(mf.base.__dict__)

    H_1_ao = - mf.mol.intor("int1e_r")
    if mf.D.ndim == 2:
        H_1_mo = mf.mo_coeff.T @ H_1_ao @ mf.mo_coeff
    else:
        H_1_mo = np.array([mf.mo_coeff[σ].T @ H_1_ao @ mf.mo_coeff[σ] for σ in range(2)])
    mf.tensors.create("H_1_ao", H_1_ao)
    mf.tensors.create("H_1_mo", H_1_mo)

    Y_mo = _get_Y_mo(mf.df_jk, mf.df_ri, mf.mo_coeff,
                          mf.base.eval_pt2, mf._incore_Y_mo,
                          max_memory=mf.max_memory)
    if mf.D.ndim == 2:
        eri = _r_get_eri_cpks(Y_mo["Y_mo_jk"], mf.nocc, mf.cx, mf._incore_Y_mo, mf.max_memory)
    else:
        eri = _u_get_eri_cpks([Y_mo["Y_mo_jk" + str(σ)] for σ in range(2)],
                               mf.nocc, mf.cx, mf._incore_Y_mo, mf.max_memory)
    mf.tensors.consume(Y_mo).consume(eri)

    mf.prepare_xc_kernel()
    if mf.D.ndim == 2:
        pt2_res, eng_bi = _prepare_pt2_r(mf, dump_t_ijab=True)
    else:
        pt2_res, eng_bi = _prepare_pt2_u(mf, dump_t_ijab=True)
    mf.tensors.consume(pt2_res)
    if mf.base.eng_tot is None and eng_bi is not None:
        DHBase.kernel(mf.base, eng_bi=(None,) + tuple(eng_bi))
    mf.prepare_lagrangian(gen_W=False)
    mf.prepare_D_r()
    mf.prepare_U_1()
    if mf.ni._xc_type(mf.xc) == "GGA":
        mf.prepare_dms()
        mf.prepare_polar_Ax1_gga()
    mf.prepare_pdA_F_0_mo()
    if mf.base.eval_pt2:
        mf.prepare_pdA_Y_ia_ri()
    mf.prepare_pt2_deriv()
    mf.prepare_polar()
    return mf.de


def _rks_gga_wv2(rho0, rho1, rho2, fxc, kxc, weight):
    frr, frg, fgg = fxc[:3]
    frrr, frrg, frgg, fggg = kxc

    sigma01 = 2 * np.einsum("rg, rg -> g", rho0[1:], rho1[1:], optimize=True)
    sigma02 = 2 * np.einsum("rg, rg -> g", rho0[1:], rho2[1:], optimize=True)
    sigma12 = 2 * np.einsum("rg, rg -> g", rho1[1:], rho2[1:], optimize=True)
    r1r2 = rho1[0] * rho2[0]
    r1s2 = rho1[0] * sigma02
    s1r2 = sigma01 * rho2[0]
    s1s2 = sigma01 * sigma02

    wv = np.zeros((4, frr.size))
    wv1_tmp = np.zeros(frr.size)

    wv[0] += frrr * r1r2
    wv[0] += frrg * r1s2
    wv[0] += frrg * s1r2
    wv[0] += frgg * s1s2
    wv[0] += frg * sigma12

    wv1_tmp += frrg * r1r2
    wv1_tmp += frgg * r1s2
    wv1_tmp += frgg * s1r2
    wv1_tmp += fggg * s1s2
    wv1_tmp += fgg * sigma12
    wv[1:] += wv1_tmp * rho0[1:]

    wv[1:] += frg * rho1[0] * rho2[1:]
    wv[1:] += frg * rho2[0] * rho1[1:]
    wv[1:] += fgg * sigma01 * rho2[1:]
    wv[1:] += fgg * sigma02 * rho1[1:]

    wv[0] *= 0.5
    wv[1:] *= 2

    wv *= weight
    return wv


class Polar(RDHRespMixin):

    def __init__(self, method):
        self.__dict__.update(method.__dict__)
        self.base = method
        self.pol_scf = None
        self.pol_corr = None
        self.pol_tot = None
        self.de = None

    @property
    def nprop(self):
        return self.tensors["H_1_ao"].shape[0]

    def prepare_U_1(self):
        tensors = self.tensors
        sv, so = self.sv, self.so

        H_1_mo = tensors.load("H_1_mo")
        U_1_vo = self.solve_cpks(H_1_mo[:, sv, so])
        U_1 = np.zeros_like(H_1_mo)
        U_1[:, sv, so] = U_1_vo
        U_1[:, so, sv] = - U_1_vo.swapaxes(-1, -2)
        tensors.create("U_1", U_1)
        return self

    def prepare_dms(self):
        tensors = self.tensors
        U_1 = tensors.load("U_1")
        D_r = tensors.load("D_r")
        rho = tensors.load("rho")
        C, Co = self.mo_coeff, self.Co
        so = self.so
        mol, grids, xc = self.mol, self.grids, self.xc
        # ni = dft.numint.NumInt()  # intended not to use self.ni, and xcfun as engine
        # ni.libxc = dft.xcfun
        ni = self.ni
        dmU = C @ U_1[:, :, so] @ Co.T
        dmU += dmU.swapaxes(-1, -2)
        dmR = C @ D_r @ C.T
        dmR += dmR.swapaxes(-1, -2)
        dmX = np.concatenate([dmU, [dmR]])
        rhoX = get_rho_from_dm_gga(ni, mol, grids, dmX)
        _, _, _, kxc = ni.eval_xc(xc, rho, spin=0, deriv=3)
        tensors.create("rhoU", rhoX[:-1])
        tensors.create("rhoR", rhoX[-1])
        tensors.create("kxc" + xc, kxc)
        return self

    def prepare_pdA_F_0_mo(self):
        tensors = self.tensors
        so, sa = self.so, self.sa

        pdA_F_0_mo = tensors.load("H_1_mo").copy()
        U_1 = tensors.load("U_1")

        pdA_F_0_mo += einsum("Apq, p -> Apq", U_1, self.mo_energy)
        pdA_F_0_mo += einsum("Aqp, q -> Apq", U_1, self.mo_energy)
        pdA_F_0_mo += self.Ax0_Core(sa, sa, sa, so)(U_1[:, :, so])
        tensors.create("pdA_F_0_mo", pdA_F_0_mo)

        if self.xc_n:
            F_0_mo_n = einsum("up, uv, vq -> pq", self.mo_coeff, self.mf_n.get_fock(dm=self.D), self.mo_coeff)
            pdA_F_0_mo_n = np.array(tensors.load("H_1_mo"))
            pdA_F_0_mo_n += einsum("Amp, mq -> Apq", U_1, F_0_mo_n)
            pdA_F_0_mo_n += einsum("Amq, pm -> Apq", U_1, F_0_mo_n)
            pdA_F_0_mo_n += self.Ax0_Core(sa, sa, sa, so, xc=self.xc_n)(U_1[:, :, so])
            tensors.create("pdA_F_0_mo_n", pdA_F_0_mo_n)
        return self

    def prepare_pdA_Y_ia_ri(self):
        tensors = self.tensors
        U_1 = tensors.load("U_1")
        Y_mo_ri = tensors["Y_mo_ri"]
        nocc, nvir, nmo, naux = self.nocc, self.nvir, self.nmo, self.df_ri.get_naoaux()
        so, sv = self.so, self.sv
        nprop = self.nprop

        pdA_Y_ia_ri = np.zeros((nprop, naux, nocc, nvir))
        nbatch = self.base.calc_batch_size(8 * nmo**2, U_1.size)
        for saux in gen_batch(0, naux, nbatch):
            pdA_Y_ia_ri[:, saux] = (
                + einsum("Ami, Pma -> APia", U_1[:, :, so], Y_mo_ri[saux, :, sv])
                + einsum("Ama, Pmi -> APia", U_1[:, :, sv], Y_mo_ri[saux, :, so]))
        tensors.create("pdA_Y_ia_ri", pdA_Y_ia_ri)
        return self

    def prepare_pt2_deriv(self):
        tensors = self.tensors
        nocc, nvir, nmo, naux = self.nocc, self.nvir, self.nmo, self.df_ri.get_naoaux()
        so, sv = self.so, self.sv
        eo, ev = self.eo, self.ev
        nprop = self.nprop

        pdA_D_rdm1 = tensors.create("pdA_D_rdm1", shape=(nprop, nmo, nmo))
        if not self.base.eval_pt2:
            return self

        pdA_F_0_mo = tensors.load("pdA_F_0_mo")
        Y_ia_ri = np.asarray(tensors["Y_mo_ri"][:, so, sv])
        pdA_Y_ia_ri = tensors["pdA_Y_ia_ri"]

        pdA_G_ia_ri = tensors.create("pdA_G_ia_ri", shape=(nprop, naux, nocc, nvir))

        nbatch = self.base.calc_batch_size(8*nocc*nvir**2, Y_ia_ri.size + pdA_F_0_mo.size + pdA_Y_ia_ri.size)
        D_jab = eo[None, :, None, None] - ev[None, None, :, None] - ev[None, None, None, :]
        for sI in gen_batch(0, nocc, nbatch):
            t_ijab = np.asarray(tensors["t_ijab"][sI])
            D_ijab = eo[sI, None, None, None] + D_jab

            pdA_t_ijab = einsum("APia, Pjb -> Aijab", pdA_Y_ia_ri[:, :, sI], Y_ia_ri)
            pdA_t_ijab += einsum("APjb, Pia -> Aijab", pdA_Y_ia_ri, Y_ia_ri[:, sI])

            for sK in gen_batch(0, nocc, nbatch):
                t_kjab = t_ijab if sK == sI else tensors["t_ijab"][sK]
                pdA_t_ijab -= einsum("Aki, kjab -> Aijab", pdA_F_0_mo[:, sK, sI], t_kjab)
            pdA_t_ijab -= einsum("Akj, ikab -> Aijab", pdA_F_0_mo[:, so, so], t_ijab)
            pdA_t_ijab += einsum("Acb, ijac -> Aijab", pdA_F_0_mo[:, sv, sv], t_ijab)
            pdA_t_ijab += einsum("Aca, ijcb -> Aijab", pdA_F_0_mo[:, sv, sv], t_ijab)
            pdA_t_ijab /= D_ijab

            c_os, c_ss = self.c_os, self.c_ss
            # T_ijab = (c_os + c_ss) * t_ijab - c_ss * t_ijab.swapaxes(-1, -2)
            # pdA_T_ijab = (c_os + c_ss) * pdA_t_ijab - c_ss * pdA_t_ijab.swapaxes(-1, -2)
            T_ijab = restricted_biorthogonalize(t_ijab, c_os, c_ss)
            pdA_T_ijab = restricted_biorthogonalize(pdA_t_ijab, c_os, c_ss)

            pdA_G_ia_ri[:, :, sI] += einsum("Aijab, Pjb -> APia", pdA_T_ijab, Y_ia_ri)
            pdA_G_ia_ri[:, :, sI] += einsum("ijab, APjb -> APia", T_ijab, pdA_Y_ia_ri)

            pdA_D_rdm1[:, so, so] -= 2 * einsum("kiab, Akjab -> Aij", T_ijab, pdA_t_ijab)
            pdA_D_rdm1[:, sv, sv] += 2 * einsum("ijac, Aijbc -> Aab", T_ijab, pdA_t_ijab)
        pdA_D_rdm1[:] += pdA_D_rdm1.swapaxes(-1, -2)
        return self

    def prepare_polar_Ax1_gga(self):
        tensors = self.tensors
        nprop = self.nprop

        rho = tensors.load("rho")
        rhoU = tensors.load("rhoU")
        rhoR = tensors.load("rhoR")
        fxc = tensors["fxc" + self.xc]
        kxc = tensors["kxc" + self.xc]

        grids = self.grids

        wv2 = np.empty((nprop, 4, grids.weights.size))
        for i in range(nprop):
            wv2[i] = np.asarray(_rks_gga_wv2(rho, rhoU[i], rhoR, fxc, kxc, grids.weights), dtype=np.float64)
            wv2[i, 0] *= 2
        res = 2 * einsum("Arg, Brg -> AB", rhoU, wv2)
        # --- The following code is old code, transform wv2 to AO basis, then contract U_1 ---
        # U_1 = tensors.load("U_1")
        # nao = self.nao
        # C, Co, so = self.mo_coeff, self.Co, self.so
        # Ax1 = np.zeros((3, nao, nao))
        # ip = 0
        # for ao, mask, weight, _ in ni.block_loop(mol, grids, nao, deriv=1):
        #     sg = slice(ip, ip + weight.size)
        #     for i in range(3):
        #         # v = einsum("rg, rgu, gv -> uv", wv2[i, :, sg], ao, ao[0])
        #         aow = _scale_ao(ao, wv2[i, :, sg])
        #         v = _dot_ao_ao(mol, aow, ao[0], mask, None, None)
        #         Ax1[i] += 2 * (v + v.T)
        #     ip += weight.size
        # res = lib.einsum("Auv, um, vi, Bmi -> AB", Ax1, C, Co, U_1[:, :, so])
        tensors.create("Ax1_contrib", res)
        return self

    def get_SCR3(self):
        tensors = self.tensors
        so, sv = self.so, self.sv
        naux = self.df_ri.get_naoaux()
        nprop = self.nprop

        SCR3 = np.zeros((nprop, self.nvir, self.nocc))
        if self.xc_n:
            pdA_F_0_mo_n = tensors.load("pdA_F_0_mo_n")
            SCR3 += 4 * pdA_F_0_mo_n[:, sv, so]
        if not self.base.eval_pt2:
            return SCR3

        U_1 = tensors.load("U_1")
        G_ia_ri = tensors.load("G_ia_ri")
        pdA_G_ia_ri = tensors.load("pdA_G_ia_ri")
        Y_mo_ri = tensors["Y_mo_ri"]

        nbatch = self.base.calc_batch_size(10 * self.nmo**2, G_ia_ri.size + pdA_G_ia_ri.size)
        for saux in gen_batch(0, naux, nbatch):
            G_blk = G_ia_ri[saux]
            Y_blk = np.asarray(Y_mo_ri[saux])
            pdA_G_blk = np.asarray(pdA_G_ia_ri[:, saux])
            # pdA_Y_ij part
            pdA_Y_blk = einsum("Ami, Pmj -> APij", U_1[:, :, so], Y_blk[:, :, so])
            pdA_Y_blk += pdA_Y_blk.swapaxes(-1, -2)
            SCR3 -= 4 * einsum("APja, Pij -> Aai", pdA_G_blk, Y_blk[:, so, so])
            SCR3 -= 4 * einsum("Pja, APij -> Aai", G_blk, pdA_Y_blk)
            # pdA_Y_ab part
            pdA_Y_blk = einsum("Ama, Pmb -> APab", U_1[:, :, sv], Y_blk[:, :, sv])
            pdA_Y_blk += pdA_Y_blk.swapaxes(-1, -2)
            SCR3 += 4 * einsum("APib, Pab -> Aai", pdA_G_blk, Y_blk[:, sv, sv])
            SCR3 += 4 * einsum("Pib, APab -> Aai", G_blk, pdA_Y_blk)

        return SCR3

    def prepare_polar(self):
        tensors = self.tensors
        so, sv, sa = self.so, self.sv, self.sa

        H_1_mo = tensors.load("H_1_mo")
        U_1 = tensors.load("U_1")
        pdA_F_0_mo = tensors.load("pdA_F_0_mo")
        D_r = tensors.load("D_r")
        pdA_D_rdm1 = tensors.load("pdA_D_rdm1")

        # SCR1 = self.Ax0_Core(sa, sa, sa, sa)(D_r)
        SCR1 = self.Ax0_Core_resp(sa, sa, sa, sa)(D_r)  # resp is faster in this case
        SCR2 = H_1_mo + self.Ax0_Core(sa, sa, sv, so)(U_1[:, sv, so])
        SCR3 = self.get_SCR3()

        pol_scf = - 4 * einsum("Api, Bpi -> AB", H_1_mo[:, :, so], U_1[:, :, so])
        pol_corr = - (
            + einsum("Aai, Bma, mi -> AB", U_1[:, sv, so], U_1[:, :, sv], SCR1[:, so])
            + einsum("Aai, Bmi, ma -> AB", U_1[:, sv, so], U_1[:, :, so], SCR1[:, sv])
            + einsum("Apm, Bmq, pq -> AB", SCR2, U_1, D_r)
            + einsum("Amq, Bmp, pq -> AB", SCR2, U_1, D_r)
            + einsum("Apq, Bpq -> AB", SCR2, pdA_D_rdm1)
            + einsum("Bai, Aai -> AB", SCR3, U_1[:, sv, so])
            - einsum("Bki, Aai, ak -> AB", pdA_F_0_mo[:, so, so], U_1[:, sv, so], D_r[sv, so])
            + einsum("Bca, Aai, ci -> AB", pdA_F_0_mo[:, sv, sv], U_1[:, sv, so], D_r[sv, so]))
        if not xc_equal(self.xc, "HF"):
            pol_corr -= tensors.load("Ax1_contrib")

        self.pol_scf = pol_scf
        self.pol_corr = pol_corr
        self.de = self.pol_tot = pol_scf + pol_corr
        return self

    def base_method(self):
        return self.base

    kernel = kernel
