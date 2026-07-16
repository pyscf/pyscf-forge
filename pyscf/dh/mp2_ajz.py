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
#          Shirong Wang <srwang20@fudan.edu.cn>
#

from pyscf import lib
from pyscf.ao2mo import _ao2mo
from pyscf.dh.dhutil import calc_batch_size, gen_batch, timing, tot_size
import numpy as np
einsum = lib.einsum


def _loop_t_ijab(mf, Y_ia_ri, mo_energy, nocc, nvir, callback):
    so = slice(0, nocc)
    sv = slice(nocc, nocc + nvir)
    D_jab = mo_energy[so, None, None] - mo_energy[None, sv, None] - mo_energy[None, None, sv]
    nbatch = mf.calc_batch_size(2 * nocc * nvir ** 2, Y_ia_ri.size + D_jab.size)
    for sI in gen_batch(0, nocc, nbatch):
        D_ijab = mo_energy[sI, None, None, None] + D_jab
        g_ijab = einsum("Pia, Pjb -> ijab", Y_ia_ri[:, sI], Y_ia_ri)
        t_ijab = g_ijab / D_ijab
        callback(sI, t_ijab, g_ijab)


@timing
def get_cderi_mo(dfobj, C, Y_mo=None, pqslice=None, max_memory=2000):
    mol = dfobj.mol
    naux = dfobj.get_naoaux()
    nao = mol.nao
    if pqslice is None:
        pqslice = (0, nao, 0, nao)
        nump, numq = nao, nao
    else:
        nump, numq = pqslice[1] - pqslice[0], pqslice[3] - pqslice[2]
    if Y_mo is None:
        Y_mo = np.empty((naux, nump, numq))

    def save(r0, r1, buf):
        Y_mo[r0:r1] = buf.reshape(r1 - r0, nump, numq)

    p0, p1 = 0, 0
    preflop = 0 if not isinstance(Y_mo, np.ndarray) else Y_mo.size
    nbatch = calc_batch_size(2 * nump * numq, max_memory, preflop)
    with lib.call_in_background(save) as bsave:
        for Y_ao in dfobj.loop(nbatch):
            p1 = p0 + Y_ao.shape[0]
            Y_mo_buf = _ao2mo.nr_e2(Y_ao, C, pqslice, aosym="s2", mosym="s1")
            bsave(p0, p1, Y_mo_buf)
            p0 = p1
    return Y_mo


@timing
def energy_elec_mp2_ajz(mf, mo_coeff=None, mo_energy=None, dfobj=None,
                         Y_ia_ri=None, t_ijab_blk=None, eval_ss=True, **kwargs):
    if mf.frozen:
        raise NotImplementedError(
            "Frozen core is not supported by the AJZ MP2 backend. "
            "Use mp2_backend='dfmp2' or 'dfmp2_native'."
        )
    if mo_coeff is None:
        if mf.mf_s.e_tot == 0:
            mf.run_scf()
        mo_coeff = mf.mo_coeff
    if mo_energy is None:
        if mf.mf_s.e_tot == 0:
            mf.run_scf()
        mo_energy = mf.mo_energy
    if Y_ia_ri is None:
        nmo = mo_coeff.shape[1]
        nocc = mf.nocc
        nvir = nmo - nocc
    else:
        nocc, nvir = Y_ia_ri.shape[1:]
        nmo = nocc + nvir
    iaslice = (0, nocc, nocc, nmo)
    if Y_ia_ri is None:
        if dfobj is None:
            dfobj = mf.df_ri
        Y_ia_ri = get_cderi_mo(dfobj, mo_coeff, pqslice=iaslice, max_memory=mf.get_memory())
    eng_bi1 = [0]
    eng_bi2 = [0]

    def accumulate(sI, t_ijab, g_ijab):
        eng_bi1[0] += einsum("ijab, ijab ->", t_ijab, g_ijab)
        if eval_ss:
            eng_bi2[0] += einsum("ijab, ijba ->", t_ijab, g_ijab)
        if t_ijab_blk:
            t_ijab_blk[sI] = t_ijab

    _loop_t_ijab(mf, Y_ia_ri, mo_energy, nocc, nvir, accumulate)
    return None, eng_bi1[0], eng_bi2[0]


@timing
def energy_elec_ump2_ajz(mf, mo_coeff=None, mo_energy=None, dfobj=None,
                           Y_ia_ri=None, t_ijab_blk=None, eval_ss=True, **_):
    if mf.frozen:
        raise NotImplementedError(
            "Frozen core is not supported by the AJZ MP2 backend. "
            "Use mp2_backend='dfmp2' or 'dfmp2_native'."
        )
    α, β = 0, 1
    αα, αβ, ββ = 0, 1, 2
    if mo_coeff is None:
        if mf.mf_s.e_tot == 0:
            mf.run_scf()
        mo_coeff = mf.mo_coeff
    if mo_energy is None:
        if mf.mf_s.e_tot == 0:
            mf.run_scf()
        mo_energy = mf.mo_energy
    if Y_ia_ri is None:
        nmo = mo_coeff.shape[-1]
        nocc = mf.nocc
        nvir = nmo - nocc[α], nmo - nocc[β]
    else:
        nocc = Y_ia_ri[α].shape[1], Y_ia_ri[β].shape[1]
        nvir = Y_ia_ri[α].shape[2], Y_ia_ri[β].shape[2]
        nmo = nocc[α] + nvir[α]
    so = slice(0, nocc[α]), slice(0, nocc[β])
    sv = slice(nocc[α], nmo), slice(nocc[β], nmo)
    eo = mo_energy[α, so[α]], mo_energy[β, so[β]]
    ev = mo_energy[α, sv[α]], mo_energy[β, sv[β]]
    iaslice = (0, nocc[α], nocc[α], nmo), (0, nocc[β], nocc[β], nmo)
    if Y_ia_ri is None:
        if dfobj is None:
            dfobj = mf.df_ri
        Y_ia_ri = [get_cderi_mo(dfobj, mo_coeff[σ], pqslice=iaslice[σ], max_memory=mf.get_memory()) for σ in (α, β)]
    eng_bi1, eng_bi2 = [0, 0, 0], [0, 0, 0]
    mocc, mvir = max(nocc), max(nvir)
    nbatch = mf.calc_batch_size(2 * mocc * mvir ** 2, tot_size(Y_ia_ri) + mocc * mvir ** 2)
    for σς, σ, ς in (αα, α, α), (αβ, α, β), (ββ, β, β):
        D_jab = eo[ς][:, None, None] - ev[σ][None, :, None] - ev[ς][None, None, :]
        for sI in gen_batch(0, nocc[σ], nbatch):
            if σς == αβ or eval_ss:
                D_ijab = eo[σ][sI, None, None, None] + D_jab
                g_ijab = einsum("Pia, Pjb -> ijab", Y_ia_ri[σ][:, sI], Y_ia_ri[ς])
                t_ijab = g_ijab / D_ijab
                eng_bi1[σς] += einsum("ijab, ijab ->", t_ijab, g_ijab)
                if t_ijab_blk:
                    t_ijab_blk[σς][sI] = t_ijab
                if σς in (αα, ββ):
                    eng_bi2[σς] += einsum("ijab, ijba ->", t_ijab, g_ijab)
    return None, tuple(eng_bi1), tuple(eng_bi2)
