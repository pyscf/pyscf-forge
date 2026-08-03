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

# dh import
from pyscf.dh.dh import DHBase
from pyscf.dh.dhutil import timing
from pyscf.dh.xccode import xc_equal
from pyscf.dh.mp2_ajz import energy_elec_ump2_ajz
from pyscf.dh.dh import energy_elec_mp2_dfump2_native, energy_elec_mp2_dfump2
# pyscf import
from pyscf import lib, dft
# other import
import h5py
import numpy as np
from functools import partial

einsum = lib.einsum
α, β = 0, 1
αα, αβ, ββ = 0, 1, 2
ndarray = np.ndarray or h5py.Dataset


@timing
def energy_elec_pt2(mf, params=None, eng_bi=None, **kwargs):
    c_os, c_ss = params if params else mf.c_os, mf.c_ss
    emp2_0, eng_bi1, eng_bi2 = eng_bi if eng_bi else mf.energy_elec_mp2(eval_ss=mf.eval_ss, **kwargs)
    if getattr(mf, 'mp2_backend', None) == "dfmp2_native":
        return emp2_0, None, None
    if getattr(mf, 'mp2_backend', None) == "dfmp2":
        return c_os * eng_bi1 + c_ss * eng_bi2, eng_bi1, eng_bi2
    eng_os = eng_bi1[αβ]
    eng_ss = 0.5 * (eng_bi1[αα] + eng_bi1[ββ] - eng_bi2[αα] - eng_bi2[ββ])
    eng_pt2 = c_os * eng_os + c_ss * eng_ss
    return eng_pt2, eng_os, eng_ss


def energy_elec(mf, params=None, **kwargs):
    eng_nc = mf.energy_elec_nc(**kwargs)[0]
    nocc, nvir = mf.nocc, mf.nvir
    _, c_ss = params if params else mf.c_os, mf.c_ss
    eval_ss = True if abs(c_ss) > 1e-7 else False
    t_ijab_blk = None
    if mf.with_t_ijab:
        t_ijab_blk = [0, 0, 0]
        for σς, σ, ς in (αα, α, α), (αβ, α, β), (ββ, β, β):
            if σς == αβ or eval_ss:
                t_ijab_blk[σς] = mf.tensors.create("t_ijab" + str(σς), shape=(nocc[σ], nocc[ς], nvir[σ], nvir[ς]), incore=mf._incore_t_ijab)
    eng_pt2, eng_os, eng_ss = energy_elec_pt2(mf, t2_blk=t_ijab_blk, **kwargs)
    eng_elec = eng_nc + eng_pt2
    return eng_elec, eng_nc, eng_pt2, eng_os, eng_ss


class UDFDH(DHBase):
    def __init__(self,
                 mf_or_mol,
                 xc: str or tuple = "XYG3",
                 auxbasis_jk: str or dict or None = None,
                 auxbasis_ri: str or dict or None = None,
                 grids: dft.Grids = None,
                 grids_cpks: dft.Grids = None,
                 mp2_backend: str = "ajz",
                 frozen: int = None,
                 ):
        super().__init__(mf_or_mol, xc, auxbasis_jk, auxbasis_ri, mp2_backend, frozen)
        mol = self.mol
        if self._scf is not None:
            mf_s = self._scf
            if hasattr(mf_s, 'xc'):
                mf_s.xc = self.xc
        else:
            mf_s = dft.UKS(mol, xc=self.xc).density_fit(auxbasis=self.auxbasis_jk)
        self.grids = grids if grids else (getattr(mf_s, 'grids', dft.Grids(mol)))
        self.grids_cpks = grids_cpks if grids_cpks else self.grids
        self.mf_s = mf_s
        self.xc_n = None if xc_equal(self.xc_n, self.xc) else self.xc_n
        self.mf_n = self.mf_s
        self.nocc = mol.nelec
        self.mocc = max(max(self.nocc), 1)
        self.nmo = self.nao
        self.nvir = (self.nmo - self.nocc[α], self.nmo - self.nocc[β])
        if self.xc_n:
            self.mf_n = self.mf_s.copy()
            self.mf_n.xc = self.xc_n
            self.mf_n._numint = dft.numint.NumInt()
        self._init_common()
        if mp2_backend == "dfmp2_native":
            self.energy_elec_mp2 = partial(energy_elec_mp2_dfump2_native, self)
        elif mp2_backend == "dfmp2":
            self.energy_elec_mp2 = partial(energy_elec_mp2_dfump2, self)
        else:
            self.energy_elec_mp2 = partial(energy_elec_ump2_ajz, self)

    @timing
    def run_scf(self, **kwargs):
        self.mf_s.grids = self.mf_n.grids = self.grids
        self.build()
        mf = self.mf_s
        if mf.e_tot == 0:
            mf.kernel(**kwargs)
        C = self.mo_coeff = mf.mo_coeff
        e = self.mo_energy = mf.mo_energy
        self.mo_occ = mf.mo_occ
        self.D = mf.make_rdm1(mf.mo_coeff)
        nocc = self.nocc
        nmo = self.nmo = self.mo_coeff.shape[-1]
        self.nvir = nmo - nocc[α], nmo - nocc[β]
        self.mvir = max(max(self.nvir), 1)
        so = self.so = slice(0, nocc[α]), slice(0, nocc[β])
        sv = self.sv = slice(nocc[α], nmo), slice(nocc[β], nmo)
        self.sa = slice(0, nmo), slice(0, nmo)
        self.Co = C[α, :, so[α]], C[β, :, so[β]]
        self.Cv = C[α, :, sv[α]], C[β, :, sv[β]]
        self.eo = e[α, so[α]], e[β, so[β]]
        self.ev = e[α, sv[α]], e[β, sv[β]]
        return self


    def nuc_grad_method(self):
        from pyscf.dh.grad.udfdh import Gradients
        return Gradients(self)

    def polar_method(self):
        from pyscf.dh.polar.udfdh import Polar
        return Polar(self)

    energy_elec_pt2 = energy_elec_pt2
    energy_elec_mp2 = energy_elec_ump2_ajz
    energy_elec = energy_elec

