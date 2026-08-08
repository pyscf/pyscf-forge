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

from pyscf import lib, gto, dft, df
from pyscf.dh.dhutil import calc_batch_size, timing, HybridDict
from pyscf.dispersion.dftd3 import DFTD3Dispersion
from pyscf.dh.xccode import parse_xc_dh, xc_equal


class DHBase(lib.StreamObject):

    def __init__(self,
                 mf_or_mol,
                 xc: str or tuple = "XYG3",
                 auxbasis_jk: str or dict or None = None,
                 auxbasis_ri: str or dict or None = None,
                 mp2_backend: str = "ajz",
                 frozen: int = None,
                 ):
        if isinstance(mf_or_mol, gto.Mole):
            mol = mf_or_mol
            self._scf = None
        else:
            mol = mf_or_mol.mol
            self._scf = mf_or_mol
        self.with_t_ijab = False
        self._incore_t_ijab = False
        self.max_memory = mol.max_memory
        self.mp2_backend = mp2_backend
        self.frozen = frozen
        self.xc_dh = xc
        if isinstance(xc, str):
            xc_list, xc_add = parse_xc_dh(xc)
        elif isinstance(xc, (tuple, list)) and len(xc) == 2 and isinstance(xc[0], str):
            xc_list, xc_add = parse_xc_dh(xc)
        elif len(xc) == 5:
            import warnings
            warnings.warn(
                "5-tuple XC format is deprecated. "
                "Use a string name, a code string, or a 2-tuple (code_scf, code_eng) for xDH.",
                FutureWarning
            )
            xc_list = xc[0], xc[1], xc[2] * xc[3], xc[2] * xc[4]
            xc_add = {}
        elif len(xc) == 4:
            xc_list = xc
            xc_add = {}
        else:
            xc_list, xc_add = xc
        self.xc, self.xc_n, self.c_os, self.c_ss = xc_list
        self.xc_add = xc_add
        if self._scf is not None:
            if not hasattr(self._scf, 'xc'):
                raise TypeError(
                    "DFDH(mf) requires a dft.KS or dft.UKS object; "
                    "scf.HF is not supported."
                )
            if not hasattr(self._scf, 'with_df'):
                raise TypeError(
                    "DFDH(mf) requires density-fitting (with_df). "
                    "Call mf.density_fit() before passing to DFDH."
                )
            if not xc_equal(self._scf.xc, self.xc):
                raise ValueError(
                    f"SCF functional '{self._scf.xc}' does not match "
                    f"DH SCF functional '{self.xc}'. "
                    "Use a SCF converged with the same functional."
                )
        self.auxbasis_jk = auxbasis_jk = auxbasis_jk if auxbasis_jk else df.make_auxbasis(mol, mp2fit=False)
        self.auxbasis_ri = auxbasis_ri = auxbasis_ri if auxbasis_ri else df.make_auxbasis(mol, mp2fit=True)
        self.same_aux = bool(auxbasis_jk == auxbasis_ri or auxbasis_ri is None)
        self.ni = None
        self.cx = None
        self.cx_n = None
        self.mol = mol
        self.nao = mol.nao
        self.tensors = HybridDict()
        self.mo_coeff = None
        self.mo_energy = None
        self.mo_occ = None
        self.Co = self.Cv = None
        self.eo = self.ev = None
        self.D = None
        self.nmo = self.nvir = None
        self.so = self.sv = self.sa = None
        self.e_tot = None
        self.eng_tot = self.eng_nc = self.eng_pt2 = self.eng_nuc = self.eng_os = self.eng_ss = None

    def _init_common(self):
        """Shared initialization after subclass creates mf_s / mf_n."""
        mf_s = self.mf_s
        mol = self.mol
        if not hasattr(self, 'grids') or self.grids is None:
            self.grids = getattr(mf_s, 'grids', dft.Grids(mol))
        if not hasattr(self, 'grids_cpks') or self.grids_cpks is None:
            self.grids_cpks = self.grids
        self.mf_s.grids = self.grids
        self.ni = getattr(mf_s, '_numint', dft.numint.NumInt())
        self.cx = self.ni.hybrid_coeff(self.xc)
        self.cx_n = self.ni.hybrid_coeff(self.xc_n)
        self.df_jk = mf_s.with_df
        self.df_ri = df.DF(mol, self.auxbasis_ri) if not self.same_aux else self.df_jk
        if self._scf is not None and self._scf.e_tot != 0:
            self.run_scf()
        self.base = self

    @property
    def converged(self):
        return self.mf_s.converged

    def get_memory(self):
        return max(self.max_memory - lib.current_memory()[0], 500)

    def calc_batch_size(self, unit_flop, pre_flop=0, fixed_mem=None):
        if fixed_mem:
            return calc_batch_size(unit_flop, fixed_mem, pre_flop)
        else:
            return calc_batch_size(unit_flop, self.get_memory(), pre_flop)

    @property
    def eval_ss(self):
        return abs(self.c_ss) > 1e-7

    @property
    def eval_os(self):
        return abs(self.c_os) > 1e-7

    @property
    def eval_pt2(self):
        return self.eval_ss or self.eval_os

    def __del__(self):
        try:
            self.tensors.close()
        except Exception:
            pass

    @timing
    def energy_elec_nc(self, mo_coeff=None, h1e=None, vhf=None, **_):
        if mo_coeff is None:
            if self.mf_s.e_tot == 0:
                self.run_scf()
                if self.xc_n is None:
                    return self.mf_s.e_tot - self.mf_s.energy_nuc(), None
            mo_coeff = self.mf_s.mo_coeff
        mo_occ = self.mf_s.mo_occ
        if mo_occ is None:
            mo_occ = self.mf_s.get_occ()
        dm = self.mf_s.make_rdm1(mo_coeff, mo_occ)
        dm = lib.tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
        eng_nc = self.mf_n.energy_elec(dm=dm, h1e=h1e, vhf=vhf)
        return eng_nc

    def energy_nuc(self, **_):
        mol = self.mol
        eng_nuc = mol.energy_nuc()
        if "D3" in self.xc_add:
            d3_info = self.xc_add["D3"]
            model = DFTD3Dispersion(mol, xc=d3_info["xc"], version=d3_info["version"])
            eng_nuc += model.get_dispersion()["energy"]
        if "D4" in self.xc_add:
            from pyscf.dispersion.dftd4 import DFTD4Dispersion
            d4_info = self.xc_add["D4"]
            model = DFTD4Dispersion(mol, xc=d4_info["xc"], version=d4_info["version"])
            eng_nuc += model.get_dispersion()["energy"]
        return eng_nuc

    def energy_tot(self, **kwargs):
        eng_elec, eng_nc, eng_pt2, eng_os, eng_ss = self.energy_elec(**kwargs)
        eng_nuc = self.energy_nuc()
        eng_tot = eng_elec + eng_nuc
        return eng_tot, eng_nc, eng_pt2, eng_nuc, eng_os, eng_ss

    def kernel(self, **kwargs):
        self.build()
        eng_tot, eng_nc, eng_pt2, eng_nuc, eng_os, eng_ss = self.energy_tot(**kwargs)
        self.e_tot = self.eng_tot = eng_tot
        self.eng_nc = eng_nc
        self.eng_pt2 = eng_pt2
        self.eng_nuc = eng_nuc
        self.eng_os = eng_os
        self.eng_ss = eng_ss
        return eng_tot

    @timing
    def build(self):
        self.mf_s.grids = self.mf_n.grids = self.grids
        if self.df_jk.auxmol is None:
            self.df_jk.build()
        if self.df_ri.auxmol is None:
            self.df_ri.build()
        return self


@timing
def energy_elec_mp2_dfmp2_native(mf, **kwargs):
    from pyscf.mp.dfmp2_native import DFRMP2
    mp2 = DFRMP2(mf.mf_s, frozen=mf.frozen)
    mp2.ps = mf.c_os
    mp2.pt = mf.c_ss
    emp2 = mp2.kernel()
    return emp2, None, None


@timing
def energy_elec_mp2_dfump2_native(mf, **kwargs):
    from pyscf.mp.dfump2_native import DFUMP2
    mp2 = DFUMP2(mf.mf_s)
    mp2.ps = mf.c_os
    mp2.pt = mf.c_ss
    emp2 = mp2.kernel()
    return emp2, None, None


@timing
def energy_elec_mp2_dfmp2(mf, **kwargs):
    from pyscf.mp.dfmp2 import DFRMP2
    mp2 = DFRMP2(mf.mf_s, frozen=mf.frozen)
    mp2.kernel()
    return None, mp2.e_corr_os, mp2.e_corr_ss


@timing
def energy_elec_mp2_dfump2(mf, **kwargs):
    from pyscf.mp.dfump2 import DFUMP2
    mp2 = DFUMP2(mf.mf_s, frozen=mf.frozen)
    mp2.kernel()
    return None, mp2.e_corr_os, mp2.e_corr_ss


def to_dh(mf, xc="XYG3", **kwargs):
    from pyscf.dh import rdfdh, udfdh

    xc_list, _ = parse_xc_dh(xc)
    dh_xc = xc_list[0]

    can_reuse = (
        hasattr(mf, 'xc')
        and xc_equal(mf.xc, dh_xc)
        and hasattr(mf, 'e_tot')
        and mf.e_tot != 0
        and getattr(mf, 'with_df', None) is not None
        and mf.converged
    )

    if not can_reuse:
        mol = mf.mol
        if not mf.istype('RHF'):
            mf = dft.UKS(mol, xc=dh_xc).density_fit()
        else:
            mf = dft.KS(mol, xc=dh_xc).density_fit()
        mf.kernel()

    if mf.istype('RHF'):
        return rdfdh.RDFDH(mf, xc, **kwargs)
    return udfdh.UDFDH(mf, xc, **kwargs)
