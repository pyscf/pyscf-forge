from pyscf import lib, gto, dft, df
from pyscf.dh.dhutil import parse_xc_dh, calc_batch_size, timing, HybridDict, get_rho_from_dm_gga
import os
import pickle
import numpy as np


class DHBase(lib.StreamObject):

    def __init__(self,
                 mol: gto.Mole,
                 xc: str or tuple = "XYG3",
                 auxbasis_jk: str or dict or None = None,
                 auxbasis_ri: str or dict or None = None,
                 mp2_backend: str = "ajz",
                 frozen: int = None,
                 ):
        self.with_t_ijab = False
        self._incore_t_ijab = False
        self._incore_Y_mo = False
        self._incore_eri_cpks = False
        self._fixed_batch = False
        self.cpks_tol = 1e-8
        self.cpks_cyc = 100
        self.max_memory = mol.max_memory
        self.mp2_backend = mp2_backend
        self.frozen = frozen
        self.xc_dh = xc
        if isinstance(xc, str):
            xc_list, xc_add = parse_xc_dh(xc)
        elif len(xc) == 5:
            xc_list = xc
            xc_add = {}
        else:
            xc_list, xc_add = xc
        self.xc, self.xc_n, self.cc, self.c_os, self.c_ss = xc_list
        self.xc_add = xc_add
        self.auxbasis_jk = auxbasis_jk = auxbasis_jk if auxbasis_jk else df.make_auxbasis(mol, mp2fit=False)
        self.auxbasis_ri = auxbasis_ri = auxbasis_ri if auxbasis_ri else df.make_auxbasis(mol, mp2fit=True)
        self.same_aux = bool(auxbasis_jk == auxbasis_ri or auxbasis_ri is None)
        self.ni = NotImplemented
        self.cx = NotImplemented
        self.cx_n = NotImplemented
        self.mol = mol
        self.nao = mol.nao
        self.tensors = HybridDict()
        self.mo_coeff = NotImplemented
        self.mo_energy = NotImplemented
        self.mo_occ = NotImplemented
        self.Co = self.Cv = NotImplemented
        self.eo = self.ev = NotImplemented
        self.D = NotImplemented
        self.nmo = self.nvir = NotImplemented
        self.so = self.sv = self.sa = NotImplemented
        self.e_tot = NotImplemented
        self.eng_tot = self.eng_nc = self.eng_pt2 = self.eng_nuc = self.eng_os = self.eng_ss = NotImplemented

    @property
    def base(self):
        return self

    @property
    def converged(self):
        return self.mf_s.converged

    def get_memory(self):
        return max(self.max_memory - lib.current_memory()[0], 500)

    def calc_batch_size(self, unit_flop, pre_flop=0, fixed_mem=None):
        if self._fixed_batch:
            return self._fixed_batch
        if fixed_mem:
            return calc_batch_size(unit_flop, fixed_mem, pre_flop)
        else:
            return calc_batch_size(unit_flop, self.get_memory(), pre_flop)

    @property
    def eval_ss(self):
        return abs(self.cc * self.c_ss) > 1e-7

    @property
    def eval_os(self):
        return abs(self.cc * self.c_os) > 1e-7

    @property
    def eval_pt2(self):
        return self.eval_ss or self.eval_os

    @timing
    def build(self):
        self.mf_s.grids = self.mf_n.grids = self.grids
        if self.df_jk.auxmol is None:
            self.df_jk.build()
            self.aux_jk = self.df_jk.auxmol
        if self.df_ri.auxmol is None:
            self.df_ri.build()
            self.aux_ri = self.df_ri.auxmol

    @timing
    def prepare_xc_kernel(self):
        mol = self.mol
        tensors = self.tensors
        ni = self.ni
        spin = len(self.D.shape) - 2
        if "rho" in tensors:
            return self
        if ni._xc_type(self.xc) == "GGA":
            rho = get_rho_from_dm_gga(ni, mol, self.grids, self.D)
            _, vxc, fxc, _ = ni.eval_xc(self.xc, rho, spin=spin, deriv=2)
            tensors.create("rho", rho)
            tensors.create("vxc" + self.xc, vxc)
            tensors.create("fxc" + self.xc, fxc)
            rho = get_rho_from_dm_gga(ni, mol, self.grids_cpks, self.D)
            _, vxc, fxc, _ = ni.eval_xc(self.xc, rho, spin=spin, deriv=2)
            tensors.create("rho" + "in cpks", rho)
            tensors.create("vxc" + self.xc + "in cpks", vxc)
            tensors.create("fxc" + self.xc + "in cpks", fxc)
        if self.xc_n and ni._xc_type(self.xc_n) == "GGA":
            if "rho" in tensors:
                vxc, fxc = ni.eval_xc(self.xc_n, tensors["rho"], deriv=2, verbose=0, spin=spin)[1:3]
                tensors.create("vxc" + self.xc_n, vxc)
                tensors.create("fxc" + self.xc_n, fxc)
            else:
                rho = get_rho_from_dm_gga(ni, mol, self.grids_cpks, self.D)
                _, vxc, fxc, _ = ni.eval_xc(self.xc_n, rho, spin=spin, deriv=2)
                tensors.create("rho", rho)
                tensors.create("vxc" + self.xc_n, vxc)
                tensors.create("fxc" + self.xc_n, fxc)
        return self

    def dump_intermediates(self, dir_path="scratch"):
        os.makedirs(dir_path, exist_ok=True)
        tensors = self.tensors
        h5_path = dir_path + "/tensors.h5"
        dat_path = dir_path + "/tensors.dat"
        tensors.dump(h5_path, dat_path)
        att_path = dir_path + "/attributes.dat"
        dct = {
            "mo_coeff": self.mo_coeff,
            "mo_energy": self.mo_energy,
            "D": self.D,
            "mo_occ": self.mo_occ,
            "mf_s_e_tot": self.mf_s.e_tot,
        }
        with open(att_path, "wb") as f:
            pickle.dump(dct, f)

    def load_intermediates(self, dir_path="scratch", rerun_scf=False):
        h5_path = dir_path + "/tensors.h5"
        dat_path = dir_path + "/tensors.dat"
        self.tensors = HybridDict.pick(h5_path, dat_path)
        att_path = dir_path + "/attributes.dat"
        with open(att_path, "rb") as f:
            dct = pickle.load(f)
        self.mf_s.mo_coeff = dct["mo_coeff"]
        self.mf_s.mo_energy = dct["mo_energy"]
        self.mf_s.mo_occ = dct["mo_occ"]
        self.mf_s.e_tot = dct["mf_s_e_tot"]
        if rerun_scf:
            self.mf_s.kernel(dm=self.mf_s.make_rdm1())
        self.run_scf()
        return self


@timing
def energy_elec_mp2_dfmp2_native(mf, **kwargs):
    from pyscf.mp.dfmp2_native import DFRMP2
    mp2 = DFRMP2(mf.mf_s)
    mp2.ps = mf.c_ss
    mp2.pt = mf.c_os
    emp2 = mp2.kernel()
    return emp2, emp2


@timing
def energy_elec_mp2_dfump2_native(mf, **kwargs):
    from pyscf.mp.dfump2_native import DFUMP2
    mp2 = DFUMP2(mf.mf_s)
    mp2.ps = mf.c_ss
    mp2.pt = mf.c_os
    emp2 = mp2.kernel()
    return emp2, emp2


@timing
def energy_elec_mp2_dfmp2(mf, **kwargs):
    from pyscf.mp.dfmp2 import DFRMP2
    mp2 = DFRMP2(mf.mf_s, frozen=mf.frozen)
    mp2.kernel()
    return mp2.e_corr_os, mp2.e_corr_ss


@timing
def energy_elec_mp2_dfump2(mf, **kwargs):
    from pyscf.mp.dfump2 import DFUMP2
    mp2 = DFUMP2(mf.mf_s, frozen=mf.frozen)
    mp2.kernel()
    return mp2.e_corr_os, mp2.e_corr_ss
