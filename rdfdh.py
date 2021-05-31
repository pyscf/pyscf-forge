from __future__ import annotations
# dh import
try:
    from dh.dhutil import parse_xc_dh, gen_batch, calc_batch_size, HybridDict, timing, restricted_biorthogonalize
except ImportError:
    from pyscf.dh.dhutil import parse_xc_dh, gen_batch, calc_batch_size, HybridDict, timing, restricted_biorthogonalize
# typing import
from typing import Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from dh.grad.rdfdh import Gradients
    from dh.polar.rdfdh import Polar
# pyscf import
from pyscf.scf import cphf
from pyscf import lib, gto, df, dft, scf
from pyscf.ao2mo import _ao2mo
from pyscf.scf._response_functions import _gen_rhf_response
from pyscf.dftd3 import itrf
# other import
import os
import pickle
import numpy as np
import ctypes

einsum = lib.einsum


# region energy evaluation


def kernel(mf: RDFDH, **kwargs):
    mf.build()
    eng_tot, eng_nc, eng_pt2, eng_nuc, eng_os, eng_ss = mf.energy_tot(**kwargs)
    mf.e_tot = mf.eng_tot = eng_tot
    mf.eng_nc = eng_nc
    mf.eng_pt2 = eng_pt2
    mf.eng_nuc = eng_nuc
    mf.eng_os = eng_os
    mf.eng_ss = eng_ss
    return eng_tot


@timing
def energy_elec_nc(mf: RDFDH, mo_coeff=None, h1e=None, vhf=None, restricted=True, **_):
    if mo_coeff is None:
        if mf.mf_s.e_tot == 0:
            mf.run_scf()
            if mf.xc_n is None:  # if bDH-like functional, just return SCF energy
                return mf.mf_s.e_tot - mf.mf_s.energy_nuc(), None
        mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    if mo_occ is NotImplemented:
        if restricted:
            mo_occ = scf.hf.get_occ(mf.mf_s)
        else:
            mo_occ = scf.uhf.get_occ(mf.mf_s)
    dm = mf.mf_s.make_rdm1(mo_coeff, mo_occ)
    dm = lib.tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
    eng_nc = mf.mf_n.energy_elec(dm=dm, h1e=h1e, vhf=vhf)
    return eng_nc


@timing
def energy_elec_mp2(mf: RDFDH, mo_coeff=None, mo_energy=None, dfobj=None, Y_ia_ri=None, t_ijab_blk=None, eval_ss=True, **_):
    # prepare mo_coeff, mo_energy
    if mo_coeff is None:
        if mf.mf_s.e_tot == 0:
            mf.run_scf()
        mo_coeff = mf.mo_coeff
    if mo_energy is None:
        if mf.mf_s.e_tot == 0:
            mf.run_scf()
        mo_energy = mf.mo_energy
    # prepare essential dimensions
    if Y_ia_ri is None:
        nmo = mo_coeff.shape[1]
        nocc = mf.nocc
        nvir = nmo - nocc
    else:
        nocc, nvir = Y_ia_ri.shape[1:]
        nmo = nocc + nvir
    so, sv = slice(0, nocc), slice(nocc, nmo)
    iaslice = (0, nocc, nocc, nmo)
    # prepare Y_ia_ri (cderi in MO occ-vir block)
    if Y_ia_ri is None:
        if dfobj is None:
            dfobj = mf.df_ri
        Y_ia_ri = get_cderi_mo(dfobj, mo_coeff, pqslice=iaslice, max_memory=mf.get_memory())
    # evaluate energy
    eng_bi1 = eng_bi2 = 0
    D_jab = mo_energy[so, None, None] - mo_energy[None, sv, None] - mo_energy[None, None, sv]
    nbatch = mf.calc_batch_size(2 * nocc * nvir ** 2, Y_ia_ri.size + D_jab.size)
    for sI in gen_batch(0, nocc, nbatch):  # batch (i)
        D_ijab = mo_energy[sI, None, None, None] + D_jab
        g_ijab = einsum("Pia, Pjb -> ijab", Y_ia_ri[:, sI], Y_ia_ri)
        t_ijab = g_ijab / D_ijab
        eng_bi1 += einsum("ijab, ijab ->", t_ijab, g_ijab)
        if eval_ss:
            eng_bi2 += einsum("ijab, ijba ->", t_ijab, g_ijab)
        if t_ijab_blk:
            t_ijab_blk[sI] = t_ijab
    return eng_bi1, eng_bi2


def energy_elec_pt2(mf: RDFDH, params=None, eng_bi=None, **kwargs):
    cc, c_os, c_ss = params if params else mf.cc, mf.c_os, mf.c_ss
    eval_ss = True if abs(c_ss) > 1e-7 else False
    eng_bi1, eng_bi2 = eng_bi if eng_bi else mf.energy_elec_mp2(eval_ss=eval_ss, **kwargs)
    return (cc * ((c_os + c_ss) * eng_bi1 - c_ss * eng_bi2),  # Total
            eng_bi1,                                          # OS
            eng_bi1 - eng_bi2)                                # SS


def energy_nuc(mf: RDFDH, **_):
    mol = mf.mol
    eng_nuc = mol.energy_nuc()
    # handle dftd3 situation
    if "D3" in mf.xc_add:
        drv = itrf.libdftd3.wrapper_params
        params = np.asarray(mf.xc_add["D3"][0], order="F")
        version = mf.xc_add["D3"][1]
        coords = np.asarray(mol.atom_coords(), order="F")
        itype = np.asarray(mol.atom_charges(), order="F")
        edisp = np.zeros(1)
        grad = np.zeros((mol.natm, 3))
        drv(
            ctypes.c_int(mol.natm),                  # natoms
            coords.ctypes.data_as(ctypes.c_void_p),  # coords
            itype.ctypes.data_as(ctypes.c_void_p),   # itype
            params.ctypes.data_as(ctypes.c_void_p),  # params
            ctypes.c_int(version),                   # version
            edisp.ctypes.data_as(ctypes.c_void_p),   # edisp
            grad.ctypes.data_as(ctypes.c_void_p))    # grads)
        eng_nuc += float(edisp)
    return eng_nuc


def energy_elec(mf: RDFDH, **kwargs):
    eng_nc = mf.energy_elec_nc(**kwargs)[0]
    nocc, nvir = mf.nocc, mf.nvir
    t_ijab_blk = None
    if mf.with_t_ijab:
        t_ijab_blk = mf.tensors.create("t_ijab", shape=(nocc, nocc, nvir, nvir), incore=mf._incore_t_ijab)
    eng_pt2, eng_os, eng_ss = mf.energy_elec_pt2(t2_blk=t_ijab_blk, **kwargs)
    eng_elec = eng_nc + eng_pt2
    return eng_elec, eng_nc, eng_pt2, eng_os, eng_ss


def energy_tot(mf: RDFDH, **kwargs):
    eng_elec, eng_nc, eng_pt2, eng_os, eng_ss = mf.energy_elec(**kwargs)
    eng_nuc = mf.energy_nuc()
    eng_tot = eng_elec + eng_nuc
    return eng_tot, eng_nc, eng_pt2, eng_nuc, eng_os, eng_ss


@timing
def get_cderi_mo(dfobj: df.DF, C, Y_mo=None, pqslice=None, max_memory=2000):
    nmo, naux = dfobj.mol.nao, dfobj.get_naoaux()
    if pqslice is None:
        pqslice = (0, nmo, 0, nmo)
        nump, numq = nmo, nmo
    else:
        nump, numq = pqslice[1] - pqslice[0], pqslice[3] - pqslice[2]
    if Y_mo is None:
        Y_mo = np.empty((naux, nump, numq))

    def save(r0, r1, buf):
        Y_mo[r0:r1] = buf.reshape(r1-r0, nump, numq)

    p0, p1 = 0, 0
    preflop = 0 if not isinstance(Y_mo, np.ndarray) else Y_mo.size
    nbatch = calc_batch_size(2*nump*numq, max_memory, preflop)
    with lib.call_in_background(save) as bsave:
        for Y_ao in dfobj.loop(nbatch):
            p1 = p0 + Y_ao.shape[0]
            Y_mo_buf = _ao2mo.nr_e2(Y_ao, C, pqslice, aosym="s2", mosym="s1")
            bsave(p0, p1, Y_mo_buf)
            p0 = p1
    return Y_mo


# endregion energy evaluation

# region first derivative related


@timing
def get_eri_cpks(Y_mo_jk, nocc, cx, eri_cpks=None, max_memory=2000):
    naux, nmo, _ = Y_mo_jk.shape
    nvir = nmo - nocc
    so, sv = slice(0, nocc), slice(nocc, nmo)
    # prepare space if bulk of eri_cpks is not provided
    if eri_cpks is None:
        eri_cpks = np.empty((nvir, nocc, nvir, nocc))
    # copy some tensors to memory
    Y_ai_jk = np.asarray(Y_mo_jk[:, sv, so])
    Y_ij_jk = np.asarray(Y_mo_jk[:, so, so])

    nbatch = calc_batch_size(nvir*naux + 2*nocc**2*nvir, max_memory, Y_ai_jk.size + Y_ij_jk.size)
    for sA in gen_batch(nocc, nmo, nbatch):
        sAvir = slice(sA.start - nocc, sA.stop - nocc)
        eri_cpks[sAvir] = (
            + 4 * einsum("Pai, Pbj -> aibj", Y_ai_jk[:, sAvir], Y_ai_jk)
            - cx * einsum("Paj, Pbi -> aibj", Y_ai_jk[:, sAvir], Y_ai_jk)
            - cx * einsum("Pij, Pab -> aibj", Y_ij_jk, Y_mo_jk[:, sA, sv]))



def Ax0_Core_HF(si, sa, sj, sb, cx, Y_mo_jk, max_memory=2000):
    naux, nmo, _ = Y_mo_jk.shape
    ni, na = si.stop - si.start, sa.stop - sa.start

    @timing
    def Ax0_Core_HF_inner(X):
        X_shape = X.shape
        X = X.reshape((-1, X_shape[-2], X_shape[-1]))
        res = np.zeros((X.shape[0], ni, na))
        nbatch = calc_batch_size(nmo**2, max_memory, X.size + res.size)
        for saux in gen_batch(0, naux, nbatch):
            Y_mo_blk = np.asarray(Y_mo_jk[saux])
            for A in range(X.shape[0]):  # explicitly split X to X[A] to avoid einsum more than 2 oprehends
                res[A] += (
                    + 4 * einsum("Pia, Pjb, jb -> ia", Y_mo_blk[:, si, sa], Y_mo_blk[:, sj, sb], X[A])
                    - cx * einsum("Pib, Pja, jb -> ia", Y_mo_blk[:, si, sb], Y_mo_blk[:, sj, sa], X[A])
                    - cx * einsum("Pij, Pab, jb -> ia", Y_mo_blk[:, si, sj], Y_mo_blk[:, sa, sb], X[A]))
        res.shape = list(X_shape[:-2]) + [res.shape[-2], res.shape[-1]]
        return res
    return Ax0_Core_HF_inner


def Ax0_Core_KS(si, sa, sj, sb, mo_coeff, xc_setting, xc_kernel):
    C = mo_coeff
    ni, mol, grids, xc, dm = xc_setting
    rho, vxc, fxc = xc_kernel

    @timing
    def Ax0_Core_KS_inner(X):
        X_shape = X.shape
        X = X.reshape((-1, X_shape[-2], X_shape[-1]))
        dmX = C[:, sj] @ X @ C[:, sb].T
        dmX += dmX.swapaxes(-1, -2)
        ax_ao = ni.nr_rks_fxc(mol, grids, xc, dm, dmX, hermi=1, rho0=rho, vxc=vxc, fxc=fxc)
        res = 2 * C[:, si].T @ ax_ao @ C[:, sa]
        res.shape = list(X_shape[:-2]) + [res.shape[-2], res.shape[-1]]
        return res
    return Ax0_Core_KS_inner


def Ax0_Core_resp(si, sa, sj, sb, mf, mo_coeff, max_memory=2000):
    C = mo_coeff
    resp = _gen_rhf_response(mf, mo_coeff=C, hermi=1, max_memory=max_memory)

    @timing
    def Ax0_Core_resp_inner(X):
        X_shape = X.shape
        X = X.reshape((-1, X_shape[-2], X_shape[-1]))
        dmX = C[:, sj] @ X @ C[:, sb].T
        dmX += dmX.swapaxes(-1, -2)
        ax_ao = resp(dmX)
        res = 2 * C[:, si].T @ ax_ao @ C[:, sa]
        res.shape = list(X_shape[:-2]) + [res.shape[-2], res.shape[-1]]
        return res
    return Ax0_Core_resp_inner


def Ax0_cpks_HF(eri_cpks, max_memory=2000):
    nvir, nocc = eri_cpks.shape[:2]

    @timing
    def Ax0_cpks_HF_inner(X):
        X_shape = X.shape
        X = X.reshape((-1, X_shape[-2], X_shape[-1]))
        res = np.zeros_like(X)
        nbatch = calc_batch_size(nocc**2 * nvir, max_memory, 0)
        for sA in gen_batch(0, nvir, nbatch):
            res[:, sA] = einsum("aibj, Abj -> Aai", eri_cpks[sA], X)
        res.shape = list(X_shape[:-2]) + [res.shape[-2], res.shape[-1]]
        return res
    return Ax0_cpks_HF_inner


# endregion first derivative related


class RDFDH(lib.StreamObject):

    def __init__(self,
                 mol: gto.Mole,
                 xc: str or tuple = "XYG3",
                 auxbasis_jk: str or dict or None = None,
                 auxbasis_ri: str or dict or None = None,
                 grids: dft.Grids = None,
                 grids_cpks: dft.Grids = None,
                 unrestricted: bool = False,  # only for class initialization
                 ):
        # tunable flags
        self.with_t_ijab = False  # only in energy calculation; polarizability is forced dump t2 to disk or mem
        self._incore_t_ijab = False
        self._incore_Y_mo = False
        self._incore_eri_cpks = False
        self._fixed_batch = False
        self.cpks_tol = 1e-8
        self.cpks_cyc = 100
        self.max_memory = mol.max_memory
        # Parse xc code
        # It's tricky to say that, self.xc refers to SCF xc, and self.xc_dh refers to double hybrid xc
        # There should be three kinds of possible inputs:
        # 1) String: "XYG3"
        # 2) Tuple: ("B3LYPg", "0.8033*HF - 0.0140*LDA + 0.2107*B88, 0.6789*LYP", 0.3211, 1, 1)
        # 3) Additional: (("0.69*HF + 0.31*PBE, 0.44*P86", None, 1, 0.52, 0.22), {"D3": ([0.48, 0, 0, 5.6, 0], 4)})
        self.xc_dh = xc
        if isinstance(xc, str):
            xc_list, xc_add = parse_xc_dh(xc)
        elif len(xc) == 5:  # here should assert xc is a tuple/list with 2 or 5 elements
            xc_list = xc
            xc_add = {}
        else:  # assert len(xc) == 2
            xc_list, xc_add = xc
        self.xc, self.xc_n, self.cc, self.c_os, self.c_ss = xc_list
        self.xc_add = xc_add
        # parse auxiliary basis
        self.auxbasis_jk = auxbasis_jk = auxbasis_jk if auxbasis_jk else df.make_auxbasis(mol, mp2fit=False)
        self.auxbasis_ri = auxbasis_ri = auxbasis_ri if auxbasis_ri else df.make_auxbasis(mol, mp2fit=True)
        self.same_aux = True if auxbasis_jk == auxbasis_ri or auxbasis_ri is None else False
        # parse scf method
        self.unrestricted = unrestricted
        if unrestricted:
            mf_s = dft.UKS(mol, xc=self.xc).density_fit(auxbasis=auxbasis_jk)
        else:
            mf_s = dft.KS(mol, xc=self.xc).density_fit(auxbasis=auxbasis_jk)
        self.grids = grids if grids else mf_s.grids                        # type: dft.grid.Grids
        self.grids_cpks = grids_cpks if grids_cpks else self.grids         # type: dft.grid.Grids
        self.mf_s = mf_s                                                   # type: dft.rks.RKS
        self.mf_s.grids = self.grids
        # parse non-consistent method
        self.xc_n = None if self.xc_n == self.xc else self.xc_n            # type: str or None
        self.mf_n = self.mf_s                                              # type: dft.rks.RKS
        if self.xc_n:
            if unrestricted:
                self.mf_n = dft.UKS(mol, xc=self.xc_n).density_fit(auxbasis=auxbasis_jk)
            else:
                self.mf_n = dft.KS(mol, xc=self.xc_n).density_fit(auxbasis=auxbasis_jk)
            self.mf_n.grids = self.mf_s.grids
            self.mf_n.grids = self.grids
        # parse hybrid coefficients
        self.ni = self.mf_s._numint
        self.cx = self.ni.hybrid_coeff(self.xc)
        self.cx_n = self.ni.hybrid_coeff(self.xc_n)
        # parse density fitting object
        self.df_jk = mf_s.with_df  # type: df.DF
        self.aux_jk = self.df_jk.auxmol
        self.df_ri = df.DF(mol, auxbasis_ri) if not self.same_aux else self.df_jk
        self.aux_ri = self.df_ri.auxmol
        # other preparation
        self.tensors = HybridDict()
        self.mol = mol
        self.nao = mol.nao  # type: int
        self.nocc = mol.nelec[0]
        # variables awaits to be build
        self.mo_coeff = NotImplemented
        self.mo_energy = NotImplemented
        self.mo_occ = NotImplemented
        self.C = self.Co = self.Cv = NotImplemented
        self.e = self.eo = self.ev = NotImplemented
        self.D = NotImplemented
        self.nmo = self.nvir = NotImplemented
        self.so = self.sv = self.sa = NotImplemented
        # results
        self.e_tot = NotImplemented
        self.eng_tot = self.eng_nc = self.eng_pt2 = self.eng_nuc = self.eng_os = self.eng_ss = NotImplemented
        # DANGEROUS PLACE
        # we could first initialize nmo as nao
        self.nmo = self.nao
        self.nvir = self.nmo - self.nocc

    @property
    def base(self):
        return self

    @property
    def converged(self):
        return self.mf_s.converged

    def get_memory(self):  # leave at least 500MB space anyway
        return max(self.max_memory - lib.current_memory()[0], 500)

    def calc_batch_size(self, unit_flop, pre_flop=0, fixed_mem=None):
        if self._fixed_batch:
            return self._fixed_batch
        if fixed_mem:
            return calc_batch_size(unit_flop, fixed_mem, pre_flop)
        else:
            return calc_batch_size(unit_flop, self.get_memory(), pre_flop)

    @timing
    def build(self):
        # make sure that grids in SCF run should be the same to other energy evaluations
        self.mf_s.grids = self.mf_n.grids = self.grids
        if self.df_jk.auxmol is None:
            self.df_jk.build()
            self.aux_jk = self.df_jk.auxmol
        if self.df_ri.auxmol is None:
            self.df_ri.build()
            self.aux_ri = self.df_ri.auxmol

    @timing
    def run_scf(self, **kwargs):
        self.mf_s.grids = self.mf_n.grids = self.grids
        self.build()
        mf = self.mf_s
        if mf.e_tot == 0:
            mf.kernel(**kwargs)
        # prepare
        self.C = self.mo_coeff = mf.mo_coeff
        self.e = self.mo_energy = mf.mo_energy
        self.mo_occ = mf.mo_occ
        self.D = mf.make_rdm1(mf.mo_coeff)
        nocc = self.nocc
        nmo = self.nmo = self.C.shape[1]
        self.nvir = nmo - nocc
        self.so, self.sv, self.sa = slice(0, nocc), slice(nocc, nmo), slice(0, nmo)
        self.Co, self.Cv = self.C[:, self.so], self.C[:, self.sv]
        self.eo, self.ev = self.e[self.so], self.e[self.sv]
        return self

    # region first derivative related in class

    def Ax0_Core_HF(self, si, sa, sj, sb, cx=None):
        Y_mo_jk = self.tensors["Y_mo_jk"]
        cx = cx if cx else self.cx
        return Ax0_Core_HF(si, sa, sj, sb, cx, Y_mo_jk, max_memory=self.get_memory())

    def Ax0_Core_KS(self, si, sa, sj, sb, xc=None, cpks=False):
        xc = xc if xc else self.xc
        if self.ni._xc_type(xc) == "HF":
            return lambda _: 0
        tensors = self.tensors
        cpks_token = "in cpks" if cpks else ""
        grids = self.grids_cpks if cpks else self.grids
        xc_setting = self.ni, self.mol, grids, xc, self.D
        if "rho" + cpks_token not in tensors:
            self.prepare_xc_kernel()
        xc_kernel = tensors["rho" + cpks_token], tensors["vxc" + xc + cpks_token], tensors["fxc" + xc + cpks_token]
        mo_coeff = self.mo_coeff
        return Ax0_Core_KS(si, sa, sj, sb, mo_coeff, xc_setting, xc_kernel)

    def Ax0_Core(self, si, sa, sj, sb, xc=None, cpks=False):
        xc = xc if xc else self.xc
        cx = self.ni.hybrid_coeff(xc)
        ax0_core_hf, ax0_core_ks = self.Ax0_Core_HF(si, sa, sj, sb, cx), self.Ax0_Core_KS(si, sa, sj, sb, xc, cpks)

        def fx(X):
            return ax0_core_hf(X) + ax0_core_ks(X)
        return fx

    def Ax0_Core_resp(self, si, sa, sj, sb, mf=None, mo_coeff=None):
        # this function is only a replacement to Ax0_Core or Ax0_cpks and left for efficiency comparasion
        mf = mf if mf else self.mf_s
        mo_coeff = mo_coeff if mo_coeff else self.C
        return Ax0_Core_resp(si, sa, sj, sb, mf, mo_coeff, max_memory=self.get_memory())

    def Ax0_cpks(self):
        so, sv = self.so, self.sv
        ax0_core_ks = self.Ax0_Core_KS(sv, so, sv, so, cpks=True)
        ax0_cpks_hf = Ax0_cpks_HF(self.tensors["eri_cpks"], self.get_memory())

        def Ax0_cpks_inner(X):
            res = ax0_cpks_hf(X) + ax0_core_ks(X)
            return res
        return Ax0_cpks_inner

    def solve_cpks(self, rhs):
        return cphf.solve(self.Ax0_cpks(), self.e, self.mo_occ, rhs, max_cycle=self.cpks_cyc, tol=self.cpks_tol)[0]

    def prepare_integral(self):
        self.run_scf()
        tensors = self.tensors
        C = self.C
        nmo, nocc, nvir = self.nmo, self.nocc, self.nvir

        # part: Y_mo_jk
        tensors.create("Y_mo_jk", shape=(self.df_jk.get_naoaux(), nmo, nmo), incore=self._incore_Y_mo)
        get_cderi_mo(self.df_jk, C, tensors["Y_mo_jk"], max_memory=self.get_memory())
        # if self.same_aux:  # I decided repeat a space, not using the same.
        #     tensors["Y_mo_ri"] = tensors["Y_mo_jk"]
        # else:
        tensors.create("Y_mo_ri", shape=(self.df_ri.get_naoaux(), nmo, nmo), incore=self._incore_Y_mo)
        get_cderi_mo(self.df_ri, C, tensors["Y_mo_ri"], max_memory=self.get_memory())
        # part: cpks and Ax0_Core preparation
        eri_cpks = tensors.create("eri_cpks", shape=(nvir, nocc, nvir, nocc), incore=self._incore_Y_mo)
        get_eri_cpks(tensors["Y_mo_jk"], nocc, self.cx, eri_cpks, max_memory=self.get_memory())
        return self

    @timing
    def prepare_xc_kernel(self):
        mol = self.mol
        tensors = self.tensors
        C, mo_occ = self.C, self.mo_occ
        ni = self.ni
        if "rho" in tensors:
            return self
        if ni._xc_type(self.xc) == "GGA":
            rho, vxc, fxc = ni.cache_xc_kernel(mol, self.grids, self.xc, C, mo_occ, max_memory=self.get_memory(), spin=self.unrestricted)
            tensors.create("rho", rho)
            tensors.create("vxc" + self.xc, vxc)
            tensors.create("fxc" + self.xc, fxc)
            rho, vxc, fxc = ni.cache_xc_kernel(mol, self.grids_cpks, self.xc, C, mo_occ, max_memory=self.get_memory(), spin=self.unrestricted)
            tensors.create("rho" + "in cpks", rho)
            tensors.create("vxc" + self.xc + "in cpks", vxc)
            tensors.create("fxc" + self.xc + "in cpks", fxc)
        if self.xc_n and ni._xc_type(self.xc_n) == "GGA":
            if "rho" in tensors:
                vxc, fxc = ni.eval_xc(self.xc_n, tensors["rho"], deriv=2, verbose=0, spin=self.unrestricted)[1:3]
                tensors.create("vxc" + self.xc_n, vxc)
                tensors.create("fxc" + self.xc_n, fxc)
            else:
                rho, vxc, fxc = ni.cache_xc_kernel(mol, self.grids, self.xc_n, C, mo_occ, max_memory=self.get_memory(), spin=self.unrestricted)
                tensors.create("rho", rho)
                tensors.create("vxc" + self.xc_n, vxc)
                tensors.create("fxc" + self.xc_n, fxc)
        return self

    @timing
    def prepare_pt2(self, dump_t_ijab=True):
        tensors = self.tensors
        nvir, nocc, nmo = self.nvir, self.nocc, self.nmo
        e = self.e
        naux = self.df_ri.get_naoaux()
        so, sv = self.so, self.sv
        cc, c_os, c_ss = self.cc, self.c_os, self.c_ss

        D_rdm1 = np.zeros((nmo, nmo))
        G_ia_ri = np.zeros((naux, nocc, nvir))
        Y_ia_ri = np.asarray(tensors["Y_mo_ri"][:, so, sv])

        dump_t_ijab = False if "t_ijab" in tensors else dump_t_ijab  # t_ijab to be dumped
        # eval_t_ijab = True if "t_ijab" not in tensors else False     # t_ijab to be evaluated
        eval_t_ijab = True  # to avoid any possible conflict for `as_scanner`
        D_jab = e[so, None, None] - e[None, sv, None] - e[None, None, sv] if eval_t_ijab else None
        if dump_t_ijab:
            tensors.create("t_ijab", shape=(nocc, nocc, nvir, nvir), incore=self._incore_t_ijab)

        eng_bi1 = eng_bi2 = 0
        eval_ss = True if abs(c_ss) > 1e-7 else False
        nbatch = self.calc_batch_size(4 * nocc * nvir ** 2, Y_ia_ri.size + G_ia_ri.size)
        for sI in gen_batch(0, nocc, nbatch):
            if eval_t_ijab:
                D_ijab = e[sI, None, None, None] + D_jab
                g_ijab = einsum("Pia, Pjb -> ijab", Y_ia_ri[:, sI], Y_ia_ri)
                t_ijab = g_ijab / D_ijab
                if self.eng_pt2 is NotImplemented:
                    eng_bi1 += einsum("ijab, ijab ->", t_ijab, g_ijab)
                    if eval_ss:
                        eng_bi2 += einsum("ijab, ijba ->", t_ijab, g_ijab)
            else:
                t_ijab = tensors["t_ijab"][sI]
            # T_ijab = cc * ((c_os + c_ss) * t_ijab - c_ss * t_ijab.swapaxes(-1, -2))
            T_ijab = restricted_biorthogonalize(t_ijab, cc, c_os, c_ss)
            D_rdm1[sv, sv] += 2 * einsum("ijac, ijbc -> ab", T_ijab, t_ijab)
            D_rdm1[so, so] -= 2 * einsum("ijab, ikab -> jk", T_ijab, t_ijab)
            G_ia_ri[:, sI] = einsum("ijab, Pjb -> Pia", T_ijab, Y_ia_ri)
            if dump_t_ijab:
                tensors["t_ijab"][sI] = t_ijab
        if self.eng_tot is NotImplemented:
            kernel(self, eng_bi=(eng_bi1, eng_bi2))
        tensors.create("D_rdm1", D_rdm1)
        tensors.create("G_ia_ri", G_ia_ri)
        return self

    @timing
    def prepare_lagrangian(self, gen_W=False):
        tensors = self.tensors
        nvir, nocc, nmo, naux = self.nvir, self.nocc, self.nmo, self.df_ri.get_naoaux()
        so, sv, sa = self.so, self.sv, self.sa

        D_rdm1 = tensors.load("D_rdm1")
        G_ia_ri = tensors.load("G_ia_ri")
        Y_mo_ri = tensors["Y_mo_ri"]
        Y_ij_ri = np.asarray(Y_mo_ri[:, so, so])
        L = np.zeros((nvir, nocc))

        if gen_W:
            Y_ia = np.asarray(Y_mo_ri[:, so, sv])
            W_I = np.zeros((nmo, nmo))
            W_I[so, so] = - 2 * einsum("Pia, Pja -> ij", G_ia_ri, Y_ia)
            W_I[sv, sv] = - 2 * einsum("Pia, Pib -> ab", G_ia_ri, Y_ia)
            W_I[sv, so] = - 4 * einsum("Pja, Pij -> ai", G_ia_ri, Y_mo_ri[:, so, so])
            tensors.create("W_I", W_I)
            L += W_I[sv, so]
        else:
            L -= 4 * einsum("Pja, Pij -> ai", G_ia_ri, Y_ij_ri)

        # L += self.Ax0_Core(sv, so, sa, sa)(D_rdm1)
        L += self.Ax0_Core_resp(sv, so, sa, sa)(D_rdm1)  # resp is faster

        nbatch = self.calc_batch_size(nvir ** 2 + nocc * nvir, G_ia_ri.size + Y_ij_ri.size)
        for saux in gen_batch(0, naux, nbatch):
            L += 4 * einsum("Pib, Pab -> ai", G_ia_ri[saux], Y_mo_ri[saux, sv, sv])

        if self.xc_n:
            L += 4 * einsum("ua, uv, vi -> ai", self.Cv, self.mf_n.get_fock(dm=self.D), self.Co)

        tensors.create("L", L)
        return self

    @timing
    def prepare_D_r(self):
        tensors = self.tensors
        sv, so = self.sv, self.so
        D_r = tensors.load("D_rdm1").copy()
        L = tensors.load("L")
        D_r[sv, so] = self.solve_cpks(L)
        tensors.create("D_r", D_r)
        return self

    def dipole(self):
        if "D_r" not in self.tensors:
            self.prepare_integral().prepare_xc_kernel() \
                .prepare_pt2(dump_t_ijab=True).prepare_lagrangian() \
                .prepare_D_r()
        D_r = self.tensors["D_r"]
        mol, C, D = self.mol, self.C, self.D
        h = - mol.intor("int1e_r")
        d = einsum("tuv, uv -> t", h, D + C @ D_r @ C.T)
        d += einsum("A, At -> t", mol.atom_charges(), mol.atom_coords())
        return d

    def dump_intermediates(self, dir_path="scratch"):
        os.makedirs(dir_path, exist_ok=True)
        # tensors
        tensors = self.tensors
        h5_path = dir_path + "/tensors.h5"
        dat_path = dir_path + "/tensors.dat"
        tensors.dump(h5_path, dat_path)
        # scf
        # scf_path = dir_path + "/scf.h5"
        # if self.mf_s.chkfile:
        #     shutil.copy(self.mf_s.chkfile, scf_path)
        # class attributes, without results
        att_path = dir_path + "/attributes.dat"
        dct = {
            "C": self.C,
            "e": self.e,
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
        self.mf_s.mo_coeff = dct["C"]
        self.mf_s.mo_energy = dct["e"]
        self.mf_s.mo_occ = dct["mo_occ"]
        self.mf_s.e_tot = dct["mf_s_e_tot"]
        if rerun_scf:  # probably required for validation of dft grids
            self.mf_s.kernel(dm=self.mf_s.make_rdm1())
        self.run_scf()
        return self

    # A REALLY DIRTY WAY  https://stackoverflow.com/questions/7078134/
    # to avoid cyclic imports in typing https://stackoverflow.com/questions/39740632/

    def nuc_grad_method(self) -> Gradients:
        try:
            from dh.grad.rdfdh import Gradients
        except ImportError:
            from pyscf.dh.grad.rdfdh import Gradients
        self.__class__ = Gradients
        Gradients.__init__(self, self.mol, skip_construct=True)
        return self

    def polar_method(self) -> Polar:
        try:
            from dh.polar.rdfdh import Polar
        except ImportError:
            from pyscf.dh.polar.rdfdh import Polar
        self.__class__ = Polar
        Polar.__init__(self, self.mol, skip_construct=True)
        return self

    # endregion first derivative related in class

    energy_elec_nc = energy_elec_nc
    energy_elec_mp2 = energy_elec_mp2
    energy_elec_pt2 = energy_elec_pt2
    energy_nuc = energy_nuc
    energy_elec = energy_elec
    energy_tot = energy_tot
    kernel = kernel
    solve_cpks = solve_cpks
