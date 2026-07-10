# dh import
from pyscf.dh.dhutil import gen_batch, calc_batch_size, HybridDict, timing, restricted_biorthogonalize, \
    get_rho_from_dm_gga
from pyscf.dh.xccode import parse_xc_dh, xc_equal
from pyscf.dh.dh import DHBase, energy_elec_mp2_dfmp2_native, energy_elec_mp2_dfmp2
from pyscf.dh.mp2_ajz import get_cderi_mo, energy_elec_mp2_ajz, _loop_t_ijab
# pyscf import
from pyscf.scf import cphf
from pyscf import lib, gto, df, dft, scf
from pyscf.dft.xc_deriv import transform_vxc, transform_fxc
from pyscf.ao2mo import _ao2mo
from pyscf.scf._response_functions import _gen_rhf_response
try:
    from pyscf.dispersion.dftd3 import DFTD3Dispersion
except ImportError:
    print('Warning: pyscf-dispersion not found. D3 correction unavailable.')
# other import
import os
import pickle
import numpy as np
from functools import partial

einsum = lib.einsum


# region energy evaluation


def kernel(mf, **kwargs):
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
def energy_elec_nc(mf, mo_coeff=None, h1e=None, vhf=None, **_):
    if mo_coeff is None:
        if mf.mf_s.e_tot == 0:
            mf.run_scf()
            if mf.xc_n is None:  # if bDH-like functional, just return SCF energy
                return mf.mf_s.e_tot - mf.mf_s.energy_nuc(), None
        mo_coeff = mf.mf_s.mo_coeff
    mo_occ = mf.mf_s.mo_occ
    if mo_occ is NotImplemented:
        mo_occ = scf.hf.get_occ(mf.mf_s)
    dm = mf.mf_s.make_rdm1(mo_coeff, mo_occ)
    dm = lib.tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
    eng_nc = mf.mf_n.energy_elec(dm=dm, h1e=h1e, vhf=vhf)
    return eng_nc


@timing
def energy_elec_pt2(mf, params=None, eng_bi=None, **kwargs):
    if not mf.eval_pt2:
        return 0, 0, 0
    cc, c_os, c_ss = params if params else mf.cc, mf.c_os, mf.c_ss
    emp2_0, eng_bi1, eng_bi2 = eng_bi if eng_bi else mf.energy_elec_mp2(eval_ss=mf.eval_ss, **kwargs)
    if getattr(mf, 'mp2_backend', None) == "dfmp2_native":
        return cc * emp2_0, None, None
    if getattr(mf, 'mp2_backend', None) == "dfmp2":
        return cc * (c_os * eng_bi1 + c_ss * eng_bi2), eng_bi1, eng_bi2
    return (cc * ((c_os + c_ss) * eng_bi1 - c_ss * eng_bi2),
            eng_bi1,
            eng_bi1 - eng_bi2)


def energy_nuc(mf, **_):
    mol = mf.mol
    eng_nuc = mol.energy_nuc()
    if "D3" in mf.xc_add:
        d3_info = mf.xc_add["D3"]
        model = DFTD3Dispersion(mol, xc=d3_info["xc"], version=d3_info["version"])
        eng_nuc += model.get_dispersion()["energy"]
    if "D4" in mf.xc_add:
        from pyscf.dispersion.dftd4 import DFTD4Dispersion
        d4_info = mf.xc_add["D4"]
        model = DFTD4Dispersion(mol, xc=d4_info["xc"], version=d4_info["version"])
        eng_nuc += model.get_dispersion()["energy"]
    return eng_nuc


def energy_elec(mf, **kwargs):
    eng_nc = mf.energy_elec_nc(**kwargs)[0]
    nocc, nvir = mf.nocc, mf.nvir
    t_ijab_blk = None
    if mf.with_t_ijab:
        t_ijab_blk = mf.tensors.create("t_ijab", shape=(nocc, nocc, nvir, nvir), incore=mf._incore_t_ijab)
    eng_pt2, eng_os, eng_ss = mf.energy_elec_pt2(t2_blk=t_ijab_blk, **kwargs)
    eng_elec = eng_nc + eng_pt2
    return eng_elec, eng_nc, eng_pt2, eng_os, eng_ss


def energy_tot(mf, **kwargs):
    eng_elec, eng_nc, eng_pt2, eng_os, eng_ss = mf.energy_elec(**kwargs)
    eng_nuc = mf.energy_nuc()
    eng_tot = eng_elec + eng_nuc
    return eng_tot, eng_nc, eng_pt2, eng_nuc, eng_os, eng_ss


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
    vxc_ = transform_vxc(rho, vxc, "GGA", spin=0)
    fxc_ = transform_fxc(rho, vxc, fxc, "GGA", spin=0)

    @timing
    def Ax0_Core_KS_inner(X):
        X_shape = X.shape
        X = X.reshape((-1, X_shape[-2], X_shape[-1]))
        dmX = C[:, sj] @ X @ C[:, sb].T
        dmX += dmX.swapaxes(-1, -2)
        ax_ao = ni.nr_rks_fxc(mol, grids, xc, dm, dmX, hermi=1, rho0=rho, vxc=vxc_, fxc=fxc_)
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


class RDFDH(DHBase):

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
            mf_s = dft.KS(mol, xc=self.xc).density_fit(auxbasis=self.auxbasis_jk)
        self.grids = grids if grids else (getattr(mf_s, 'grids', dft.Grids(mol)))
        self.grids_cpks = grids_cpks if grids_cpks else self.grids
        self.mf_s = mf_s
        self.mf_s.grids = self.grids
        self.xc_n = None if xc_equal(self.xc_n, self.xc) else self.xc_n
        self.mf_n = self.mf_s
        if self.xc_n:
            self.mf_n = dft.KS(mol, xc=self.xc_n).density_fit(auxbasis=self.auxbasis_jk)
            self.mf_n.grids = self.mf_s.grids
            self.mf_n.grids = self.grids
        self.ni = getattr(self.mf_s, '_numint', dft.numint.NumInt())
        self.cx = self.ni.hybrid_coeff(self.xc)
        self.cx_n = self.ni.hybrid_coeff(self.xc_n)
        self.df_jk = mf_s.with_df
        self.aux_jk = self.df_jk.auxmol
        self.df_ri = df.DF(mol, self.auxbasis_ri) if not self.same_aux else self.df_jk
        self.aux_ri = self.df_ri.auxmol
        self.nocc = mol.nelec[0]
        self.nmo = self.nao
        self.nvir = self.nmo - self.nocc
        if self._scf is not None and self._scf.e_tot != 0:
            self.run_scf()
        if mp2_backend == "dfmp2_native":
            self.energy_elec_mp2 = partial(energy_elec_mp2_dfmp2_native, self)
        elif mp2_backend == "dfmp2":
            self.energy_elec_mp2 = partial(energy_elec_mp2_dfmp2, self)
        else:
            self.energy_elec_mp2 = partial(energy_elec_mp2_ajz, self)

    @timing
    def run_scf(self, **kwargs):
        self.mf_s.grids = self.mf_n.grids = self.grids
        self.build()
        mf = self.mf_s
        if mf.e_tot == 0:
            mf.kernel(**kwargs)
        self.mo_coeff = mf.mo_coeff
        self.mo_energy = mf.mo_energy
        self.mo_occ = mf.mo_occ
        self.D = mf.make_rdm1(mf.mo_coeff)
        nocc = self.nocc
        nmo = self.nmo = self.mo_coeff.shape[1]
        self.nvir = nmo - nocc
        self.so, self.sv, self.sa = slice(0, nocc), slice(nocc, nmo), slice(0, nmo)
        self.Co, self.Cv = self.mo_coeff[:, self.so], self.mo_coeff[:, self.sv]
        self.eo, self.ev = self.mo_energy[self.so], self.mo_energy[self.sv]
        return self

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
        mf = mf if mf else self.mf_s
        mo_coeff = mo_coeff if mo_coeff else self.mo_coeff
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
        return cphf.solve(self.Ax0_cpks(), self.mo_energy, self.mo_occ, rhs, max_cycle=self.cpks_cyc, tol=self.cpks_tol)[0]

    def prepare_integral(self):
        self.run_scf()
        tensors = self.tensors
        C = self.mo_coeff
        nmo, nocc, nvir = self.nmo, self.nocc, self.nvir

        tensors.create("Y_mo_jk", shape=(self.df_jk.get_naoaux(), nmo, nmo), incore=self._incore_Y_mo)
        get_cderi_mo(self.df_jk, C, tensors["Y_mo_jk"], max_memory=self.get_memory())
        if self.eval_pt2:
            tensors.create("Y_mo_ri", shape=(self.df_ri.get_naoaux(), nmo, nmo), incore=self._incore_Y_mo)
            get_cderi_mo(self.df_ri, C, tensors["Y_mo_ri"], max_memory=self.get_memory())
        eri_cpks = tensors.create("eri_cpks", shape=(nvir, nocc, nvir, nocc), incore=self._incore_Y_mo)
        get_eri_cpks(tensors["Y_mo_jk"], nocc, self.cx, eri_cpks, max_memory=self.get_memory())
        return self

    @timing
    def prepare_pt2(self, dump_t_ijab=True):
        tensors = self.tensors
        nvir, nocc, nmo = self.nvir, self.nocc, self.nmo
        e = self.mo_energy
        naux = self.df_ri.get_naoaux()
        so, sv = self.so, self.sv
        cc, c_os, c_ss = self.cc, self.c_os, self.c_ss

        D_rdm1 = np.zeros((nmo, nmo))

        if not self.eval_pt2:
            if self.eng_tot is NotImplemented:
                tensors.create("D_rdm1", D_rdm1)
                kernel(self, eng_bi=(None, 0, 0))
            return self

        G_ia_ri = np.zeros((naux, nocc, nvir))
        Y_ia_ri = np.asarray(tensors["Y_mo_ri"][:, so, sv])

        dump_t_ijab = False if "t_ijab" in tensors else dump_t_ijab
        if dump_t_ijab:
            tensors.create("t_ijab", shape=(nocc, nocc, nvir, nvir), incore=self._incore_t_ijab)

        eng_bi1 = [0]
        eng_bi2 = [0]

        def build(sI, t_ijab, g_ijab):
            if self.eng_pt2 is NotImplemented:
                eng_bi1[0] += einsum("ijab, ijab ->", t_ijab, g_ijab)
                if self.eval_ss:
                    eng_bi2[0] += einsum("ijab, ijba ->", t_ijab, g_ijab)
            if dump_t_ijab:
                tensors["t_ijab"][sI] = t_ijab
            T_ijab = restricted_biorthogonalize(t_ijab, cc, c_os, c_ss)
            D_rdm1[sv, sv] += 2 * einsum("ijac, ijbc -> ab", T_ijab, t_ijab)
            D_rdm1[so, so] -= 2 * einsum("ijab, ikab -> jk", T_ijab, t_ijab)
            G_ia_ri[:, sI] = einsum("ijab, Pjb -> Pia", T_ijab, Y_ia_ri)

        _loop_t_ijab(self, Y_ia_ri, e, nocc, nvir, build)

        if self.eng_tot is NotImplemented:
            kernel(self, eng_bi=(None, eng_bi1[0], eng_bi2[0]))
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

        L += self.Ax0_Core_resp(sv, so, sa, sa)(D_rdm1)

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
        mol, C, D = self.mol, self.mo_coeff, self.D
        h = - mol.intor("int1e_r")
        d = einsum("tuv, uv -> t", h, D + C @ D_r @ C.T)
        d += einsum("A, At -> t", mol.atom_charges(), mol.atom_coords())
        return d

    def nuc_grad_method(self):
        from pyscf.dh.grad.rdfdh import Gradients
        self.__class__ = Gradients
        Gradients.__init__(self, self.mol, skip_construct=True)
        return self

    def polar_method(self):
        from pyscf.dh.polar.rdfdh import Polar
        self.__class__ = Polar
        Polar.__init__(self, self.mol, skip_construct=True)
        return self

    energy_elec_nc = energy_elec_nc
    energy_elec_pt2 = energy_elec_pt2
    energy_elec_mp2 = energy_elec_mp2_ajz
    energy_nuc = energy_nuc
    energy_elec = energy_elec
    energy_tot = energy_tot
    kernel = kernel
    solve_cpks = solve_cpks
