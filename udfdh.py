from __future__ import annotations

from typing import Tuple

from pyscf.scf import ucphf
import h5py
from dh import RDFDH
from dh.dhutil import parse_xc_dh, gen_batch, calc_batch_size, HybridDict, timing
from pyscf import lib, gto, df, dft
from pyscf.ao2mo import _ao2mo
from pyscf.scf._response_functions import _gen_uhf_response
import numpy as np

from dh.rdfdh import get_cderi_mo

einsum = lib.einsum
α, β = 0, 1
αα, αβ, ββ = 0, 1, 2

ndarray = np.ndarray or h5py.Dataset


def tot_size(*args):
    size = 0
    for i in args:
        if isinstance(i, np.ndarray):
            size += i.size
        else:
            size += tot_size(*i)
    return size


# region energy evaluation


@timing
def energy_elec_mp2(mf: UDFDH,
                    mo_coeff: np.ndarray = None,
                    mo_energy: np.ndarray = None,
                    dfobj: df.DF = None,
                    Y_ia_ri: Tuple[ndarray, ...] = None,
                    t_ijab_blk: Tuple[ndarray, ...] = None,
                    eval_ss=True, **_):
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
    # prepare Y_ia_ri (cderi in MO occ-vir block)
    if Y_ia_ri is None:
        if dfobj is None:
            dfobj = mf.df_ri
        Y_ia_ri = [get_cderi_mo(dfobj, mo_coeff[σ], pqslice=iaslice[σ], max_memory=mf.get_memory()) for σ in (α, β)]
    # evaluate energy
    eng_bi1, eng_bi2 = [0, 0, 0], [0, 0, 0]
    mocc, mvir = max(nocc), max(nvir)
    nbatch = mf.calc_batch_size(2 * mocc * mvir ** 2, tot_size(Y_ia_ri) + mocc * mvir ** 2)
    # situation αβ
    for σς, σ, ς in (αα, α, α), (αβ, α, β), (ββ, β, β):
        D_jab = eo[ς][:, None, None] - ev[σ][None, :, None] - ev[ς][None, None, :]
        for sI in gen_batch(0, nocc[σ], nbatch):
            if σς == αβ or eval_ss:  # if c_ss == 0, then SS contribution is not counted
                D_ijab = eo[σ][:, None, None, None] + D_jab
                g_ijab = einsum("Pia, Pjb -> ijab", Y_ia_ri[σ][:, sI], Y_ia_ri[ς])
                t_ijab = g_ijab / D_ijab
                eng_bi1[σς] += einsum("ijab, ijab ->", t_ijab, g_ijab)
                if t_ijab_blk:
                    t_ijab_blk[σς][sI] = t_ijab
                if σς in (αα, ββ):
                    eng_bi2[σς] += einsum("ijab, ijba ->", t_ijab, g_ijab)
    return tuple(eng_bi1), tuple(eng_bi2)


def energy_elec_pt2(mf: UDFDH, params=None, eng_bi=None, **kwargs):
    cc, c_os, c_ss = params if params else mf.cc, mf.c_os, mf.c_ss
    eval_ss = True if abs(c_ss) > 1e-7 else False
    eng_bi1, eng_bi2 = eng_bi if eng_bi else energy_elec_mp2(mf, eval_ss=eval_ss, **kwargs)
    eng_os = eng_bi1[αβ]
    eng_ss = 0.5 * (eng_bi1[αα] + eng_bi1[ββ] - eng_bi2[αα] - eng_bi2[ββ])
    eng_pt2 = cc * (c_os * eng_os + c_ss * eng_ss)
    return eng_pt2, eng_os, eng_ss


def energy_elec(mf: UDFDH, params=None, **kwargs):
    eng_nc = mf.energy_elec_nc(**kwargs)[0]
    nocc, nvir = mf.nocc, mf.nvir
    cc, c_os, c_ss = params if params else mf.cc, mf.c_os, mf.c_ss
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


# end region energy evaluation

# region first derivative related


def get_eri_cpks(Y_mo_jk, nocc, cx, eri_cpks=None, max_memory=2000):
    naux, nmo, _ = Y_mo_jk[0].shape
    nvir = nmo - nocc[α], nmo - nocc[β]
    mvir, mocc = max(nvir), max(nocc)
    so = slice(0, nocc[α]), slice(0, nocc[β])
    sv = slice(nocc[α], nmo), slice(nocc[β], nmo)
    if eri_cpks is None:
        eri_cpks = [None, None, None]
        for σς, σ, ς in (αα, α, α), (αβ, α, β), (ββ, β, β):
            eri_cpks[σς] = np.empty((nvir[σ], nocc[σ], nvir[ς], nocc[ς]))
    Y_ai_jk = [np.asarray(Y_mo_jk[σ][:, sv[σ], so[σ]]) for σ in (α, β)]
    Y_ij_jk = [np.asarray(Y_mo_jk[σ][:, so[σ], so[σ]]) for σ in (α, β)]
    nbatch = calc_batch_size(mvir*naux + 2*mocc**2*mvir, max_memory, tot_size(Y_ai_jk, Y_ij_jk))
    for σς, σ, ς in (αα, α, α), (αβ, α, β), (ββ, β, β):
        for sA in gen_batch(nocc[σ], nmo, nbatch):
            sAvir = slice(sA.start - nocc[σ], sA.stop - nocc[σ])
            if σς in (αα, ββ):  # same spin
                eri_cpks[σς][sAvir] = (
                    + 2 * einsum("Pai, Pbj -> aibj", Y_ai_jk[σ][:, sAvir], Y_ai_jk[σ])
                    - cx * einsum("Paj, Pbi -> aibj", Y_ai_jk[σ][:, sAvir], Y_ai_jk[σ])
                    - cx * einsum("Pij, Pab -> aibj", Y_ij_jk[σ], Y_mo_jk[σ][:, sA, sv[σ]]))
            else:
                eri_cpks[σς][sAvir] = 2 * einsum("Pai, Pbj -> aibj", Y_ai_jk[σ][:, sAvir], Y_ai_jk[ς])


def Ax0_cpks_HF(eri_cpks, max_memory=2000):
    nvir = eri_cpks[αα].shape[0], eri_cpks[ββ].shape[0]
    nocc = eri_cpks[αα].shape[1], eri_cpks[ββ].shape[1]
    mvir, mocc = max(nvir), max(nocc)

    def Ax0_cpks_HF_inner(X):
        prop_shape = X[0].shape[:-2]
        X = [X[σ].reshape(-1, X[σ].shape[-2], X[σ].shape[-1]) for σ in (α, β)]
        res = [np.zeros_like(x) for x in X]
        nbatch = calc_batch_size(mocc**2*mvir, max_memory, 0)
        for sA in gen_batch(0, nvir[α], nbatch):
            res[α][:, sA] += einsum("aibj, Abj -> Aai", eri_cpks[αα], X[α])
        for sA in gen_batch(0, nvir[β], nbatch):
            res[β][:, sA] += einsum("aibj, Abj -> Aai", eri_cpks[ββ], X[β])
        for sA in gen_batch(0, nvir[α], nbatch):
            eri_cpks_batch = eri_cpks[αβ][sA]
            res[α][:, sA] += einsum("aibj, Abj -> Aai", eri_cpks_batch, X[β])
            res[β] += einsum("aibj, Aai -> Abj", eri_cpks_batch, X[α][:, sA])
        for σ in α, β:
            res[σ].shape = list(prop_shape) + list(res[σ].shape[-2:])
        return res
    return Ax0_cpks_HF_inner


def Ax0_Core_HF(si, sa, sj, sb, cx, Y_mo_jk, max_memory=2000):
    naux, nmo, _ = Y_mo_jk[0].shape
    ni = [si[σ].stop - si[σ].start for σ in (α, β)]
    na = [sa[σ].stop - sa[σ].start for σ in (α, β)]

    def Ax0_Core_HF_inner(X):
        prop_shape = X[0].shape[:-2]
        X = [X[σ].reshape(-1, X[σ].shape[-2], X[σ].shape[-1]) for σ in (α, β)]
        res = [np.zeros((X[0].shape[0], ni[σ], na[σ])) for σ in (α, β)]
        nbatch = calc_batch_size(nmo**2, max_memory)
        for saux in gen_batch(0, naux, nbatch):
            Y_mo_blk = [Y_mo_jk[σ][saux] for σ in (α, β)]
            for σ, ς in (α, β), (β, α):
                res[σ] += (
                    + 2  * einsum("Pia, Pjb, Ajb -> Aia", Y_mo_blk[σ][:, si[σ], sa[σ]], Y_mo_blk[σ][:, sj[σ], sb[σ]], X[σ])
                    + 2  * einsum("Pia, Pjb, Ajb -> Aia", Y_mo_blk[σ][:, si[σ], sa[σ]], Y_mo_blk[ς][:, sj[ς], sb[ς]], X[ς])
                    - cx * einsum("Pib, Pja, Ajb -> Aia", Y_mo_blk[σ][:, si[σ], sb[σ]], Y_mo_blk[σ][:, sj[σ], sa[σ]], X[σ])
                    - cx * einsum("Pij, Pab, Ajb -> Aia", Y_mo_blk[σ][:, si[σ], sj[σ]], Y_mo_blk[σ][:, sa[σ], sb[σ]], X[σ]))
        for σ in α, β:
            res[σ].shape = list(prop_shape) + list(res[σ].shape[-2:])
        return res
    return Ax0_Core_HF_inner


def Ax0_Core_KS(si, sa, sj, sb, mo_coeff, xc_setting, xc_kernel):
    C = mo_coeff
    ni, mol, grids, xc, dm = xc_setting
    rho, vxc, fxc = xc_kernel

    def Ax0_Core_KS_inner(X):
        prop_shape = X[0].shape[:-2]
        X = [X[σ].reshape(-1, X[σ].shape[-2], X[σ].shape[-1]) for σ in (α, β)]
        dmX = np.array([C[σ][:, sj[σ]] @ X[σ] @ C[σ][:, sb[σ]].T for σ in (α, β)])
        dmX += dmX.swapaxes(-1, -2)
        ax_ao = ni.nr_uks_fxc(mol, grids, xc, dm, dmX, hermi=1, rho0=rho, vxc=vxc, fxc=fxc)
        res = [C[σ][:, si[σ]].T @ ax_ao[σ] @ C[σ][:, sa[σ]] for σ in (α, β)]
        for σ in α, β:
            res[σ].shape = list(prop_shape) + list(res[σ].shape[-2:])
        return res
    return Ax0_Core_KS_inner


# end region first derivative related


class UDFDH(RDFDH):
    def __init__(self,
                 mol: gto.Mole,
                 xc: str or tuple = "XYG3",
                 auxbasis_jk: str or dict or None = None,
                 auxbasis_ri: str or dict or None = None,
                 grids: dft.Grids = None,
                 grids_cpks: dft.Grids = None,
                 unrestricted: bool = True  # only for class initialization
                 ):
        super(UDFDH, self).__init__(mol, xc, auxbasis_jk, auxbasis_ri, grids, grids_cpks, unrestricted)
        self.nocc = mol.nelec
        self.mvir = NotImplemented
        self.mocc = max(max(self.nocc), 1)

    def run_scf(self):
        self.mf_s.grids = self.mf_n.grids = self.grids
        self.build()
        mf = self.mf_s
        if mf.e_tot == 0:
            mf.run()
        # prepare
        C = self.C = self.mo_coeff = mf.mo_coeff
        e = self.e = self.mo_energy = mf.mo_energy
        self.mo_occ = mf.mo_occ
        self.D = mf.make_rdm1(mf.mo_coeff)
        nocc = self.nocc
        nmo = self.nmo = self.C.shape[-1]
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

    def Ax0_Core_HF(self, si, sa, sj, sb, cx=None):
        Y_mo_jk = [self.tensors["Y_mo_jk" + str(σ)] for σ in (α, β)]
        cx = cx if cx else self.cx
        return Ax0_Core_HF(si, sa, sj, sb, cx, Y_mo_jk, max_memory=self.get_memory())

    def Ax0_Core_KS(self, si, sa, sj, sb, xc=None, cpks=False):
        xc = xc if xc else self.xc
        if self.ni._xc_type(xc) == "HF":
            return lambda _: (0, 0)
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
            ax0_hf = ax0_core_hf(X)
            ax0_ks = ax0_core_ks(X)
            return [ax0_hf[σ] + ax0_ks[σ] for σ in (α, β)]
        return fx

    def Ax0_cpks(self):
        so, sv = self.so, self.sv
        ax0_core_ks = self.Ax0_Core_KS(sv, so, sv, so, cpks=True)
        ax0_cpks_hf = Ax0_cpks_HF([self.tensors["eri_cpks" + str(σς)] for σς in (αα, αβ, ββ)], self.get_memory())

        def Ax0_cpks_inner(X):
            ax0_hf = ax0_cpks_hf(X)
            ax0_ks = ax0_core_ks(X)
            return [ax0_hf[σ] + ax0_ks[σ] for σ in (α, β)]
        return Ax0_cpks_inner

    def solve_cpks(self, rhs):
        nocc, nvir = self.nocc, self.nvir

        def reshape_inner(X):
            X_shape = X.shape
            X = X.reshape(-1, X.shape[-1])
            nprop = X.shape[0]
            Xα = X[:, :nocc[α]*nvir[α]].reshape(nprop, nvir[α], nocc[α])
            Xβ = X[:, nocc[α]*nvir[α]:].reshape(nprop, nvir[β], nocc[β])
            res = self.Ax0_cpks()((Xα, Xβ))
            flt = np.zeros_like(X)
            for prop, res_pair in enumerate(zip(*res)):
                flt[prop] = np.concatenate([m.reshape(-1) for m in res_pair])
            flt.shape = X_shape
            return flt

        return ucphf.solve(reshape_inner, self.e, self.mo_occ, rhs, max_cycle=self.cpks_cyc, tol=self.cpks_tol)[0]

    def prepare_integral(self):
        self.run_scf()
        tensors = self.tensors
        C = self.C
        nmo, nocc, nvir = self.nmo, self.nocc, self.nvir

        for σ in α, β:
            y = tensors.create("Y_mo_jk" + str(σ), shape=(self.df_jk.get_naoaux(), nmo, nmo), incore=self._incore_Y_mo)
            get_cderi_mo(self.df_jk, C[σ], y, max_memory=self.get_memory())
            if self.same_aux:
                tensors["Y_mo_ri" + str(σ)] = y
            else:
                y = tensors.create("Y_mo_ri" + str(σ), shape=(self.df_ri.get_naoaux(), nmo, nmo), incore=self._incore_Y_mo)
                get_cderi_mo(self.df_ri, C[σ], y, max_memory=self.get_memory())
        eri_cpks = [None, None, None]
        for σς, σ, ς in (αα, α, α), (αβ, α, β), (ββ, β, β):
            eri_cpks[σς] = tensors.create("eri_cpks" + str(σς), shape=(nvir[σ], nocc[σ], nvir[ς], nocc[ς]), incore=self._incore_Y_mo)
        get_eri_cpks([tensors["Y_mo_jk" + str(σ)] for σ in (α, β)], nocc, self.cx, eri_cpks, self.get_memory())
        return self

    def prepare_pt2(self, dump_t_ijab=True):
        tensors = self.tensors
        nvir, nocc, nmo = self.nvir, self.nocc, self.nmo
        mocc, mvir = max(nocc), max(nvir)
        eo, ev = self.eo, self.ev
        naux = self.df_ri.get_naoaux()
        so, sv = self.so, self.sv
        cc, c_os, c_ss = self.cc, self.c_os, self.c_ss

        D_rdm1 = np.zeros((2, nmo, nmo))
        G_ia_ri = [np.zeros((naux, nocc[σ], nvir[σ])) for σ in (α, β)]
        Y_ia_ri = [np.asarray(tensors["Y_mo_ri" + str(σ)][:, so[σ], sv[σ]]) for σ in (α, β)]

        dump_t_ijab = False if "t_ijab" + str(αα) in tensors else dump_t_ijab  # t_ijab to be dumped
        eval_t_ijab = True if "t_ijab" + str(αα) not in tensors else False     # t_ijab to be evaluated
        if dump_t_ijab:
            for σς, σ, ς in (αα, α, α), (αβ, α, β), (ββ, β, β):
                tensors.create("t_ijab" + str(σς), shape=(nocc[σ], nocc[ς], nvir[σ], nvir[ς]), incore=self._incore_t_ijab)

        eng_bi1, eng_bi2 = [0, 0, 0], [0, 0, 0]
        eval_ss = True if abs(c_ss) > 1e-7 else False
        nbatch = self.calc_batch_size(2 * mocc * mvir ** 2, tot_size(Y_ia_ri) + mocc * mvir ** 2)
        # situation αβ
        for σς, σ, ς in (αα, α, α), (αβ, α, β), (ββ, β, β):
            if σς in (αα, ββ) and not eval_ss:
                continue
            D_jab = eo[ς][:, None, None] - ev[σ][None, :, None] - ev[ς][None, None, :] if eval_t_ijab else None
            for sI in gen_batch(0, nocc[σ], nbatch):
                if eval_t_ijab:
                    D_ijab = eo[σ][:, None, None, None] + D_jab
                    g_ijab = einsum("Pia, Pjb -> ijab", Y_ia_ri[σ][:, sI], Y_ia_ri[ς])
                    t_ijab = g_ijab / D_ijab
                    eng_bi1[σς] += einsum("ijab, ijab ->", t_ijab, g_ijab)
                    if dump_t_ijab:
                        tensors["t_ijab" + str(σς)][sI] = t_ijab
                    if σς in (αα, ββ):
                        eng_bi2[σς] += einsum("ijab, ijba ->", t_ijab, g_ijab)
                else:
                    t_ijab = tensors["t_ijab" + str(σς)][sI]
                if σς in (αα, ββ):
                    T_ijab = cc * 0.5 * c_ss * (t_ijab - t_ijab.swapaxes(-1, -2))
                    D_rdm1[σ, so[σ], so[σ]] -= 2 * einsum("ikab, jkab -> ij", T_ijab, t_ijab)
                    D_rdm1[σ, sv[σ], sv[σ]] += 2 * einsum("ijac, ijbc -> ab", T_ijab, t_ijab)
                    G_ia_ri[σ] += 4 * einsum("ijab, Pjb -> Pia", T_ijab, Y_ia_ri[σ])
                else:  # σς == αβ
                    T_ijab = cc * c_os * t_ijab
                    D_rdm1[α, so[α], so[α]] -= einsum("ikab, jkab -> ij", T_ijab, t_ijab)
                    D_rdm1[β, so[β], so[β]] -= einsum("kiba, kjba -> ij", T_ijab, t_ijab)
                    D_rdm1[α, sv[α], sv[α]] += einsum("ijac, ijbc -> ab", T_ijab, t_ijab)
                    D_rdm1[β, sv[β], sv[β]] += einsum("jica, jicb -> ab", T_ijab, t_ijab)
                    G_ia_ri[α] += 2 * einsum("ijab, Pjb -> Pia", T_ijab, Y_ia_ri[β])
                    G_ia_ri[β] += 2 * einsum("jiba, Pjb -> Pia", T_ijab, Y_ia_ri[α])

        if self.eng_tot is NotImplemented:
            self.kernel(eng_bi=(eng_bi1, eng_bi2))

        tensors.create("D_rdm1", D_rdm1)
        for σ in (α, β):
            tensors.create("G_ia_ri" + str(σ), G_ia_ri[σ])

        return self

    def prepare_lagrangian(self, gen_W=False):
        tensors = self.tensors
        nvir, nocc, nmo, naux = self.nvir, self.nocc, self.nmo, self.df_ri.get_naoaux()
        mvir, mocc = max(nvir), max(nocc)
        so, sv, sa = self.so, self.sv, self.sa
        D_rdm1 = tensors.load("D_rdm1")
        G_ia_ri = [tensors.load("G_ia_ri" + str(σ)) for σ in (α, β)]
        Y_mo_ri = [tensors["Y_mo_ri" + str(σ)] for σ in (α, β)]
        Y_ij_ri = [np.asarray(Y_mo_ri[σ][:, so[σ], so[σ]]) for σ in (α, β)]

        # initialize by directly calling Ax0_Core
        L = list(self.Ax0_Core(sv, so, sa, sa)(D_rdm1))

        nbatch = self.calc_batch_size(mvir**2 + mocc*mvir, tot_size(G_ia_ri + Y_ij_ri))
        if gen_W:
            raise NotImplementedError("generate W will be available soon!")
        else:
            for σ in (α, β):
                L[σ] -= einsum("Pja, Pij -> ai", G_ia_ri[σ], Y_mo_ri[σ][:, so[σ], so[σ]])
                for saux in gen_batch(0, naux, nbatch):
                    L[σ] += einsum("Pib, Pab -> ai", G_ia_ri[σ][saux], Y_mo_ri[σ][saux, sv[σ], sv[σ]])
            if self.xc_n:
                F_0_ao_n = self.mf_n.get_fock(dm=self.D)
                F_0_ai_n = [self.Cv[σ].T @ F_0_ao_n[σ] @ self.Co[σ] for σ in (α, β)]
                for σ in (α, β):
                    L[σ] += 2 * F_0_ai_n[σ]
        for σ in (α, β):
            tensors.create("L" + str(σ), L[σ])
        return self

    def prepare_D_r(self):
        tensors = self.tensors
        sv, so = self.sv, self.so
        D_r = tensors.load("D_rdm1").copy()
        L = [tensors.load("L" + str(σ)) for σ in (α, β)]
        D_r_ai = self.solve_cpks(L)
        for σ in (α, β):
            D_r[σ][sv[σ], so[σ]] = D_r_ai[σ]
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
        d = np.einsum("Auv, suv -> A", h, D)
        d += np.einsum("Auv, spq, sup, svq -> A", h, D_r, C, C, optimize=True)
        d += np.einsum("A, At -> t", mol.atom_charges(), mol.atom_coords())
        return d

    energy_elec_mp2 = energy_elec_mp2
    energy_elec_pt2 = energy_elec_pt2
    energy_elec = energy_elec

