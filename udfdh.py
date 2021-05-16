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
    for σς, σ, ς in ((αα, α, α), (αβ, α, β), (ββ, β, β)):
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
        for σς, σ, ς in ((αα, α, α), (αβ, α, β), (ββ, β, β)):
            if σς == αβ or eval_ss:
                t_ijab_blk[σς] = mf.tensors.create("t_ijab" + str(σς), shape=(nocc[σ], nocc[ς], nvir[σ], nvir[ς]), incore=mf._incore_t_ijab)
    eng_pt2, eng_os, eng_ss = energy_elec_pt2(mf, t2_blk=t_ijab_blk, **kwargs)
    eng_elec = eng_nc + eng_pt2
    return eng_elec, eng_nc, eng_pt2, eng_os, eng_ss


# end region energy evaluation


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

    energy_elec_mp2 = energy_elec_mp2
    energy_elec_pt2 = energy_elec_pt2
    energy_elec = energy_elec

