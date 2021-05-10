from __future__ import annotations

from dh.dhutil import parse_xc_dh, gen_batch, calc_batch_size, HybridDict
from pyscf import lib, gto, df, dft
from pyscf.ao2mo import _ao2mo
import numpy as np

einsum = lib.einsum


def kernel(mf_dh: RDFDH, **kwargs):
    eng_tot, eng_nc, eng_pt2, eng_nuc, eng_os, eng_ss = energy_tot(mf_dh, **kwargs)
    mf_dh.e_tot = mf_dh.eng_tot = eng_tot
    mf_dh.eng_nc = eng_nc
    mf_dh.eng_pt2 = eng_pt2
    mf_dh.eng_nuc = eng_nuc
    mf_dh.eng_os = eng_os
    mf_dh.eng_ss = eng_ss
    return eng_tot


def energy_elec_nc(mf_dh: RDFDH, mo_coeff=None, h1e=None, vhf=None, **kwargs):
    if mo_coeff is None:
        if mf_dh.mf.e_tot == 0:
            mf_dh.run_scf()
            if mf_dh.xc_n is None:  # if bDH-like functional, just return SCF energy
                return (mf_dh.mf.e_tot - mf_dh.mf.energy_nuc(), None)
        mo_coeff = mf_dh.mo_coeff
    mo_occ = mf_dh.mo_occ
    if mo_occ is NotImplemented:
        mo_occ = np.zeros((mo_coeff.shape[1], ))
        mo_occ[:mf_dh.mf.mol.nelec[0]] = 2
    dm = mf_dh.mf.make_rdm1(mo_coeff, mo_occ)
    eng_nc = mf_dh.mf_n.energy_elec(dm=dm, h1e=h1e, vhf=vhf)
    return eng_nc


def energy_elec_mp2(mf_dh: RDFDH, mo_coeff=None, mo_energy=None, dfobj=None, Y_ia=None, t2_blk=None, eval_ss=True, **kwargs):
    # prepare mo_coeff, mo_energy
    if mo_coeff is None:
        if mf_dh.mf.e_tot == 0:
            mf_dh.run_scf()
        mo_coeff = mf_dh.mo_coeff
    if mo_energy is None:
        if mf_dh.mf.e_tot == 0:
            mf_dh.run_scf()
        mo_energy = mf_dh.mo_energy
    # prepare essential dimensions
    if Y_ia is None:
        nmo = mo_coeff.shape[1]
        nocc = mf_dh.nocc
        nvir = nmo - nocc
    else:
        nocc, nvir = Y_ia.shape[1:]
        nmo = nocc + nvir
    so, sv = slice(0, nocc), slice(nocc, nmo)
    iaslice = (0, nocc, nocc, nmo)
    # prepare Y_ia (cderi in MO occ-vir block)
    if Y_ia is None:
        if dfobj is None:
            dfobj = mf_dh.df_ri
        Y_ia = get_cderi_mo(dfobj, mo_coeff, pqslice=iaslice, max_memory=mf_dh.get_max_memory())
    # evaluate energy
    eng_bi1, eng_bi2 = 0, 0
    D_jab = mo_energy[so, None, None] - mo_energy[None, sv, None] - mo_energy[None, None, sv]
    nbatch = calc_batch_size(2*nocc*nvir**2, mf_dh.get_max_memory(), Y_ia.size + D_jab.size)
    for sI in gen_batch(0, nocc, nbatch):  # batch (i)
        D_ijab = mo_energy[sI, None, None, None] + D_jab
        g_ijab = einsum("Pia, Pjb -> ijab", Y_ia[:, sI], Y_ia)
        t_ijab = g_ijab / D_ijab
        eng_bi1 += einsum("ijab, ijab ->", t_ijab, g_ijab)
        if eval_ss:
            eng_bi2 += einsum("ijab, ijba ->", t_ijab, g_ijab)
        if t2_blk:
            t2_blk[sI] = t_ijab
    return eng_bi1, eng_bi2


def energy_elec_pt2(mf_dh: RDFDH, params=None, *args, **kwargs):
    if params is None:
        cc, c_os, c_ss = mf_dh.cc, mf_dh.c_os, mf_dh.c_ss
    else:
        cc, c_os, c_ss = params
    eval_ss = True if abs(c_ss) > 1e-7 else False
    eng_bi1, eng_bi2 = energy_elec_mp2(mf_dh, eval_ss=eval_ss, *args, **kwargs)
    return (cc * ((c_os + c_ss) * eng_bi1 - c_ss * eng_bi2),  # Total
            eng_bi1,                                          # OS
            eng_bi1 - eng_bi2)                                # SS


def energy_nuc(mf_dh: RDFDH, **kwargs):
    return mf_dh.mol.energy_nuc()


def energy_elec(mf_dh: RDFDH, **kwargs):
    eng_nc = energy_elec_nc(mf_dh, **kwargs)[0]
    nocc, nvir = mf_dh.nocc, mf_dh.nvir
    t2_blk = None
    if mf_dh.with_t2:
        if "t_ijab" in mf_dh.tensors:
            if mf_dh.tensors["t_ijab"].shape == (nocc, nocc, nvir, nvir):
                t2_blk = mf_dh.tensors["t_ijab"]
            else:
                mf_dh.tensors.delete("t_ijab")
                t2_blk = mf_dh.tensors.create("t_ijab", shape=(nocc, nocc, nvir, nvir), incore=mf_dh._incore_t2)
        else:
            t2_blk = mf_dh.tensors.create("t_ijab", shape=(nocc, nocc, nvir, nvir), incore=mf_dh._incore_t2)
    eng_pt2, eng_os, eng_ss = energy_elec_pt2(mf_dh, t2_blk=t2_blk, **kwargs)
    eng_elec = eng_nc + eng_pt2
    return eng_elec, eng_nc, eng_pt2, eng_os, eng_ss


def energy_tot(mf_dh: RDFDH, **kwargs):
    eng_elec, eng_nc, eng_pt2, eng_os, eng_ss = energy_elec(mf_dh, **kwargs)
    eng_nuc = energy_nuc(mf_dh)
    eng_tot = eng_elec + eng_nuc
    return eng_tot, eng_nc, eng_pt2, eng_nuc, eng_os, eng_ss


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


class RDFDH(lib.StreamObject):

    def __init__(self,
                 mol: gto.Mole,
                 xc: str or tuple = "XYG3",
                 auxbasis_jk: str or dict or None = None,
                 auxbasis_ri: str or dict or None = None,
                 grids: dft.Grids = None,
                 max_memory: float = None):
        # tune flags
        self.with_t2 = True  # only in energy calculation; force or dipole is forced dump t2 to disk or mem
        self._incore_t2 = False
        # Parse xc code
        # It's tricky to say that, self.xc refers to SCF xc, and self.xc_dh refers to double hybrid xc
        self.xc_dh = xc
        xc_list = parse_xc_dh(xc) if isinstance(xc, str) else xc
        self.xc, self.xc_n, self.cc, self.c_os, self.c_ss = xc_list
        # parse scf method
        mf = dft.RKS(mol, xc=self.xc).density_fit(auxbasis=auxbasis_jk)
        mf.grids = grids if grids else mf.grids
        self.mf = mf
        # parse auxiliary basis
        auxbasis_jk = auxbasis_jk if auxbasis_jk else df.make_auxbasis(mol, mp2fit=False)
        auxbasis_ri = auxbasis_ri if auxbasis_ri else df.make_auxbasis(mol, mp2fit=True)
        self.df_jk = mf.with_df
        self.aux_jk = self.df_jk.auxmol
        self.same_aux = True if auxbasis_jk == auxbasis_ri or auxbasis_ri is None else False
        self.df_ri = df.DF(mol, auxbasis_ri) if not self.same_aux else self.df_jk
        self.aux_ri = self.df_ri.auxmol
        # parse non-consistent method
        self.xc_n = None if self.xc_n == self.xc else self.xc_n
        self.mf_n = self.mf
        if self.xc_n:
            self.mf_n = dft.RKS(mol, xc=self.xc_n).density_fit(auxbasis=auxbasis_jk)
            self.mf_n.grids = self.mf.grids
        # parse maximum memory
        self.max_memory = max_memory if max_memory else mol.max_memory
        # other preparation
        self.tensors = HybridDict()
        self.mol = mol
        self.nao = mol.nao
        self.nocc = mol.nelec[0]
        # variables awaits to be build
        self.mo_coeff = NotImplemented
        self.mo_energy = NotImplemented
        self.mo_occ = NotImplemented
        self.C = self.Co = self.Cv = NotImplemented
        self.e = self.eo = self.ev = NotImplemented
        self.D = NotImplemented
        self.nvir = self.nmo = NotImplemented
        self.so = self.sv = self.sa = NotImplemented
        # results
        self.e_tot = NotImplemented
        self.eng_tot = self.eng_nc = self.eng_pt2 = self.eng_nuc = self.eng_os = self.eng_ss = NotImplemented
        # DANGEROUS PLACE
        # we could first initialize nmo as nao
        self.nmo = self.nao
        self.nvir = self.nmo - self.nocc

    def get_max_memory(self):  # leave at least 500MB space anyway
        return max(self.max_memory - lib.current_memory()[0], 500)

    def build(self):
        if self.df_jk.auxmol is None:
            self.df_jk.build()
            self.aux_jk = self.df_jk.auxmol
        if self.df_ri.auxmol is None:
            self.df_ri.build()
            self.aux_ri = self.df_ri.auxmol

    def run_scf(self):
        mf = self.mf

        if mf.e_tot == 0:
            mf.run()
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
        self.Co = self.Co.transpose(0, 1)

    energy_elec_nc = energy_elec_nc
    energy_elec_mp2 = energy_elec_mp2
    energy_elec_pt2 = energy_elec_pt2
    energy_nuc = energy_nuc
    energy_elec = energy_elec
    energy_tot = energy_tot
    kernel = kernel
