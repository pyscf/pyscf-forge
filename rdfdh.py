from __future__ import annotations

from pyscf.scf import cphf

from dh.dhutil import parse_xc_dh, gen_batch, calc_batch_size, HybridDict
from pyscf import lib, gto, df, dft
from pyscf.ao2mo import _ao2mo
from pyscf.scf._response_functions import _gen_rhf_response
import numpy as np

einsum = lib.einsum


# region energy evaluation


def kernel(mf_dh: RDFDH, **kwargs):
    mf_dh.build()
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


def energy_elec_mp2(mf_dh: RDFDH, mo_coeff=None, mo_energy=None, dfobj=None, Y_ia=None, t_ijab_blk=None, eval_ss=True, **kwargs):
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
        Y_ia = get_cderi_mo(dfobj, mo_coeff, pqslice=iaslice, max_memory=mf_dh.get_memory())
    # evaluate energy
    eng_bi1 = eng_bi2 = 0
    D_jab = mo_energy[so, None, None] - mo_energy[None, sv, None] - mo_energy[None, None, sv]
    nbatch = calc_batch_size(2 * nocc * nvir ** 2, mf_dh.get_memory(), Y_ia.size + D_jab.size)
    for sI in gen_batch(0, nocc, nbatch):  # batch (i)
        D_ijab = mo_energy[sI, None, None, None] + D_jab
        g_ijab = einsum("Pia, Pjb -> ijab", Y_ia[:, sI], Y_ia)
        t_ijab = g_ijab / D_ijab
        eng_bi1 += einsum("ijab, ijab ->", t_ijab, g_ijab)
        if eval_ss:
            eng_bi2 += einsum("ijab, ijba ->", t_ijab, g_ijab)
        if t_ijab_blk:
            t_ijab_blk[sI] = t_ijab
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
    t_ijab_blk = None
    if mf_dh.with_t_ijab:
        if "t_ijab" in mf_dh.tensors:
            if mf_dh.tensors["t_ijab"].shape == (nocc, nocc, nvir, nvir):
                t_ijab_blk = mf_dh.tensors["t_ijab"]
            else:
                mf_dh.tensors.delete("t_ijab")
                t_ijab_blk = mf_dh.tensors.create("t_ijab", shape=(nocc, nocc, nvir, nvir), incore=mf_dh._incore_t_ijab)
        else:
            t_ijab_blk = mf_dh.tensors.create("t_ijab", shape=(nocc, nocc, nvir, nvir), incore=mf_dh._incore_t_ijab)
    eng_pt2, eng_os, eng_ss = energy_elec_pt2(mf_dh, t2_blk=t_ijab_blk, **kwargs)
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


# endregion energy evaluation

# region first derivative related


def get_eri_cpks(Y_mo, nocc, cx, eri_cpks=None, max_memory=2000):
    naux, nmo, _ = Y_mo.shape
    nvir = nmo - nocc
    so, sv = slice(0, nocc), slice(nocc, nmo)
    # prepare space if bulk of eri_cpks is not provided
    if eri_cpks is None:
        eri_cpks = np.empty((nvir, nocc, nvir, nocc))
    # copy some tensors to memory
    Y_ai = np.asarray(Y_mo[:, sv, so])
    Y_ij = np.asarray(Y_mo[:, so, so])

    nbatch = calc_batch_size(nvir*naux + 2*nocc**2*nvir, max_memory, Y_ai.size + Y_ij.size)
    for sA in gen_batch(nocc, nmo, nbatch):
        sAvir = slice(sA.start - nocc, sA.stop - nocc)
        eri_cpks[sAvir] = (
            + 4 * einsum("Pai, Pbj -> aibj", Y_ai[:, sAvir], Y_ai)
            - cx * einsum("Paj, Pbi -> aibj", Y_ai[:, sAvir], Y_ai)
            - cx * einsum("Pij, Pab -> aibj", Y_ij, Y_mo[:, sA, sv]))


def Ax0_Core_HF(si, sa, sj, sb, cx, Y_mo, max_memory=2000):
    naux, nmo, _ = Y_mo.shape
    nI, nA = si.stop - si.start, sa.stop - sa.start

    def Ax0_Core_HF_inner(X):
        X_shape = X.shape
        X = X.reshape((-1, X_shape[-2], X_shape[-1]))
        res = np.zeros((X.shape[0], nI, nA))
        nbatch = calc_batch_size(nmo**2, max_memory, X.size + res.size)
        for saux in gen_batch(0, naux, nbatch):
            Y_mo_blk = np.asarray(Y_mo[saux])
            res += (
                + 4 * einsum("Pia, Pjb, Ajb -> Aia", Y_mo_blk[:, si, sa], Y_mo_blk[:, sj, sb], X)
                - cx * einsum("Pib, Pja, Ajb -> Aia", Y_mo_blk[:, si, sb], Y_mo_blk[:, sj, sa], X)
                - cx * einsum("Pij, Pab, Ajb -> Aia", Y_mo_blk[:, si, sj], Y_mo_blk[:, sa, sb], X))
        res.shape = list(X_shape[:-2]) + [res.shape[-2], res.shape[-1]]
        return res
    return Ax0_Core_HF_inner


def Ax0_Core_KS(si, sa, sj, sb, mo_coeff, xc_setting, xc_kernel):
    C = mo_coeff
    ni, mol, grids, xc, dm = xc_setting
    rho, vxc, fxc = xc_kernel

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


def solve_cpks(mf_dh: RDFDH, rhs):
    return cphf.solve(mf_dh.Ax0_cpks(), mf_dh.e, mf_dh.mo_occ, rhs, max_cycle=mf_dh.cpks_cyc, tol=mf_dh.cpks_tol)[0]


# endregion first derivative related


class RDFDH(lib.StreamObject):

    def __init__(self,
                 mol: gto.Mole,
                 xc: str or tuple = "XYG3",
                 auxbasis_jk: str or dict or None = None,
                 auxbasis_ri: str or dict or None = None,
                 grids: dft.Grids = None,
                 grids_cpks: dft.Grids = None):
        # tunable flags
        self.with_t_ijab = False  # only in energy calculation; polarizability is forced dump t2 to disk or mem
        self._incore_t_ijab = False
        self._incore_Y_mo = False
        self._incore_eri_cpks = False
        self.cpks_tol = 1e-8
        self.cpks_cyc = 100
        self.max_memory = mol.max_memory
        # Parse xc code
        # It's tricky to say that, self.xc refers to SCF xc, and self.xc_dh refers to double hybrid xc
        self.xc_dh = xc
        xc_list = parse_xc_dh(xc) if isinstance(xc, str) else xc
        self.xc, self.xc_n, self.cc, self.c_os, self.c_ss = xc_list
        self.cx = dft.numint.NumInt().hybrid_coeff(self.xc)
        self.cx_n = dft.numint.NumInt().hybrid_coeff(self.xc_n)
        # parse scf method
        mf = dft.RKS(mol, xc=self.xc).density_fit(auxbasis=auxbasis_jk)
        self.grids = grids if grids else mf.grids
        self.grids_cpks = grids_cpks if grids_cpks else self.grids
        self.mf = mf
        self.mf.grids = self.grids
        # parse auxiliary basis
        auxbasis_jk = auxbasis_jk if auxbasis_jk else df.make_auxbasis(mol, mp2fit=False)
        auxbasis_ri = auxbasis_ri if auxbasis_ri else df.make_auxbasis(mol, mp2fit=True)
        self.df_jk = mf.with_df  # type: df.DF
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
            self.mf_n.grids = self.grids
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

    def get_memory(self):  # leave at least 500MB space anyway
        return max(self.max_memory - lib.current_memory()[0], 500)

    def build(self):
        if self.df_jk.auxmol is None:
            self.df_jk.build()
            self.aux_jk = self.df_jk.auxmol
        if self.df_ri.auxmol is None:
            self.df_ri.build()
            self.aux_ri = self.df_ri.auxmol

    def run_scf(self):
        self.build()
        mf = self.mf
        if mf.e_tot == 0:
            mf.run()
        # make sure that grids in SCF run should be the same to other energy evaluations
        self.grids = self.mf_n.grids = self.mf.grids
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

    # region first derivative related in class

    def Ax0_Core_HF(self, si, sa, sj, sb, cx=None):
        Y_mo = self.tensors["Y_mo_jk"]
        cx = cx if cx else self.cx
        return Ax0_Core_HF(si, sa, sj, sb, cx, Y_mo, max_memory=self.get_memory())

    def Ax0_Core_KS(self, si, sa, sj, sb, xc=None, cpks=False):
        tensors = self.tensors
        xc = xc if xc else self.xc
        cpks_token = "in cpks" if cpks else ""
        grids = self.grids_cpks if cpks else self.grids
        xc_setting = self.mf._numint, self.mol, grids, xc, self.D
        xc_kernel = tensors["rho" + cpks_token], tensors["vxc" + xc + cpks_token], tensors["fxc" + xc + cpks_token]
        mo_coeff = self.mo_coeff
        return Ax0_Core_KS(si, sa, sj, sb, mo_coeff, xc_setting, xc_kernel)

    def Ax0_Core(self, si, sa, sj, sb, xc=None, cpks=False):
        xc = xc if xc else self.xc
        if xc == "HF":
            return self.Ax0_Core_HF(si, sa, sj, sb, cx=1)
        else:
            cx = self.mf._numint.hybrid_coeff(xc)
            ax0_core_hf, ax0_core_ks = self.Ax0_Core_HF(si, sa, sj, sb, cx), self.Ax0_Core_KS(si, sa, sj, sb, xc, cpks)

            def fx(X):
                return ax0_core_hf(X) + ax0_core_ks(X)
            return fx

    def Ax0_Core_resp(self, si, sa, sj, sb, mf=None, mo_coeff=None):
        # this function is only a replacement to Ax0_Core or Ax0_cpks and left for efficiency comparasion
        mf = mf if mf else self.mf
        mo_coeff = mo_coeff if mo_coeff else self.C
        return Ax0_Core_resp(si, sa, sj, sb, mf, mo_coeff, max_memory=self.get_memory())

    def Ax0_cpks(self):
        so, sv = self.so, self.sv
        ax0_core_ks = self.Ax0_Core_KS(sv, so, sv, so, cpks=True) if self.xc != "HF" else None
        ax0_cpks_hf = Ax0_cpks_HF(self.tensors["eri_cpks"], self.get_memory())

        def Ax0_cpks_inner(X):
            res = ax0_cpks_hf(X)
            if ax0_core_ks:
                res += ax0_core_ks(X)
            return res
        return Ax0_cpks_inner

    def prepare_integral(self):
        self.run_scf()
        tensors = self.tensors
        C = self.C
        nmo, nocc, nvir = self.nmo, self.nocc, self.nvir

        # part: Y_mo
        if "Y_mo_jk" not in tensors:
            tensors.create("Y_mo_jk", shape=(self.df_jk.get_naoaux(), nmo, nmo), incore=self._incore_Y_mo)
            get_cderi_mo(self.df_jk, C, tensors["Y_mo_jk"], max_memory=self.get_memory())
        if "Y_mo_ri" not in tensors:
            if self.same_aux:
                tensors["Y_mo_ri"] = tensors["Y_mo_jk"]
            else:
                tensors.create("Y_mo_ri", shape=(self.df_ri.get_naoaux(), nmo, nmo), incore=self._incore_Y_mo)
                get_cderi_mo(self.df_ri, C, tensors["Y_mo_ri"], max_memory=self.get_memory())
        # part: cpks and Ax0_Core preparation
        eri_cpks = tensors.create("eri_cpks", shape=(nvir, nocc, nvir, nocc), incore=self._incore_Y_mo)
        get_eri_cpks(tensors["Y_mo_jk"], nocc, self.cx, eri_cpks, max_memory=self.get_memory())

    def prepare_xc_kernel(self):
        mol = self.mol
        tensors = self.tensors
        C, mo_occ = self.C, self.mo_occ
        ni = self.mf._numint
        if ni._xc_type(self.xc) == "GGA":
            rho, vxc, fxc = ni.cache_xc_kernel(mol, self.grids, self.xc, C, mo_occ, max_memory=self.get_memory())
            tensors.create("rho", rho)
            tensors.create("vxc" + self.xc, vxc)
            tensors.create("fxc" + self.xc, fxc)
            rho, vxc, fxc = ni.cache_xc_kernel(mol, self.grids_cpks, self.xc, C, mo_occ,
                                               max_memory=self.get_memory())
            tensors.create("rho" + "in cpks", rho)
            tensors.create("vxc" + self.xc + "in cpks", vxc)
            tensors.create("fxc" + self.xc + "in cpks", fxc)
        if self.xc_n and ni._xc_type(self.xc_n) == "GGA":
            if "rho" in tensors:
                vxc, fxc = ni.eval_xc(self.xc_n, tensors["rho"], deriv=2, verbose=0)[1:3]
                tensors.create("vxc" + self.xc_n, vxc)
                tensors.create("fxc" + self.xc_n, fxc)
            else:
                rho, vxc, fxc = ni.cache_xc_kernel(mol, self.grids, self.xc_n, C, mo_occ, max_memory=self.get_memory())
                tensors.create("rho", rho)
                tensors.create("vxc" + self.xc, vxc)
                tensors.create("fxc" + self.xc, fxc)

    def prepare_pt2(self, dump_t_ijab=True):
        tensors = self.tensors
        nvir, nocc, nmo = self.nvir, self.nocc, self.nmo
        mo_energy = self.mo_energy
        naux = self.df_ri.get_naoaux()
        so, sv = self.so, self.sv
        cc, c_os, c_ss = self.cc, self.c_os, self.c_ss

        D_rdm1 = np.zeros((nmo, nmo))
        G_ia = np.zeros((naux, nocc, nvir))
        Y_ia = np.asarray(tensors["Y_mo_ri"][:, so, sv])

        dump_t_ijab = False if "t_ijab" in tensors else dump_t_ijab
        flag_t_ijab = False
        if "t_ijab" not in tensors:
            D_jab = mo_energy[so, None, None] - mo_energy[None, sv, None] - mo_energy[None, None, sv]
            flag_t_ijab = True
        else:
            D_jab = None
        if dump_t_ijab:
            tensors.create("t_ijab", shape=(nocc, nocc, nvir, nvir), incore=self._incore_t_ijab)

        eng_bi1 = eng_bi2 = 0
        eval_ss = True if abs(c_ss) > 1e-7 else False
        nbatch = calc_batch_size(4 * nocc * nvir ** 2, self.get_memory(), Y_ia.size + G_ia.size)
        for sI in gen_batch(0, nocc, nbatch):
            if flag_t_ijab:
                D_ijab = mo_energy[sI, None, None, None] + D_jab
                g_ijab = einsum("Pia, Pjb -> ijab", Y_ia[:, sI], Y_ia)
                t_ijab = g_ijab / D_ijab
                if self.eng_pt2 is NotImplemented:
                    eng_bi1 += einsum("ijab, ijab ->", t_ijab, g_ijab)
                    if eval_ss:
                        eng_bi2 += einsum("ijab, ijba ->", t_ijab, g_ijab)
            else:
                t_ijab = tensors["t_ijab"][sI]
            T_ijab = cc * ((c_os + c_ss) * t_ijab - c_ss * t_ijab.swapaxes(-1, -2))
            D_rdm1[sv, sv] += 2 * einsum("ijac, ijbc -> ab", T_ijab, t_ijab)
            D_rdm1[so, so] -= 2 * einsum("ijab, ikab -> jk", T_ijab, t_ijab)
            G_ia[:, sI] = einsum("ijab, Pjb -> Pia", T_ijab, Y_ia)
            if dump_t_ijab:
                tensors["t_ijab"][sI] = t_ijab
        tensors.create("D_rdm1", D_rdm1)
        tensors.create("G_ia", G_ia)

    def prepare_lagrangian(self, gen_W=False):
        tensors = self.tensors

        nvir, nocc, nmo, naux = self.nvir, self.nocc, self.nmo, self.df_ri.get_naoaux()
        so, sv, sa = self.so, self.sv, self.sa

        D_rdm1 = tensors.load("D_rdm1")
        G_ia = tensors.load("G_ia")
        Y_mo = tensors["Y_mo_ri"]
        Y_ij = np.asarray(Y_mo[:, so, so])
        L = np.zeros((nvir, nocc))

        if gen_W:
            Y_ia = np.asarray(Y_mo[:, so, sv])
            W_I = np.zeros((nmo, nmo))
            W_I[so, so] = - 2 * einsum("Pia, Pja -> ij", G_ia, Y_ia)
            W_I[sv, sv] = - 2 * einsum("Pia, Pib -> ab", G_ia, Y_ia)
            W_I[sv, so] = - 4 * einsum("Pja, Pij -> ai", G_ia, Y_mo[:, so, so])
            tensors.create("W_I", W_I)
            L += W_I[sv, so]
        else:
            L -= 4 * einsum("Pja, Pij -> ai", G_ia, Y_ij)

        L += self.Ax0_Core(sv, so, sa, sa)(D_rdm1)

        nbatch = calc_batch_size(nvir ** 2 + nocc * nvir, self.get_memory(), G_ia.size + Y_ij.size)
        for saux in gen_batch(0, naux, nbatch):
            L += 4 * einsum("Pib, Pab -> ai", G_ia[saux], Y_mo[saux, sv, sv])

        if self.xc_n:
            L += 4 * einsum("ua, uv, vi -> ai", self.Cv, self.mf_n.get_fock(dm=self.D), self.Co)

        tensors.create("L", L)

    def prepare_D_r(self):
        tensors = self.tensors
        sv, so = self.sv, self.so
        D_r = tensors.load("D_rdm1").copy()
        L = tensors.load("L")
        D_r[sv, so] = self.solve_cpks(L)
        tensors.create("D_r", D_r)

    def dipole(self):
        D_r = self.tensors["D_r"]
        mol, C, D = self.mol, self.C, self.D
        h = - mol.intor("int1e_r")
        d = einsum("tuv, uv -> t", h, D + C @ D_r @ C.T)
        d += einsum("A, At -> t", mol.atom_charges(), mol.atom_coords())
        return d

    def nuc_grad_method(self):
        # A REALLY DIRTY WAY
        # https://stackoverflow.com/questions/7078134/
        from dh.grad.rdfdh import Gradients
        self.__class__ = Gradients
        Gradients.__init__(self, self.mol, skip_construct=True)
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
