# dh import
from pyscf.dh.dh import DHBase
from pyscf.dh.dhutil import gen_batch, calc_batch_size, timing, tot_size, hermi_sum_last2dim
from pyscf.dh.xccode import xc_equal
from pyscf.dh.rdfdh import kernel, energy_nuc, energy_tot
from pyscf.dh.mp2_ajz import get_cderi_mo, energy_elec_ump2_ajz
from pyscf.dh.dh import energy_elec_mp2_dfump2_native, energy_elec_mp2_dfump2
# pyscf import
from pyscf import lib, gto, df, dft, scf
from pyscf.lib.numpy_helper import ANTIHERMI
from pyscf.dft.xc_deriv import transform_vxc, transform_fxc
# other import
import h5py
import numpy as np
from functools import partial

einsum = lib.einsum
α, β = 0, 1
αα, αβ, ββ = 0, 1, 2
ndarray = np.ndarray or h5py.Dataset


# region energy evaluation


@timing
def energy_elec_nc(mf, mo_coeff=None, h1e=None, vhf=None, **_):
    if mo_coeff is None:
        if mf.mf_s.e_tot == 0:
            mf.run_scf()
            if mf.xc_n is None:
                return mf.mf_s.e_tot - mf.mf_s.energy_nuc(), None
        mo_coeff = mf.mf_s.mo_coeff
    mo_occ = mf.mf_s.mo_occ
    if mo_occ is NotImplemented:
        mo_occ = scf.uhf.get_occ(mf.mf_s)
    dm = mf.mf_s.make_rdm1(mo_coeff, mo_occ)
    dm = lib.tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
    eng_nc = mf.mf_n.energy_elec(dm=dm, h1e=h1e, vhf=vhf)
    return eng_nc


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


# end region energy evaluation

# region first derivative related



# end region first derivative related


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
        self.mf_s.grids = self.grids
        self.xc_n = None if xc_equal(self.xc_n, self.xc) else self.xc_n
        self.mf_n = self.mf_s
        if self.xc_n:
            self.mf_n = dft.UKS(mol, xc=self.xc_n).density_fit(auxbasis=self.auxbasis_jk)
            self.mf_n.grids = self.mf_s.grids
            self.mf_n.grids = self.grids
        self.ni = getattr(self.mf_s, '_numint', dft.numint.NumInt())
        self.cx = self.ni.hybrid_coeff(self.xc)
        self.cx_n = self.ni.hybrid_coeff(self.xc_n)
        self.df_jk = mf_s.with_df
        self.aux_jk = self.df_jk.auxmol
        self.df_ri = df.DF(mol, self.auxbasis_ri) if not self.same_aux else self.df_jk
        self.aux_ri = self.df_ri.auxmol
        self.nocc = mol.nelec
        self.mvir = NotImplemented
        self.mocc = max(max(self.nocc), 1)
        self.nmo = self.nao
        self.nvir = (self.nmo - self.nocc[α], self.nmo - self.nocc[β])
        if self._scf is not None and self._scf.e_tot != 0:
            self.run_scf()
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


    @timing
    def prepare_pt2(self, dump_t_ijab=True, fast_trans=True):
        tensors = self.tensors
        nvir, nocc, nmo = self.nvir, self.nocc, self.nmo
        mocc, mvir = max(nocc), max(nvir)
        eo, ev = self.eo, self.ev
        naux = self.df_ri.get_naoaux()
        so, sv = self.so, self.sv
        c_os, c_ss = self.c_os, self.c_ss
        eval_ss = True if abs(c_ss) > 1e-7 else False

        D_rdm1 = np.zeros((2, nmo, nmo))
        if not self.eval_pt2:
            tensors.create("D_rdm1", D_rdm1)
            return self

        G_ia_ri = [np.zeros((naux, nocc[σ], nvir[σ])) for σ in (α, β)]
        Y_ia_ri = [np.asarray(tensors["Y_mo_ri" + str(σ)][:, so[σ], sv[σ]]) for σ in (α, β)]

        dump_t_ijab = False if "t_ijab" + str(αα) in tensors else dump_t_ijab  # t_ijab to be dumped
        eval_t_ijab = True if "t_ijab" + str(αα) not in tensors else False     # t_ijab to be evaluated
        if dump_t_ijab:
            for σς, σ, ς in (αα, α, α), (αβ, α, β), (ββ, β, β):
                if σς in (αα, ββ) and not eval_ss:
                    continue
                tensors.create("t_ijab" + str(σς), shape=(nocc[σ], nocc[ς], nvir[σ], nvir[ς]), incore=self._incore_t_ijab)

        eng_bi1, eng_bi2 = [0, 0, 0], [0, 0, 0]
        nbatch = self.calc_batch_size(2 * mocc * mvir ** 2, tot_size(Y_ia_ri) + mocc * mvir ** 2)
        # situation αβ
        for σς, σ, ς in (αα, α, α), (αβ, α, β), (ββ, β, β):
            if σς in (αα, ββ) and not eval_ss:
                continue
            D_jab = eo[ς][:, None, None] - ev[σ][None, :, None] - ev[ς][None, None, :] if eval_t_ijab else None
            for sI in gen_batch(0, nocc[σ], nbatch):
                if eval_t_ijab:
                    D_ijab = eo[σ][sI, None, None, None] + D_jab
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
                    # T_ijab = cc * 0.5 * c_ss * (t_ijab - t_ijab.swapaxes(-1, -2))
                    T_ijab = 0.5 * c_ss * hermi_sum_last2dim(t_ijab, hermi=ANTIHERMI, inplace=False)
                    D_rdm1[σ, so[σ], so[σ]] -= 2 * einsum("kiab, kjab -> ij", T_ijab, t_ijab)
                    D_rdm1[σ, sv[σ], sv[σ]] += 2 * einsum("ijac, ijbc -> ab", T_ijab, t_ijab)
                    G_ia_ri[σ][:, sI] += 4 * einsum("ijab, Pjb -> Pia", T_ijab, Y_ia_ri[σ])
                else:  # σς == αβ
                    T_ijab = c_os * t_ijab
                    # D_rdm1[α, so[α], so[α]] -= einsum("ikab, jkab -> ij", T_ijab, t_ijab)
                    # D_rdm1[β, so[β], so[β]] -= einsum("kiba, kjba -> ij", T_ijab, t_ijab)
                    # D_rdm1[α, sv[α], sv[α]] += einsum("ijac, ijbc -> ab", T_ijab, t_ijab)
                    # D_rdm1[β, sv[β], sv[β]] += einsum("jica, jicb -> ab", T_ijab, t_ijab)
                    # G_ia_ri[α][:, sI] += 2 * einsum("ijab, Pjb -> Pia", T_ijab, Y_ia_ri[β])
                    # G_ia_ri[β][:, sI] += 2 * einsum("jiba, Pjb -> Pia", T_ijab, Y_ia_ri[α])
                    for sJ in gen_batch(0, nocc[α], nbatch):
                        if sI == sJ:
                            t_jkab = t_ijab
                        elif sI.start < sJ.start:
                            continue
                        else:
                            t_jkab = tensors["t_ijab" + str(αβ)][sJ]
                        D_tmp = einsum("ikab, jkab -> ij", T_ijab, t_jkab)
                        D_rdm1[α, sI, sJ] -= D_tmp
                        if sI != sJ:
                            D_rdm1[α, sJ, sI] -= D_tmp.swapaxes(-1, -2)
                    D_rdm1[β, so[β], so[β]] -= einsum("kiba, kjba -> ij", T_ijab, t_ijab)
                    D_rdm1[α, sv[α], sv[α]] += einsum("ijac, ijbc -> ab", T_ijab, t_ijab)
                    D_rdm1[β, sv[β], sv[β]] += einsum("jica, jicb -> ab", T_ijab, t_ijab)
                    G_ia_ri[α][:, sI] += 2 * einsum("ijab, Pjb -> Pia", T_ijab, Y_ia_ri[β])
                    G_ia_ri[β] += 2 * einsum("jiba, Pjb -> Pia", T_ijab, Y_ia_ri[α][:, sI])

        if self.eng_tot is NotImplemented:
            kernel(self, eng_bi=(None, eng_bi1, eng_bi2))

        tensors.create("D_rdm1", D_rdm1)
        for σ in (α, β):
            tensors.create("G_ia_ri" + str(σ), G_ia_ri[σ])

        return self

    @timing
    # A REALLY DIRTY WAY transform to son class https://stackoverflow.com/questions/7078134/
    def nuc_grad_method(self):
        from pyscf.dh.grad.udfdh import Gradients
        self.__class__ = Gradients
        Gradients.__init__(self, self.mol, skip_construct=True)
        return self

    def polar_method(self):
        from pyscf.dh.polar.udfdh import Polar
        self.__class__ = Polar
        Polar.__init__(self, self.mol, skip_construct=True)
        return self

    energy_elec_nc = energy_elec_nc
    energy_elec_pt2 = energy_elec_pt2
    energy_elec_mp2 = energy_elec_ump2_ajz
    energy_nuc = energy_nuc
    energy_elec = energy_elec
    energy_tot = energy_tot
    kernel = kernel

