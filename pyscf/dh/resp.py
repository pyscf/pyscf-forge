from pyscf.scf import cphf, ucphf
from pyscf import lib
from pyscf.dh.dhutil import gen_batch, calc_batch_size, timing, tot_size, HybridDict, get_rho_from_dm_gga
import numpy as np
einsum = lib.einsum

α, β = 0, 1
αα, αβ, ββ = 0, 1, 2


# ---- Restricted module-level helpers (from rdfdh.py) ----

def _r_get_eri_cpks(Y_mo_jk, nocc, cx, eri_cpks=None, max_memory=2000):
    naux, nmo, _ = Y_mo_jk.shape
    nvir = nmo - nocc
    so, sv = slice(0, nocc), slice(nocc, nmo)
    if eri_cpks is None:
        eri_cpks = np.empty((nvir, nocc, nvir, nocc))
    Y_ai_jk = np.asarray(Y_mo_jk[:, sv, so])
    Y_ij_jk = np.asarray(Y_mo_jk[:, so, so])
    nbatch = calc_batch_size(nvir*naux + 2*nocc**2*nvir, max_memory, Y_ai_jk.size + Y_ij_jk.size)
    for sA in gen_batch(nocc, nmo, nbatch):
        sAvir = slice(sA.start - nocc, sA.stop - nocc)
        eri_cpks[sAvir] = (
            + 4 * einsum("Pai, Pbj -> aibj", Y_ai_jk[:, sAvir], Y_ai_jk)
            - cx * einsum("Paj, Pbi -> aibj", Y_ai_jk[:, sAvir], Y_ai_jk)
            - cx * einsum("Pij, Pab -> aibj", Y_ij_jk, Y_mo_jk[:, sA, sv]))


def _r_Ax0_Core_HF(si, sa, sj, sb, cx, Y_mo_jk, max_memory=2000):
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
            for A in range(X.shape[0]):
                res[A] += (
                    + 4 * einsum("Pia, Pjb, jb -> ia", Y_mo_blk[:, si, sa], Y_mo_blk[:, sj, sb], X[A])
                    - cx * einsum("Pib, Pja, jb -> ia", Y_mo_blk[:, si, sb], Y_mo_blk[:, sj, sa], X[A])
                    - cx * einsum("Pij, Pab, jb -> ia", Y_mo_blk[:, si, sj], Y_mo_blk[:, sa, sb], X[A]))
        res.shape = list(X_shape[:-2]) + [res.shape[-2], res.shape[-1]]
        return res
    return Ax0_Core_HF_inner


def _r_Ax0_Core_KS(si, sa, sj, sb, mo_coeff, xc_setting, xc_kernel):
    from pyscf.dft.xc_deriv import transform_vxc, transform_fxc
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


def _r_Ax0_Core_resp(si, sa, sj, sb, mf, mo_coeff, max_memory=2000):
    from pyscf.scf._response_functions import _gen_rhf_response
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


def _r_Ax0_cpks_HF(eri_cpks, max_memory=2000):
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


# ---- Unrestricted module-level helpers (from udfdh.py) ----

def _u_get_eri_cpks(Y_mo_jk, nocc, cx, eri_cpks=None, max_memory=2000):
    naux, nmo, _ = Y_mo_jk[0].shape
    nvir = nmo - nocc[α], nmo - nocc[β]
    mvir, mocc = max(nvir), max(nocc)
    so = slice(0, nocc[α]), slice(0, nocc[β])
    sv = slice(nocc[α], nmo), slice(nocc[β], nmo)
    if eri_cpks is None:
        eri_cpks = [np.empty((nv_σ, nc_σ, nv_ς, nc_ς)) for nc_σ, nv_σ, nc_ς, nv_ς in
                    zip(nocc, nvir, nocc, nvir)]
    Y_ai_jk = [np.asarray(Y_mo_jk[σ][:, sv[σ], so[σ]]) for σ in (α, β)]
    Y_ij_jk = [np.asarray(Y_mo_jk[σ][:, so[σ], so[σ]]) for σ in (α, β)]
    nbatch = calc_batch_size(mvir * naux + 2 * mocc ** 2 * mvir, max_memory, tot_size(Y_ai_jk + Y_ij_jk))
    for σς, σ, ς in (αα, α, α), (αβ, α, β), (ββ, β, β):
        for sA in gen_batch(nocc[σ], nmo, nbatch):
            sAvir = slice(sA.start - nocc[σ], sA.stop - nocc[σ])
            if σς in (αα, ββ):
                eri_cpks[σς][sAvir] = (
                    + 2 * einsum("Pai, Pbj -> aibj", Y_ai_jk[σ][:, sAvir], Y_ai_jk[σ])
                    - cx * einsum("Paj, Pbi -> aibj", Y_ai_jk[σ][:, sAvir], Y_ai_jk[ς])
                    - cx * einsum("Pij, Pab -> aibj", Y_ij_jk[σ], Y_mo_jk[ς][:, sA, sv[ς]]))
            else:
                eri_cpks[σς][sAvir] = 2 * einsum("Pai, Pbj -> aibj", Y_ai_jk[σ][:, sAvir], Y_ai_jk[ς])


def _u_Ax0_cpks_HF(eri_cpks, max_memory=2000):
    nvir = eri_cpks[αα].shape[0], eri_cpks[ββ].shape[0]
    nocc = eri_cpks[αα].shape[1], eri_cpks[ββ].shape[1]
    mvir, mocc = max(nvir), max(nocc)

    @timing
    def Ax0_cpks_HF_inner(X):
        prop_shape = X[0].shape[:-2]
        X = [X[σ].reshape(-1, X[σ].shape[-2], X[σ].shape[-1]) for σ in (α, β)]
        res = [np.zeros_like(x) for x in X]
        nbatch = calc_batch_size(mocc**2*mvir, max_memory, 0)
        for sA in gen_batch(0, nvir[α], nbatch):
            res[α][:, sA] += einsum("aibj, Abj -> Aai", eri_cpks[αα][sA], X[α])
        for sA in gen_batch(0, nvir[β], nbatch):
            res[β][:, sA] += einsum("aibj, Abj -> Aai", eri_cpks[ββ][sA], X[β])
        for sA in gen_batch(0, nvir[α], nbatch):
            eri_cpks_batch = eri_cpks[αβ][sA]
            res[α][:, sA] += einsum("aibj, Abj -> Aai", eri_cpks_batch, X[β])
            res[β] += einsum("aibj, Aai -> Abj", eri_cpks_batch, X[α][:, sA])
        for σ in α, β:
            res[σ].shape = list(prop_shape) + list(res[σ].shape[-2:])
        return res
    return Ax0_cpks_HF_inner


def _u_Ax0_Core_HF(si, sa, sj, sb, cx, Y_mo_jk, max_memory=2000):
    naux, nmo, _ = Y_mo_jk[0].shape
    ni = [si[σ].stop - si[σ].start for σ in (α, β)]
    na = [sa[σ].stop - sa[σ].start for σ in (α, β)]

    @timing
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


def _u_Ax0_Core_KS(si, sa, sj, sb, mo_coeff, xc_setting, xc_kernel):
    from pyscf.dft.xc_deriv import transform_vxc, transform_fxc
    C = mo_coeff
    ni, mol, grids, xc, dm = xc_setting
    rho, vxc, fxc = xc_kernel
    vxc_ = transform_vxc(rho, vxc, "GGA", spin=1)
    fxc_ = transform_fxc(rho, vxc, fxc, "GGA", spin=1)

    @timing
    def Ax0_Core_KS_inner(X):
        prop_shape = X[0].shape[:-2]
        X = [X[σ].reshape(-1, X[σ].shape[-2], X[σ].shape[-1]) for σ in (α, β)]
        dmX = np.array([C[σ][:, sj[σ]] @ X[σ] @ C[σ][:, sb[σ]].T for σ in (α, β)])
        dmX += dmX.swapaxes(-1, -2)
        ax_ao = ni.nr_uks_fxc(mol, grids, xc, dm, dmX, hermi=1, rho0=rho, vxc=vxc_, fxc=fxc_)
        res = [C[σ][:, si[σ]].T @ ax_ao[σ] @ C[σ][:, sa[σ]] for σ in (α, β)]
        for σ in α, β:
            res[σ].shape = list(prop_shape) + list(res[σ].shape[-2:])
        return res
    return Ax0_Core_KS_inner


# ---- RespMixin classes ----

class RespMixin:
    """Mixin providing CPHF/CPKS solver, prepare_D_r, and dipole."""

    cpks_tol = 1e-8
    cpks_cyc = 100
    _incore_Y_mo = False

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

    def solve_cpks(self, rhs):
        if isinstance(rhs, int) and rhs == 0:
            return 0
        if isinstance(rhs, tuple) and isinstance(rhs[0], int) and all(r == 0 for r in rhs):
            return rhs
        if self.D.ndim == 2:
            return cphf.solve(self.Ax0_cpks(), self.mo_energy, self.mo_occ, rhs,
                              max_cycle=self.cpks_cyc, tol=self.cpks_tol)[0]
        else:
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

            return ucphf.solve(reshape_inner, self.mo_energy, self.mo_occ, rhs,
                               max_cycle=self.cpks_cyc, tol=self.cpks_tol)[0]

    def prepare_D_r(self):
        tensors = self.tensors
        sv, so = self.sv, self.so
        D_r = tensors.load("D_rdm1").copy()
        if D_r.ndim == 2:
            L = tensors.load("L")
            D_r[sv, so] = self.solve_cpks(L)
        else:
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
        mol, C, D = self.mol, self.mo_coeff, self.D
        h = - mol.intor("int1e_r")
        if D.ndim == 2:
            d = einsum("tuv, uv -> t", h, D + C @ D_r @ C.T)
        else:
            d = einsum("tuv, suv -> t", h, D)
            d += einsum("tuv, spq, sup, svq -> t", h, D_r, C, C, optimize=True)
        d += einsum("A, At -> t", mol.atom_charges(), mol.atom_coords())
        return d


class RDHRespMixin(RespMixin):
    """Restricted response mixin: prepare_integral, Ax0_*, prepare_lagrangian."""

    def prepare_integral(self):
        from pyscf.dh.mp2_ajz import get_cderi_mo
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
        _r_get_eri_cpks(tensors["Y_mo_jk"], nocc, self.cx, eri_cpks, max_memory=self.get_memory())
        return self

    def Ax0_Core_HF(self, si, sa, sj, sb, cx=None):
        Y_mo_jk = self.tensors["Y_mo_jk"]
        cx = cx if cx else self.cx
        return _r_Ax0_Core_HF(si, sa, sj, sb, cx, Y_mo_jk, max_memory=self.get_memory())

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
        return _r_Ax0_Core_KS(si, sa, sj, sb, mo_coeff, xc_setting, xc_kernel)

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
        return _r_Ax0_Core_resp(si, sa, sj, sb, mf, mo_coeff, max_memory=self.get_memory())

    def Ax0_cpks(self):
        so, sv = self.so, self.sv
        ax0_core_ks = self.Ax0_Core_KS(sv, so, sv, so, cpks=True)
        ax0_cpks_hf = _r_Ax0_cpks_HF(self.tensors["eri_cpks"], self.get_memory())

        def Ax0_cpks_inner(X):
            return ax0_cpks_hf(X) + ax0_core_ks(X)
        return Ax0_cpks_inner

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


class UDHRespMixin(RespMixin):
    """Unrestricted response mixin: prepare_integral, Ax0_*, prepare_lagrangian."""

    def prepare_integral(self):
        from pyscf.dh.mp2_ajz import get_cderi_mo
        self.run_scf()
        tensors = self.tensors
        C = self.mo_coeff
        nmo, nocc, nvir = self.nmo, self.nocc, self.nvir

        for σ in α, β:
            y = tensors.create("Y_mo_jk" + str(σ), shape=(self.df_jk.get_naoaux(), nmo, nmo), incore=self._incore_Y_mo)
            get_cderi_mo(self.df_jk, C[σ], y, max_memory=self.get_memory())
            if self.eval_pt2:
                y = tensors.create("Y_mo_ri" + str(σ), shape=(self.df_ri.get_naoaux(), nmo, nmo), incore=self._incore_Y_mo)
                get_cderi_mo(self.df_ri, C[σ], y, max_memory=self.get_memory())
        eri_cpks = [None, None, None]
        for σς, σ, ς in (αα, α, α), (αβ, α, β), (ββ, β, β):
            eri_cpks[σς] = tensors.create("eri_cpks" + str(σς), shape=(nvir[σ], nocc[σ], nvir[ς], nocc[ς]), incore=self._incore_Y_mo)
        _u_get_eri_cpks([tensors["Y_mo_jk" + str(σ)] for σ in (α, β)], nocc, self.cx, eri_cpks, self.get_memory())
        return self

    def Ax0_Core_HF(self, si, sa, sj, sb, cx=None):
        Y_mo_jk = [self.tensors["Y_mo_jk" + str(σ)] for σ in (α, β)]
        cx = cx if cx else self.cx
        return _u_Ax0_Core_HF(si, sa, sj, sb, cx, Y_mo_jk, max_memory=self.get_memory())

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
        return _u_Ax0_Core_KS(si, sa, sj, sb, mo_coeff, xc_setting, xc_kernel)

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
        ax0_cpks_hf = _u_Ax0_cpks_HF([self.tensors["eri_cpks" + str(σς)] for σς in (αα, αβ, ββ)], self.get_memory())

        def Ax0_cpks_inner(X):
            ax0_hf = ax0_cpks_hf(X)
            ax0_ks = ax0_core_ks(X)
            return [ax0_hf[σ] + ax0_ks[σ] for σ in (α, β)]
        return Ax0_cpks_inner

    def prepare_lagrangian(self, gen_W=False):
        tensors = self.tensors
        nvir, nocc, nmo, naux = self.nvir, self.nocc, self.nmo, self.df_ri.get_naoaux()
        mvir, mocc = max(nvir), max(nocc)
        so, sv, sa = self.so, self.sv, self.sa
        D_rdm1 = tensors.load("D_rdm1")

        L = [np.zeros((nvir[σ], nocc[σ])) for σ in (α, β)]
        if self.xc_n:
            F_0_ao_n = self.mf_n.get_fock(dm=self.D)
            F_0_ai_n = [self.Cv[σ].T @ F_0_ao_n[σ] @ self.Co[σ] for σ in (α, β)]
            for σ in (α, β):
                L[σ] += 2 * F_0_ai_n[σ]
        if not self.eval_pt2:
            for σ in (α, β):
                tensors.create("L" + str(σ), L[σ])
            return self

        G_ia_ri = [tensors.load("G_ia_ri" + str(σ)) for σ in (α, β)]
        Y_mo_ri = [tensors["Y_mo_ri" + str(σ)] for σ in (α, β)]
        Y_ij_ri = [np.asarray(Y_mo_ri[σ][:, so[σ], so[σ]]) for σ in (α, β)]
        Y_ia_ri = [np.asarray(Y_mo_ri[σ][:, so[σ], sv[σ]]) for σ in (α, β)]

        r = self.Ax0_Core(sv, so, sa, sa)(D_rdm1)
        for σ in (α, β):
            L[σ] += r[σ]

        if gen_W:
            W_I = np.zeros((2, nmo, nmo))
            for σ in (α, β):
                W_I[σ][so[σ], so[σ]] = - 0.5 * einsum("Pia, Pja -> ij", G_ia_ri[σ], Y_ia_ri[σ])
                W_I[σ][sv[σ], sv[σ]] = - 0.5 * einsum("Pia, Pib -> ab", G_ia_ri[σ], Y_ia_ri[σ])
                W_I[σ][sv[σ], so[σ]] = - einsum("Pja, Pij -> ai", G_ia_ri[σ], Y_ij_ri[σ])
                L[σ] += W_I[σ][sv[σ], so[σ]]
                tensors.create("W_I", W_I)
        else:
            for σ in (α, β):
                L[σ] -= einsum("Pja, Pij -> ai", G_ia_ri[σ], Y_ij_ri[σ])

        nbatch = self.calc_batch_size(mvir ** 2 + mocc * mvir, tot_size(G_ia_ri + Y_ij_ri))
        for σ in (α, β):
            for saux in gen_batch(0, naux, nbatch):
                L[σ] += einsum("Pib, Pab -> ai", G_ia_ri[σ][saux], Y_mo_ri[σ][saux, sv[σ], sv[σ]])

        for σ in (α, β):
            tensors.create("L" + str(σ), L[σ])
        return self



