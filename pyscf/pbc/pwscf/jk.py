""" J/K builder for PW-SCF
"""

import tempfile
import numpy as np

from pyscf.pbc import tools
from pyscf.pbc.pwscf.pw_helper import (get_kcomp, set_kcomp, acc_kcomp,
                                       scale_kcomp)
from pyscf.pbc.lib.kpts_helper import member, is_zero
from pyscf import lib
from pyscf import __config__


THR_OCC = 1e-10


def get_rho_R(C_ks, mocc_ks, mesh):
    nkpts = len(C_ks)
    rho_R = 0.
    for k in range(nkpts):
        occ = np.where(mocc_ks[k] > THR_OCC)[0].tolist()
        Co_k = get_kcomp(C_ks, k, occ=occ)
        Co_k_R = tools.ifft(Co_k, mesh)
        rho_R += np.einsum("ig,ig->g", Co_k_R.conj(), Co_k_R).real
    return rho_R


def apply_j_kpt(C_k, mesh, vj_R, C_k_R=None):
    if C_k_R is None: C_k_R = tools.ifft(C_k, mesh)
    return tools.fft(C_k_R * vj_R, mesh)


# def apply_j(C_ks, mesh, vj_R, C_ks_R=None, out=None):
#     nkpts = len(C_ks)
#     if out is None: out = [None] * nkpts
#     for k in range(nkpts):
#         C_k = get_kcomp(C_ks, k)
#         C_k_R = None if C_ks_R is None else get_kcomp(C_ks_R, k)
#         Cbar_k = apply_j_kpt(C_k, mesh, vj_R, C_k_R=C_k_R)
#         set_kcomp(Cbar_k, out, k)
#
#     return out


def apply_k_kpt(cell, C_k, kpt1, C_ks, mocc_ks, kpts, mesh, Gv,
                C_k_R=None, C_ks_R=None, exxdiv=None):
    r""" Apply the EXX operator to given MOs

    Math:
        Cbar_k(G) = \sum_{j,k'} \sum_{G'} rho_{jk',ik}(G') v(k-k'+G') C_k(G-G')
    Code:
        rho_r = C_ik_r * C_jk'_r.conj()
        rho_G = FFT(rho_r)
        coulG = get_coulG(k-k')
        v_r = iFFT(rho_G * coulG)
        Cbar_ik_G = FFT(v_r * C_jk'_r)
    """
    ngrids = Gv.shape[0]
    nkpts = len(kpts)
    fac = ngrids**2./(cell.vol*nkpts)

    Cbar_k = np.zeros_like(C_k)
    if C_k_R is None: C_k_R = tools.ifft(C_k, mesh)

    for k2 in range(nkpts):
        kpt2 = kpts[k2]
        coulG = tools.get_coulG(cell, kpt1-kpt2, exx=False, mesh=mesh, Gv=Gv)

        occ = np.where(mocc_ks[k2]>THR_OCC)[0]
        no_k2 = occ.size
        if C_ks_R is None:
            Co_k2 = get_kcomp(C_ks, k2, occ=occ)
            Co_k2_R = tools.ifft(Co_k2, mesh)
            Co_k2 = None
        else:
            Co_k2_R = get_kcomp(C_ks_R, k2, occ=occ)
        for j in range(no_k2):
            Cj_k2_R = Co_k2_R[j]
            vij_R = tools.ifft(
                tools.fft(C_k_R * Cj_k2_R.conj(), mesh) * coulG, mesh)
            Cbar_k += vij_R * Cj_k2_R

    Cbar_k = tools.fft(Cbar_k, mesh) * fac

    return Cbar_k


def apply_k_kpt_support_vec(C_k, W_k):
    Cbar_k = lib.dot(lib.dot(C_k, W_k.conj().T), W_k)
    return Cbar_k


def apply_k_s1(cell, C_ks, mocc_ks, kpts, Ct_ks, ktpts, mesh, Gv, out=None,
               outcore=False):
    nkpts = len(kpts)
    nktpts = len(ktpts)
    ngrids = np.prod(mesh)
    fac = ngrids**2./(cell.vol*nkpts)
    occ_ks = [np.where(mocc_ks[k] > THR_OCC)[0] for k in range(nkpts)]

    if out is None: out = [None] * nktpts

# swap file to hold FFTs
    if outcore:
        swapfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        fswap = lib.H5TmpFile(swapfile.name)
        swapfile = None
        Co_ks_R = fswap.create_group("Co_ks_R")
        Ct_ks_R = fswap.create_group("Ct_ks_R")
    else:
        Co_ks_R = [None] * nkpts
        Ct_ks_R = [None] * nktpts

    for k in range(nkpts):
        Co_k = get_kcomp(C_ks, k, occ=occ_ks[k])
        set_kcomp(tools.ifft(Co_k, mesh), Co_ks_R, k)
        Co_k = None

    for k in range(nktpts):
        Ct_k = get_kcomp(Ct_ks, k)
        set_kcomp(tools.ifft(Ct_k, mesh), Ct_ks_R, k)
        Ct_k = None

    for k1,kpt1 in enumerate(ktpts):
        Ct_k1_R = get_kcomp(Ct_ks_R, k1)
        Ctbar_k1 = np.zeros_like(Ct_k1_R)
        for k2,kpt2 in enumerate(kpts):
            coulG = tools.get_coulG(cell, kpt1-kpt2, exx=False, mesh=mesh,
                                    Gv=Gv)
            Co_k2_R = get_kcomp(Co_ks_R, k2)
            for j in occ_ks[k2]:
                Cj_k2_R = Co_k2_R[j]
                vij_R = tools.ifft(tools.fft(Ct_k1_R * Cj_k2_R.conj(), mesh) *
                                   coulG, mesh)
                Ctbar_k1 += vij_R * Cj_k2_R

        Ctbar_k1 = tools.fft(Ctbar_k1, mesh) * fac
        set_kcomp(Ctbar_k1, out, k1)
        Ctbar_k1 = None

    return out


def apply_k_s2(cell, C_ks, mocc_ks, kpts, mesh, Gv, out=None, outcore=False):
    nkpts = len(kpts)
    ngrids = np.prod(mesh)
    fac = ngrids**2./(cell.vol*nkpts)
    occ_ks = [np.where(mocc_ks[k] > THR_OCC)[0] for k in range(nkpts)]

    if out is None: out = [None] * nkpts

    if isinstance(C_ks, list):
        n_ks = [C_ks[k].shape[0] for k in range(nkpts)]
    else:
        n_ks = [C_ks["%d"%k].shape[0] for k in range(nkpts)]
    no_ks = [np.sum(mocc_ks[k]>THR_OCC) for k in range(nkpts)]
    n_max = np.max(n_ks)
    no_max = np.max(no_ks)

# TODO: non-aufbau configurations
    for k in range(nkpts):
        if np.sum(mocc_ks[k][:no_ks[k]]>THR_OCC) != no_ks[k]:
            raise NotImplementedError("Non-aufbau configurations are not supported.")

    if outcore:
        swapfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        fswap = lib.H5TmpFile(swapfile.name)
        swapfile = None
        C_ks_R = fswap.create_group("C_ks_R")
    else:
        C_ks_R = [None] * nkpts

    for k in range(nkpts):
        C_k = get_kcomp(C_ks, k)
        set_kcomp(tools.ifft(C_k, mesh), C_ks_R, k)
        set_kcomp(np.zeros_like(C_k), out, k)
        C_k = None

    dtype = np.complex128

    buf1 = np.empty(n_max*ngrids, dtype=dtype)
    buf2 = np.empty(no_max*ngrids, dtype=dtype)
    for k1,kpt1 in enumerate(kpts):
        C_k1_R = get_kcomp(C_ks_R, k1)
        no_k1 = no_ks[k1]
        n_k1 = n_ks[k1]
        Cbar_k1 = np.ndarray((n_k1,ngrids), dtype=dtype, buffer=buf1)
        Cbar_k1.fill(0)
        for k2,kpt2 in enumerate(kpts):
            if n_k1 == no_k1 and k2 > k1: continue

            C_k2_R = get_kcomp(C_ks_R, k2)
            no_k2 = no_ks[k2]

            coulG = tools.get_coulG(cell, kpt1-kpt2, exx=False, mesh=mesh,
                                    Gv=Gv)

            # o --> o
            if k2 <= k1:
                Cbar_k2 = np.ndarray((no_k2,ngrids), dtype=dtype, buffer=buf2)
                Cbar_k2.fill(0)

                for i in range(no_k1):
                    jmax = i+1 if k2 == k1 else no_k2
                    jmax2 = i if k2 == k1 else no_k2
                    vji_R = tools.ifft(tools.fft(C_k2_R[:jmax].conj() *
                                       C_k1_R[i], mesh) * coulG, mesh)
                    Cbar_k1[i] += np.sum(vji_R * C_k2_R[:jmax], axis=0)
                    if jmax2 > 0:
                        Cbar_k2[:jmax2] += vji_R[:jmax2].conj() * C_k1_R[i]

                acc_kcomp(Cbar_k2, out, k2, occ=occ_ks[k2])

            # o --> v
            if n_k1 > no_k1:
                for j in range(no_ks[k2]):
                    vij_R = tools.ifft(tools.fft(C_k1_R[no_k1:] *
                                                 C_k2_R[j].conj(), mesh) *
                                       coulG, mesh)
                    Cbar_k1[no_k1:] += vij_R  * C_k2_R[j]

        acc_kcomp(Cbar_k1, out, k1)

    for k in range(nkpts):
        set_kcomp(tools.fft(get_kcomp(out, k), mesh) * fac, out, k)

    return out


def apply_k(cell, C_ks, mocc_ks, kpts, mesh, Gv, Ct_ks=None, ktpts=None,
            exxdiv=None, out=None, outcore=False):
    if Ct_ks is None:
        return apply_k_s2(cell, C_ks, mocc_ks, kpts, mesh, Gv, out, outcore)
    else:
        return apply_k_s1(cell, C_ks, mocc_ks, kpts, Ct_ks, ktpts, mesh, Gv, out, outcore)


def jk(mf, with_jk=None, ace_exx=True, outcore=False):
    if with_jk is None:
        with_jk = PWJK(mf.cell, mf.kpts, exxdiv=mf.exxdiv)
        with_jk.ace_exx = ace_exx
        with_jk.outcore = outcore

    mf.with_jk = with_jk

    return mf


def get_ace_support_vec(cell, C1_ks, mocc1_ks, k1pts, C2_ks=None, k2pts=None,
                        out=None, mesh=None, Gv=None, exxdiv=None, method="cd",
                        outcore=False):
    """ Compute the ACE support vectors for orbitals given by C2_ks and the
    corresponding k-points given by k2pts, using the Fock matrix obtained from
    C1_ks, mocc1_ks, k1pts. If C2_ks and/or k2pts are not provided, their
    values will be set to the C1_ks and/or k1pts. The results are saved to out
    and returned.
    """
    from pyscf.pbc.pwscf.pseudo import get_support_vec
    if mesh is None: mesh = cell.mesh
    if Gv is None: Gv = cell.get_Gv(mesh)

    if outcore:
        swapfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        fswap = lib.H5TmpFile(swapfile.name)
        dname0 = "W_ks"
        W_ks = fswap.create_group(dname0)
    else:
        W_ks = None

    W_ks = apply_k(cell, C1_ks, mocc1_ks, k1pts, mesh, Gv,
                   Ct_ks=C2_ks, ktpts=k2pts, exxdiv=exxdiv, out=W_ks,
                   outcore=outcore)

    if C2_ks is None: C2_ks = C1_ks
    if k2pts is None: k2pts = k1pts
    nk2pts = len(k2pts)

    for k in range(nk2pts):
        C_k = get_kcomp(C2_ks, k)
        W_k = get_kcomp(W_ks, k)
        W_k = get_support_vec(C_k, W_k, method=method)
        set_kcomp(W_k, out, k)
        W_k = None

    if outcore:
        del fswap[dname0]

    return out


class PWJK:

    def __init__(self, cell, kpts, mesh=None, exxdiv=None, **kwargs):
        self.cell = cell
        self.kpts = kpts
        if mesh is None: mesh = cell.mesh
        self.mesh = mesh
        self.Gv = cell.get_Gv(mesh)
        self.exxdiv = exxdiv

        # kwargs
        self.ace_exx = kwargs.get("ace_exx", True)
        self.outcore = kwargs.get("outcore", False)

        # the following are not input options
        self.exx_W_ks = None

    def __init_exx(self):
        if self.outcore:
            self.swapfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
            self.fswap = lib.H5TmpFile(self.swapfile.name)
            self.exx_W_ks = self.fswap.create_group("exx_W_ks")
        else:
            self.exx_W_ks = {}

    def get_Gv(self, mesh):
        if is_zero(np.asarray(mesh)-np.asarray(self.mesh)):
            return self.Gv
        else:
            return self.cell.get_Gv(mesh)

    def get_rho_R(self, C_ks, mocc_ks, mesh=None, Gv=None, ncomp=1):
        if mesh is None: mesh = self.mesh
        if Gv is None: Gv = self.get_Gv(mesh)
        if ncomp == 1:
            rho_R = get_rho_R(C_ks, mocc_ks, mesh)
        else:
            rho_R = 0.
            for comp in range(ncomp):
                C_ks_comp = get_kcomp(C_ks, comp, load=False)
                rho_R += get_rho_R(C_ks_comp, mocc_ks[comp], mesh)
            rho_R *= 1./ncomp
        return rho_R

    def get_vj_R_from_rho_R(self, rho_R, mesh=None, Gv=None):
        if mesh is None: mesh = self.mesh
        if Gv is None: Gv = self.get_Gv(mesh)
        cell = self.cell
        nkpts = len(self.kpts)
        ngrids = Gv.shape[0]
        fac = ngrids**2 / (cell.vol*nkpts)
        vj_R = tools.ifft(tools.fft(rho_R, mesh) * tools.get_coulG(cell, Gv=Gv),
                          mesh).real * fac
        return vj_R

    def get_vj_R(self, C_ks, mocc_ks, mesh=None, Gv=None, ncomp=1):
        if mesh is None: mesh = self.mesh
        if Gv is None: Gv = self.get_Gv(mesh)
        rho_R = self.get_rho_R(C_ks, mocc_ks, mesh, Gv, ncomp)
        vj_R = self.get_vj_R_from_rho_R(rho_R, mesh, Gv)

        return vj_R

    def update_k_support_vec(self, C_ks, mocc_ks, kpts, Ct_ks=None,
                             mesh=None, Gv=None, exxdiv=None, comp=None):
        if self.exx_W_ks is None:
            self.__init_exx()

        nkpts = len(kpts)

        if comp is None:
            out = self.exx_W_ks
        elif isinstance(comp, int):
            keycomp = "%d" % comp
            if keycomp not in self.exx_W_ks:
                if self.outcore:
                    self.exx_W_ks.create_group(keycomp)
                else:
                    self.exx_W_ks[keycomp] = {}
            out = self.exx_W_ks[keycomp]
        else:
            raise RuntimeError("comp must be None or int")

        if self.ace_exx:
            out = get_ace_support_vec(self.cell, C_ks, mocc_ks, kpts,
                                      C2_ks=Ct_ks, k2pts=kpts, out=out,
                                      mesh=mesh, Gv=Gv, exxdiv=exxdiv,
                                      method="cd", outcore=self.outcore)
        else:   # store ifft of Co_ks
            if mesh is None: mesh = self.mesh
            for k in range(nkpts):
                occ = np.where(mocc_ks[k]>THR_OCC)[0]
                Co_k = get_kcomp(C_ks, k, occ=occ)
                set_kcomp(tools.ifft(Co_k, mesh), out, k)

    def apply_j_kpt(self, C_k, mesh=None, vj_R=None, C_k_R=None):
        if mesh is None: mesh = self.mesh
        if vj_R is None: vj_R = self.vj_R
        return apply_j_kpt(C_k, mesh, vj_R, C_k_R=None)

    # NOTE seems this was never used, and since we are adding MGGA term
    # to apply_j_kpt we should remove this to avoid accidentally calling it.
    # def apply_j(self, C_ks, mesh=None, vj_R=None, C_ks_R=None, out=None):
    #     if mesh is None: mesh = self.mesh
    #     if vj_R is None: vj_R = self.vj_R
    #     return apply_j(C_ks, mesh, vj_R, C_ks_R=out, out=out)

    def apply_k_kpt(self, C_k, kpt, mesh=None, Gv=None, exxdiv=None, comp=None):
        if comp is None:
            W_ks = self.exx_W_ks
        elif isinstance(comp, int):
            W_ks = get_kcomp(self.exx_W_ks, comp, load=False)
        else:
            raise RuntimeError("comp must be None or int.")

        if self.ace_exx:
            k = member(kpt, self.kpts)[0]
            W_k = get_kcomp(W_ks, k)
            return apply_k_kpt_support_vec(C_k, W_k)
        else:
            cell = self.cell
            kpts = self.kpts
            nkpts = len(kpts)
            if mesh is None: mesh = self.mesh
            if Gv is None: Gv = self.get_Gv(mesh)
            if exxdiv is None: exxdiv = self.exxdiv
            mocc_ks = [np.ones(get_kcomp(W_ks, k, load=False).shape[0])*2
                       for k in range(nkpts)]
            return apply_k_kpt(cell, C_k, kpt, None, mocc_ks, kpts, mesh, Gv,
                               C_ks_R=W_ks, exxdiv=exxdiv)

    def apply_k(self, C_ks, mocc_ks, kpts, Ct_ks=None, mesh=None, Gv=None,
                exxdiv=None, out=None):
        cell = self.cell
        if mesh is None: mesh = self.mesh
        if Gv is None: Gv = self.get_Gv(mesh)
        if exxdiv is None: exxdiv = self.exxdiv
        return apply_k(cell, C_ks, mocc_ks, kpts, mesh, Gv, Ct_ks=Ct_ks,
                       exxdiv=exxdiv, out=out)
