""" All actual implementation of PW-related PPs go here.
    The wrapper for calling the functions here go to pw_helper.py
"""

import tempfile
import numpy as np
import scipy.linalg
from scipy.special import dawsn
from scipy.interpolate import make_interp_spline

from pyscf.pbc.pwscf.pw_helper import (get_kcomp, set_kcomp, get_C_ks_G, orth,
                                       get_mesh_map, wf_fft, wf_ifft)
from pyscf.pbc.gto import pseudo as gth_pseudo
from pyscf.pbc import tools
from pyscf.pbc.lib.kpts_helper import member
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__


IOBLK = getattr(__config__, "pbc_pwscf_pseudo_IOBLK", 4000) # unit MB


""" Wrapper functions
"""
def get_vpplocR(cell, mesh=None, Gv=None):
    if mesh is None: mesh = cell.mesh
    if Gv is None: Gv = cell.get_Gv(mesh=mesh)
    SI = cell.get_SI(Gv=Gv)
    ngrids = Gv.shape[0]
    fac = ngrids / cell.vol
    vpplocG = np.einsum("ag,ag->g", SI, get_vpplocG(cell, mesh, Gv))
    vpplocR = tools.ifft(vpplocG, mesh).real * fac

    return vpplocR


def get_vpplocG(cell, mesh=None, Gv=None):
    if mesh is None: mesh = cell.mesh
    if Gv is None: Gv = cell.get_Gv(mesh=mesh)

    if len(cell._ecp) > 0:
        return get_vpplocG_ccecp(cell, Gv)
    elif cell.pseudo is not None:
        if "GTH" in cell.pseudo.upper():
            return get_vpplocG_gth(cell, Gv)
        elif cell.pseudo == "SG15":
            return get_vpplocG_sg15(cell, Gv)
        else:
            raise NotImplementedError("Pseudopotential %s is currently not supported." % (str(cell.pseudo)))
    else:
        return get_vpplocG_alle(cell, Gv)


def apply_vppl_kpt(cell, C_k, mesh=None, vpplocR=None, C_k_R=None, basis=None):
    if mesh is None: mesh = cell.mesh
    if vpplocR is None: vpplocR = get_vpplocR(cell, mesh)
    if C_k_R is None: C_k_R = wf_ifft(C_k, mesh, basis=basis)
    return wf_fft(C_k_R * vpplocR, mesh, basis=basis)


def apply_vppnl_kpt(cell, C_k, kpt, mesh=None, Gv=None, basis=None):
    if mesh is None: mesh = cell.mesh
    if Gv is None: Gv = cell.get_Gv(mesh=mesh)

    if len(cell._ecp) > 0:
        return apply_vppnl_kpt_ccecp(cell, C_k, kpt, Gv)
    elif cell.pseudo is not None:
        if "GTH" in cell.pseudo.upper():
            return apply_vppnl_kpt_gth(cell, C_k, kpt, Gv, basis=basis)
        elif cell.pseudo == "SG15":
            return apply_vppnl_kpt_sg15(cell, C_k, kpt, Gv, basis=basis)
        else:
            raise NotImplementedError("Pseudopotential %s is currently not "
                                      "supported." % (str(cell.pseudo)))
    else:
        return apply_vppnl_kpt_alle(cell, C_k, kpt, Gv)


""" PW-PP class implementation goes here
"""
def get_pp_type(cell):
    hasecp = len(cell._ecp) > 0
    haspp = len(cell._pseudo) > 0
    if not (hasecp or haspp):
        return "alle"
    elif haspp:
        if isinstance(cell.pseudo, str):
            if cell.pseudo == "SG15":
                return "SG15"
            assert("GTH" in cell.pseudo.upper())
        elif isinstance(cell.pseudo, dict):
            for key,pp in cell.pseudo.items():
                assert("GTH" in pp.upper())
        else:
            raise RuntimeError("Unknown pseudo type %s" % (str(cell.pseudo)))
        return "gth"
    else:
        if isinstance(cell.ecp, str):
            assert("CCECP" in cell.ecp.upper())
        elif isinstance(cell.ecp, dict):
            for key,pp in cell.ecp.items():
                assert("CCECP" in pp.upper())
        else:
            raise RuntimeError("Unknown ecp type %s" % (str(cell.ecp)))
        return "ccecp"


def pseudopotential(mf, with_pp=None, mesh=None, outcore=False, **kwargs):
    def set_kw(with_pp_, key):
        val = kwargs.get(key, None)
        if val is not None: setattr(with_pp_, key, val)

    if with_pp is None:
        with_pp = PWPP(mf.cell, mf.kpts, mesh=mesh, outcore=outcore)
        set_kw(with_pp, "ecpnloc_method")
        set_kw(with_pp, "ecpnloc_kbbas")
        set_kw(with_pp, "ecpnloc_ke_cutoff")
        set_kw(with_pp, "ecpnloc_use_numexpr")

    mf.with_pp = with_pp

    return mf


class PWPP:

    ecpnloc_method = getattr(__config__, "pbc_pwscf_pseudo_PWPP_ecpnloc_method",
                             "kb")  # other options: "direct"
    ecpnloc_kbbas = getattr(__config__, "pbc_pwscf_pseudo_PWPP_ecpnloc_method",
                            "ccecp-cc-pvqz")
    ecpnloc_ke_cutoff = getattr(__config__,
                                "pbc_pwscf_pseudo_PWPP_ecpnloc_ke_cutoff", None)
    threshold_svec = getattr(__config__, "pbc_pwscf_pseudo_PWPP_threshold_svec",
                             1e-12)

    def __init__(self, cell, kpts, mesh=None, **kwargs):
        self.cell = cell
        self.stdout = cell.stdout
        self.verbose = cell.verbose
        self.kpts = kpts
        if mesh is None: mesh = cell.mesh
        self.mesh = mesh
        self.Gv = cell.get_Gv(mesh)
        logger.debug(self, "Initializing PP local part")
        self.vpplocR = get_vpplocR(cell, self.mesh, self.Gv)

        self.pptype = get_pp_type(cell)
        self._ecp = None
        self.vppnlocGG = None
        self.vppnlocWks = None
        self._ecpnloc_initialized = False

        # kwargs
        self.outcore = kwargs.get("outcore", False)

        # debug options
        self.ecpnloc_use_numexpr = False

    def initialize_ecpnloc(self):
        if self.pptype == "ccecp":
            logger.debug(self, "Initializing ccECP non-local part")
            cell = self.cell
            dtype = np.complex128
            self._ecp = format_ccecp_param(cell)
            if self.outcore:
                self.swapfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
                self.fswap = lib.H5TmpFile(self.swapfile.name)
                if self.ecpnloc_method in ["direct", "kb", "kb2"]:
                    self.vppnlocWks = self.fswap.create_group("vppnlocWks")
                else:
                    raise RuntimeError("Unknown ecpnloc_method %s" %
                                       (self.ecp_nloc_item))
            else:
                if self.ecpnloc_method in ["direct", "kb", "kb2"]:
                    self.vppnlocWks = {}
                else:
                    raise RuntimeError("Unknown ecpnloc_method %s" %
                                       (self.ecp_nloc_item))
        self._ecpnloc_initialized = True

    def update_vppnloc_support_vec(self, C_ks, ncomp=1, out=None):
        if self.pptype == "ccecp":
            if not self._ecpnloc_initialized:
                self.initialize_ecpnloc()
            nkpts = len(self.kpts)
            cell = self.cell

            if self.ecpnloc_method == "kb":
                if len(self.vppnlocWks) > 0:
                    return
                if out is None:
                    out = self.vppnlocWks
                get_ccecp_kb_support_vec(cell, self.ecpnloc_kbbas, self.kpts,
                                         out,
                                         ke_cutoff_nloc=self.ecpnloc_ke_cutoff,
                                         ncomp=ncomp, _ecp=self._ecp,
                                         thr_eig=self.threshold_svec,
                                         use_numexpr=self.ecpnloc_use_numexpr)
            elif self.ecpnloc_method == "kb2":
                raise NotImplementedError
                if len(self.vppnlocWks) > 0:
                    return
                if ncomp == 1:
                    out = self.vppnlocWks
                else:
                    out = self.vppnlocWks.create_group("0")
                kb_basis = self.ecpnloc_kbbas
                kpts = self.kpts
                get_ccecp_kb_support_vec(cell, kb_basis, kpts, out=out)
                if ncomp > 1:
                    for comp in range(1,ncomp):
                        self.vppnlocWks["%d"%comp] = out
            else:
                if out is None: out = self.vppnlocWks
                get_ccecp_support_vec(cell, C_ks, self.kpts, out,
                                      _ecp=self._ecp,
                                      ke_cutoff_nloc=self.ecpnloc_ke_cutoff,
                                      ncomp=ncomp, thr_eig=self.threshold_svec,
                                      use_numexpr=self.ecpnloc_use_numexpr)

    def apply_vppl_kpt(self, C_k, mesh=None, vpplocR=None, C_k_R=None,
                       basis=None):
        if mesh is None: mesh = self.mesh
        if vpplocR is None: vpplocR = self.vpplocR
        return apply_vppl_kpt(self, C_k, mesh=mesh, vpplocR=vpplocR,
                              C_k_R=C_k_R, basis=basis)

    def apply_vppnl_kpt(self, C_k, kpt, mesh=None, Gv=None, comp=None,
                        basis=None):
        cell = self.cell
        if self.pptype == "ccecp":
            k = member(kpt, self.kpts)[0]
            if self.vppnlocWks is None:
                return lib.dot(C_k.conj(), self.vppnlocGG[k]).conj()
            else:
                if comp is None:
                    W_k = get_kcomp(self.vppnlocWks, k)
                elif isinstance(comp, int):
                    W_k = get_kcomp(self.vppnlocWks["%d"%comp], k)
                else:
                    raise RuntimeError("comp must be None or int")
                return lib.dot(lib.dot(C_k, W_k.T.conj()), W_k)
        elif self.pptype == "gth":
            return apply_vppnl_kpt_gth(cell, C_k, kpt, Gv, basis=basis)
        elif self.pptype == "SG15":
            return apply_vppnl_kpt_sg15(cell, C_k, kpt, Gv, basis=basis)
        elif self.pptype == "alle":
            return apply_vppnl_kpt_alle(cell, C_k, kpt, Gv)
        else:
            raise NotImplementedError("Pseudopotential %s is currently not supported." % (str(cell.pseudo)))


""" All-electron implementation starts here
"""
def get_vpplocG_alle(cell, Gv):
    Zs = cell.atom_charges()
    coulG = tools.get_coulG(cell, Gv=Gv)
    vpplocG = -np.einsum("a,g->ag", Zs, coulG)
    return vpplocG


def apply_vppnl_kpt_alle(cell, C_k, kpt, Gv):
    return np.zeros_like(C_k)


""" GTH implementation starts here
"""
def get_vpplocG_gth(cell, Gv):
    return -gth_pseudo.get_vlocG(cell, Gv)


def get_vpplocG_sg15(cell, Gv):
    coulG = tools.get_coulG(cell, Gv=Gv)
    G2 = np.einsum('ix,ix->i', Gv, Gv)
    G = np.sqrt(G2)
    G0idx = np.where(G2==0)[0]
    vlocG = np.zeros((cell.natm, len(G2)))
    for ia in range(cell.natm):
        Zia = cell.atom_charge(ia)
        symb = cell.atom_symbol(ia)
        vlocG[ia] = Zia * coulG
        if symb in cell._pseudo:
            pp = cell._pseudo[symb]
            spline = make_interp_spline(pp["grids"]["k"], pp["local_part"]["recip"])
            vlocG[ia] *= spline(G) / Zia  # spline is normalized to Zia
            # alpha parameters from the non-divergent Hartree+Vloc G=0 term.
            # TODO this needed? Should compute limit of second deriv.
            # How to figure out if this is working?
            # vlocG[ia,G0idx] = pp["local_part"]["finite_g0"]
    vlocG[:] *= -1
    return vlocG


def apply_vppnl_kpt_gth(cell, C_k, kpt, Gv, basis=None):
    no = C_k.shape[0]

    # non-local pp
    from pyscf import gto
    fakemol = gto.Mole()
    fakemol._atm = np.zeros((1,gto.ATM_SLOTS), dtype=np.int32)
    fakemol._bas = np.zeros((1,gto.BAS_SLOTS), dtype=np.int32)
    ptr = gto.PTR_ENV_START
    fakemol._env = np.zeros(ptr+10)
    fakemol._bas[0,gto.NPRIM_OF ] = 1
    fakemol._bas[0,gto.NCTR_OF  ] = 1
    fakemol._bas[0,gto.PTR_EXP  ] = ptr+3
    fakemol._bas[0,gto.PTR_COEFF] = ptr+4

    if basis is None:
        Gk = Gv + kpt
        SI = cell.get_SI(Gv=Gv)
    else:
        Gk = basis.Gk
        SI = cell.get_SI(Gv=Gk-kpt)
    ngrids = Gk.shape[0]
    buf = np.empty((48,ngrids), dtype=np.complex128)
    Cbar_k = np.zeros_like(C_k)

    G_rad = lib.norm(Gk, axis=1)
    #:vppnl = 0
    for ia in range(cell.natm):
        symb = cell.atom_symbol(ia)
        if symb not in cell._pseudo:
            continue
        pp = cell._pseudo[symb]
        p1 = 0
        for l, proj in enumerate(pp[5:]):
            rl, nl, hl = proj
            if nl > 0:
                fakemol._bas[0,gto.ANG_OF] = l
                fakemol._env[ptr+3] = .5*rl**2
                fakemol._env[ptr+4] = rl**(l+1.5)*np.pi**1.25
                pYlm_part = fakemol.eval_gto('GTOval', Gk)

                p0, p1 = p1, p1+nl*(l*2+1)
                # pYlm is real, SI[ia] is complex
                pYlm = np.ndarray((nl,l*2+1,ngrids), dtype=np.complex128, buffer=buf[p0:p1])
                for k in range(nl):
                    qkl = gth_pseudo.pp._qli(G_rad*rl, l, k)
                    pYlm[k] = pYlm_part.T * qkl
                #:SPG_lmi = np.einsum('g,nmg->nmg', SI[ia].conj(), pYlm)
                #:SPG_lm_aoG = np.einsum('nmg,gp->nmp', SPG_lmi, aokG)
                #:tmp = np.einsum('ij,jmp->imp', hl, SPG_lm_aoG)
                #:vppnl += np.einsum('imp,imq->pq', SPG_lm_aoG.conj(), tmp)
        if p1 > 0:
            SPG_lmi = buf[:p1]
            SPG_lmi *= SI[ia].conj()
            p1 = 0
            for l, proj in enumerate(pp[5:]):
                rl, nl, hl = proj
                if nl > 0:
                    p0, p1 = p1, p1+nl*(l*2+1)
                    hl = np.asarray(hl)
                    SPG_lmi_ = SPG_lmi[p0:p1].reshape(nl,l*2+1,-1)
                    tmp = np.einsum("imG,IG->Iim", SPG_lmi_, C_k)
                    tmp = np.einsum("ij,Iim->Ijm", hl, tmp)
                    Cbar_k += np.einsum("Iim,imG->IG", tmp, SPG_lmi_.conj())
    Cbar_k /= cell.vol

    return Cbar_k


def apply_vppnl_kpt_sg15(cell, C_k, kpt, Gv, basis=None):
    no = C_k.shape[0]
    from pyscf.pbc.gto.pseudo.pp import Ylm, Ylm_real, cart2polar

    if basis is None:
        Gk = Gv + kpt
        SI = cell.get_SI(Gv=Gv)
    else:
        Gk = basis.Gk
        SI = cell.get_SI(Gv=Gk-kpt)
    ngrids = Gk.shape[0]
    # buf = np.empty((48,ngrids), dtype=np.complex128)
    Cbar_k = np.zeros_like(C_k)

    G_rad, G_theta, G_phi = cart2polar(Gk)
    G_phi[:] = G_phi % (2 * np.pi)
    lmax = np.max([[proj["l"] for proj in pp["projectors"]]
                  for pp in cell._pseudo.values()])
    G_ylm = np.empty(((lmax + 1) * (lmax + 1), ngrids), dtype=np.float64)
    lm = 0
    for l in range(lmax + 1):
        for m in range(2 * l + 1):
            G_ylm[lm] = Ylm(l, m, G_theta, G_phi)
            lm += 1

    for ia in range(cell.natm):
        symb = cell.atom_symbol(ia)
        if symb not in cell._pseudo:
            continue
        pp = cell._pseudo[symb]
        kmesh = pp["grids"]["k"]
        for iproj, proj in enumerate(pp["projectors"]):
            l = proj["l"]
            pfunc = proj["kproj"]
            spline = make_interp_spline(kmesh, pfunc)
            radpart = spline(G_rad)
            sphpart = G_ylm[l*l:(l+1)*(l+1)]
            d = pp["dij"][iproj, iproj]
            SPG_mi = radpart * sphpart * SI[ia].conj()
            tmp = np.einsum("mG,IG->Im", SPG_mi, C_k)
            tmp *= d
            Cbar_k += np.einsum("Im,mG->IG", tmp, SPG_mi.conj())
    Cbar_k /= cell.vol

    return Cbar_k


""" ccECP implementation starts here
"""
def fast_SphBslin(n, xs, thr_switch=20, thr_overflow=700, out=None):
    if out is None: out = np.zeros_like(xs)
    with np.errstate(over="ignore", invalid="ignore"):
        if n == 0:
            out[:] = np.sinh(xs) / xs
        elif n == 1:
            out[:] = (xs * np.cosh(xs) - np.sinh(xs)) / xs**2.
        elif n == 2:
            out[:] = ((xs**2.+3.)*np.sinh(xs) - 3.*xs*np.cosh(xs)) / xs**3.
        elif n == 3:
            out[:] = ((xs**3.+15.*xs)*np.cosh(xs) -
                      (6.*xs**2.+15.)*np.sinh(xs)) / xs**4.
        else:
            raise NotImplementedError("fast_SphBslin with n=%d is not implemented." % n)

    np.nan_to_num(out, copy=False, nan=0., posinf=0., neginf=0.)

    return out


def fast_SphBslin_numexpr(n, xs, thr_switch=20, thr_overflow=700, out=None):
    import numexpr
    if out is None: out = np.zeros_like(xs)
    with np.errstate(over="ignore", invalid="ignore"):
        if n == 0:
            numexpr.evaluate("sinh(xs)/xs", out=out)
        elif n == 1:
            numexpr.evaluate("(xs * cosh(xs) - sinh(xs)) / xs**2.", out=out)
        elif n == 2:
            numexpr.evaluate("((xs**2.+3.)*sinh(xs) - 3.*xs*cosh(xs)) / xs**3.",
                             out=out)
        elif n == 3:
            numexpr.evaluate("((xs**3.+15.*xs)*cosh(xs) -(6.*xs**2.+15.)*sinh(xs)) / xs**4.", out=out)
        else:
            raise NotImplementedError("fast_SphBslin with n=%d is not implemented." % n)

    np.nan_to_num(out, copy=False, nan=0., posinf=0., neginf=0.)

    return out


def fast_SphBslin_c(n, xs, out=None):
    if n > 3:
        raise NotImplementedError("fast_SphBslin with n=%d is not implemented." % n)

    if out is None: out = np.zeros_like(xs)

    import ctypes
    libpw = lib.load_library("libpwscf")
    libpw.fast_SphBslin(
        xs.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(xs.size),
        ctypes.c_int(n),
        out.ctypes.data_as(ctypes.c_void_p),
    )
    np.nan_to_num(out, copy=False, nan=0., posinf=0., neginf=0.)

    return out


def format_ccecp_param(cell):
    r""" Format the ecp data into the following dictionary:
        _ecp = {
                    atm1: [_ecpl_atm1, _ecpnl_atm1],
                    atm2: [_ecpl_atm2, _ecpnl_atm2],
                    ...
                }
        _ecpl  = [
                    [alp1_1, c1_1, alp2_1, c2_1, ...],
                    [alp1_2, c1_2, alp2_2, c2_2, ...],
                    [alp1_3, c1_3, alp2_3, c2_3, ...],
                ]
        _ecpnl = [
                    [l1, alp1_l1, c1_l1, alp2_l1, c2_l1, ...],
                    [l2, alp1_l2, c1_l2, alp2_l2, c2_l2, ...],
                    ...
                ]
        where
            Zeff = \sum_k ck_1
            Vl(r)  = -Zeff/r + c_1/r*exp(-alp_1*r^2) + c_2*r*exp(-alp_2*r^2) +
                        \sum_{k} ck_3*exp(-alpk_3*r^2)
            Vnl(r) = \sum_l \sum_k ck_l * exp(-alpk_l*r^2) \sum_m |lm><lm|
    """
    uniq_atms = cell._basis.keys()
    _ecp = {}
    for iatm in range(cell.natm):
        atm = cell.atom_symbol(iatm)
        if atm not in cell._ecp: continue
        if atm in _ecp: continue
        ncore, ecp_dic = cell._ecp[atm]
# local part
        ecp_loc = ecp_dic[0]
        _ecp_loc = []
        ecp_loc_item = ecp_loc[1]
        _ecp_loc = [np.concatenate([*ecp_loc_item[iloc]]) for iloc in [1,3,2]]
# non-local part
        _ecp_nloc = []
        for ecp_nloc_litem in ecp_dic[1:]:
            l = ecp_nloc_litem[0]
            _ecp_nloc_item = [l]
            for ecp_nloc_item in ecp_nloc_litem[1]:
                if len(ecp_nloc_item) > 0:
                    for ecp_nloc_item2 in ecp_nloc_item:
                        _ecp_nloc_item += ecp_nloc_item2
            _ecp_nloc.append(_ecp_nloc_item)
        _ecp[atm] = [_ecp_loc, _ecp_nloc]

    return _ecp


def get_vpplocG_ccecp(cell, Gv, _ecp=None):
    if _ecp is None: _ecp = format_ccecp_param(cell)
    G_rad = np.linalg.norm(Gv, axis=1)
    coulG = tools.get_coulG(cell, Gv=Gv)
    G0_idx = np.where(G_rad==0)[0]
    with np.errstate(divide="ignore"):
        invG = 4*np.pi / G_rad
        invG[G0_idx] = 0
    ngrids = coulG.size
    vlocG = np.zeros((cell.natm,ngrids))
    for iatm in range(cell.natm):
        atm = cell.atom_symbol(iatm)
        if atm not in _ecp:
            continue
        _ecpi = _ecp[atm][0]
# Zeff / r
        Zeff = sum(_ecpi[0][1::2])
        vlocG[iatm] += -coulG * Zeff
        v0 = -coulG[G0_idx] * Zeff
# c1 / r * exp(-a1 * r^2)
        n1 = len(_ecpi[0]) // 2
        for i1 in range(n1):
            a1, c1 = _ecpi[0][i1*2:(i1+1)*2]
            vlocG[iatm] += c1 * invG * a1**-0.5 * dawsn(G_rad*(0.5/a1**0.5))
            v0 += 2*np.pi / a1 * c1
# c2 * r * exp(-a2 * r^2)
        n2 = len(_ecpi[1]) // 2
        for i2 in range(n2):
            a2, c2 = _ecpi[1][i2*2:(i2+1)*2]
            vlocG[iatm] += c2 * (np.pi/a2**2. + ((0.5/a2**1.5) * invG -
                                                 (np.pi/a2**2.5)*G_rad) *
                                 dawsn(G_rad*(0.5/a2**0.5)))
            v0 += 2*np.pi / a2**2 * c2
# \sum_k c3_k * exp(-a3_k * r^2)
        n3 = len(_ecpi[2]) // 2
        if n3 > 0:
            for i3 in range(n3):
                a3, c3 = _ecpi[2][i3*2:(i3+1)*2]
                vlocG[iatm] += c3 * (np.pi/a3)**1.5 * np.exp(-G_rad**2.*
                                                             (0.25/a3))
                v0 += (np.pi/a3)**1.5 * c3
# G = 0
        vlocG[iatm][G0_idx] = v0

    return vlocG


def apply_vppnlocGG_kpt_ccecp(cell, C_k, kpt, _ecp=None, use_numexpr=False):
    log = logger.Logger(cell.stdout, cell.verbose)

    if _ecp is None: _ecp = format_ccecp_param(cell)
    Gv = cell.get_Gv()
    SI = cell.get_SI(Gv)
    ngrids = Gv.shape[0]

    from pyscf import gto
    fakemol = gto.Mole()
    fakemol._atm = np.zeros((1,gto.ATM_SLOTS), dtype=np.int32)
    fakemol._bas = np.zeros((1,gto.BAS_SLOTS), dtype=np.int32)
    ptr = gto.PTR_ENV_START
    fakemol._env = np.zeros(ptr+10)
    fakemol._bas[0,gto.NPRIM_OF ] = 1
    fakemol._bas[0,gto.NCTR_OF  ] = 1
    fakemol._bas[0,gto.PTR_EXP  ] = ptr+3
    fakemol._bas[0,gto.PTR_COEFF] = ptr+4

    uniq_atm_map = dict()
    for iatm in range(cell.natm):
        atm = cell.atom_symbol(iatm)
        if atm not in uniq_atm_map:
            uniq_atm_map[atm] = []
        uniq_atm_map[atm].append(iatm)

    nmo = C_k.shape[0]

    angls_nl = [_ecpnlitem[0] for _ecpitem in _ecp.values()
                for _ecpnlitem in _ecpitem[1]]
    if len(angls_nl) == 0:
        return np.zeros_like(C_k)

    lmax = np.max(angls_nl)
    natmmax = np.max([len(iatm_lst) for iatm_lst in uniq_atm_map.values()])

    dtype0 = np.float64
    dtype = np.complex128
    dsize = 16
    max_memory = (cell.max_memory - lib.current_memory()[0]) * 0.8
    Gblksize = min(int(np.floor((max_memory*1e6/dsize/ngrids -
                                 ((2*lmax+1)*natmmax+10+nmo))*0.2)), ngrids)
    buf = np.empty(Gblksize*ngrids, dtype=dtype)
    buf2 = np.empty(Gblksize*ngrids, dtype=dtype0)
    buf3 = np.empty(Gblksize*ngrids, dtype=dtype0)
    log.debug1("Computing v^nl*C_k in %d segs with blksize %d",
               (ngrids-1)//Gblksize+1, Gblksize)

    Gk = Gv + kpt
    G_rad = lib.norm(Gk, axis=1)
    if abs(kpt).sum() < 1e-8: G_rad += 1e-40    # avoid inverting zero
    if lmax > 0: invG_rad = 1./G_rad

    tspans = np.zeros((4,2))
    TICK = np.array([logger.process_clock(), logger.perf_counter()])

    # if use_numexpr:
    #     fSBin = fast_SphBslin_c
    # else:
    #     fSBin = fast_SphBslin
    fSBin = fast_SphBslin_c
    # fSBin = fast_SphBslin

    Cbar_k = np.zeros_like(C_k)
    for atm,iatm_lst in uniq_atm_map.items():
        if atm not in _ecp:
            continue
        _ecpnl_lst = _ecp[atm][1]
        for _ecpnl in _ecpnl_lst:
            l = _ecpnl[0]
            nl = (len(_ecpnl) - 1) // 2
            for il in range(nl):
                al, cl = _ecpnl[(1+il*2):(3+il*2)]
                fakemol._bas[0,gto.ANG_OF] = l
                fakemol._env[ptr+3] = 0.25 / al
                fakemol._env[ptr+4] = 2.*np.pi**1.25 * abs(cl)**0.5 / al**0.75
                flip_sign = cl < 0
                # pYlm_part.shape = (ngrids, (2*l+1)*len(iatm_lst))
                pYlm_part = np.einsum("gl,ag->gla",
                                      fakemol.eval_gto('GTOval', Gk),
                                      SI[iatm_lst]).reshape(ngrids,-1)
                if l > 0:
                    pYlm_part[:] *= (invG_rad**l)[:,None]
                G_red = G_rad * (0.5 / al)
                iblk = 0
                for p0,p1 in lib.prange(0,ngrids,Gblksize):
                    log.debug2("Gblk [%d/%d], %d ~ %d", iblk,
                               (ngrids-1)//Gblksize+1, p0, p1)
                    iblk += 1
                    vnlGG = np.ndarray((p1-p0,ngrids), dtype=dtype, buffer=buf)
                    G_rad2 = np.ndarray((p1-p0,ngrids), dtype=dtype0,
                                        buffer=buf2)
                    SBin = np.ndarray((p1-p0,ngrids), dtype=dtype0, buffer=buf3)
                    np.multiply(G_rad[p0:p1,None], G_red, out=G_rad2)
                    # use np.dot since a slice is neither F nor C-contiguous
                    if flip_sign:
                        vnlGG = np.dot(pYlm_part[p0:p1], -pYlm_part.conj().T,
                                       out=vnlGG)
                    else:
                        vnlGG = np.dot(pYlm_part[p0:p1], pYlm_part.conj().T,
                                       out=vnlGG)
                    tick = np.array([logger.process_clock(), logger.perf_counter()])
                    SBin = fSBin(l, G_rad2, out=SBin)
                    tock = np.array([logger.process_clock(), logger.perf_counter()])
                    tspans[0] += tock - tick
                    np.multiply(vnlGG, SBin, out=vnlGG)
                    tick = np.array([logger.process_clock(), logger.perf_counter()])
                    Cbar_k[:,p0:p1] += lib.dot(vnlGG, C_k.T).T
                    tock = np.array([logger.process_clock(), logger.perf_counter()])
                    tspans[1] += tock - tick
                    G_rad2 = vnlGG = SBin = None
                G_red = pYlm_part = None
    Cbar_k /= cell.vol

    TOCK = np.array([logger.process_clock(), logger.perf_counter()])
    tspans[3] += TOCK - TICK
    tspans[2] = tspans[3] - np.sum(tspans[:2], axis=0)

    tnames = ["SBin", "dot", "other", "total"]
    for tname, tspan in zip(tnames, tspans):
        tc, tw = tspan
        rc, rw = tspan / tspans[-1] * 100
        log.debug1('CPU time for %10s %9.2f  ( %6.2f%% ), wall time '
                   '%9.2f  ( %6.2f%% )', tname.ljust(10), tc, rc, tw, rw)

    return Cbar_k


def apply_vppnlocGG_kpt_ccecp_full(cell, C_k, k, vppnlocGG):
    ngrids = C_k.shape[1]
    max_memory = (cell.max_memory - lib.current_memory()[0]) * 0.8
    Gblksize = min(int(np.floor(max_memory*1e6/16/ngrids)), ngrids)
    W_k = np.zeros_like(C_k)
    for p0,p1 in lib.prange(0,ngrids,Gblksize):
        W_k += lib.dot(C_k[:,p0:p1].conj(), vppnlocGG[k,p0:p1]).conj()
    return W_k


def apply_vppnl_kpt_ccecp(cell, C_k, kpt, Gv, _ecp=None):
    """ very slow implementation
    """
    vppnlocGG = get_vppnlocGG_kpt_ccecp(cell, kpt, Gv, _ecp=_ecp)
    return lib.dot(C_k, vppnlocGG)


def get_ccecp_support_vec(cell, C_ks, kpts, out, _ecp=None, ke_cutoff_nloc=None,
                          ncomp=1, thr_eig=1e-12, use_numexpr=False):
    log = logger.Logger(cell.stdout, cell.verbose)

    if out is None:
        out = {}
    if isinstance(out, dict):
        outcore = False
    else:
        outcore = True

    if ncomp > 1:
        for comp in range(ncomp):
            key = "%d"%comp
            if outcore:
                if key in out: del out[key]
                out.create_group(key)
            else:
                out[key] = {}

    if _ecp is None: _ecp = format_ccecp_param(cell0)

    mesh_map = cell_nloc = None
    if ke_cutoff_nloc is not None:
        if ke_cutoff_nloc < cell.ke_cutoff:
            log.debug1("Using ke_cutoff_nloc %s for KB support vector", ke_cutoff_nloc)
            mesh_map = get_mesh_map(cell, cell.ke_cutoff, ke_cutoff_nloc)
            cell_nloc = cell.copy()
            cell_nloc.ke_cutoff = ke_cutoff_nloc
            cell_nloc.build()
        else:
            log.warn("Input ke_cutoff_nloc %s is greater than cell.ke_cutoff "
                     "%s and will be ignored.", ke_cutoff_nloc, cell.ke_cutoff)

    nkpts = len(kpts)
    for k in range(nkpts):
        if ncomp == 1:
            C_k = get_kcomp(C_ks, k)
        else:
            # concatenate all kpts
            comp_loc = [0] * (ncomp+1)
            C_k = [None] * ncomp
            for comp in range(ncomp):
                C_k[comp] = get_kcomp(C_ks["%d"%comp], k)
                comp_loc[comp+1] = comp_loc[comp] + C_k[comp].shape[0]
            C_k = np.vstack(C_k)

        kpt = kpts[k]
        if cell_nloc is None:
            W_k = apply_vppnlocGG_kpt_ccecp(cell, C_k, kpt, _ecp=_ecp,
                                            use_numexpr=use_numexpr)
        else:
            W_k = np.zeros_like(C_k)
            W_k[:,mesh_map] = apply_vppnlocGG_kpt_ccecp(cell_nloc,
                                                        C_k[:,mesh_map],
                                                        kpt, _ecp=_ecp,
                                                        use_numexpr=use_numexpr)

        if ncomp == 1:
            W_k = get_support_vec(C_k, W_k, method="eig", thr_eig=thr_eig)
            set_kcomp(W_k, out, k)
        else:
            # deconcatenate all kpts
            for comp in range(ncomp):
                p0, p1 = comp_loc[comp:comp+2]
                w_k = get_support_vec(C_k[p0:p1], W_k[p0:p1],
                                      method="eig", thr_eig=thr_eig)
                set_kcomp(w_k, out["%d"%comp], k)
                w_k = None

        C_k = W_k = None

    return out


def get_ccecp_kb_support_vec(cell, kb_basis, kpts, out, ke_cutoff_nloc=None,
                             ncomp=1, _ecp=None, thr_eig=1e-12,
                             use_numexpr=False, ioblk=IOBLK):

    log = logger.Logger(cell.stdout, cell.verbose)

    if out is None:
        out = {}
    outcore = not isinstance(out, dict)

    if ncomp == 1:
        W_ks = out
    else:
        if outcore:
            W_ks = out.create_group("0")
        else:
            out["0"] = {}
            W_ks = out["0"]

    nkpts = len(kpts)
    cell_kb = cell.copy()
    cell_kb.basis = kb_basis
    cell_kb.build()
    log.debug("Using basis %s for KB-ccECP (%d AOs)", kb_basis,
              cell_kb.nao_nr())

    nao = cell_kb.nao_nr()

# batching kpts to avoid high peak disk usage
    ngrids = np.prod(cell_kb.mesh)
    kblk = min(int(np.floor(ioblk/(ngrids*nao*16/1024**2.))), nkpts)
    nblk = int(np.ceil(nkpts / kblk))
    log.debug("Calculating KB support vec for all kpts in %d segments with "
              "kptblk size %d", nblk, kblk)
    log.debug("KB outcore: %s", outcore)

    tmpgroupname = "tmp"
    iblk = 0
    for k0,k1 in lib.prange(0,nkpts,kblk):
        log.debug1("BLK %d  kpt range %d ~ %d  kpts %s", iblk, k0, k1,
                   kpts[k0:k1])
        iblk += 1
        nkpts01 = k1 - k0
        kpts01 = kpts[k0:k1]
        Cg_ks = [np.eye(nao) + 0.j for k in range(nkpts01)]
        ng_ks = [nao] * nkpts01
        if outcore:
            W_ks_blk = W_ks.create_group(tmpgroupname)
            Cg_ks = get_C_ks_G(cell_kb, kpts01, Cg_ks, ng_ks, out=W_ks_blk)
        else:
            W_ks_blk = {}
            Cg_ks = get_C_ks_G(cell_kb, kpts01, Cg_ks, ng_ks)
        for k in range(nkpts01):
            Cg_k = get_kcomp(Cg_ks, k)
            Cg_k = orth(cell_kb, Cg_k)
            set_kcomp(Cg_k, Cg_ks, k)
        Cg_k = None
        log.debug("keeping %s SOAOs", ng_ks)

        get_ccecp_support_vec(cell, Cg_ks, kpts01, W_ks_blk, _ecp=_ecp,
                              ke_cutoff_nloc=ke_cutoff_nloc, ncomp=1,
                              thr_eig=thr_eig, use_numexpr=use_numexpr)

        for k in range(k0,k1):
            set_kcomp(get_kcomp(W_ks_blk, k-k0), W_ks, k)
        if outcore:
            del W_ks[tmpgroupname]
        else:
            Cg_ks = W_ks_blk = None

    nsv_ks = np.array([get_kcomp(W_ks, k, load=False).shape[0]
                       for k in range(nkpts)])
    mem_W_ks = nsv_ks.sum() * ngrids * 16 / 1024**2.

    log.debug("keeping %s KB support vectors", nsv_ks)
    log.debug("estimated %s usage: %.2f MB", "disk" if outcore else "memory",
              mem_W_ks)

    if ncomp > 1:
        for comp in range(1,ncomp):
            key = "%d"%comp
            if key in out: del out[key]
            out[key] = W_ks


def get_ccecp_kb2_support_vec(cell0, kb_basis, kpts, out=None, thr=1e-12):
    from pyscf.pbc.gto import ecp
    cell = cell0.copy()
    cell.basis = kb_basis
    cell.pseudo = "ccecp"   # make sure
    cell.verbose = 0
    cell.build()

# remove local part of the ecp
    cell = cell.copy()
    for bas in cell._ecpbas:
        if bas[1] == -1:
            idx = list(range(bas[5],bas[6]+1))
            cell._env[idx] = 0.

    nkpts = len(kpts)
    if out is None: out = [None] * nkpts

    ovlp = cell.pbc_intor("int1e_ovlp", kpts=kpts)
    vecp = ecp.ecp_int(cell, kpts)

# get Sinv and gto bas vecs (SOAO)
    c = [None] * nkpts
    Sinv = [None] * nkpts
    for k in range(nkpts):
        e, u = scipy.linalg.eigh(ovlp[k])
        c[k] = lib.dot(u*e**-0.5, u.T.conj())
        Sinv[k] = lib.dot(u*e**-1, u.T.conj())

# gto -> pw
    swapfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    fswap = lib.H5TmpFile(swapfile.name)
    swapfile = None
    C = fswap.create_group("C")
    D = fswap.create_group("D")

    n_ks = [c[k].shape[1] for k in range(nkpts)]
    get_C_ks_G(cell, kpts, c, n_ks, out=C)
    n_ks = [Sinv[k].shape[1] for k in range(nkpts)]
    get_C_ks_G(cell, kpts, Sinv, n_ks, out=D)

# get W
    for k in range(nkpts):
        C_k = get_kcomp(C, k)
        D_k = get_kcomp(D, k)
        DC_k = lib.dot(D_k.conj(), C_k.T)
        w_k = lib.dot(Sinv[k], lib.dot(vecp[k], DC_k))
        W_k = get_C_ks_G(cell, [kpts[k]], [w_k], [w_k.shape[1]])[0]
        C_k = get_kcomp(C, k)
        W_k = get_support_vec(C_k, W_k, method="eig")
        set_kcomp(W_k, out, k)
        C_k = D_k = W_k = None


def get_support_vec(C, W, method="cd", thr_eig=1e-12):
    M = lib.dot(C.conj(), W.T)
    if np.sum(np.abs(M)) < 1e-10:
        svec = np.zeros_like(C)
    else:
        if method == "cd":
            svec = scipy.linalg.cholesky(M, lower=True)
            svec = scipy.linalg.solve_triangular(svec.conj(), W, lower=True)
        elif method == "eig":
            e, u = scipy.linalg.eigh(M)
            idx_keep = np.where(e > thr_eig)[0]
            svec = lib.dot((u[:,idx_keep]*e[idx_keep]**-0.5).T, W)
        else:
            raise RuntimeError("Unknown method %s" % str(method))

    return svec
