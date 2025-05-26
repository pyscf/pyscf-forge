#!/usr/bin/env python
# Copyright 2014-2024 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors:
#          Shuhang Li <shuhangli98@gmail.com>
#          Zijun Zhao <brian.zhaozijun@gmail.com>
#

"""
Driven similarity renormalization group second-order multireference perturbation theory (DSRG-MRPT2)

References:
[1] Multireference Driven Similarity Renormalization Group: A Second-Order Perturbative Analysis
    Chenyang Li and Francesco A. Evangelista
    J. Chem. Theory Comput. 2015, 11, 2097-2108
[2] Driven similarity renormalization group for excited states: A state-averaged perturbation theory
    Chenyang Li and Francesco A. Evangelista
    J. Chem. Phys. 2018, 148, 124106
[3] Spin-free formulation of the multireference driven similarity renormalization group:
    A benchmark study of first-row diatomic molecules and spin-crossover energetics
    Chenyang Li and Francesco A. Evangelista
    J. Chem. Phys. 2021, 155, 114111
"""


import ctypes
import numpy as np
from pyscf import lib, mcscf
from pyscf import fci
from pyscf import ao2mo
from pyscf import df

MACHEPS = 1e-9
TAYLOR_THRES = 1e-3

libdsrg = lib.load_library("libdsrg")


def taylor_exp(z):
    """
    Taylor expansion of (1-exp(-z^2))/z for small z.
    """
    n = int(0.5 * (15.0 / TAYLOR_THRES + 1)) + 1
    if n > 0:
        value = z
        tmp = z
        for x in range(n - 1):
            tmp *= -1.0 * z * z / (x + 2)
            value += tmp

        return value
    else:
        return 0.0


def regularized_denominator(x, s):
    """
    Returns (1-exp(-s*x^2))/x
    """
    z = np.sqrt(s) * x
    if abs(z) <= MACHEPS:
        return taylor_exp(z) * np.sqrt(s)
    else:
        return (1.0 - np.exp(-s * x**2)) / x


def get_SF_RDM_SA(ci_vecs, weights, norb, nelec):
    """
    Returns the state-averaged spin-free active space 1-/2-/3-RDM.
    Reordered 2-rdm <p+ r+ s q> in Pyscf is stored as: dm2[pqrs]
    Forte stores it as rdm[prqs]
    """
    G1 = np.zeros((norb,) * 2)
    G2 = np.zeros((norb,) * 4)
    G3 = np.zeros((norb,) * 6)

    for i in range(len(ci_vecs)):
        # Unlike fcisolver.make_rdm1, make_dm123 doesn't automatically return the state-averaged RDM.
        _dm1, _dm2, _dm3 = fci.rdm.make_dm123(
            "FCI3pdm_kern_sf", ci_vecs[i], ci_vecs[i], norb, nelec
        )
        _dm1, _dm2, _dm3 = fci.rdm.reorder_dm123(_dm1, _dm2, _dm3)
        _G1 = np.einsum("pq->qp", _dm1)
        _G2 = np.einsum("pqrs->prqs", _dm2)
        _G3 = np.einsum("pqrstu->prtqsu", _dm3)
        G1 += weights[i] * _G1
        G2 += weights[i] * _G2
        G3 += weights[i] * _G3
    return G1, G2, G3


def get_SF_cu2(G1, G2):
    """
    Returns the spin-free active space 2-body cumulant.
    """
    L2 = G2.copy()
    L2 -= np.einsum("pr,qs->pqrs", G1, G1)
    L2 += 0.5 * np.einsum("ps,qr->pqrs", G1, G1)
    return L2


def get_SF_cu3(G1, G2, G3):
    """
    Returns the spin-free active space 3-body cumulant.
    """
    L3 = G3.copy()
    L3 -= (
        np.einsum("ps,qrtu->pqrstu", G1, G2)
        + np.einsum("qt,prsu->pqrstu", G1, G2)
        + np.einsum("ru,pqst->pqrstu", G1, G2)
    )
    L3 += 0.5 * (
        np.einsum("pt,qrsu->pqrstu", G1, G2)
        + np.einsum("pu,qrts->pqrstu", G1, G2)
        + np.einsum("qs,prtu->pqrstu", G1, G2)
        + np.einsum("qu,prst->pqrstu", G1, G2)
        + np.einsum("rs,pqut->pqrstu", G1, G2)
        + np.einsum("rt,pqsu->pqrstu", G1, G2)
    )
    L3 += 2 * np.einsum("ps,qt,ru->pqrstu", G1, G1, G1)
    L3 -= (
        np.einsum("ps,qu,rt->pqrstu", G1, G1, G1)
        + np.einsum("pu,qt,rs->pqrstu", G1, G1, G1)
        + np.einsum("pt,qs,ru->pqrstu", G1, G1, G1)
    )
    L3 += 0.5 * (
        np.einsum("pt,qu,rs->pqrstu", G1, G1, G1)
        + np.einsum("pu,qs,rt->pqrstu", G1, G1, G1)
    )
    return L3


def projected_fcisolver(mc, h1, h2, ncas, nelecas, ecore, nroots, ss=0):
    """
    Project out states with a given S^2 eigenvalue.
    """
    _fcisolver = (
        fci.direct_spin1_symm.FCISolver(mol=mc.mol)
        if mc.fcisolver.wfnsym is not None
        else fci.direct_spin1.FCISolver(mol=mc.mol)
    )
    _fcisolver.orbsym = mc.fcisolver.orbsym
    _fcisolver.wfnsym = mc.fcisolver.wfnsym

    _nroots = max(5, nroots * 2)
    nss = 0
    while True:
        evals, evecs = _fcisolver.kernel(
            h1, h2, ncas, nelecas, ecore=ecore, nroots=_nroots
        )
        evecs_proj = []
        evals_proj = []
        for i in range(_nroots):
            if np.isclose(_fcisolver.spin_square(evecs[i], ncas, nelecas)[0], ss):
                evecs_proj.append(evecs[i])
                evals_proj.append(evals[i])
                nss += 1
        if nss >= nroots:
            return np.array(evals_proj[:nroots]), np.array(evecs_proj[:nroots])
        else:
            if _nroots > evecs[0].shape[0]:
                raise ValueError(
                    "projected_fcisolver failed to find enough states with the specified S^2 eigenvalue."
                )
            _nroots += 5
            nss = 0


class DSRG_MRPT2(lib.StreamObject):
    """
    DSRG-MRPT2

    Attributes:
        s : float (default: 0.5)
            The flow parameter, which controls the extent to which
            the Hamiltonian is block-diagonalized.
        relax : str (default: 'none')
            Reference relaxation method. Options: 'none', 'once', 'twice', 'iterate'.
        density_fit: bool (default: False)
            To control whether density fitting to be used.
            For CCVV, CAVV, and CCAV terms, V and T2 will not be stored explicitly.
        batch: bool(default: False)
            To control whether the CCVV term to be computed in batches.
            CCVV: for a given m and n, form B(ef) = Bm(L|e) * Bn(L|f)
            This is only available with density fitting.

    Examples:

    >>> mf = gto.M('N 0 0 0; N 0 0 1.4', basis='6-31g').apply(scf.RHF).run()
    >>> mc = mcscf.CASCI(mf, 4, 4).run()
    >>> DSRG_MRPT2(mc, s=0.5).kernel()
    -0.15708345625685638
    """

    def __init__(
        self,
        mc,
        s=0.5,
        relax="none",
        relax_maxiter=10,
        relax_conv=1e-8,
        batch=False,
        ss=None,
    ):
        if not mc.converged:
            raise RuntimeError("MCSCF not converged or not performed.")
        self.mc = mc
        self.flow_param = s
        self.relax = relax
        if relax not in ["none", "once", "twice", "iterate"]:
            raise RuntimeError(
                f"Relaxation method '{relax}' not recognized. \
                    Supported methods are 'none', 'once', 'twice', and 'iterate'."
            )

        self.with_df = None
        self.df = False
        if getattr(mc, "with_df", None):
            self = self.density_fit()
        self.batch = batch
        # self.wfnsym = mc.fcisolver.wfnsym
        # self.orbsym = mc.fcisolver.orbsym

        if isinstance(mc.fcisolver, mcscf.addons.StateAverageFCISolver):
            if relax == "none":
                raise RuntimeError(
                    "State-averaged MCSCF is detected, please set relax to 'once', 'twice' or 'iterate'."
                )
            self.state_average = True
            self.state_average_weights = mc.fcisolver.weights
            self.state_average_nstates = mc.fcisolver.nstates
            self.ci_vecs = mc.ci
        else:
            self.state_average = False
            self.state_average_weights = [1.0]
            self.state_average_nstates = 1
            self.ci_vecs = [mc.ci]

        if relax == "none":
            self.nrelax = 0
        elif relax == "once":
            self.nrelax = 1
        elif relax == "twice":
            self.nrelax = 2
        elif relax == "iterate":
            self.nrelax = relax_maxiter

        self.relax_ref = self.nrelax > 0
        self.relax_conv = relax_conv

        self.form_hbar = self.relax_ref or self.state_average

        self.converged = False

        self.nao = mc.mol.nao
        self.ncore = mc.ncore
        self.nact = mc.ncas
        self.nelecas = mc.nelecas  # Tuple of (nalpha, nbeta)

        self.nvirt = self.nao - self.nact - self.ncore
        self.flow_param = s

        self.nhole = self.ncore + self.nact
        self.npart = self.nact + self.nvirt

        self.core = slice(0, self.ncore)
        self.active = slice(self.ncore, self.ncore + self.nact)
        self.virt = slice(self.ncore + self.nact, mc.mol.nao)
        self.hole = slice(0, self.ncore + self.nact)
        self.part = slice(self.ncore, mc.mol.nao)

        self.hc = self.core
        self.ha = self.active
        self.pa = slice(0, self.nact)
        self.pv = slice(self.nact, self.nact + self.nvirt)

        self.e_casci = mc.e_tot
        self.e_corr = None
        self.h1e_cas, self.ecore = mc.get_h1eff()
        self.h2e_cas = mc.get_h2eff()

        if not self.df:
            self.eri = self.mc.mol.intor("int2e", aosym="s8")
            _p = self.mc.mo_coeff[:, self.part]
            _h = self.mc.mo_coeff[:, self.hole]
            self.eri = (
                ao2mo.incore.general(self.eri, (_p, _h, _p, _h))
                .reshape((self.npart, self.nhole, self.npart, self.nhole))
                .swapaxes(1, 2)
            )

        self.ss = ss
        if ss is None:
            try:
                self.ss = self.mc.fcisolver.ss_value
            except Exception:
                try:
                    self.ss = self.mc.fcisolver.spin_square(
                        self.mc.ci, self.nact, self.nelecas
                    )[0]
                except Exception:
                    try:
                        self.ss = self.mc.fcisolver.spin_square(
                            self.mc.ci[0], self.nact, self.nelecas
                        )[0]
                    except Exception:
                        raise RuntimeError(
                            "Spin square value cannot be determined from the mc object. \
                                Please provide the value using the 'ss' kwarg."
                        )

    def density_fit(self, auxbasis=None, with_df=None):
        self.df = True
        if with_df is None:
            if getattr(self.mc, "with_df", None) and (
                auxbasis is None or auxbasis == self.mc.with_df.auxbasis
            ):
                self.with_df = self.mc.with_df
            else:
                self.with_df = df.DF(self.mc.mol, auxbasis)
                self.with_df.build()
        else:
            self.with_df = with_df

        return self

    def semi_canonicalize(self):
        # get_fock() uses the state-averaged RDM by default, via mc.fcisolver.make_rdm1()
        _G1_canon, _G2_canon, _G3_canon = get_SF_RDM_SA(
            self.ci_vecs, self.state_average_weights, self.nact, self.nelecas
        )
        _fock_canon = np.einsum(
            "pi,pq,qj->ij",
            self.mc.mo_coeff,
            self.mc.get_fock(casdm1=_G1_canon),
            self.mc.mo_coeff,
            optimize="optimal",
        )
        self.semicanonicalizer = np.zeros((self.nao, self.nao), dtype="float64")
        _, self.semicanonicalizer[self.core, self.core] = np.linalg.eigh(
            _fock_canon[self.core, self.core]
        )
        _, self.semicanonicalizer[self.active, self.active] = np.linalg.eigh(
            _fock_canon[self.active, self.active]
        )
        _, self.semicanonicalizer[self.virt, self.virt] = np.linalg.eigh(
            _fock_canon[self.virt, self.virt]
        )

        self.semicanonicalizer = self.semicanonicalizer.T  #  semicanonical * canonical

        self.fock = np.einsum(
            "ip,pq,jq->ij",
            self.semicanonicalizer,
            _fock_canon,
            self.semicanonicalizer,
            optimize="optimal",
        )

        # RDMs in semi-canonical basis.
        # This should be fine since all indices are active.

        _G1_semi_canon = np.einsum(
            "ip,jq,pq->ij",
            self.semicanonicalizer[self.active, self.active],
            self.semicanonicalizer[self.active, self.active],
            _G1_canon,
            optimize="optimal",
        )
        _G2_semi_canon = np.einsum(
            "ip,jq,kr,ls,pqrs->ijkl",
            self.semicanonicalizer[self.active, self.active],
            self.semicanonicalizer[self.active, self.active],
            self.semicanonicalizer[self.active, self.active],
            self.semicanonicalizer[self.active, self.active],
            _G2_canon,
            optimize="optimal",
        )
        _G3_semi_canon = np.einsum(
            "ip,jq,kr,ls,mt,nu,pqrstu->ijklmn",
            self.semicanonicalizer[self.active, self.active],
            self.semicanonicalizer[self.active, self.active],
            self.semicanonicalizer[self.active, self.active],
            self.semicanonicalizer[self.active, self.active],
            self.semicanonicalizer[self.active, self.active],
            self.semicanonicalizer[self.active, self.active],
            _G3_canon,
            optimize="optimal",
        )

        self.Eta = 2.0 * np.identity(self.nact) - _G1_semi_canon
        self.L1 = _G1_semi_canon.copy()
        self.L2 = get_SF_cu2(_G1_semi_canon, _G2_semi_canon)
        self.L3 = get_SF_cu3(_G1_semi_canon, _G2_semi_canon, _G3_semi_canon)
        del (
            _G1_canon,
            _G2_canon,
            _G3_canon,
            _G1_semi_canon,
            _G2_semi_canon,
            _G3_semi_canon,
        )

        if self.df:
            # Shuhang: I don't think batching will help here since a N^3 tensor (Bpq_ao) has to be construct explicitly.
            # If we want to avoid storing tensors with N^3 elements, DiskDF should be implemented.
            self.semi_coeff = np.einsum(
                "ip,up->iu",
                self.semicanonicalizer,
                self.mc.mo_coeff,
                optimize="optimal",
            )  # semicanonical * ao
            self.V = dict.fromkeys(
                ["vvaa", "aacc", "avca", "avac", "vaaa", "aaca", "aaaa"]
            )
            Bpq_ao = lib.unpack_tril(self.with_df._cderi)  # Aux * ao * ao
            self.Bpq = np.einsum(
                "ip,lpq,jq->ijl",
                self.semi_coeff[self.part, :],
                Bpq_ao,
                self.semi_coeff[self.hole, :],
                optimize="optimal",
            )  # Particle * Hole * Aux
            self.V["vvaa"] = np.einsum(
                "aig,bjg->abij",
                self.Bpq[self.pv, self.ha, :],
                self.Bpq[self.pv, self.ha, :],
                optimize="optimal",
            ).copy()
            self.V["aacc"] = np.einsum(
                "aig,bjg->abij",
                self.Bpq[self.pa, self.hc, :],
                self.Bpq[self.pa, self.hc, :],
                optimize="optimal",
            ).copy()
            self.V["avca"] = np.einsum(
                "aig,bjg->abij",
                self.Bpq[self.pa, self.hc, :],
                self.Bpq[self.pv, self.ha, :],
                optimize="optimal",
            ).copy()
            self.V["avac"] = np.einsum(
                "aig,bjg->abij",
                self.Bpq[self.pa, self.ha, :],
                self.Bpq[self.pv, self.hc, :],
                optimize="optimal",
            ).copy()
            self.V["vaaa"] = np.einsum(
                "aig,bjg->abij",
                self.Bpq[self.pv, self.ha, :],
                self.Bpq[self.pa, self.ha, :],
                optimize="optimal",
            ).copy()
            self.V["aaca"] = np.einsum(
                "aig,bjg->abij",
                self.Bpq[self.pa, self.hc, :],
                self.Bpq[self.pa, self.ha, :],
                optimize="optimal",
            ).copy()
            self.V["aaaa"] = np.einsum(
                "aig,bjg->abij",
                self.Bpq[self.pa, self.ha, :],
                self.Bpq[self.pa, self.ha, :],
                optimize="optimal",
            ).copy()
            del Bpq_ao
        else:
            self.V = dict.fromkeys(
                [
                    "vvaa",
                    "aacc",
                    "avca",
                    "avac",
                    "vaaa",
                    "aaca",
                    "aaaa",
                    "vvcc",
                    "vvac",
                    "vacc",
                ]
            )
            _eri = np.einsum(
                "ip,jq,pqrs,kr,ls->ijkl",
                self.semicanonicalizer[self.part, self.part],
                self.semicanonicalizer[self.part, self.part],
                self.eri,
                self.semicanonicalizer[self.hole, self.hole],
                self.semicanonicalizer[self.hole, self.hole],
                optimize="optimal",
            )
            self.V["vvaa"] = _eri[self.pv, self.pv, self.ha, self.ha].copy()
            self.V["aacc"] = _eri[self.pa, self.pa, self.hc, self.hc].copy()
            self.V["avca"] = _eri[self.pa, self.pv, self.hc, self.ha].copy()
            self.V["avac"] = _eri[self.pa, self.pv, self.ha, self.hc].copy()
            self.V["vaaa"] = _eri[self.pv, self.pa, self.ha, self.ha].copy()
            self.V["aaca"] = _eri[self.pa, self.pa, self.hc, self.ha].copy()
            self.V["aaaa"] = _eri[self.pa, self.pa, self.ha, self.ha].copy()
            self.V["vvcc"] = _eri[self.pv, self.pv, self.hc, self.hc].copy()
            self.V["vvac"] = _eri[self.pv, self.pv, self.ha, self.hc].copy()
            self.V["vacc"] = _eri[self.pv, self.pa, self.hc, self.hc].copy()
            del _eri

    def compute_T2(self):
        self.e_orb = {
            "c": np.diagonal(self.fock)[self.core].copy(),
            "a": np.diagonal(self.fock)[self.active].copy(),
            "v": np.diagonal(self.fock)[self.virt].copy(),
        }
        self.T2 = {}
        self.S = {}
        # Density fitting: these T2 blocks are stored:
        # aavv, ccaa, caav, acav, aava, caaa. Internal exciation (aaaa) tensor is zero.
        # Direct: three more blocks are stored: ccvv, acvv, ccva
        for Vblock, tensor in self.V.items():
            if Vblock != "aaaa":
                block = Vblock[2] + Vblock[3] + Vblock[0] + Vblock[1]
                self.T2[block] = np.einsum("abij->ijab", tensor).copy()
                libdsrg.compute_T2_block(
                    self.T2[block].ctypes.data_as(ctypes.c_void_p),
                    self.e_orb[block[0]].ctypes.data_as(ctypes.c_void_p),
                    self.e_orb[block[1]].ctypes.data_as(ctypes.c_void_p),
                    self.e_orb[block[2]].ctypes.data_as(ctypes.c_void_p),
                    self.e_orb[block[3]].ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_double(self.flow_param),
                    ctypes.c_int(tensor.shape[2]),
                    ctypes.c_int(tensor.shape[3]),
                    ctypes.c_int(tensor.shape[0]),
                    ctypes.c_int(tensor.shape[1]),
                )
        # form S2 = 2 * J - K
        # aavv, ccaa, caav, acav, aava, caaa
        self.S["aavv"] = 2.0 * self.T2["aavv"] - np.einsum(
            "uvef->uvfe", self.T2["aavv"]
        )
        self.S["ccaa"] = 2.0 * self.T2["ccaa"] - np.einsum(
            "mnuv->mnvu", self.T2["ccaa"]
        )
        self.S["caav"] = 2.0 * self.T2["caav"] - np.einsum(
            "umve->muve", self.T2["acav"]
        )
        self.S["acav"] = 2.0 * self.T2["acav"] - np.einsum(
            "muve->umve", self.T2["caav"]
        )
        self.S["aava"] = 2.0 * self.T2["aava"] - np.einsum(
            "vuex->uvex", self.T2["aava"]
        )
        self.S["caaa"] = 2.0 * self.T2["caaa"] - np.einsum(
            "muvx->muxv", self.T2["caaa"]
        )
        # ccvv, acvv, ccva
        if not self.df:
            self.S["ccvv"] = 2.0 * self.T2["ccvv"] - np.einsum(
                "mnef->mnfe", self.T2["ccvv"]
            )
            self.S["acvv"] = 2.0 * self.T2["acvv"] - np.einsum(
                "umef->umfe", self.T2["acvv"]
            )
            self.S["ccva"] = 2.0 * self.T2["ccva"] - np.einsum(
                "mnue->nmue", self.T2["ccva"]
            )

    def renormalize_V(self):
        for block, tensor in self.V.items():
            libdsrg.renormalize_V(
                tensor.ctypes.data_as(ctypes.c_void_p),
                self.e_orb[block[0]].ctypes.data_as(ctypes.c_void_p),
                self.e_orb[block[1]].ctypes.data_as(ctypes.c_void_p),
                self.e_orb[block[2]].ctypes.data_as(ctypes.c_void_p),
                self.e_orb[block[3]].ctypes.data_as(ctypes.c_void_p),
                ctypes.c_double(self.flow_param),
                ctypes.c_int(tensor.shape[0]),
                ctypes.c_int(tensor.shape[1]),
                ctypes.c_int(tensor.shape[2]),
                ctypes.c_int(tensor.shape[3]),
            )

    def compute_T1(self):
        # initialize T1 with F + [H0, A]
        self.T1 = self.fock[self.hole, self.part].copy()
        self.T1[self.hc, self.pa] += 0.5 * np.einsum(
            "ivaw, wu, uv->ia",
            self.S["caaa"],
            self.fock[self.active, self.active],
            self.L1,
            optimize="optimal",
        )
        self.T1[self.hc, self.pv] += 0.5 * np.einsum(
            "vmwe, wu, uv->me",
            self.S["acav"],
            self.fock[self.active, self.active],
            self.L1,
            optimize="optimal",
        )
        self.T1[self.ha, self.pv] += 0.5 * np.einsum(
            "ivaw, wu, uv->ia",
            self.S["aava"],
            self.fock[self.active, self.active],
            self.L1,
            optimize="optimal",
        )

        self.T1[self.hc, self.pa] -= 0.5 * np.einsum(
            "iwau,vw,uv->ia",
            self.S["caaa"],
            self.fock[self.active, self.active],
            self.L1,
            optimize="optimal",
        )
        self.T1[self.hc, self.pv] -= 0.5 * np.einsum(
            "wmue,vw,uv->me",
            self.S["acav"],
            self.fock[self.active, self.active],
            self.L1,
            optimize="optimal",
        )
        self.T1[self.ha, self.pv] -= 0.5 * np.einsum(
            "iwau,vw,uv->ia",
            self.S["aava"],
            self.fock[self.active, self.active],
            self.L1,
            optimize="optimal",
        )

        _ei = np.diagonal(self.fock)[self.hole].copy()
        _ea = np.diagonal(self.fock)[self.part].copy()
        libdsrg.compute_T1(
            self.T1.ctypes.data_as(ctypes.c_void_p),
            _ei.ctypes.data_as(ctypes.c_void_p),
            _ea.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_double(self.flow_param),
            ctypes.c_int(self.nhole),
            ctypes.c_int(self.npart),
        )
        self.T1[self.ha, self.pa] = 0

    def renormalize_F(self):
        _tmp = np.zeros((self.npart, self.nhole), dtype="float64")
        _tmp = self.fock[self.part, self.hole].copy()
        _tmp[self.pa, self.hc] += 0.5 * np.einsum(
            "ivaw,wu,uv->ai",
            self.S["caaa"],
            self.fock[self.active, self.active],
            self.L1,
            optimize="optimal",
        )
        _tmp[self.pa, self.hc] -= 0.5 * np.einsum(
            "iwau,vw,uv->ai",
            self.S["caaa"],
            self.fock[self.active, self.active],
            self.L1,
            optimize="optimal",
        )

        _tmp[self.pv, self.hc] += 0.5 * np.einsum(
            "vmwe,wu,uv->em",
            self.S["acav"],
            self.fock[self.active, self.active],
            self.L1,
            optimize="optimal",
        )
        _tmp[self.pv, self.hc] -= 0.5 * np.einsum(
            "wmue,vw,uv->em",
            self.S["acav"],
            self.fock[self.active, self.active],
            self.L1,
            optimize="optimal",
        )

        _tmp[self.pv, self.ha] += 0.5 * np.einsum(
            "ivaw,wu,uv->ai",
            self.S["aava"],
            self.fock[self.active, self.active],
            self.L1,
            optimize="optimal",
        )
        _tmp[self.pv, self.ha] -= 0.5 * np.einsum(
            "iwau,vw,uv->ai",
            self.S["aava"],
            self.fock[self.active, self.active],
            self.L1,
            optimize="optimal",
        )

        _eh = np.float64(np.diagonal(self.fock))[self.hole].copy()
        _ep = np.float64(np.diagonal(self.fock))[self.part].copy()
        libdsrg.renormalize_F(
            _tmp.ctypes.data_as(ctypes.c_void_p),
            _eh.ctypes.data_as(ctypes.c_void_p),
            _ep.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_double(self.flow_param),
            ctypes.c_int(self.nhole),
            ctypes.c_int(self.npart),
        )

        self.F_tilde = np.zeros((self.npart, self.nhole), dtype="float64")
        self.F_tilde = self.fock[self.part, self.hole].copy()
        self.F_tilde += _tmp
        del _tmp

    def H1_T1_C0(self):
        E = 0.0
        E = 2.0 * np.einsum(
            "am,ma->", self.F_tilde[:, self.hc], self.T1[self.hc, :], optimize="optimal"
        )
        temp = np.einsum(
            "ev,ue->uv",
            self.F_tilde[self.pv, self.ha],
            self.T1[self.ha, self.pv],
            optimize="optimal",
        )
        temp -= np.einsum(
            "um,mv->uv",
            self.F_tilde[self.pa, self.hc],
            self.T1[self.hc, self.pa],
            optimize="optimal",
        )
        E += np.einsum("uv,uv->", self.L1, temp, optimize="optimal")
        return E

    def H1_T2_C0(self):
        E = 0.0
        temp = np.einsum(
            "ex,uvey->uvxy",
            self.F_tilde[self.pv, self.ha],
            self.T2["aava"],
            optimize="optimal",
        )
        temp -= np.einsum(
            "vm,muyx->uvxy",
            self.F_tilde[self.pa, self.hc],
            self.T2["caaa"],
            optimize="optimal",
        )
        E = np.einsum("uvxy,uvxy->", self.L2, temp, optimize="optimal")
        return E

    def H2_T1_C0(self):
        E = 0.0
        temp = np.einsum(
            "evxy,ue->uvxy",
            self.V["vaaa"],
            self.T1[self.ha, self.pv],
            optimize="optimal",
        )
        temp -= np.einsum(
            "uvmy,mx->uvxy",
            self.V["aaca"],
            self.T1[self.hc, self.pa],
            optimize="optimal",
        )
        E = np.einsum("uvxy,uvxy->", self.L2, temp)
        return E

    def H2_T2_C0(self):
        E = np.einsum("efmn,mnef->", self.V["vvcc"], self.S["ccvv"], optimize="optimal")
        E += np.einsum(
            "feum,vmfe,uv->",
            self.V["vvac"],
            self.S["acvv"],
            self.L1,
            optimize="optimal",
        )
        E += np.einsum(
            "evnm,nmeu,uv->",
            self.V["vacc"],
            self.S["ccva"],
            self.Eta,
            optimize="optimal",
        )
        E += self.H2_T2_C0_T2small()
        return E

    def H2_T2_C0_T2small(self):
        #  Note the following blocks should be available in memory.
        #  V : vvaa, aacc, avca, avac, vaaa, aaca
        #  T2: aavv, ccaa, caav, acav, aava, caaa
        #  S : aavv, ccaa, caav, acav, aava, caaa
        E = 0.0
        # [H2, T2] L1 from aavv
        E += 0.25 * np.einsum(
            "efxu,yvef,uv,xy->",
            self.V["vvaa"],
            self.S["aavv"],
            self.L1,
            self.L1,
            optimize="optimal",
        )
        # [H2, T2] L1 from ccaa
        E += 0.25 * np.einsum(
            "vymn,mnux,uv,xy->",
            self.V["aacc"],
            self.S["ccaa"],
            self.Eta,
            self.Eta,
            optimize="optimal",
        )
        # [H2, T2] L1 from caav
        temp = 0.5 * np.einsum(
            "vemx,myue->uxyv", self.V["avca"], self.S["caav"], optimize="optimal"
        )
        temp += 0.5 * np.einsum(
            "vexm,ymue->uxyv", self.V["avac"], self.S["acav"], optimize="optimal"
        )
        E += np.einsum("uxyv,uv,xy->", temp, self.Eta, self.L1, optimize="optimal")
        # [H2, T2] L1 from caaa and aaav
        temp = 0.25 * np.einsum(
            "evwx,zyeu,wz->uxyv",
            self.V["vaaa"],
            self.S["aava"],
            self.L1,
            optimize="optimal",
        )
        temp += 0.25 * np.einsum(
            "vzmx,myuw,wz->uxyv",
            self.V["aaca"],
            self.S["caaa"],
            self.Eta,
            optimize="optimal",
        )
        E += np.einsum("uxyv,uv,xy->", temp, self.Eta, self.L1, optimize="optimal")

        # <[Hbar2, T2]> C_4 (C_2)^2
        # HH
        temp = 0.5 * np.einsum(
            "uvmn,mnxy->uvxy", self.V["aacc"], self.T2["ccaa"], optimize="optimal"
        )
        temp += 0.5 * np.einsum(
            "uvmw,mzxy,wz->uvxy",
            self.V["aaca"],
            self.T2["caaa"],
            self.L1,
            optimize="optimal",
        )

        # PP
        temp += 0.5 * np.einsum(
            "efxy,uvef->uvxy", self.V["vvaa"], self.T2["aavv"], optimize="optimal"
        )
        temp += 0.5 * np.einsum(
            "ezxy,uvew,wz->uvxy",
            self.V["vaaa"],
            self.T2["aava"],
            self.Eta,
            optimize="optimal",
        )

        # HP
        temp += np.einsum(
            "uexm,vmye->uvxy", self.V["avac"], self.S["acav"], optimize="optimal"
        )
        temp -= np.einsum(
            "uemx,vmye->uvxy", self.V["avca"], self.T2["acav"], optimize="optimal"
        )
        temp -= np.einsum(
            "vemx,muye->uvxy", self.V["avca"], self.T2["caav"], optimize="optimal"
        )

        # HP with Gamma1
        temp += 0.5 * np.einsum(
            "euwx,zvey,wz->uvxy",
            self.V["vaaa"],
            self.S["aava"],
            self.L1,
            optimize="optimal",
        )
        temp -= 0.5 * np.einsum(
            "euxw,zvey,wz->uvxy",
            self.V["vaaa"],
            self.T2["aava"],
            self.L1,
            optimize="optimal",
        )
        temp -= 0.5 * np.einsum(
            "evxw,uzey,wz->uvxy",
            self.V["vaaa"],
            self.T2["aava"],
            self.L1,
            optimize="optimal",
        )

        # HP with Eta1
        temp += 0.5 * np.einsum(
            "wumx,mvzy,wz->uvxy",
            self.V["aaca"],
            self.S["caaa"],
            self.Eta,
            optimize="optimal",
        )
        temp -= 0.5 * np.einsum(
            "uwmx,mvzy,wz->uvxy",
            self.V["aaca"],
            self.T2["caaa"],
            self.Eta,
            optimize="optimal",
        )
        temp -= 0.5 * np.einsum(
            "vwmx,muyz,wz->uvxy",
            self.V["aaca"],
            self.T2["caaa"],
            self.Eta,
            optimize="optimal",
        )

        E += np.einsum("uvxy,uvxy->", temp, self.L2)

        #
        E += np.einsum(
            "ewxy,uvez,xyzuwv->",
            self.V["vaaa"],
            self.T2["aava"],
            self.L3,
            optimize="optimal",
        )
        E -= np.einsum(
            "uvmz,mwxy,xyzuwv->",
            self.V["aaca"],
            self.T2["caaa"],
            self.L3,
            optimize="optimal",
        )
        return E

    def E_V_T2_CCVV_batch(self):
        E = 0.0
        # The three-index integral is created in the semicanonicalization step.
        # (me|nf) * [2 * (me|nf) - (mf|ne)] * [1 - e^(-2 * s * D)] / D
        # Batching: for a given m and n, form B(ef) = Bm(L|e) * Bn(L|f)
        _ec = self.e_orb["c"]
        _ev = self.e_orb["v"]
        for m in range(self.ncore):
            for n in range(m, self.ncore):
                if m == n:
                    factor = 1.0
                else:
                    factor = 2.0

                J_mn = np.einsum(
                    "eL,fL->ef",
                    np.squeeze(self.Bpq[self.pv, m, :]),
                    np.squeeze(self.Bpq[self.pv, n, :]),
                    optimize="optimal",
                ).copy()
                JK_mn = 2.0 * J_mn - J_mn.T

                libdsrg.renormalize_CCVV_batch(
                    J_mn.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_double(_ec[m] + _ev[n]),
                    _ev.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_double(self.flow_param),
                    ctypes.c_int(self.nvirt),
                )

                E += factor * np.einsum("ef,ef->", J_mn, JK_mn, optimize="optimal")
        return E

    def E_V_T2_CCVV(self):
        E = 0.0
        B_Lfn = self.Bpq[self.pv, self.hc, :].copy()
        _ec = self.e_orb["c"]
        _ev = self.e_orb["v"]
        for m in range(self.ncore):
            J_m = np.einsum(
                "eL,fnL->efn",
                np.squeeze(self.Bpq[self.pv, m, :]),
                B_Lfn,
                optimize="optimal",
            ).copy()
            JK_m = 2.0 * J_m - np.einsum("efn->fen", J_m.copy())

            libdsrg.renormalize_CCVV(
                J_m.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_double(_ec[m]),
                _ev.ctypes.data_as(ctypes.c_void_p),
                _ec.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_double(self.flow_param),
                ctypes.c_int(self.nvirt),
                ctypes.c_int(self.ncore),
            )

            E += np.einsum("efn,efn->", J_m, JK_m, optimize="optimal")
        return E

    def E_V_T2_CAVV(self):
        E = 0.0
        B_Lfv = self.Bpq[self.pv, self.ha, :].copy()
        temp = np.zeros((self.nact,) * 2)
        _ec = self.e_orb["c"]
        _ev = self.e_orb["v"]
        _ea = self.e_orb["a"]

        for m in range(self.ncore):
            J_m = np.einsum(
                "eL,fvL->efv",
                np.squeeze(self.Bpq[self.pv, m, :]),
                B_Lfv,
                optimize="optimal",
            ).copy()
            JK_m = 2.0 * J_m - np.einsum("efv->fev", J_m).copy()

            libdsrg.renormalize_CAVV(
                JK_m.ctypes.data_as(ctypes.c_void_p),
                J_m.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_double(_ec[m]),
                _ev.ctypes.data_as(ctypes.c_void_p),
                _ea.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_double(self.flow_param),
                ctypes.c_int(self.nvirt),
                ctypes.c_int(self.nact),
            )
            temp += np.einsum("efu,efv->uv", J_m, JK_m, optimize="optimal")

        E += np.einsum("uv,uv->", temp, self.L1, optimize="optimal")

        if self.form_hbar:
            self.C1_VT2_CAVV = temp.copy()
        del temp

        return E

    def E_V_T2_CCAV(self):
        E = 0.0
        temp = np.zeros((self.nact,) * 2)
        _ec = self.e_orb["c"]
        _ev = self.e_orb["v"]
        _ea = self.e_orb["a"]
        for m in range(self.ncore):
            for n in range(0, self.ncore):
                J_mn = np.einsum(
                    "eL,uL->eu",
                    np.squeeze(self.Bpq[self.pv, m, :]),
                    np.squeeze(self.Bpq[self.pa, n, :]),
                    optimize="optimal",
                ).copy()
                J_mn_2 = np.einsum(
                    "eL,uL->eu",
                    np.squeeze(self.Bpq[self.pv, n, :]),
                    np.squeeze(self.Bpq[self.pa, m, :]),
                    optimize="optimal",
                ).copy()
                JK_mn = 2.0 * J_mn - J_mn_2

                libdsrg.renormalize_CCAV(
                    JK_mn.ctypes.data_as(ctypes.c_void_p),
                    J_mn.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_double(_ec[m] + _ec[n]),
                    _ev.ctypes.data_as(ctypes.c_void_p),
                    _ea.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_double(self.flow_param),
                    ctypes.c_int(self.nvirt),
                    ctypes.c_int(self.nact),
                )

                temp += np.einsum("eu,ev->uv", J_mn, JK_mn, optimize="optimal")
        E += np.einsum("uv,uv->", temp, self.Eta, optimize="optimal")

        if self.form_hbar:
            self.C1_VT2_CCAV = temp.copy()

        del temp
        return E

    def H1_T_C1a_smallS(self, C1):
        C1 += 1.00 * np.einsum(
            "ev,ue->uv",
            self.F_tilde[self.pv, self.ha],
            self.T1[self.ha, self.pv],
            optimize="optimal",
        )
        C1 -= 1.00 * np.einsum(
            "um,mv->uv",
            self.F_tilde[self.pa, self.hc],
            self.T1[self.hc, self.pa],
            optimize="optimal",
        )
        C1 += 1.00 * np.einsum(
            "em,umve->uv",
            self.F_tilde[self.pv, self.hc],
            self.S["acav"],
            optimize="optimal",
        )
        C1 += 1.00 * np.einsum(
            "xm,muxv->uv",
            self.F_tilde[self.pa, self.hc],
            self.S["caaa"],
            optimize="optimal",
        )
        C1 += 0.50 * np.einsum(
            "ex,yuev,xy->uv",
            self.F_tilde[self.pv, self.ha],
            self.S["aava"],
            self.L1,
            optimize="optimal",
        )
        C1 -= 0.50 * np.einsum(
            "ym,muxv,xy->uv",
            self.F_tilde[self.pa, self.hc],
            self.S["caaa"],
            self.L1,
            optimize="optimal",
        )

    def H2_T_C1a_smallS(self, C1):
        C1 += 1.00 * np.einsum(
            "uemz,mwue->wz", self.V["avca"], self.S["caav"], optimize="optimal"
        )
        C1 += 1.00 * np.einsum(
            "uezm,wmue->wz", self.V["avac"], self.S["acav"], optimize="optimal"
        )
        C1 += 1.00 * np.einsum(
            "vumz,mwvu->wz", self.V["aaca"], self.S["caaa"], optimize="optimal"
        )

        C1 -= 1.00 * np.einsum(
            "wemu,muze->wz", self.V["avca"], self.S["caav"], optimize="optimal"
        )
        C1 -= 1.00 * np.einsum(
            "weum,umze->wz", self.V["avac"], self.S["acav"], optimize="optimal"
        )
        C1 -= 1.00 * np.einsum(
            "ewvu,vuez->wz", self.V["vaaa"], self.S["aava"], optimize="optimal"
        )

        temp = 0.5 * np.einsum(
            "wvef,efzu->wzuv", self.S["aavv"], self.V["vvaa"], optimize="optimal"
        )
        temp += 0.5 * np.einsum(
            "wvex,exzu->wzuv", self.S["aava"], self.V["vaaa"], optimize="optimal"
        )
        temp += 0.5 * np.einsum(
            "vwex,exuz->wzuv", self.S["aava"], self.V["vaaa"], optimize="optimal"
        )

        temp -= 0.5 * np.einsum(
            "wmue,vezm->wzuv", self.S["acav"], self.V["avac"], optimize="optimal"
        )
        temp -= 0.5 * np.einsum(
            "mwxu,xvmz->wzuv", self.S["caaa"], self.V["aaca"], optimize="optimal"
        )

        temp -= 0.5 * np.einsum(
            "mwue,vemz->wzuv", self.S["caav"], self.V["avca"], optimize="optimal"
        )
        temp -= 0.5 * np.einsum(
            "mwux,vxmz->wzuv", self.S["caaa"], self.V["aaca"], optimize="optimal"
        )

        temp += 0.25 * np.einsum(
            "jwxu,xy,yvjz->wzuv",
            self.S["caaa"],
            self.L1,
            self.V["aaca"],
            optimize="optimal",
        )
        temp -= 0.25 * np.einsum(
            "ywbu,xy,bvxz->wzuv",
            self.S["aava"],
            self.L1,
            self.V["vaaa"],
            optimize="optimal",
        )
        temp -= 0.25 * np.einsum(
            "wybu,xy,bvzx->wzuv",
            self.S["aava"],
            self.L1,
            self.V["vaaa"],
            optimize="optimal",
        )

        C1 += np.einsum("wzuv,uv->wz", temp, self.L1, optimize="optimal")
        temp = np.zeros((self.nact,) * 4)

        temp -= 0.5 * np.einsum(
            "mnzu,wvmn->wzuv", self.S["ccaa"], self.V["aacc"], optimize="optimal"
        )
        temp -= 0.5 * np.einsum(
            "mxzu,wvmx->wzuv", self.S["caaa"], self.V["aaca"], optimize="optimal"
        )
        temp -= 0.5 * np.einsum(
            "mxuz,vwmx->wzuv", self.S["caaa"], self.V["aaca"], optimize="optimal"
        )

        temp += 0.5 * np.einsum(
            "vmze,weum->wzuv", self.S["acav"], self.V["avac"], optimize="optimal"
        )
        temp += 0.5 * np.einsum(
            "xvez,ewxu->wzuv", self.S["aava"], self.V["vaaa"], optimize="optimal"
        )

        temp += 0.5 * np.einsum(
            "mvze,wemu->wzuv", self.S["caav"], self.V["avca"], optimize="optimal"
        )
        temp += 0.5 * np.einsum(
            "vxez,ewux->wzuv", self.S["aava"], self.V["vaaa"], optimize="optimal"
        )

        temp -= 0.25 * np.einsum(
            "yvbz,xy,bwxu->wzuv",
            self.S["aava"],
            self.Eta,
            self.V["vaaa"],
            optimize="optimal",
        )
        temp += 0.25 * np.einsum(
            "jvxz,xy,ywju->wzuv",
            self.S["caaa"],
            self.Eta,
            self.V["aaca"],
            optimize="optimal",
        )
        temp += 0.25 * np.einsum(
            "jvzx,xy,wyju->wzuv",
            self.S["caaa"],
            self.Eta,
            self.V["aaca"],
            optimize="optimal",
        )

        C1 += np.einsum("wzuv,uv->wz", temp, self.Eta, optimize="optimal")

        C1 += 0.50 * np.einsum(
            "vujz,jwyx,xyuv->wz",
            self.V["aaca"],
            self.T2["caaa"],
            self.L2,
            optimize="optimal",
        )
        C1 += 0.50 * np.einsum(
            "auzx,wvay,xyuv->wz",
            self.V["vaaa"],
            self.S["aava"],
            self.L2,
            optimize="optimal",
        )
        C1 -= 0.50 * np.einsum(
            "auxz,wvay,xyuv->wz",
            self.V["vaaa"],
            self.T2["aava"],
            self.L2,
            optimize="optimal",
        )
        C1 -= 0.50 * np.einsum(
            "auxz,vway,xyvu->wz",
            self.V["vaaa"],
            self.T2["aava"],
            self.L2,
            optimize="optimal",
        )

        C1 -= 0.50 * np.einsum(
            "bwyx,vubz,xyuv->wz",
            self.V["vaaa"],
            self.T2["aava"],
            self.L2,
            optimize="optimal",
        )
        C1 -= 0.50 * np.einsum(
            "wuix,ivzy,xyuv->wz",
            self.V["aaca"],
            self.S["caaa"],
            self.L2,
            optimize="optimal",
        )
        C1 += 0.50 * np.einsum(
            "uwix,ivzy,xyuv->wz",
            self.V["aaca"],
            self.T2["caaa"],
            self.L2,
            optimize="optimal",
        )
        C1 += 0.50 * np.einsum(
            "uwix,ivyz,xyvu->wz",
            self.V["aaca"],
            self.T2["caaa"],
            self.L2,
            optimize="optimal",
        )

        C1 += 0.50 * np.einsum(
            "avxy,uwaz,xyuv->wz",
            self.V["vaaa"],
            self.S["aava"],
            self.L2,
            optimize="optimal",
        )
        C1 -= 0.50 * np.einsum(
            "uviy,iwxz,xyuv->wz",
            self.V["aaca"],
            self.S["caaa"],
            self.L2,
            optimize="optimal",
        )

    def H2_T_C1a_smallG(self, C1):
        G2 = dict.fromkeys(["avac", "aaac", "avaa"])
        G2["avac"] = 2.0 * self.V["avac"] - np.einsum(
            "uemv->uevm", self.V["avca"], optimize="optimal"
        )
        G2["aaac"] = 2.0 * np.einsum(
            "vumw->uvwm", self.V["aaca"], optimize="optimal"
        ) - np.einsum("uvmw->uvwm", self.V["aaca"], optimize="optimal")
        G2["avaa"] = 2.0 * np.einsum(
            "euyx->uexy", self.V["vaaa"], optimize="optimal"
        ) - np.einsum("euxy->uexy", self.V["vaaa"], optimize="optimal")

        C1 += np.einsum(
            "ma,uavm->uv", self.T1[self.hc, self.pa], G2["aaac"], optimize="optimal"
        )
        C1 += np.einsum(
            "ma,uavm->uv", self.T1[self.hc, self.pv], G2["avac"], optimize="optimal"
        )
        C1 += 0.50 * np.einsum(
            "xe,yx,uevy->uv",
            self.T1[self.ha, self.pv],
            self.L1,
            G2["avaa"],
            optimize="optimal",
        )
        C1 -= 0.50 * np.einsum(
            "mx,xy,uyvm->uv",
            self.T1[self.hc, self.pa],
            self.L1,
            G2["aaac"],
            optimize="optimal",
        )

        C1 += 0.50 * np.einsum(
            "wezx,uvey,xyuv->wz",
            G2["avaa"],
            self.T2["aava"],
            self.L2,
            optimize="optimal",
        )
        C1 -= 0.50 * np.einsum(
            "wuzm,mvxy,xyuv->wz",
            G2["aaac"],
            self.T2["caaa"],
            self.L2,
            optimize="optimal",
        )

    def H_T_C2a_smallS(self, C2):
        C2 += np.einsum(
            "efxy,uvef->uvxy", self.V["vvaa"], self.T2["aavv"], optimize="optimal"
        )
        # C2["uvxy"] += H2["wzxy"] * T2["uvwz"];
        C2 += np.einsum(
            "ewxy,uvew->uvxy", self.V["vaaa"], self.T2["aava"], optimize="optimal"
        )
        C2 += np.einsum(
            "ewyx,vuew->uvxy", self.V["vaaa"], self.T2["aava"], optimize="optimal"
        )

        C2 += np.einsum(
            "uvmn,mnxy->uvxy", self.V["aacc"], self.T2["ccaa"], optimize="optimal"
        )
        # C2["uvxy"] += H2["uvwz"] * T2["wzxy"];
        C2 += np.einsum(
            "vumw,mwyx->uvxy", self.V["aaca"], self.T2["caaa"], optimize="optimal"
        )
        C2 += np.einsum(
            "uvmw,mwxy->uvxy", self.V["aaca"], self.T2["caaa"], optimize="optimal"
        )

        temp = np.einsum(
            "ax,uvay->uvxy",
            self.F_tilde[self.pv, self.ha],
            self.T2["aava"],
            optimize="optimal",
        )
        temp -= np.einsum(
            "ui,ivxy->uvxy",
            self.F_tilde[self.pa, self.hc],
            self.T2["caaa"],
            optimize="optimal",
        )
        temp += np.einsum(
            "ua,avxy->uvxy",
            self.T1[self.ha, self.pv],
            self.V["vaaa"],
            optimize="optimal",
        )
        temp -= np.einsum(
            "ix,uviy->uvxy",
            self.T1[self.hc, self.pa],
            self.V["aaca"],
            optimize="optimal",
        )

        temp -= 0.50 * np.einsum(
            "wz,vuaw,azyx->uvxy",
            self.L1,
            self.T2["aava"],
            self.V["vaaa"],
            optimize="optimal",
        )
        temp -= 0.50 * np.einsum(
            "wz,izyx,vuiw->uvxy",
            self.Eta,
            self.T2["caaa"],
            self.V["aaca"],
            optimize="optimal",
        )

        temp += np.einsum(
            "uexm,vmye->uvxy", self.V["avac"], self.S["acav"], optimize="optimal"
        )
        temp += np.einsum(
            "wumx,mvwy->uvxy", self.V["aaca"], self.S["caaa"], optimize="optimal"
        )

        temp += 0.50 * np.einsum(
            "wz,zvay,auwx->uvxy",
            self.L1,
            self.S["aava"],
            self.V["vaaa"],
            optimize="optimal",
        )
        temp -= 0.50 * np.einsum(
            "wz,ivwy,zuix->uvxy",
            self.L1,
            self.S["caaa"],
            self.V["aaca"],
            optimize="optimal",
        )

        temp -= np.einsum(
            "uemx,vmye->uvxy", self.V["avca"], self.T2["acav"], optimize="optimal"
        )
        temp -= np.einsum(
            "uwmx,mvwy->uvxy", self.V["aaca"], self.T2["caaa"], optimize="optimal"
        )

        temp -= 0.50 * np.einsum(
            "wz,zvay,auxw->uvxy",
            self.L1,
            self.T2["aava"],
            self.V["vaaa"],
            optimize="optimal",
        )
        temp += 0.50 * np.einsum(
            "wz,ivwy,uzix->uvxy",
            self.L1,
            self.T2["caaa"],
            self.V["aaca"],
            optimize="optimal",
        )

        temp -= np.einsum(
            "vemx,muye->uvxy", self.V["avca"], self.T2["caav"], optimize="optimal"
        )
        temp -= np.einsum(
            "vwmx,muyw->uvxy", self.V["aaca"], self.T2["caaa"], optimize="optimal"
        )

        temp -= 0.50 * np.einsum(
            "wz,uzay,avxw->uvxy",
            self.L1,
            self.T2["aava"],
            self.V["vaaa"],
            optimize="optimal",
        )
        temp += 0.50 * np.einsum(
            "wz,iuyw,vzix->uvxy",
            self.L1,
            self.T2["caaa"],
            self.V["aaca"],
            optimize="optimal",
        )

        C2 += temp
        C2 += np.einsum("uvxy->vuyx", temp, optimize="optimal")

    def compute_hbar(self):
        hbar1_temp = np.zeros((self.nact,) * 2)
        hbar2_temp = np.zeros((self.nact,) * 4)

        self.H1_T_C1a_smallS(hbar1_temp)
        self.H2_T_C1a_smallS(hbar1_temp)
        self.H2_T_C1a_smallG(hbar1_temp)
        self.H_T_C2a_smallS(hbar2_temp)

        self.hbar1 += 0.5 * hbar1_temp
        self.hbar1 += 0.5 * hbar1_temp.T

        self.hbar2 += 0.5 * hbar2_temp
        self.hbar2 += 0.5 * np.einsum("uvxy->xyuv", hbar2_temp, optimize="optimal")

        if self.df:
            self.hbar1 += 0.5 * self.C1_VT2_CAVV
            self.hbar1 += 0.5 * self.C1_VT2_CAVV.T
            self.hbar1 -= 0.5 * self.C1_VT2_CCAV
            self.hbar1 -= 0.5 * self.C1_VT2_CCAV.T
        else:
            hbar1_temp = np.einsum(
                "efzm,wmef->wz", self.V["vvac"], self.S["acvv"], optimize="optimal"
            )
            hbar1_temp -= np.einsum(
                "ewnm,nmez->wz", self.V["vacc"], self.S["ccva"], optimize="optimal"
            )
            self.hbar1 += 0.5 * hbar1_temp
            self.hbar1 += 0.5 * hbar1_temp.T
        del hbar1_temp, hbar2_temp

    def deGNO_ints(self):
        hbar2_temp = 2 * self.hbar2 - np.einsum(
            "pqrs->pqsr", self.hbar2, optimize="optimal"
        )

        self.e_scalar1 = -np.einsum("vu,vu->", self.hbar1, self.L1)
        self.e_scalar2 = 0.25 * np.einsum(
            "uv,vyux,xy->", self.L1, hbar2_temp, self.L1
        ) - 0.5 * np.einsum("xyuv,uvxy->", self.hbar2, self.L2)
        self.relax_e_scalar = self.e_scalar1 + self.e_scalar2

        self.hbar1 -= 0.5 * np.einsum("uxvy,yx->uv", hbar2_temp, self.L1)

        del hbar2_temp

        _active_semicanonicalizer = self.semicanonicalizer[
            self.active, self.active
        ].T  # Canonical * Semicanonical

        self.hbar1_canon = np.einsum(
            "ip,pq,jq->ij",
            _active_semicanonicalizer,
            self.hbar1,
            _active_semicanonicalizer,
            optimize="optimal",
        )
        self.hbar2_canon = np.einsum(
            "ip,jq,pqrs,kr,ls->ijkl",
            _active_semicanonicalizer,
            _active_semicanonicalizer,
            self.hbar2,
            _active_semicanonicalizer,
            _active_semicanonicalizer,
            optimize="optimal",
        )

    def drsg_mrpt2_iteration(self):
        # 1. Semicanonicalize orbitals from CASSCF
        self.semi_canonicalize()
        if self.relax_ref:
            self.hbar1 = self.fock[self.active, self.active].copy()
            self.hbar2 = self.V["aaaa"].copy()
        # 2. Compute regularized amplitudes and dressed integrals
        self.compute_T2()
        self.compute_T1()
        self.renormalize_V()
        self.renormalize_F()
        # 3. Compute energy
        self.e_h1_t1 = self.H1_T1_C0()
        self.e_h1_t2 = self.H1_T2_C0()
        self.e_h2_t1 = self.H2_T1_C0()
        if self.df:
            self.e_h2_t2_small = (
                self.H2_T2_C0_T2small()
            )  # Blocks with more than two active indices are available in memory
            self.e_h2_t2_cavv = self.E_V_T2_CAVV()
            self.e_h2_t2_ccav = self.E_V_T2_CCAV()
            # [todo]: unified interface for batching: give a list of indices to batch over
            if self.batch:
                self.e_h2_t2_ccvv = self.E_V_T2_CCVV_batch()
            else:
                self.e_h2_t2_ccvv = self.E_V_T2_CCVV()
            self.e_h2_t2 = (
                self.e_h2_t2_small
                + self.e_h2_t2_cavv
                + self.e_h2_t2_ccav
                + self.e_h2_t2_ccvv
            )
        else:
            self.e_h2_t2 = self.H2_T2_C0()

        # this is the correlation energy wrt the relaxed reference, NOT the original reference
        _e_corr = self.e_h1_t1 + self.e_h1_t2 + self.e_h2_t1 + self.e_h2_t2
        self.e_tot = self.e_casci + _e_corr

    def relax_reference(self):
        self.compute_hbar()
        # De-normal ordering:
        # Express transformed Hamiltonian using operators normal-ordered with respect to the true vacuum.
        self.deGNO_ints()

        self.relax_eigval, self.ci_vecs = projected_fcisolver(
            self.mc,
            self.hbar1_canon,
            self.hbar2_canon.swapaxes(1, 2),
            self.mc.ncas,
            self.mc.nelecas,
            ecore=self.relax_e_scalar,
            nroots=self.state_average_nstates,
            ss=self.ss,
        )

        if self.state_average_nstates == 1:
            self.relax_eigval = [self.relax_eigval]
            self.ci_vecs = [self.ci_vecs]
        _eci_avg = np.dot(
            self.relax_eigval[: self.state_average_nstates], self.state_average_weights
        )
        self.e_relax_eigval_shifted = list(
            np.array(self.relax_eigval[: self.state_average_nstates]) + self.e_tot
        )
        self.e_tot += _eci_avg

        self.e_casci = self.get_casci_energy(self.ci_vecs)

    def get_casci_energy(self, ci_vecs):
        e_casci = 0.0
        for i in range(self.state_average_nstates):
            e_casci += (
                self.mc.fcisolver.energy(
                    self.h1e_cas,
                    self.h2e_cas,
                    ci_vecs[i],
                    self.mc.ncas,
                    self.mc.nelecas,
                )
                + self.ecore
            ) * self.state_average_weights[i]

        return e_casci

    def test_relaxation_convergence(self, n):
        """
        Test convergence for reference relaxation.
        :param n: iteration number (start from 0)
        :return: True if converged
        """
        if n == 1 and self.nrelax == 2:
            self.converged = True

        if n != 0 and self.nrelax > 2:
            e_diff_u = abs(self.relax_energies[n][0] - self.relax_energies[n - 1][0])
            e_diff_r = abs(self.relax_energies[n][1] - self.relax_energies[n - 1][1])
            e_diff = abs(self.relax_energies[n][0] - self.relax_energies[n][1])
            if all(e < self.relax_conv for e in [e_diff_u, e_diff_r, e_diff]):
                self.converged = True

        return self.converged

    def kernel(self):
        self.drsg_mrpt2_iteration()

        if self.relax_ref:
            self.relax_energies = np.zeros(
                (self.nrelax, 3)
            )  # [iter, [unrelaxed, relaxed, Eref]]
        else:
            self.relax_energies = np.zeros((1, 3))
            self.relax_energies[0, 0] = self.e_tot
            self.relax_energies[0, 2] = self.e_casci
            self.converged = True
            self.e_corr = self.e_tot - self.e_casci

        for irelax in range(self.nrelax):
            self.relax_energies[irelax, 0] = self.e_tot
            self.relax_energies[irelax, 2] = self.e_casci

            self.relax_reference()
            self.relax_energies[irelax, 1] = (
                self.e_tot[0] if isinstance(self.e_tot, np.ndarray) else self.e_tot
            )  # fix numpy depreciation warning
            self.e_corr = (
                self.relax_energies[irelax, 1] - self.relax_energies[0, 2]
            )  # correlation energy wrt the original unrelaxed reference

            if self.test_relaxation_convergence(irelax):
                break
            if self.nrelax == 1:
                self.converged = True
                break  # don't do another DSRG calculation if we're just doing partial relaxation

            self.drsg_mrpt2_iteration()

        if not self.converged:
            print(
                "Warning! relax_maxiter has been reached, DSRG-MRPT2 did not converge!"
            )

        return self.e_tot
