#!/usr/bin/env python

from __future__ import annotations

from functools import reduce

import numpy as np
from numpy.typing import ArrayLike

from pyscf import lib, lo
from pyscf.lib import logger
from pyscf.lno import LNO
from pyscf.lno.lnoccsd import CCSD

from .afqmc import run_afqmc_lno_helper

_fdot = np.dot
fdot = lambda *args: reduce(_fdot, args)
einsum = lib.einsum


def impurity_solve(
    mcc,
    mo_coeff,
    uocc_loc,
    mo_occ,
    maskact,
    eris,
    log=None,
    verbose_imp=None,
    max_las_size_afqmc=None,
    kwargs_imp=None,
    frozen: int | ArrayLike | None = None,
    n_blocks=50,
    n_walkers=10,
    seed=52,
    chol_cut=1e-4,
    maxError=1e-3,
    dt=0.01,
    n_eql=2,
):
    r"""Solve impurity problem and calculate local correlation energy."""
    mf = mcc._scf
    log = logger.new_logger(mcc if log is None else log)
    cput1 = (logger.process_clock(), logger.perf_counter())

    maskocc = mo_occ > 1e-10
    nmo = mo_occ.size

    orbfrzocc = mo_coeff[:, ~maskact & maskocc]
    orbactocc = mo_coeff[:, maskact & maskocc]
    orbactvir = mo_coeff[:, maskact & ~maskocc]
    orbfrzvir = mo_coeff[:, ~maskact & ~maskocc]
    nfrzocc, nactocc, nactvir, nfrzvir = [
        orb.shape[1] for orb in [orbfrzocc, orbactocc, orbactvir, orbfrzvir]
    ]
    nlo = uocc_loc.shape[1]
    nactmo = nactocc + nactvir
    log.debug(
        "    impsol:  %d LOs  %d/%d MOs  %d occ  %d vir",
        nlo,
        nactmo,
        nmo,
        nactocc,
        nactvir,
    )

    sig_dec_orbe = 6
    sig_err_orbe = 0.0
    sig_e_orbe = 0.0

    if nactocc == 0 or nactvir == 0:
        elcorr_pt2 = lib.tag_array(0.0, spin_comp=np.array((0.0, 0.0)))
        elcorr_afqmc = 0.0
        err_afqmc = 0.0
    else:
        if nactmo > max_las_size_afqmc:
            log.warn(
                "Number of active space orbitals (%d) exceed "
                "`_max_las_size_afqmc` (%d). Impurity AFQMC calculations "
                "will NOT be performed.",
                nactmo,
                max_las_size_afqmc,
            )
            elcorr_pt2 = lib.tag_array(0.0, spin_comp=np.array((0.0, 0.0)))
            elcorr_afqmc = 0.0
            err_afqmc = 0.0
        else:
            imp_eris = mcc.ao2mo()
            if isinstance(imp_eris.ovov, np.ndarray):
                ovov = imp_eris.ovov
            else:
                ovov = imp_eris.ovov[()]
            oovv = ovov.reshape(nactocc, nactvir, nactocc, nactvir).transpose(0, 2, 1, 3)
            ovov = None
            cput1 = log.timer_debug1("imp sol - eri    ", *cput1)

            t1, t2 = mcc.init_amps(eris=imp_eris)[1:]
            cput1 = log.timer_debug1("imp sol - mp2 amp", *cput1)
            elcorr_pt2 = get_fragment_energy(oovv, t2, uocc_loc).real

            prjlo = np.array([uocc_loc.flatten()])
            elcorr_afqmc, err_afqmc = run_afqmc_lno_helper(
                mf,
                mo_coeff=mo_coeff,
                norb_act=(nactocc + nactvir),
                nelec_act=nactocc * 2,
                frozen_orbitals=frozen,
                n_walkers=n_walkers,
                nblocks=n_blocks,
                seed=seed,
                chol_cut=chol_cut,
                target_error=maxError,
                dt=dt,
                prjlo=prjlo,
                n_eql=n_eql,
            )
            if err_afqmc > 0:
                sig_dec_orbe = int(abs(np.floor(np.log10(err_afqmc))))
                sig_err_orbe = np.around(
                    np.round(err_afqmc * 10**sig_dec_orbe) * 10 ** (-sig_dec_orbe),
                    sig_dec_orbe,
                )
                sig_e_orbe = np.around(elcorr_afqmc, sig_dec_orbe)
            else:
                sig_err_orbe = 0.0
                sig_e_orbe = float(elcorr_afqmc)

    frag_msg = "  ".join(
        [
            f"E_corr(MP2) = {elcorr_pt2:.15g}",
            f"E_corr(AFQMC) = {sig_e_orbe:.{sig_dec_orbe}f} +/- {sig_err_orbe:.{sig_dec_orbe}f}",
        ]
    )

    return (elcorr_pt2, elcorr_afqmc, err_afqmc), frag_msg


def get_maskact(frozen, nmo):
    if frozen is None:
        frozen = 0
    elif isinstance(frozen, (list, tuple, np.ndarray)) and len(frozen) == 0:
        frozen = 0

    if isinstance(frozen, (int, np.integer)):
        maskact = np.hstack([np.zeros(frozen, dtype=bool), np.ones(nmo - frozen, dtype=bool)])
    elif isinstance(frozen, (list, tuple, np.ndarray)):
        frozen_arr = np.asarray(frozen, dtype=np.int64).reshape(-1)
        frozen_set = {int(i) for i in frozen_arr}
        maskact = np.array([i not in frozen_set for i in range(nmo)])
    else:
        raise RuntimeError

    return frozen, maskact


def get_fragment_energy(oovv, t2, uocc_loc):
    m = fdot(uocc_loc, uocc_loc.T.conj())
    ed = einsum("ijab,kjab,ik->", t2, oovv, m) * 2
    ex = -einsum("ijab,kjba,ik->", t2, oovv, m)
    ed = ed.real
    ex = ex.real
    ess = ed * 0.5 + ex
    eos = ed * 0.5
    return lib.tag_array(ess + eos, spin_comp=np.array((ess, eos)))


class LNOAFQMC(LNO):
    _max_las_size_afqmc = 300

    def __init__(self, mf, lo_coeff, frag_lolist, lno_type=None, lno_thresh=None, frozen=None):
        super().__init__(
            mf,
            lo_coeff,
            frag_lolist,
            lno_type=lno_type,
            lno_thresh=lno_thresh,
            frozen=frozen,
        )

        self.efrag_afqmc = None
        self.efrag_pt2 = None
        self.errfrag_afqmc = None

        self.kwargs_imp = None
        self.verbose_imp = 2

        self._h1e = None
        self._vhf = None

        self._max_las_size_afqmc = type(self)._max_las_size_afqmc
        self.n_walkers = 20
        self.n_blocks = 200
        self.seed = np.random.randint(1, 1000000)
        self.chol_cut = 1e-4
        self.maxError = 1e-4
        self.dt = 0.01
        self.n_eql = 2

    @property
    def h1e(self):
        if self._h1e is None:
            self._h1e = self._scf.get_hcore()
        return self._h1e

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose=verbose)
        log = logger.new_logger(self, verbose)
        log.info("_max_las_size_afqmc = %s", self._max_las_size_afqmc)
        return self

    def impurity_solve(self, mf, mo_coeff, uocc_loc, eris, frozen=None, log=None):
        if log is None:
            log = logger.new_logger(self)
        mo_occ = self.mo_occ
        frozen, maskact = get_maskact(frozen, mo_occ.size)
        mcc = CCSD(mf, mo_coeff=mo_coeff, frozen=frozen).set(verbose=self.verbose_imp)
        mcc._s1e = self._s1e
        mcc._h1e = self._h1e
        mcc._vhf = self._vhf
        if self.kwargs_imp is not None:
            mcc = mcc.set(**self.kwargs_imp)

        return impurity_solve(
            mcc,
            mo_coeff,
            uocc_loc,
            mo_occ,
            maskact,
            eris,
            log=log,
            verbose_imp=self.verbose_imp,
            max_las_size_afqmc=self._max_las_size_afqmc,
            frozen=frozen,
            n_blocks=self.n_blocks,
            n_walkers=self.n_walkers,
            seed=self.seed,
            chol_cut=self.chol_cut,
            maxError=self.maxError,
            dt=self.dt,
            n_eql=self.n_eql,
        )

    def _post_proc(self, frag_res, frag_wghtlist):
        nfrag = len(frag_res)
        efrag_pt2 = np.zeros(nfrag)
        efrag_afqmc = np.zeros(nfrag)
        errfrag_afqmc = np.zeros(nfrag)
        for i in range(nfrag):
            ept2, eafqmc, err_afqmc = frag_res[i]
            efrag_pt2[i] = float(ept2)
            efrag_afqmc[i] = float(eafqmc)
            errfrag_afqmc[i] = float(err_afqmc)
        self.efrag_pt2 = efrag_pt2 * frag_wghtlist
        self.efrag_afqmc = efrag_afqmc * frag_wghtlist
        self.errfrag_afqmc = errfrag_afqmc * frag_wghtlist

    def _finalize(self):
        logger.note(self, "E(%s) = %.15g  E_corr = %.15g", "LNOMP2", self.e_tot_pt2, self.e_corr_pt2)
        if self.afqmc_error_ecorr > 0:
            sig_dec_corr = int(abs(np.floor(np.log10(self.afqmc_error_ecorr))))
            sig_err_corr = np.around(
                np.round(self.afqmc_error_ecorr * 10**sig_dec_corr) * 10 ** (-sig_dec_corr),
                sig_dec_corr,
            )
            sig_e_corr = np.around(self.e_corr, sig_dec_corr)
            sig_e_tot = np.around(self.e_tot, sig_dec_corr)
        else:
            sig_err_corr = 0.0
            sig_e_corr = float(self.e_corr)
            sig_e_tot = float(self.e_tot)
        logger.note(
            self,
            "E(%s) = %.15g  E_corr = %.15g +/- %.15g",
            "LNOAFQMC",
            sig_e_tot,
            sig_e_corr,
            sig_err_corr,
        )
        return self

    @property
    def e_tot_scf(self):
        return self._scf.e_tot

    @property
    def e_corr(self):
        return self.e_corr_afqmc

    @property
    def e_tot(self):
        return self.e_corr + self.e_tot_scf

    @property
    def e_corr_afqmc(self):
        return np.sum(self.efrag_afqmc)

    @property
    def e_corr_pt2(self):
        return np.sum(self.efrag_pt2)

    @property
    def e_tot_afqmc(self):
        return self.e_corr_afqmc + self.e_tot_scf

    @property
    def e_tot_pt2(self):
        return self.e_corr_pt2 + self.e_tot_scf

    @property
    def afqmc_error_ecorr(self):
        return np.sqrt(np.sum(self.errfrag_afqmc**2))

    def e_corr_pt2corrected(self, ept2):
        return self.e_corr - self.e_corr_pt2 + ept2

    def e_tot_pt2corrected(self, ept2):
        return self.e_tot_scf + self.e_corr_pt2corrected(ept2)

    def e_corr_afqmc_pt2corrected(self, ept2):
        return self.e_corr_afqmc - self.e_corr_pt2 + ept2

    def e_tot_afqmc_pt2corrected(self, ept2):
        return self.e_tot_scf + self.e_corr_afqmc_pt2corrected(ept2)


def prep_local_orbitals(mf, frozen=0, localization_method="pm"):
    if localization_method not in ["pm"]:
        raise ValueError(
            f"Localization method '{localization_method}' is not supported. Make LOs by yourself."
        )

    orbocc = mf.mo_coeff[:, frozen : np.count_nonzero(mf.mo_occ)]
    mlo = lo.PipekMezey(mf.mol, orbocc)
    lo_coeff = mlo.kernel()
    while True:
        stable, lo_coeff1 = mlo.stability_jacobi()
        if stable:
            break
        mlo = lo.PipekMezey(mf.mol, lo_coeff1).set(verbose=2)
        mlo.init_guess = None
        lo_coeff = mlo.kernel()

    frag_lolist = [[i] for i in range(lo_coeff.shape[1])]
    return lo_coeff, frag_lolist


AfqmcLno = LNOAFQMC

__all__ = [
    "LNOAFQMC",
    "AfqmcLno",
    "get_fragment_energy",
    "get_maskact",
    "impurity_solve",
    "prep_local_orbitals",
]
