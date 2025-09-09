#!/usr/bin/env python
# Copyright 2014-2025 The PySCF Developers. All Rights Reserved.
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
# Author: Hong-Zhou Ye <osirpt.sun@gmail.com>
#

""" kpt-sampled periodic MP2 using a plane wave basis
"""

import h5py
import tempfile
import numpy as np

from pyscf.pbc.pwscf.pw_helper import (get_nocc_ks_from_mocc, get_kcomp,
                                       set_kcomp, wf_ifft)
from pyscf.pbc import tools
from pyscf import lib
from pyscf.lib import logger


def read_fchk(chkfile_name):
    from pyscf.lib.chkfile import load
    scf_dict = load(chkfile_name, "scf")
    mocc_ks = scf_dict["mo_occ"]
    moe_ks = scf_dict["mo_energy"]
    scf_dict = None

    fchk = h5py.File(chkfile_name, "r")
    C_ks = fchk["mo_coeff"]

    return fchk, C_ks, moe_ks, mocc_ks


def kconserv(kptija, reduce_latvec, kdota):
    tmp = lib.dot(kptija.reshape(1,-1), reduce_latvec) - kdota
    return np.where(abs(tmp - np.rint(tmp)).sum(axis=1)<1e-6)[0][0]


def fill_oovv(oovv, v_ia, Co_kj_R, Cv_kb_R, fac=None):
    r"""
    Math:
        oovv = \sum_G rho_ia^kika(G)*coulG(ki-ka) * rho_jb^kjkb(kptijab-G)
             = \sum_G V_ia^kika(G) * rho_jb^kjkb(kptijab-G)
             = \sum_r V_ia^kika(r)*phase * rho_jb^kjkb(r)
             = \sum_r v_ia^kika(r) * rho_jb^kjkb(r)
    """
    nocc_i, nocc_j = oovv.shape[:2]
    rho_shape = Cv_kb_R.shape
    rho_dtype = Cv_kb_R.dtype
    buf = np.empty(rho_shape, dtype=rho_dtype)
    for j in range(nocc_j):
        # rho_jb_R = Co_kj_R[j].conj() * Cv_kb_R
        rho_jb_R = np.ndarray(rho_shape, rho_dtype, buffer=buf)
        np.multiply(Co_kj_R[j].conj(), Cv_kb_R, out=rho_jb_R)
        for i in range(nocc_i):
            # oovv[i,j] = lib.dot(v_ia[i], rho_jb_R.T)
            lib.dot(v_ia[i], rho_jb_R.T, c=oovv[i,j])
    if fac is not None: oovv *= fac

    return oovv


def kernel_dx_(cell, kpts, chkfile_name, summary, nvir=None, nvir_lst=None,
               frozen=None, basis_ks=None):
    """ Compute both direct (d) and exchange (x) contributions together.

    Args:
        nvir_lst (array-like of int):
            If given, the MP2 correlation energies using the number of virtual
            orbitals specified by the list will be returned.
        frozen (int):
            Number of core orbitals to be frozen.
    """
    log = logger.Logger(cell.stdout, cell.verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())

    dtype = np.complex128
    dsize = 16

    fchk, C_ks, moe_ks, mocc_ks = read_fchk(chkfile_name)

    if frozen is not None:
        if isinstance(frozen, int):
            log.info("freezing %d orbitals", frozen)
            moe_ks = [moe_k[frozen:] for moe_k in moe_ks]
            mocc_ks = [mocc_k[frozen:] for mocc_k in mocc_ks]
        else:
            raise NotImplementedError

    nkpts = len(kpts)
    if basis_ks is None:
        basis_ks = [None] * nkpts
        mesh = cell.mesh
    else:
        assert len(basis_ks) == nkpts
        mesh = basis_ks[0].mesh
    coords = cell.get_uniform_grids(mesh=mesh)
    ngrids = coords.shape[0]

    reduce_latvec = cell.lattice_vectors() / (2*np.pi)
    kdota = lib.dot(kpts, reduce_latvec)

    fac = ngrids**2. / cell.vol
    fac_oovv = fac * ngrids / nkpts

    nocc_ks = get_nocc_ks_from_mocc(mocc_ks)
    if nvir is None:
        n_ks = [len(mocc_ks[k]) for k in range(nkpts)]
        nvir_ks = [n_ks[k] - nocc_ks[k] for k in range(nkpts)]
    else:
        nvir_ks = [nvir] * nkpts
        n_ks = [nocc_ks[k] + nvir_ks[k] for k in range(nkpts)]
    occ_ks = [list(range(nocc)) for nocc in nocc_ks]
    vir_ks = [list(range(nocc,n)) for nocc,n in zip(nocc_ks,n_ks)]
    nocc_max = np.max(nocc_ks)
    nvir_max = np.max(nvir_ks)
    if nvir_lst is None:
        nvir_lst = [nvir_max]
    nvir_lst = np.asarray(nvir_lst)
    nnvir = len(nvir_lst)
    log.info("Compute emp2 for these nvir's: %s", nvir_lst)

    # estimate memory requirement if done outcore
    est_mem = (nocc_max*nvir_max)**2*4      # for caching oovv_ka/kb, eijab, wijab
    est_mem += nocc_max*nvir_max*ngrids     # for caching v_ia_R
    est_mem += (nocc_max+nvir_max)*ngrids*2 # for caching MOs
    est_mem *= dsize / 1e6
    est_mem_outcore = est_mem
    # estimate memory requirement if done incore
    est_mem_incore = nkpts * (
                nocc_max*nvir_max*ngrids +  # for caching v_ia_ks_R
                (nocc_max+nvir_max)*ngrids  # for caching C_ks_R
            ) * dsize / 1e6
    est_mem_incore += est_mem
    # get currently available memory
    frac = 0.6
    cur_mem = cell.max_memory - lib.current_memory()[0]
    safe_mem = cur_mem * frac
    # check if incore mode is possible
    incore = est_mem_incore < cur_mem
    est_mem = est_mem_incore if incore else est_mem_outcore

    log.debug("Currently available memory total   %9.2f MB, "
              "safe   %9.2f MB", cur_mem, safe_mem)
    log.debug("Estimated required  memory outcore %9.2f MB, "
              "incore %9.2f MB", est_mem_outcore, est_mem_incore)
    log.debug("Incore mode: %r", incore)
    if est_mem > safe_mem:
        rec_mem = est_mem / frac + lib.current_memory()[0]
        log.warn("Estimate memory (%.2f MB) exceeds %.0f%% of currently "
                 "available memory (%.2f MB). Calculations may fail and "
                 "`cell.max_memory = %.2f` is recommended.",
                 est_mem, frac*100, safe_mem, rec_mem)

    buf1 = np.empty(nocc_max*nvir_max*ngrids, dtype=dtype)
    buf2 = np.empty(nocc_max*nocc_max*nvir_max*nvir_max, dtype=dtype)
    buf3 = np.empty(nocc_max*nocc_max*nvir_max*nvir_max, dtype=dtype)

    if incore:
        C_ks_R = [None] * nkpts
        v_ia_ks_R = [None] * nkpts
    else:
        swapfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        fswap = lib.H5TmpFile(swapfile.name)
        swapfile = None

        C_ks_R = fswap.create_group("C_ks_R")
        v_ia_ks_R = fswap.create_group("v_ia_ks_R")

    for k in range(nkpts):
        C_k = get_kcomp(C_ks, k)
        if frozen is not None:
            C_k = C_k[frozen:]
        # C_k = tools.ifft(C_k, mesh)
        C_k = wf_ifft(C_k, mesh, basis_ks[k])
        set_kcomp(C_k, C_ks_R, k)
        C_k = None

    C_ks = None
    fchk.close()

    cput1 = log.timer('initialize pwmp2', *cput0)

    tick = np.zeros(2)
    tock = np.zeros(2)
    tspans = np.zeros((7,2))
    tcomps = summary["tcomps"] = ["init", "v_ks_R", "khelper", "IO", "oovv",
                                  "energy", "tot"]
    tspans[0] = np.asarray(cput1) - np.asarray(cput0)

    emp2_d = np.zeros(nnvir)
    emp2_x = np.zeros(nnvir)
    emp2_ss = np.zeros(nnvir)
    emp2_os = np.zeros(nnvir)
    for ki in range(nkpts):
        kpti = kpts[ki]
        nocc_i = nocc_ks[ki]
        occ_i = occ_ks[ki]

        tick[:] = logger.process_clock(), logger.perf_counter()

        Co_ki_R = get_kcomp(C_ks_R, ki, occ=occ_i)

        for ka in range(nkpts):
            kpta = kpts[ka]
            nvir_a = nvir_ks[ka]
            vir_a = vir_ks[ka]
            coulG = tools.get_coulG(cell, kpta-kpti, exx=False, mesh=mesh)

            Cv_ka_R = get_kcomp(C_ks_R, ka, occ=vir_a)
            if incore:
                # if from buffer, an extra "copy" is needed in "set_kcomp"
                # below, which can be 1000x slower than allocating new mem.
                v_ia_R = np.empty((nocc_i,nvir_a,ngrids), dtype=dtype)
            else:
                v_ia_R = np.ndarray((nocc_i,nvir_a,ngrids), dtype=dtype,
                                    buffer=buf1)

            for i in range(nocc_i):
                v_ia = tools.fft(Co_ki_R[i].conj() * Cv_ka_R, mesh) * coulG
                v_ia_R[i] = tools.ifft(v_ia, mesh)

            set_kcomp(v_ia_R, v_ia_ks_R, ka)
            v_ia_R = Cv_ka_R = None

        Co_ki_R = None

        tock[:] = logger.process_clock(), logger.perf_counter()
        tspans[1] += tock - tick

        for kj in range(nkpts):
            nocc_j = nocc_ks[kj]
            occ_j = occ_ks[kj]
            kptij = kpti + kpts[kj]

            tick[:] = logger.process_clock(), logger.perf_counter()

            Co_kj_R = get_kcomp(C_ks_R, kj, occ=occ_j)

            tock[:] = logger.process_clock(), logger.perf_counter()
            tspans[3] += tock - tick

            done = [False] * nkpts
            kab_lst = []
            kptijab_lst = []
            for ka in range(nkpts):
                if done[ka]: continue
                kptija = kptij - kpts[ka]
                kb = kconserv(kptija, reduce_latvec, kdota)
                kab_lst.append((ka,kb))
                kptijab_lst.append(kptija-kpts[kb])
                done[ka] = done[kb] = True

            tick[:] = logger.process_clock(), logger.perf_counter()
            tspans[2] += tick - tock

            nkab = len(kab_lst)
            for ikab in range(nkab):
                ka,kb = kab_lst[ikab]
                kptijab = kptijab_lst[ikab]

                nvir_a = nvir_ks[ka]
                nvir_b = nvir_ks[kb]
                occ_a = occ_ks[ka]
                vir_a = vir_ks[ka]
                occ_b = occ_ks[kb]
                vir_b = vir_ks[kb]

                tick[:] = logger.process_clock(), logger.perf_counter()
                Cv_kb_R = get_kcomp(C_ks_R, kb, occ=vir_b)
                v_ia = get_kcomp(v_ia_ks_R, ka)
                tock[:] = logger.process_clock(), logger.perf_counter()
                tspans[3] += tock - tick

                phase = np.exp(-1j*lib.dot(coords,
                                           kptijab.reshape(-1,1))).reshape(-1)
                if incore:
                    # two possible schemes: 1) make a copy in "get_kcomp" above
                    # and use "a*=b" here. 2) (currently used) no copy in
                    # "get_kcomp", init v_ia from buf, and use multiply with
                    # "out".
                    # numerical tests found that: a) copy is 2x expensive than
                    # "a*=b" and 1000x than init from buf. b) mutiply with
                    # "out" is as fast as "a*=b", which is half the cost of
                    # "a*b".
                    # conclusion: scheme 2 will be >3x faster.
                    v_ia_ = v_ia
                    v_ia = np.ndarray((nocc_i,nvir_a,ngrids), dtype=dtype,
                                      buffer=buf1)
                    np.multiply(v_ia_, phase, out=v_ia)
                    v_ia_ = None
                else:
                    v_ia *= phase
                oovv_ka = np.ndarray((nocc_i,nocc_j,nvir_a,nvir_b), dtype=dtype,
                                     buffer=buf2)
                fill_oovv(oovv_ka, v_ia, Co_kj_R, Cv_kb_R, fac_oovv)
                tick[:] = logger.process_clock(), logger.perf_counter()
                tspans[4] += tick - tock

                Cv_kb_R = v_ia = None

                if ka != kb:
                    Cv_ka_R = get_kcomp(C_ks_R, ka, occ=vir_a)
                    v_ib = get_kcomp(v_ia_ks_R, kb)
                    tock[:] = logger.process_clock(), logger.perf_counter()
                    tspans[3] += tock - tick

                    if incore:
                        v_ib_ = v_ib
                        v_ib = np.ndarray((nocc_i,nvir_b,ngrids), dtype=dtype,
                                          buffer=buf1)
                        np.multiply(v_ib_, phase, out=v_ib)
                        v_ib_ = None
                    else:
                        v_ib *= phase
                    oovv_kb = np.ndarray((nocc_i,nocc_j,nvir_b,nvir_a), dtype=dtype,
                                         buffer=buf3)
                    fill_oovv(oovv_kb, v_ib, Co_kj_R, Cv_ka_R, fac_oovv)
                    tick[:] = logger.process_clock(), logger.perf_counter()
                    tspans[4] += tick - tock

                    Cv_ka_R = v_ib = None
                else:
                    oovv_kb = oovv_ka

# KMP2 energy evaluation starts here
                tick[:] = logger.process_clock(), logger.perf_counter()
                mo_e_o = moe_ks[ki][occ_i]
                mo_e_v = moe_ks[ka][vir_a]
                eia = mo_e_o[:,None] - mo_e_v

                if ka != kb:
                    mo_e_o = moe_ks[kj][occ_j]
                    mo_e_v = moe_ks[kb][vir_b]
                    ejb = mo_e_o[:,None] - mo_e_v
                else:
                    ejb = eia

                eijab = lib.direct_sum('ia,jb->ijab',eia,ejb)
                t2_ijab = np.conj(oovv_ka/eijab)

                for invir_,nvir_ in enumerate(nvir_lst):
                    eijab_d = 2 * np.einsum('ijab,ijab->',
                                            t2_ijab[:,:,:nvir_,:nvir_],
                                            oovv_ka[:,:,:nvir_,:nvir_]).real
                    eijab_x = - np.einsum('ijab,ijba->',
                                          t2_ijab[:,:,:nvir_,:nvir_],
                                          oovv_kb[:,:,:nvir_,:nvir_]).real
                    if ka != kb:
                        eijab_d *= 2
                        eijab_x *= 2

                    emp2_d[invir_] += eijab_d
                    emp2_x[invir_] += eijab_x
                    emp2_ss[invir_] += eijab_d * 0.5 + eijab_x
                    emp2_os[invir_] += eijab_d * 0.5

                tock[:] = logger.process_clock(), logger.perf_counter()
                tspans[5] += tock - tick

                oovv_ka = oovv_kb = eijab = woovv = None

        cput1 = log.timer('kpt %d (%6.3f %6.3f %6.3f)'%(ki,*kpti), *cput1)

    buf1 = buf2 = buf3 = None

    emp2_d /= nkpts
    emp2_x /= nkpts
    emp2_ss /= nkpts
    emp2_os /= nkpts
    emp2 = emp2_d + emp2_x
    summary["e_corr_d"] = emp2_d[-1]
    summary["e_corr_x"] = emp2_x[-1]
    summary["e_corr_ss"] = emp2_ss[-1]
    summary["e_corr_os"] = emp2_os[-1]
    summary["e_corr"] = emp2[-1]
    summary["nvir_lst"] = nvir_lst
    summary["e_corr_d_lst"] = emp2_d
    summary["e_corr_x_lst"] = emp2_x
    summary["e_corr_ss_lst"] = emp2_ss
    summary["e_corr_os_lst"] = emp2_os
    summary["e_corr_lst"] = emp2

    cput1 = log.timer('pwmp2', *cput0)
    tspans[6] = np.asarray(cput1) - np.asarray(cput0)
    for tspan, tcomp in zip(tspans,tcomps):
        summary["t-%s"%tcomp] = tspan

    return emp2[-1]


def PWKRMP2_from_gtomf(mf, chkfile=None):
    """ PWMP2 from a GTO-RHF object.
    """
    from pyscf.pbc.pwscf.pw_helper import gtomf2pwmf

    return PWKRMP2(gtomf2pwmf(mf, chkfile=chkfile))


class PWKRMP2:
    """
    Restriced MP2 perturbation theory in a plane-wave basis.
    """
    def __init__(self, mf, nvir=None, frozen=None):
        self.cell = self.mol = mf.cell
        self._scf = mf

        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.nvir = nvir
        self.frozen = frozen

##################################################
# don't modify the following attributes, they are not input options
        self.kpts = mf.kpts
        self.nkpts = len(self.kpts)
        self.mp2_summary = dict()
        self.e_hf = self._scf.e_tot
        self.e_corr = None
        self.t2 = None
        self._keys = set(self.__dict__.keys())

    @property
    def e_tot(self):
        if self.e_corr is None:
            return None
        else:
            return self.e_hf + self.e_corr

    def dump_mp2_summary(self, verbose=logger.DEBUG):
        log = logger.new_logger(self, verbose)
        summary = self.mp2_summary
        def write(fmt, key):
            if key in summary:
                log.info(fmt, summary[key])
        log.info('**** MP2 Summaries ****')
        log.info('Number of virtuals =              %d', summary["nvir_lst"][-1])
        log.info('Total Energy (HF+MP2) =           %24.15f', self.e_tot)
        log.info('Correlation Energy =              %24.15f', self.e_corr)
        write('Direct Energy =                   %24.15f', 'e_corr_d')
        write('Exchange Energy =                 %24.15f', 'e_corr_x')
        write('Same-spin Energy =                %24.15f', 'e_corr_ss')
        write('Opposite-spin Energy =            %24.15f', 'e_corr_os')

        nvir_lst = summary["nvir_lst"]
        if len(nvir_lst) > 1:
            log.info('%sNvirt  Ecorr', "\n")
            ecorr_lst = summary["e_corr_lst"]
            for nvir,ecorr in zip(nvir_lst,ecorr_lst):
                log.info("%5d  %24.15f", nvir, ecorr)
            log.info("%s", "")

        def write_time(comp, t_comp, t_tot):
            tc, tw = t_comp
            tct, twt = t_tot
            rc = tc / tct * 100
            rw = tw / twt * 100
            log.info('CPU time for %10s %9.2f  ( %6.2f%% ), wall time %9.2f  '
                     '( %6.2f%% )', comp.ljust(10), tc, rc, tw, rw)

        t_tot = summary["t-tot"]
        for icomp,comp in enumerate(summary["tcomps"]):
            write_time(comp, summary["t-%s"%comp], t_tot)

    def kernel(self, nvir=None, nvir_lst=None, frozen=None):
        cell = self.cell
        kpts = self.kpts
        chkfile = self._scf.chkfile
        summary = self.mp2_summary
        if nvir is None: nvir = self.nvir
        if frozen is None: frozen = self.frozen

        self.e_corr = kernel_dx_(cell, kpts, chkfile, summary, nvir=nvir,
                                 nvir_lst=nvir_lst, frozen=frozen,
                                 basis_ks=self._scf._basis_data)

        self._finalize()

        return self.e_corr

    def _finalize(self):
        logger.note(self, "KMP2 energy = %.15g", self.e_corr)


if __name__ == "__main__":
    from pyscf.pbc import gto, scf, mp, pwscf

    atom = "H 0 0 0; H 0.9 0 0"
    a = np.eye(3) * 3
    basis = "gth-szv"
    pseudo = "gth-pade"

    ke_cutoff = 50

    cell = gto.Cell(atom=atom, a=a, basis=basis, pseudo=pseudo,
                    ke_cutoff=ke_cutoff)
    cell.build()
    cell.verbose = 6

    nk = 2
    kmesh = [nk] * 3
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)

    pwmf = pwscf.PWKRHF(cell, kpts)
    pwmf.nvir = 20
    pwmf.kernel()

    es = {"5": -0.01363871, "10": -0.01873622, "20": -0.02461560}

    pwmp = PWKRMP2(pwmf)
    pwmp.kernel(nvir_lst=[5,10,20])
    pwmp.dump_mp2_summary()
    nvir_lst = pwmp.mp2_summary["nvir_lst"]
    ecorr_lst = pwmp.mp2_summary["e_corr_lst"]
    for nvir,ecorr in zip(nvir_lst,ecorr_lst):
        err = abs(ecorr - es["%d"%nvir])
        print(err)
        assert(err < 1e-6)
