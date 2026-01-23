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
# Author: Hong-Zhou Ye <hzyechem@gmail.com>
#

""" kpt-sampled periodic MP2 using a plane wave basis and spin-unrestricted HF
"""

import tempfile
import numpy as np

from pyscf.pbc.pwscf import kmp2
from pyscf.pbc.pwscf.pw_helper import (
    get_nocc_ks_from_mocc, wf_ifft, wf_fft, get_mesh_map
)
from pyscf.pbc.pwscf.kuhf import get_spin_component
from pyscf.pbc import tools
from pyscf import lib
from pyscf.lib import logger


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
    for j in range(nocc_j):
        rho_jb_R = Co_kj_R[j].conj() * Cv_kb_R
        for i in range(nocc_i):
            oovv[i,j] = lib.dot(v_ia[i], rho_jb_R.T)
    if fac is not None: oovv *= fac

    return oovv


def kernel_dx_(cell, kpts, chkfile_name, summary, nvir=None, nvir_lst=None,
               basis_ks=None, ecut_eri=None):
    """ Compute both direct (d) and exchange (x) contributions together.
    """
    log = logger.Logger(cell.stdout, cell.verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())

    dtype = np.complex128
    dsize = 16

    fchk, C_ks, moe_ks, mocc_ks = kmp2.read_fchk(chkfile_name)

    nkpts = len(kpts)
    if basis_ks is None:
        basis_ks = [None] * nkpts
        mesh = cell.mesh
    else:
        assert len(basis_ks) == nkpts
        mesh = basis_ks[0].mesh
    if ecut_eri is not None:
        eri_mesh = cell.cutoff_to_mesh(ecut_eri)
        if (eri_mesh > mesh).any():
            log.warn("eri_mesh larger than mesh, so eri_mesh is ignored.")
            eri_mesh = mesh
    else:
        eri_mesh = mesh

    coords = cell.get_uniform_grids(mesh=eri_mesh)
    ngrids = coords.shape[0]

    reduce_latvec = cell.lattice_vectors() / (2*np.pi)
    kdota = lib.dot(kpts, reduce_latvec)

    fac = ngrids**2. / cell.vol
    fac_oovv = fac * ngrids / nkpts

    nocc_ks = np.asarray([get_nocc_ks_from_mocc(mocc_ks[s]) for s in [0,1]])
    if nvir is None:
        n_ks = np.asarray([[len(mocc_ks[s][k]) for k in range(nkpts)]
                          for s in [0,1]])
        nvir_ks = n_ks - nocc_ks
    else:
        if isinstance(nvir,int): nvir = [nvir] * 2
        nvir_ks = np.asarray([[nvir[s]] * nkpts for s in [0,1]])
        n_ks = nocc_ks + nvir_ks
    nocc_max = np.max(nocc_ks)
    nvir_max = np.max(nvir_ks)
    nocc_sps = np.asarray([[nocc_ks[0][k],nocc_ks[1][k]] for k in range(nkpts)])
    nvir_sps = np.asarray([[nvir_ks[0][k],nvir_ks[1][k]] for k in range(nkpts)])
    if nvir_lst is None:
        nvir_lst = [nvir_max]
    nvir_lst = np.asarray(nvir_lst)
    nnvir = len(nvir_lst)
    logger.info(cell, "Compute emp2 for these nvir's: %s", nvir_lst)

    # estimate memory requirement
    est_mem = nocc_max*nvir_max*ngrids      # for caching v_ia_R
    est_mem += (nocc_max*nvir_max)**2*4     # for caching oovv_ka/kb, eijab, wijab
    est_mem += (nocc_max+nvir_max)*ngrids*2 # for caching MOs
    est_mem *= dsize / 1e6
    frac = 0.6
    cur_mem = cell.max_memory - lib.current_memory()[0]
    safe_mem = cur_mem * frac
    log.debug("Currently available memory %9.2f MB, safe %9.2f MB",
              cur_mem, safe_mem)
    log.debug("Estimated required memory  %9.2f MB", est_mem)
    if est_mem > safe_mem:
        rec_mem = est_mem / frac + lib.current_memory()[0]
        log.warn("Estimate memory requirement (%.2f MB) exceeds %.0f%%"
                 " of currently available memory (%.2f MB). Calculations may"
                 " fail and `cell.max_memory = %.2f` is recommended.",
                 est_mem, frac*100, safe_mem, rec_mem)

    buf1 = np.empty(nocc_max*nvir_max*ngrids, dtype=dtype)
    buf2 = np.empty(nocc_max*nocc_max*nvir_max*nvir_max, dtype=dtype)
    buf3 = np.empty(nocc_max*nocc_max*nvir_max*nvir_max, dtype=dtype)

    swapfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    fswap = lib.H5TmpFile(swapfile.name)
    swapfile = None

    use_eri_mesh = not (np.array(mesh) == eri_mesh).all()
    if use_eri_mesh:
        eri_inds = get_mesh_map(cell, None, None, mesh=mesh, mesh2=eri_mesh)

# ifft to make C(G) --> C(r)
# note the ordering of spin and k-pt indices is swapped
    C_ks_R = fswap.create_group("C_ks_R")
    for s in [0,1]:
        C_ks_s = get_spin_component(C_ks, s)
        for k in range(nkpts):
            key = "%d"%k
            C_k = C_ks_s[key][()]
            # C_ks_R["%s/%d"%(key,s)] = tools.ifft(C_k, mesh)
            C_k = wf_ifft(C_k, mesh, basis_ks[k])
            if use_eri_mesh:
                C_k = wf_fft(C_k, mesh)
                C_k = C_k[..., eri_inds]
                C_k = wf_ifft(C_k, eri_mesh)
            C_ks_R["%s/%d"%(key,s)] = C_k
            C_k = None

    v_ia_ks_R = fswap.create_group("v_ia_ks_R")

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
        nocc_i = nocc_sps[ki]

        tick[:] = logger.process_clock(), logger.perf_counter()

        Co_ki_R = [C_ks_R["%d/%d"%(ki,s)][:nocc_i[s]] for s in [0,1]]

        for ka in range(nkpts):
            kpta = kpts[ka]
            nocc_a = nocc_sps[ka]
            nvir_a = nvir_sps[ka]
            coulG = tools.get_coulG(cell, kpta-kpti, exx=False, mesh=eri_mesh)

            key_ka = "%d"%ka
            if key_ka in v_ia_ks_R: del v_ia_ks_R[key_ka]

            for s in [0,1]:
                Cv_ka_R = C_ks_R["%s/%d"%(key_ka,s)][nocc_a[s]:nocc_a[s]+nvir_a[s]]
                v_ia_R = np.ndarray((nocc_i[s],nvir_a[s],ngrids), dtype=dtype,
                                    buffer=buf1)

                for i in range(nocc_i[s]):
                    v_ia = tools.fft(Co_ki_R[s][i].conj() *
                                     Cv_ka_R, eri_mesh) * coulG
                    v_ia_R[i] = tools.ifft(v_ia, eri_mesh)

                v_ia_ks_R["%s/%d"%(key_ka,s)] = v_ia_R
                v_ia_R = Cv_ka_R = None

        Co_ki_R = None

        tock[:] = logger.process_clock(), logger.perf_counter()
        tspans[1] += tock - tick

        for kj in range(nkpts):
            nocc_j = nocc_sps[kj]
            kptij = kpti + kpts[kj]

            tick[:] = logger.process_clock(), logger.perf_counter()

            Co_kj_R = [C_ks_R["%d/%d"%(kj,s)][:nocc_j[s]] for s in [0,1]]

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

                nocc_a = nocc_sps[ka]
                nvir_a = nvir_sps[ka]
                nocc_b = nocc_sps[kb]
                nvir_b = nvir_sps[kb]

                tick[:] = logger.process_clock(), logger.perf_counter()
                phase = np.exp(-1j*lib.dot(coords,
                                           kptijab.reshape(-1,1))).reshape(-1)
                tock[:] = logger.process_clock(), logger.perf_counter()
                tspans[4] += tock - tick

                for s in [0,1]:

                    tick[:] = logger.process_clock(), logger.perf_counter()
                    Cv_kb_R = C_ks_R["%d/%d"%(kb,s)][nocc_b[s]:nocc_b[s]+nvir_b[s]]
                    v_ia = v_ia_ks_R["%d/%d"%(ka,s)][:]
                    tock[:] = logger.process_clock(), logger.perf_counter()
                    tspans[3] += tock - tick

                    v_ia *= phase
                    oovv_ka = np.ndarray((nocc_i[s],nocc_j[s],nvir_a[s],nvir_b[s]),
                                         dtype=dtype, buffer=buf2)
                    fill_oovv(oovv_ka, v_ia, Co_kj_R[s], Cv_kb_R, fac_oovv)
                    tick[:] = logger.process_clock(), logger.perf_counter()
                    tspans[4] += tick - tock

                    Cv_kb_R = None

                    if ka != kb:
                        Cv_ka_R = C_ks_R["%d/%d"%(ka,s)][nocc_a[s]:
                                                         nocc_a[s]+nvir_a[s]]
                        v_ib = v_ia_ks_R["%d/%s"%(kb,s)][:]
                        tock[:] = logger.process_clock(), logger.perf_counter()
                        tspans[3] += tock - tick

                        v_ib *= phase
                        oovv_kb = np.ndarray((nocc_i[s],nocc_j[s],nvir_b[s],nvir_a[s]),
                                             dtype=dtype, buffer=buf3)
                        fill_oovv(oovv_kb, v_ib, Co_kj_R[s], Cv_ka_R, fac_oovv)
                        tick[:] = logger.process_clock(), logger.perf_counter()
                        tspans[4] += tick - tock

                        Cv_ka_R = v_ib = None
                    else:
                        oovv_kb = oovv_ka

# Same-spin contribution to KUMP2 energy
                    tick[:] = logger.process_clock(), logger.perf_counter()
                    mo_e_o = moe_ks[s][ki][:nocc_i[s]]
                    mo_e_v = moe_ks[s][ka][nocc_a[s]:nocc_a[s]+nvir_a[s]]
                    eia = mo_e_o[:,None] - mo_e_v

                    if ka != kb:
                        mo_e_o = moe_ks[s][kj][:nocc_j[s]]
                        mo_e_v = moe_ks[s][kb][nocc_b[s]:nocc_b[s]+nvir_b[s]]
                        ejb = mo_e_o[:,None] - mo_e_v
                    else:
                        ejb = eia

                    eijab = lib.direct_sum('ia,jb->ijab',eia,ejb)
                    t2_ijab = np.conj(oovv_ka/eijab)
                    for invir_,nvir_ in enumerate(nvir_lst):
                        eijab_d = np.einsum('ijab,ijab->',
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
                        emp2_ss[invir_] += eijab_d + eijab_x
                    tock[:] = logger.process_clock(), logger.perf_counter()
                    tspans[5] += tock - tick

                    oovv_ka = oovv_kb = eijab = None

# Opposite-spin contribution to KUMP2 energy
                    if s == 0:
                        t = 1 - s
                        tick[:] = logger.process_clock(), logger.perf_counter()
                        Cv_kb_R = C_ks_R["%d/%d"%(kb,t)][nocc_b[t]:
                                                         nocc_b[t]+nvir_b[t]]
                        tock[:] = logger.process_clock(), logger.perf_counter()
                        tspans[3] += tock - tick

                        oovv_ka = np.ndarray((nocc_i[s],nocc_j[t],nvir_a[s],nvir_b[t]),
                                             dtype=dtype, buffer=buf2)
                        fill_oovv(oovv_ka, v_ia, Co_kj_R[t], Cv_kb_R, fac_oovv)
                        tick[:] = logger.process_clock(), logger.perf_counter()
                        tspans[4] += tick - tock

                        Cv_kb_R = v_ia = None

                        mo_e_o = moe_ks[t][kj][:nocc_j[t]]
                        mo_e_v = moe_ks[t][kb][nocc_b[t]:nocc_b[t]+nvir_b[t]]
                        ejb = mo_e_o[:,None] - mo_e_v

                        eijab = lib.direct_sum('ia,jb->ijab',eia,ejb)
                        t2_ijab = np.conj(oovv_ka/eijab)
                        for invir_,nvir_ in enumerate(nvir_lst):
                            eijab_d = np.einsum('ijab,ijab->',
                                                t2_ijab[:,:,:nvir_,:nvir_],
                                                oovv_ka[:,:,:nvir_,:nvir_]).real
                            if ka != kb:
                                eijab_d *= 2
                            eijab_d *= 2    # alpha,beta <-> beta,alpha
                            emp2_d[invir_] += eijab_d
                            emp2_os[invir_] += eijab_d
                        tock[:] = logger.process_clock(), logger.perf_counter()
                        tspans[5] += tock - tick

                        oovv_ka = eijab = None
                    else:
                        v_ia = None

        cput1 = log.timer('kpt %d (%6.3f %6.3f %6.3f)'%(ki,*kpti), *cput1)

    buf1 = buf2 = buf3 = None

    emp2_d *= 0.5 / nkpts
    emp2_x *= 0.5 / nkpts
    emp2_ss *= 0.5 / nkpts
    emp2_os *= 0.5 / nkpts
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


class PWKUMP2(kmp2.PWKRMP2):
    """
    Spin-unrestriced MP2 in a plane-wave basis.
    """
    def __init__(self, mf, nvir=None):
        kmp2.PWKRMP2.__init__(self, mf, nvir=nvir)

    def kernel(self, nvir=None, nvir_lst=None):
        cell = self.cell
        kpts = self.kpts
        chkfile = self._scf.chkfile
        summary = self.mp2_summary
        if nvir is None: nvir = self.nvir

        self.e_corr = kernel_dx_(cell, kpts, chkfile, summary, nvir=nvir,
                                 nvir_lst=nvir_lst,
                                 basis_ks=self._scf._basis_data,
                                 ecut_eri=self.ecut_eri)

        self._finalize()

        return self.e_corr


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
    cell.verbose = 5

    nk = 2
    kmesh = [nk] * 3
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)

    pwmf = pwscf.PWKUHF(cell, kpts)
    pwmf.nvir = 5
    pwmf.kernel()

    es = {"5": -0.01363871}

    pwmp = PWKUMP2(pwmf)
    pwmp.kernel(nvir_lst=[5])
    pwmp.dump_mp2_summary()
    nvir_lst = pwmp.mp2_summary["nvir_lst"]
    ecorr_lst = pwmp.mp2_summary["e_corr_lst"]
    for nvir,ecorr in zip(nvir_lst,ecorr_lst):
        err = abs(ecorr - es["%d"%nvir])
        print(err)
        assert(err < 1e-5)
