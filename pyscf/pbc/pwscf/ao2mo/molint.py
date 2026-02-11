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


""" Generating MO integrals
"""

import time
import h5py
import tempfile
import numpy as np

from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import tools

from pyscf.pbc.pwscf.pw_helper import get_kcomp, set_kcomp, wf_ifft, wf_fft, \
                                      get_mesh_map

dot = lib.dot
einsum = np.einsum


def kconserv(kpt123, reduce_latvec, kdota):
    tmp = dot(kpt123.reshape(1,-1), reduce_latvec) + kdota
    return np.where(abs(tmp - np.rint(tmp)).sum(axis=1)<1e-6)[0][0]


def get_molint_from_C(cell, C_ks, kpts, mo_slices=None, exxdiv=None,
                      erifile=None, dataname="eris", basis_ks=None,
                      ecut_eri=None):
    """
    Args:
        C_ks : list or h5py group
            If list, the MO coeff for the k-th kpt is C_ks[k]
            If h5py, the MO coeff for the k-th kpt is C_ks["%d"%k][()]
            Note: this function assumes that MOs from different kpts are appropriately padded.
        mo_slices
        erifile: str, h5py File or h5py Group
            The file to store the ERIs. If not given, the ERIs are held in memory.
    """
    cput0 = (logger.process_clock(), logger.perf_counter())

    nkpts = len(kpts)
    if basis_ks is None:
        mesh = cell.mesh
    else:
        mesh = basis_ks[0].mesh

    reduce_latvec = cell.lattice_vectors() / (2*np.pi)
    kdota = dot(kpts, reduce_latvec)

    fswap = lib.H5TmpFile(dir=lib.param.TMPDIR)

    if ecut_eri is not None:
        eri_mesh = cell.cutoff_to_mesh(ecut_eri)
        if (np.array(mesh) == eri_mesh).all():
            eri_mesh = None
        if (eri_mesh > mesh).any():
            log.warn("eri_mesh larger than mesh, so eri_mesh is ignored.")
            eri_mesh = None
        if eri_mesh is not None:
            eri_inds = get_mesh_map(cell, None, None, mesh=mesh, mesh2=eri_mesh)
    else:
        eri_mesh = None

    C_ks_R = fswap.create_group("C_ks_R")
    for k in range(nkpts):
        C_k = get_kcomp(C_ks, k)
        basis = None if basis_ks is None else basis_ks[k]
        # C_k = tools.ifft(C_k, mesh)
        C_k = wf_ifft(C_k, mesh, basis=basis)
        if eri_mesh is not None:
            C_k = wf_fft(C_k, mesh)
            C_k = C_k[..., eri_inds]
            C_k = wf_ifft(C_k, eri_mesh)
        set_kcomp(C_k, C_ks_R, k)
        C_k = None

    if eri_mesh is not None:
        mesh = eri_mesh

    coords = cell.get_uniform_grids(mesh=mesh)
    ngrids = coords.shape[0]
    fac = ngrids**3. / cell.vol / nkpts

    dtype = np.complex128
    if mo_slices is None:
        nmo = get_kcomp(C_ks, 0, load=False).shape[0]
        mo_slices = [(0,nmo)] * 4
    nmos = [mo_slice[1]-mo_slice[0] for mo_slice in mo_slices]
    buf = np.empty(nmos[0]*nmos[1]*ngrids, dtype=dtype)
    mo_ranges = [list(range(mo_slice[0],mo_slice[1])) for mo_slice in mo_slices]

    if erifile is None:
        incore = True
        deri = np.zeros((nkpts,nkpts,nkpts,*nmos), dtype=dtype)
    elif isinstance(erifile, (str, h5py.Group)):
        incore = False
        if isinstance(erifile, str):
            if h5py.is_hdf5(erifile):
                feri = h5py.File(erifile, "a")
            else:
                feri = h5py.File(erifile, "w")
        else:
            assert(isinstance(erifile, h5py.Group))
            feri = erifile
        if dataname in feri: del feri[dataname]
        deri = feri.create_dataset(dataname, (nkpts,nkpts,nkpts,*nmos),
                                   dtype=dtype)
        buf2 = np.empty(nmos, dtype=dtype)
    else:
        raise RuntimeError

    cput1 = logger.timer(cell, 'initialize pwmolint', *cput0)

    tick = np.zeros(2)
    tock = np.zeros(2)
    tspans = np.zeros((4,2))
    tcomps = ["init", "v_ks_R", "eri", "tot"]

    for k1 in range(nkpts):
        kpt1 = kpts[k1]
        p0,p1 = mo_slices[0]
        C_k1_R = get_kcomp(C_ks_R, k1, occ=mo_ranges[0])
        for k2 in range(nkpts):
            tick[:] = (logger.process_clock(), logger.perf_counter())

            kpt2 = kpts[k2]
            kpt12 = kpt2 - kpt1
            q0,q1 = mo_slices[1]
            C_k2_R = get_kcomp(C_ks_R, k2, occ=mo_ranges[1])
            coulG_k12 = tools.get_coulG(cell, kpt12, exx=exxdiv, mesh=mesh)
# FIXME: batch appropriately
            v_pq_k12 = np.ndarray((nmos[0],nmos[1],ngrids), dtype=dtype,
                                  buffer=buf)
            for p in range(p0,p1):
                ip = p - p0
                v_pq_k12[ip] = tools.ifft(tools.fft(C_k1_R[ip].conj() * C_k2_R,
                                          mesh) * coulG_k12, mesh)

            tock[:] = (logger.process_clock(), logger.perf_counter())
            tspans[1] += tock - tick

            for k3 in range(nkpts):
                kpt3 = kpts[k3]
                kpt123 = kpt12 - kpt3
                k4 = kconserv(kpt123, reduce_latvec, kdota)
                kpt4 = kpts[k4]
                kpt1234 = kpt123 + kpt4
                phase = np.exp(1j*lib.dot(coords,
                               kpt1234.reshape(-1,1))).reshape(-1)

                r0,r1 = mo_slices[2]
                C_k3_R = get_kcomp(C_ks_R, k3, occ=mo_ranges[2])
                s0,s1 = mo_slices[3]
                C_k4_R = get_kcomp(C_ks_R, k4, occ=mo_ranges[3]) * phase

                if incore:
                    vpqrs = deri[k1,k2,k3]
                else:
                    vpqrs = np.ndarray(nmos, dtype=dtype, buffer=buf2)
                for r in range(r0,r1):
                    ir = r - r0
                    rho_rs_k34 = C_k3_R[ir].conj() * C_k4_R
                    vpqrs[:,:,ir] = dot(v_pq_k12.reshape(-1,ngrids),
                                        rho_rs_k34.T).reshape(nmos[0],nmos[1],nmos[-1])
                vpqrs *= fac
                if not incore:
                    deri[k1,k2,k3,:] = vpqrs
                    vpqrs = None
            tick[:] = (logger.process_clock(), logger.perf_counter())
            tspans[2] += tick - tock

        tock[:] = (logger.process_clock(), logger.perf_counter())
        cput1 = logger.timer(cell, 'kpt %d (%6.3f %6.3f %6.3f)'%(k1,*kpt1),
                             *cput1)

    fswap.close()

    cput1 = logger.timer(cell, 'pwmolint', *cput0)
    tspans[3] = np.asarray(cput1) - np.asarray(cput0)

# dump timing
    def write_time(comp, t_comp, t_tot):
        tc, tw = t_comp
        tct, twt = t_tot
        rc = tc / tct * 100
        rw = tw / twt * 100
        fmtstr = 'CPU time for %10s %9.2f  ( %6.2f%% ), wall time %9.2f  ( %6.2f%% )'
        logger.debug1(cell, fmtstr, comp.ljust(10), tc, rc, tw, rw)

    t_tot = tspans[-1]
    for icomp,comp in enumerate(tcomps):
        write_time(comp, tspans[icomp], t_tot)

    return deri


if __name__ == "__main__":
    from pyscf.pbc import pwscf, gto

    atom = "H 0 0 0; H 0.9 0 0"
    a = np.eye(3) * 3
    basis = "gth-szv"
    pseudo = "gth-pade"

    ke_cutoff = 30

    cell = gto.Cell(atom=atom, a=a, basis=basis, pseudo=pseudo,
                    ke_cutoff=ke_cutoff)
    cell.build()
    cell.verbose = 5

    kmesh = [2,1,1]
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)

    nvir = 5
    chkfile = "mf.chk"
    mf = pwscf.KRHF(cell, kpts)
    mf.nvir = nvir
    # mf.init_guess = "chk"
    mf.chkfile = chkfile
    mf.kernel()

    mmp = pwscf.KMP2(mf)
    mmp.kernel()

    fchk = h5py.File(chkfile, "r")
    C_ks = fchk["mo_coeff"]

    swapfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    erifile = swapfile.name
    swapfile = None

    no = cell.nelectron // 2
    nmo = no + nvir
    mo_slices = [(0,no),(no,nmo),(0,no),(no,nmo)]
    feri = get_molint_from_C(cell, C_ks, kpts, mo_slices, exxdiv=None,
                             erifile=erifile, dataname="eris")

    fchk.close()
