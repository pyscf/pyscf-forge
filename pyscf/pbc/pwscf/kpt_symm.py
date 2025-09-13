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
# Author: Kyle Bystrom <kylebystrom@gmail.com>
#

""" k-point symmetry for plane-wave HF and DFT
"""

import tempfile
from pyscf import lib
import numpy as np
import ctypes
from pyscf.pbc.lib import kpts as libkpts
from pyscf.pbc import tools
from pyscf.pbc.pwscf import jk
from pyscf.pbc.pwscf import khf, kuhf, krks, kuks
from pyscf.lib import logger
from pyscf.pbc.pwscf.pw_helper import wf_fft, wf_ifft


libpw = lib.load_library("libpwscf")


def add_rotated_realspace_func_(fin, fout, mesh, rot, wt):
    """
    For real-valued functions fin and fout on mesh,

    fout(rot * x) = fout(rot * x) + wt * fin(x)

    where rot is a rotation operator represented as a 3x3 integer matrix,
    and x is an integer-valued 3D vector representing the position
    of the function on the mesh.
    """
    assert fin.dtype == np.float64
    assert fout.dtype == np.float64
    assert fin.flags.c_contiguous
    assert fout.flags.c_contiguous
    shape = np.asarray(mesh, dtype=np.int32, order="C")
    assert fout.size == np.prod(shape)
    assert fin.size == np.prod(shape)
    rot = np.asarray(rot, dtype=np.int32, order="C")
    assert rot.shape == (3, 3)
    libpw.add_rotated_realspace_func(
        fin.ctypes, fout.ctypes, shape.ctypes, rot.ctypes, ctypes.c_double(wt)
    )


def get_rotated_complex_func(fin, mesh, rot, shift=None, fout=None):
    """
    For complex-valued fin on a given mesh, store a rotated
    and shifted function in fout.

    fout(rot * x + shift) = fin(x)

    where rot and x are defined as in add_rotated_realspace_func_,
    and shift is an integer-valued 3D vector.
    """
    if shift is None:
        shift = [0, 0, 0]
    assert fin.dtype == np.complex128
    fout = np.ndarray(shape=mesh, dtype=np.complex128, order="C", buffer=fout)
    assert fin.flags.c_contiguous
    shape = np.asarray(mesh, dtype=np.int32, order="C")
    assert fin.size == np.prod(shape), f"{fin.shape} {shape}"
    rot = np.asarray(rot, dtype=np.int32, order="C")
    assert rot.shape == (3, 3)
    shift = np.asarray(shift, dtype=np.int32, order="C")
    libpw.get_rotated_complex_func(
        fin.ctypes, fout.ctypes, shape.ctypes, rot.ctypes, shift.ctypes
    )
    return fout


def get_rho_R_ksym(C_ks, mocc_ks, mesh, kpts, basis_ks=None):
    """
    Get the real-space density from C_ks and mocc_ks, where the
    set of kpts is reduced to the IBZ. kpts is a Kpoints object
    storing both the IBZ and BZ k-points along with symmetry
    mappings between them.
    """
    rho_R = np.zeros(np.prod(mesh), dtype=np.float64, order="C")
    tmp_R = np.empty_like(rho_R)
    nelec = 0
    if basis_ks is None:
        basis_ks = [None] * len(C_ks)
    for k, mocc_k in enumerate(mocc_ks):
        nelec += mocc_k.sum() * kpts.weights_ibz[k]
    for k in range(kpts.nkpts_ibz):
        occ = np.where(mocc_ks[k] > jk.THR_OCC)[0].tolist()
        Co_k = jk.get_kcomp(C_ks, k, occ=occ)
        Co_k_R = wf_ifft(Co_k, mesh, basis=basis_ks[k])
        jk._mul_by_occ_(Co_k_R, mocc_ks[k], occ)
        tmp_R[:] = lib.einsum("ig,ig->g", Co_k_R.conj(), Co_k_R).real
        for istar, iop in enumerate(kpts.stars_ops[k]):
            rot = kpts.ops[iop].rot
            add_rotated_realspace_func_(tmp_R, rho_R, mesh, rot, 1.0)
    return rho_R


def get_ibz2bz_info(C_ks_ibz, kpts, k_bz, occ_ks=None):
    k_ibz = kpts.bz2ibz[k_bz]
    iop = kpts.stars_ops_bz[k_bz]
    rot = kpts.ops[iop].rot
    if occ_ks is not None:
        occ = occ_ks[k_ibz]
    else:
        occ = None
    C_k_ibz = jk.get_kcomp(C_ks_ibz, k_ibz, occ=occ)
    return (
        kpts.kpts_scaled_ibz[k_ibz],
        kpts.kpts_scaled[k_bz],
        rot,
        kpts.time_reversal_symm_bz[k_bz],
        C_k_ibz,
    )


def get_ibz2bz_info_v2(kpts, k_ibz):
    maps = []
    for istar, iop in enumerate(kpts.stars_ops[k_ibz]):
        k_bz = kpts.stars[k_ibz][istar]
        rot = kpts.ops[iop].rot
        maps.append([
            kpts.kpts_scaled_ibz[k_ibz],
            kpts.kpts_scaled[k_bz],
            rot,
            kpts.time_reversal_symm_bz[k_bz],
            k_bz,
        ])
    return maps


def get_C_from_ibz2bz_info(mesh, kpt_ibz, kpt_bz, rot, tr, C_k_ibz,
                           out=None, realspace=False):
    """
    From a set of bands C_k_ibz at a k-point in the IBZ (kpt_ibz), get the
    bands at a symmetrically equivalent k-point kpt_bz.

    kpt_ibz and kpt_bz are the scaled k-points (fractional coords in bz).
    tr is a bool indicating whether the symmetry operation
    includes time-reversal.

    If tr == True, kpt_bz = -rot * kpt_ibz.
    If tr == False, kpt_bz = rot * kpt_ibz.
    In both cases, the k-points are equivalent modulo 1
    (in fractional coordinates).
    """
    out = np.ndarray(C_k_ibz.shape, dtype=np.complex128, order="C", buffer=out)
    rrot = rot.copy()
    krot = np.rint(np.linalg.inv(rot).T)
    if tr:
        krot[:] *= -1
    if not realspace:
        rot = krot
    else:
        rot = rrot
    new_kpt = krot.dot(kpt_ibz)
    shift = [0, 0, 0]
    for v in range(3):
        while np.round(new_kpt[v] - kpt_bz[v]) < 0:
            shift[v] += 1
            new_kpt[v] += 1
        while np.round(new_kpt[v] - kpt_bz[v]) > 0:
            shift[v] -= 1
            new_kpt[v] -= 1
        assert np.abs(new_kpt[v] - kpt_bz[v]) < 1e-8, f"{v}, {new_kpt} {kpt_bz}"
    kshift = [-1 * v for v in shift]
    shift = [0, 0, 0] if realspace else kshift
    for i in range(out.shape[0]):
        get_rotated_complex_func(C_k_ibz[i], mesh, rot, shift, fout=out[i])
        if tr:
            out[i] = out[i].conj()
    if realspace:
        outshape = out.shape
        out.shape = (-1, mesh[0], mesh[1], mesh[2])
        wt = 1.0 / mesh[v]
        phases = []
        for v in range(3):
            phases.append(np.exp(2j * np.pi * kshift[v] * np.arange(mesh[v]) * wt))
        out[:] *= phases[0][None, :, None, None]
        out[:] *= phases[1][None, None, :, None]
        out[:] *= phases[2][None, None, None, :]
        out.shape = outshape
    return out


def get_C_from_symm(C_ks_ibz, mesh, kpts, k_bz, out=None, occ_ks=None,
                    realspace=False):
    """
    Get C_k in the full BZ from C_ks_ibz in the IBZ, at the k-point index k_bz.
    """
    kpt_ibz, kpt_bz, rot, tr, C_k_ibz = get_ibz2bz_info(C_ks_ibz, kpts, k_bz,
                                                        occ_ks=occ_ks)
    return get_C_from_ibz2bz_info(mesh, kpt_ibz, kpt_bz, rot, tr, C_k_ibz,
                                  out=out, realspace=realspace)


def get_C_from_C_ibz(C_ks_ibz, mesh, kpts, realspace=False):
    """
    Get C_ks in the full BZ from C_kz_ibz in the IBZ.
    Assumes that C_ks_ibz is incore.
    """
    C_ks = []
    for k in range(kpts.nkpts):
        C_ks.append(get_C_from_symm(
            C_ks_ibz, mesh, kpts, k, realspace=realspace
        ))
    return C_ks


def apply_k_sym_s1(cell, C_ks, mocc_ks, kpts_obj, Ct_ks, ktpts, mesh, Gv,
                   out=None, outcore=False, basis_ks=None):
    kpts = kpts_obj.kpts
    nkpts = len(kpts)
    nktpts = len(ktpts)
    ngrids = np.prod(mesh)
    fac = ngrids**2./(cell.vol*nkpts)
    mocc_ks = [mocc_ks[kpts_obj.bz2ibz[k]] for k in range(nkpts)]
    occ_ks = [np.where(mocc_ks[k] > jk.THR_OCC)[0] for k in range(nkpts)]

    if out is None: out = [None] * nktpts
    if basis_ks is None:
        basis_ks = [None] * len(C_ks)
        use_basis = False
    else:
        use_basis = True

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

    if use_basis:
        # TODO this is probably a bit memory-intensive
        for k_ibz in range(len(C_ks)):
            Co_k_ibz = jk.get_kcomp(C_ks, k_ibz, occ=occ_ks[k_ibz])
            jk._mul_by_occ_(Co_k_ibz, mocc_ks[k_ibz], occ_ks[k_ibz])
            Co_k_ibz_R = wf_ifft(Co_k_ibz, mesh, basis=basis_ks[k_ibz])
            maps = get_ibz2bz_info_v2(kpts_obj, k_ibz)
            for kmap in maps:
                k_bz = kmap[-1]
                kmap[-1] = Co_k_ibz_R
                Co_k_R = get_C_from_ibz2bz_info(mesh, *kmap, realspace=True)
                jk.set_kcomp(Co_k_R, Co_ks_R, k_bz)
    else:
        for k_ibz in range(len(C_ks)):
            Co_k_ibz = jk.get_kcomp(C_ks, k_ibz, occ=occ_ks[k_ibz])
            jk._mul_by_occ_(Co_k_ibz, mocc_ks[k_ibz], occ_ks[k_ibz])
            maps = get_ibz2bz_info_v2(kpts_obj, k_ibz)
            for kmap in maps:
                k_bz = kmap[-1]
                kmap[-1] = Co_k_ibz
                Co_k = get_C_from_ibz2bz_info(mesh, *kmap, realspace=False)
                jk.set_kcomp(wf_ifft(Co_k, mesh), Co_ks_R, k_bz)
        """
        Below is a draft of an alternate approach for the above loop
        for k in range(nkpts):
            # Co_k = jk.set_kcomp(C_ks, k, occ=occ_ks[k])
            # TODO need to make a new basis for symmetrized calculation,
            # or perhaps just rotate it in real space?
            Co_k = get_C_from_symm(C_ks, mesh, kpts_obj, k, occ_ks=occ_ks)
            jk._mul_by_occ_(Co_k, mocc_ks[k], occ_ks[k])
            jk.set_kcomp(wf_ifft(Co_k, mesh), Co_ks_R, k)
            Co_k = None
        """

    for k in range(nktpts):
        Ct_k = jk.get_kcomp(Ct_ks, k)
        jk.set_kcomp(wf_ifft(Ct_k, mesh, basis=basis_ks[k]), Ct_ks_R, k)
        Ct_k = None

    for k1,kpt1 in enumerate(ktpts):
        Ct_k1_R = jk.get_kcomp(Ct_ks_R, k1)
        Ctbar_k1 = np.zeros_like(Ct_k1_R)
        for k2,kpt2 in enumerate(kpts):
            coulG = tools.get_coulG(cell, kpt1-kpt2, exx=False, mesh=mesh,
                                    Gv=Gv)
            Co_k2_R = jk.get_kcomp(Co_ks_R, k2)
            for j in occ_ks[k2]:
                Cj_k2_R = Co_k2_R[j]
                vij_R = tools.ifft(tools.fft(Ct_k1_R * Cj_k2_R.conj(), mesh) *
                                   coulG, mesh)
                Ctbar_k1 += vij_R * Cj_k2_R

        Ctbar_k1 = wf_fft(Ctbar_k1, mesh, basis=basis_ks[k1]) * fac
        jk.set_kcomp(Ctbar_k1, out, k1)
        Ctbar_k1 = None

    return out


def apply_k_sym(cell, C_ks, mocc_ks, kpts, mesh, Gv, Ct_ks=None, ktpts=None,
                exxdiv=None, out=None, outcore=False, basis_ks=None):
    """
    Apply the EXX operator with symmetry-reduced k-points.
    """
    if Ct_ks is None:
        # TODO s2 symmetry
        Ct_ks = C_ks
        ktpts = kpts.kpts_ibz
        return apply_k_sym_s1(cell, C_ks, mocc_ks, kpts, Ct_ks, ktpts, mesh, Gv,
                              out, outcore, basis_ks)
    else:
        return apply_k_sym_s1(cell, C_ks, mocc_ks, kpts, Ct_ks, ktpts, mesh, Gv,
                              out, outcore, basis_ks)


def get_ace_support_vec(cell, C1_ks, mocc1_ks, k1pts, C2_ks=None, k2pts=None,
                        out=None, mesh=None, Gv=None, exxdiv=None, method="cd",
                        outcore=False, basis_ks=None):
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

    W_ks = apply_k_sym(cell, C1_ks, mocc1_ks, k1pts, mesh, Gv,
                       Ct_ks=C2_ks, ktpts=k2pts, exxdiv=exxdiv, out=W_ks,
                       outcore=outcore, basis_ks=basis_ks)

    if C2_ks is None: C2_ks = C1_ks
    if k2pts is None: k2pts = k1pts
    nk2pts = len(k2pts)

    for k in range(nk2pts):
        C_k = jk.get_kcomp(C2_ks, k)
        W_k = jk.get_kcomp(W_ks, k)
        W_k = get_support_vec(C_k, W_k, method=method)
        jk.set_kcomp(W_k, out, k)
        W_k = None

    if outcore:
        del fswap[dname0]

    return out


class KsymAdaptedPWJK(jk.PWJK):
    """
    Lattice symmetry-adapted PWJK module.
    """
    _ace_kpts = None

    def __init__(self, cell, kpts, mesh=None, exxdiv=None, **kwargs):
        if cell.space_group_symmetry and not cell.symmorphic:
            raise NotImplementedError(
                "Plane-wave calculation with k-point symmetry only "
                "supports symmorphic symmetry operations"
            )
        super().__init__(cell, kpts, mesh=mesh, exxdiv=exxdiv, **kwargs)

    def __init_exx(self):
        if self.outcore:
            self.swapfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
            self.fswap = lib.H5TmpFile(self.swapfile.name)
            self.exx_W_ks = self.fswap.create_group("exx_W_ks")
        else:
            self.exx_W_ks = {}

    def get_rho_R(self, C_ks, mocc_ks, mesh=None, Gv=None, ncomp=1):
        if mesh is None: mesh = self.mesh
        if Gv is None: Gv = self.get_Gv(mesh)
        if ncomp == 1:
            rho_R = get_rho_R_ksym(
                C_ks, mocc_ks, mesh, self.kpts, basis_ks=self.basis_ks
            )
        else:
            rho_R = 0.
            for comp in range(ncomp):
                C_ks_comp = jk.get_kcomp(C_ks, comp, load=False)
                rho_R += get_rho_R_ksym(
                    C_ks_comp, mocc_ks[comp], mesh, self.kpts,
                    basis_ks=self.basis_ks
                )
            rho_R *= 1./ncomp
        return rho_R

    def get_vj_R_from_rho_R(self, rho_R, mesh=None, Gv=None):
        if mesh is None: mesh = self.mesh
        if Gv is None: Gv = self.get_Gv(mesh)
        cell = self.cell
        nkpts = self.kpts.nkpts
        ngrids = Gv.shape[0]
        fac = ngrids**2 / (cell.vol*nkpts)
        vj_R = tools.ifft(tools.fft(rho_R, mesh) * tools.get_coulG(cell, Gv=Gv),
                          mesh).real * fac
        return vj_R

    def update_k_support_vec(self, C_ks, mocc_ks, kpts, Ct_ks=None,
                             mesh=None, Gv=None, exxdiv=None, comp=None):
        """
        kpts are the kpts in the bz, or those for which you want to calculate
        the support vectors.
        """
        if self.exx_W_ks is None:
            self.__init_exx()

        if mesh is None:
            mesh = self.mesh

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
            self._ace_kpts = kpts
            out = get_ace_support_vec(self.cell, C_ks, mocc_ks, self.kpts,
                                      C2_ks=Ct_ks, k2pts=kpts, out=out,
                                      mesh=mesh, Gv=Gv, exxdiv=exxdiv,
                                      method="cd", outcore=self.outcore,
                                      basis_ks=self.basis_ks)
        else:   # store ifft of Co_ks
            # TODO kpt_symm without ACE
            raise NotImplementedError("kpt_symm only supports ACE for EXX")
            """
            if mesh is None: mesh = self.mesh
            for k in range(nkpts):
                occ = np.where(mocc_ks[k]>jk.THR_OCC)[0]
                Co_k = jk.get_kcomp(C_ks, k, occ=occ)
                jk.set_kcomp(tools.ifft(Co_k, mesh), out, k)
            """

    def apply_k_kpt(self, C_k, kpt, mesh=None, Gv=None, exxdiv=None, comp=None,
                    basis=None):
        if comp is None:
            W_ks = self.exx_W_ks
        elif isinstance(comp, int):
            W_ks = jk.get_kcomp(self.exx_W_ks, comp, load=False)
        else:
            raise RuntimeError("comp must be None or int.")

        if self.ace_exx:
            if self._ace_kpts is None:
                kpts_ibz = self.kpts.kpts_ibz
            else:
                kpts_ibz = self._ace_kpts
            k = jk.member(kpt, kpts_ibz)[0]
            W_k = jk.get_kcomp(W_ks, k)
            return jk.apply_k_kpt_support_vec(C_k, W_k)
        else:
            # TODO kpt_symm without ACE
            raise NotImplementedError("kpt_symm only supports ACE for EXX")
            """
            cell = self.cell
            kpts = self.kpts
            nkpts = len(kpts)
            if mesh is None: mesh = self.mesh
            if Gv is None: Gv = self.get_Gv(mesh)
            if exxdiv is None: exxdiv = self.exxdiv
            mocc_ks = [np.ones(jk.get_kcomp(W_ks, k, load=False).shape[0])*2
                       for k in range(nkpts)]
            return apply_k_kpt(cell, C_k, kpt, None, mocc_ks, kpts, mesh, Gv,
                               C_ks_R=W_ks, exxdiv=exxdiv)
            """


def jksym(mf, with_jk=None, ace_exx=True, outcore=False, mesh=None,
          basis_ks=None):
    if with_jk is None:
        with_jk = KsymAdaptedPWJK(mf.cell, mf.kpts_obj, exxdiv=mf.exxdiv,
                                  mesh=mesh, basis_ks=basis_ks)
        with_jk.ace_exx = ace_exx
        with_jk.outcore = outcore

    mf.with_jk = with_jk

    return mf


class KsymMixin:
    """
    This mixin can be inherited to make a PWKSCF object support
    symmetry reduction of the k-points to the
    irreducible Brillouin zone (IBZ).
    """
    def _set_madelung(self):
        self._madelung = tools.pbc.madelung(self.cell, self.all_kpts)
        self._etot_shift_ewald = -0.5*self._madelung*self.cell.nelectron

    @property
    def kpts(self):
        return self._kpts.kpts_ibz
    @property
    def all_kpts(self):
        return self._kpts.kpts
    @property
    def kpts_obj(self):
        return self._kpts
    @property
    def weights(self):
        return self._kpts.weights_ibz
    @kpts.setter
    def kpts(self, x):
        if isinstance(x, np.ndarray):
            kpts = libkpts.make_kpts(
                self.cell,
                kpts=np.reshape(x, (-1,3)),
                space_group_symmetry=False,
                time_reversal_symmetry=False,
            )
        elif isinstance(x, libkpts.KPoints):
            kpts = x
        else:
            raise TypeError("Input kpts have wrong type: %s" % type(kpts))
        self._kpts = kpts
        # update madelung constant and energy shift for exxdiv
        self._set_madelung()
        if self._ecut_wf is None:
            self._wf_mesh = None
            self._xc_mesh = None
            self._wf2xc = None
            self._basis_data = None
        else:
            self.set_meshes()

    def init_jk(self, with_jk=None, ace_exx=None):
        if ace_exx is None: ace_exx = self.ace_exx
        return jksym(self, with_jk=with_jk, ace_exx=ace_exx,
                     outcore=self.outcore, mesh=self.wf_mesh,
                     basis_ks=self._basis_data)

    def get_init_guess_key(self, cell=None, kpts=None, basis=None, pseudo=None,
                           nvir=None, key="hcore", out=None):
        if cell is None: cell = self.cell
        if kpts is None: kpts = self.kpts
        if nvir is None: nvir = self.nvir

        if key in ["h1e","hcore","cycle1","scf"]:
            C_ks, mocc_ks = khf.get_init_guess(cell, kpts,
                                               basis=basis, pseudo=pseudo,
                                               nvir=nvir, key=key, out=out,
                                               kpts_obj=self.kpts_obj,
                                               mesh=self.wf_mesh)
        else:
            logger.warn(self, "Unknown init guess %s", key)
            raise RuntimeError

        if self._basis_data is not None:
            for k, kpt in enumerate(self.kpts):
                inds = self.get_basis_kpt(kpt).indexes
                jk.set_kcomp(np.ascontiguousarray(C_ks[k][:, inds]), C_ks, k)

        return C_ks, mocc_ks


class KsymAdaptedPWKRHF(KsymMixin, khf.PWKRHF):
    pass


class KsymAdaptedPWKUHF(KsymMixin, kuhf.PWKUHF):
    pass


class KsymAdaptedPWKRKS(KsymMixin, krks.PWKRKS):
    pass


class KsymAdaptedPWKUKS(KsymMixin, kuks.PWKUKS):
    pass


if __name__ == "__main__":
    from pyscf.pbc import gto
    from pyscf.pbc.pwscf.khf import PWKRHF
    import time

    cell = gto.Cell(
        atom = "C 0 0 0; C 0.89169994 0.89169994 0.89169994",
        a = np.asarray([
                [0.       , 1.78339987, 1.78339987],
                [1.78339987, 0.        , 1.78339987],
                [1.78339987, 1.78339987, 0.        ]]),
        basis="gth-szv",
        ke_cutoff=50,
        pseudo="gth-pade",
        space_group_symmetry=True,
        symmorphic=True,
    )
    cell.build()
    cell.verbose = 6

    kmesh = [2, 2, 2]
    center = [0, 0, 0]
    kpts = cell.make_kpts(kmesh)
    skpts = cell.make_kpts(
        kmesh,
        scaled_center=center,
        space_group_symmetry=True,
        time_reversal_symmetry=True,
    )

    mf = PWKRHF(cell, kpts, ecut_wf=40)
    mf.nvir = 4
    t0 = time.monotonic()
    mf.kernel()
    t1 = time.monotonic()

    mf2 = KsymAdaptedPWKRHF(cell, skpts, ecut_wf=20)
    mf2.damp_type = "simple"
    mf2.damp_factor = 0.7
    mf2.nvir = 4
    t2 = time.monotonic()
    mf2.kernel()
    t3 = time.monotonic()

    print(mf.e_tot, mf2.e_tot)
    mf.dump_scf_summary()
    mf2.dump_scf_summary()
    print("nkpts in BZ and IBZ", skpts.nkpts, skpts.nkpts_ibz)
    print("Runtime without symmmetry", t1 - t0)
    print("Runtime with symmetry", t3 - t2)
