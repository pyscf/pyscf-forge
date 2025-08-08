#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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

""" Spin-restricted Kohn-Sham DFT in plane-wave basis
"""

from pyscf.pbc.pwscf import khf
from pyscf.pbc.dft import rks
from pyscf.pbc import gto, tools
from pyscf import __config__, lib
from pyscf.lib import logger
import numpy as np


def get_rho_for_xc(mf, xctype, C_ks, mocc_ks, mesh=None, Gv=None,
                   out=None):
    mocc_ks = np.asarray(mocc_ks)
    if mocc_ks.ndim == 2:
        spin = 0
    else:
        assert mocc_ks.ndim == 3
        spin = 1
    cell = mf.cell
    if mesh is None: mesh = mf.wf_mesh
    if Gv is None: Gv = cell.get_Gv(mesh)
    if xctype == "LDA":
        nrho = 1
    elif xctype == "GGA":
        nrho = 4
    elif xctype == "MGGA":
        nrho = 5
    elif xctype == None:
        nrho = 0
    else:
        raise ValueError(f"Unsupported xctype {xctype}")
    if spin > 0:
        nspin = len(C_ks)
        assert nspin > 0
    else:
        nspin = 1
        C_ks = [C_ks]
        mocc_ks = [mocc_ks]
    outshape = (nspin, nrho, np.prod(mesh))
    rhovec_R = np.ndarray(outshape, buffer=out)
    if nrho > 0:
        for s in range(nspin):
            rhovec_R[s, 0] = mf.with_jk.get_rho_R(
                C_ks[s], mocc_ks[s], mesh=mesh, Gv=Gv
            )
    if nrho > 1:
        for s in range(nspin):
            rho_G = tools.fft(rhovec_R[s, 0], mesh)
            for v in range(3):
                drho_G = 1j * Gv[:, v] * rho_G
                rhovec_R[s, v + 1] = tools.ifft(drho_G, mesh).real
    if nrho > 4:
        dC_ks = [np.empty_like(C_k) for C_k in C_ks[0]]
        for s in range(nspin):
            rhovec_R[s, 4] = 0
            const = 1j * np.sqrt(0.5)
            for v in range(3):
                for k, C_k in enumerate(C_ks[s]):
                    if mf.with_jk.basis_ks is None:
                        ikgv = const * (mf.kpts[k][v] + Gv[:, v])
                    else:
                        ikgv = const * mf.with_jk.basis_ks[k].Gk[:, v]
                    dC_ks[k][:] = ikgv * C_k
                rhovec_R[s, 4] += mf.with_jk.get_rho_R(
                    dC_ks, mocc_ks[s], mesh=mesh, Gv=Gv
                )
    if spin == 0:
        rhovec_R = rhovec_R[0]
    return rhovec_R


def apply_vxc_kpt(mf, C_k, kpt, vxc_R, vtau_R=None, mesh=None, Gv=None,
                  C_k_R=None, comp=None, basis=None):
    cell = mf.cell
    if mesh is None: mesh = mf.wf_mesh
    if Gv is None: Gv = cell.get_Gv(mesh)
    if comp is not None:
        vxc_R = vxc_R[comp]
        if vtau_R is not None:
            vtau_R = vtau_R[comp]
    apply_j_kpt = mf.with_jk.apply_j_kpt
    Cbar_k = apply_j_kpt(C_k, mesh, vxc_R, C_k_R=C_k_R, basis=basis)
    if vtau_R is not None:
        const = 1j * np.sqrt(0.5)
        dC_k = np.empty_like(C_k)
        for v in range(3):
            if mf.with_jk.basis_ks is None:
                ikgv = const * (kpt[v] + Gv[:, v])
            else:
                ikgv = const * basis.Gk[:, v]
            dC_k[:] = ikgv * C_k
            dC_k[:] = apply_j_kpt(dC_k, mesh, vtau_R, basis=basis)
            Cbar_k[:] += ikgv.conj() * dC_k
    return Cbar_k


def eval_xc(mf, xc_code, rhovec_R, xctype):
    if rhovec_R.ndim == 2:
        spin = 0
    else:
        assert rhovec_R.ndim == 3
        spin = 1
    exc_R, vxcvec_R = mf._numint.eval_xc_eff(xc_code, rhovec_R, deriv=1,
                                             xctype=xctype)[:2]
    dv = mf.cell.vol / exc_R.size
    if spin == 0:
        vxcvec_R = vxcvec_R[None, ...]
        rho_R = rhovec_R[0]
        rhovec_R = rhovec_R.view()[None, ...]
    else:
        rho_R = rhovec_R[:, 0].sum(0)
    exc = dv * exc_R.dot(rho_R)
    return exc, vxcvec_R


def vxc_from_vxcvec(rhovec_R, vxcvec_R, xctype, mesh, Gv, dv):
    nspin = vxcvec_R.shape[0]
    vxc_R = vxcvec_R[:, 0].copy()
    vxcdot = 0
    for s in range(nspin):
        if xctype in ["GGA", "MGGA"]:
            vrho_G = 0
            for v in range(3):
                vdrho_G = tools.fft(vxcvec_R[s, v + 1], mesh)
                vrho_G += -1j * Gv[:, v] * vdrho_G
            vxc_R[s, :] += tools.ifft(vrho_G, mesh).real
        vxcdot += vxc_R[s].dot(rhovec_R[s, 0])
    if xctype == "MGGA":
        vtau_R = vxcvec_R[:, 4].copy()
        for s in range(nspin):
            vxcdot += vtau_R[s].dot(rhovec_R[s, 4])
    else:
        vtau_R = None
    return vxcdot * dv, vxc_R, vtau_R


def apply_veff_kpt(mf, C_k, kpt, mocc_ks, kpts, mesh, Gv, vj_R, with_jk,
                   exxdiv, C_k_R=None, comp=None, ret_E=False):
    r""" Apply non-local part of the Fock opeartor to orbitals at given
    k-point. The non-local part includes the exact exchange.
    Also apply the semilocal XC part to the orbitals.
    """
    log = logger.Logger(mf.stdout, mf.verbose)

    tspans = np.zeros((3,2))
    es = np.zeros(3, dtype=np.complex128)
    ni = mf._numint
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mf.cell.spin)
    if omega != 0:
        # TODO range-separated hybrid functionals
        raise NotImplementedError

    basis = mf.get_basis_kpt(kpt)

    tick = np.asarray([logger.process_clock(), logger.perf_counter()])
    tmp = with_jk.apply_j_kpt(C_k, mesh, vj_R, C_k_R=C_k_R, basis=basis)
    Cbar_k = tmp * 2.
    es[0] = np.einsum("ig,ig->", C_k.conj(), tmp) * 2.
    tock = np.asarray([logger.process_clock(), logger.perf_counter()])
    tspans[0] = np.asarray(tock - tick).reshape(1,2)

    if ni.libxc.is_hybrid_xc(mf.xc):
        tmp = -hyb * with_jk.apply_k_kpt(C_k, kpt, mesh=mesh, Gv=Gv, exxdiv=exxdiv,
                                         comp=comp, basis=basis)
        Cbar_k += tmp
        es[1] = np.einsum("ig,ig->", C_k.conj(), tmp)
    else:
        es[1] = 0.0
    tick = np.asarray([logger.process_clock(), logger.perf_counter()])
    tspans[1] = np.asarray(tick - tock).reshape(1,2)

    tmp = mf.apply_vxc_kpt(C_k, kpt, vxc_R=vj_R.vxc_R, mesh=mesh, Gv=Gv,
                           C_k_R=C_k_R, vtau_R=vj_R.vtau_R, comp=comp,
                           basis=basis)
    Cbar_k += tmp
    es[2] = vj_R.exc
    tock = np.asarray([logger.process_clock(), logger.perf_counter()])
    tspans[2] = np.asarray(tock - tick).reshape(1,2)

    for ie_comp,e_comp in enumerate(mf.scf_summary["e_comp_name_lst"][-3:]):
        key = "t-%s" % e_comp
        if key not in mf.scf_summary:
            mf.scf_summary[key] = np.zeros(2)
        mf.scf_summary[key] += tspans[ie_comp]

    if ret_E:
        if (np.abs(es.imag) > 1e-6).any():
            e_comp = mf.scf_summary["e_comp_name_lst"][-2:]
            icomps = np.where(np.abs(es.imag) > 1e-6)[0]
            log.warn("Energy has large imaginary part:" +
                     "%s : %s\n" * len(icomps),
                     *[s for i in icomps for s in [e_comp[i],es[i]]])
        es = es.real
        return Cbar_k, es
    else:
        return Cbar_k


class PWKohnShamDFT(rks.KohnShamDFT):
    
    def __init__(self, xc='LDA,VWN'):
        rks.KohnShamDFT.__init__(self, xc)
        self.scf_summary["e_comp_name_lst"].append("xc")

    get_rho_for_xc = get_rho_for_xc
    apply_vxc_kpt = apply_vxc_kpt
    eval_xc = eval_xc
    apply_veff_kpt = apply_veff_kpt

    @property
    def etot_shift_ewald(self):
        ni = self._numint
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(
            self.xc, spin=self.cell.spin
        )
        if omega != 0:
            # TODO range-separated hybrid functionals
            raise NotImplementedError
        return hyb * self._etot_shift_ewald

    @property
    def madelung(self):
        ni = self._numint
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(
            self.xc, spin=self.cell.spin
        )
        if omega != 0:
            # TODO range-separated hybrid functionals
            raise NotImplementedError
        return hyb * self._madelung

    def nuc_grad_method(self):
        raise NotImplementedError

    def get_vj_R_from_rho_R(self, *args, **kwargs):
        # TODO
        raise NotImplementedError

    def coarse_to_dense_grid(self, func_xR, out_xr=None):
        # TODO use real FFTs here since the real-space density is real
        ratio = np.prod(self.xc_mesh) / np.prod(self.wf_mesh)
        invr = 1 / ratio
        rhovec_G = tools.fft(func_xR, self.wf_mesh)
        dense_size = np.prod(self.xc_mesh)
        if func_xR.ndim == 1:
            shape = (dense_size,)
        else:
            nrho = func_xR.shape[0]
            shape = (nrho, dense_size)
        rhovec_g = np.zeros(shape, dtype=np.complex128)
        # print(rhovec_g.shape, self._wf2xc.shape, rhovec_G.shape)
        rhovec_g[..., self._wf2xc] = rhovec_G
        if out_xr is None:
            rhovec_r = tools.ifft(rhovec_g, self.xc_mesh).real
        else:
            rhovec_r = out_xr
            rhovec_r[:] = tools.ifft(rhovec_g, self.xc_mesh).real
        rhovec_r[:] *= ratio
        return rhovec_r

    def dense_to_coarse_grid(self, func_xr, out_xR=None):
        # TODO use real FFTs here since the real-space density is real
        ratio = np.prod(self.xc_mesh) / np.prod(self.wf_mesh)
        invr = 1 / ratio
        vxcvec_g = tools.fft(func_xr, self.xc_mesh) * invr
        vxcvec_G = np.asarray(vxcvec_g[:, self._wf2xc], order="C")
        if out_xR is None:
            out_xR = tools.ifft(vxcvec_G, self.wf_mesh).real
        else:
            out_xR[:] = tools.ifft(vxcvec_G, self.wf_mesh).real
        return out_xR

    def get_vj_R(self, C_ks, mocc_ks, mesh=None, Gv=None, save_rho=False):
        # Override get_vj_R to include XC potential
        cell = self.cell
        if mesh is None: mesh = self.wf_mesh
        if Gv is None: Gv = cell.get_Gv(mesh)
        ng = np.prod(mesh)
        dv = self.cell.vol / ng
        xctype = self._numint._xc_type(self.xc)
        rhovec_R = self.get_rho_for_xc(xctype, C_ks, mocc_ks, mesh, Gv)
        if rhovec_R.ndim == 2:
            # non-spin-polarized
            spinfac = 2
            rho_R = rhovec_R[0]
            nkpts = len(C_ks)
        else:
            # spin-polarized
            spinfac = 1
            rho_R = rhovec_R[:, 0].mean(0)
            nkpts = len(C_ks[0])
        if self.kpts_obj is not None:
            nkpts = self.kpts_obj.nkpts
        vj_R = self.with_jk.get_vj_R_from_rho_R(rho_R, mesh=mesh, Gv=Gv)
        rhovec_R[:] *= (spinfac / nkpts) * ng / dv
        if save_rho:
            self._rhovec_R = rhovec_R
        if (self.wf_mesh == self.xc_mesh).all():
            # xc integration is on the same mesh as density generation
            exc, vxcvec_R = self.eval_xc(
                self.xc, rhovec_R, xctype
            )
            if hasattr(self, "_deda_r") and self._deda_r is not None:
                # TODO add this back in
                vxcvec_R[:] += self._deda_r * self._damix_r
        else:
            # xc integration is on a denser mesh than density generation
            # TODO this doesn't work for spin polarization
            """
            ratio = np.prod(self.xc_mesh) / np.prod(self.wf_mesh)
            invr = 1 / ratio
            nrho = rhovec_R.shape[0]
            rhovec_G = tools.fft(rhovec_R, self.wf_mesh)
            dense_size = np.prod(self.xc_mesh)
            rhovec_g = np.zeros((nrho, dense_size), dtype=np.complex128)
            # print(rhovec_g.shape, self._wf2xc.shape, rhovec_G.shape)
            rhovec_g[..., self._wf2xc] = rhovec_G
            rhovec_r = tools.ifft(rhovec_g, self.xc_mesh).real * ratio
            """
            rhovec_r = self.coarse_to_dense_grid(rhovec_R)
            exc, vxcvec_r = self.eval_xc(
                self.xc, rhovec_r, xctype
            )
            if hasattr(self, "_deda_r") and self._deda_r is not None:
                # TODO add this back in
                vxcvec_r[:] += self._deda_r * self._damix_r
            vxcvec_R = np.empty_like(rhovec_R)
            if vxcvec_R.ndim == 2:
                vxcvec_R = vxcvec_R[None, ...]
            for s in range(vxcvec_r.shape[0]):
                self.dense_to_coarse_grid(vxcvec_r[s], vxcvec_R[s])
                #vxcvec_g = tools.fft(vxcvec_r[s], self.xc_mesh) * invr
                #vxcvec_G = np.asarray(vxcvec_g[:, self._wf2xc], order="C")
                #vxcvec_R[s] = tools.ifft(vxcvec_G, self.wf_mesh).real
        vxcdot, vxc_R, vtau_R = vxc_from_vxcvec(
            rhovec_R, vxcvec_R, xctype, mesh, Gv, dv
        )
        vj_R = lib.tag_array(
            vj_R, exc=exc, vxcdot=vxcdot, vxc_R=vxc_R, vtau_R=vtau_R
        )
        return vj_R

    to_gpu = lib.to_gpu


class PWKRKS(PWKohnShamDFT, khf.PWKRHF):
    
    def __init__(self, cell, kpts=np.zeros((1,3)), xc='LDA,VWN',
                 ekincut=None, ecut_xc=None,
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald')):
        khf.PWKRHF.__init__(self, cell, kpts, ekincut=ekincut, ecut_xc=ecut_xc,
                            exxdiv=exxdiv)
        PWKohnShamDFT.__init__(self, xc)

    def dump_flags(self, verbose=None):
        khf.PWKRHF.dump_flags(self)
        PWKohnShamDFT.dump_flags(self, verbose)
        return self

    def to_hf(self):
        out = self._transfer_attrs_(khf.PWKRHF(self.cell, self.kpts))
        # TODO might need to setup up ACE here if xc is not hybrid
        return out

    def get_mo_energy(self, C_ks, mocc_ks, mesh=None, Gv=None, exxdiv=None,
                      vj_R=None, comp=None, ret_mocc=True, full_ham=False):
        if vj_R is None: vj_R = self.get_vj_R(C_ks, mocc_ks)
        res = khf.PWKRHF.get_mo_energy(self, C_ks, mocc_ks, mesh=mesh, Gv=Gv,
                                       exxdiv=exxdiv, vj_R=vj_R, comp=comp,
                                       ret_mocc=ret_mocc, full_ham=full_ham)
        if ret_mocc:
            moe_ks = res[0]
        else:
            moe_ks = res
        moe_ks[0] = lib.tag_array(moe_ks[0], xcdiff=vj_R.exc-vj_R.vxcdot)
        return res

    def energy_elec(self, C_ks, mocc_ks, mesh=None, Gv=None, moe_ks=None,
                    vj_R=None, exxdiv=None):
        e_scf = khf.PWKRHF.energy_elec(self, C_ks, mocc_ks, moe_ks=moe_ks,
                                       mesh=mesh, Gv=Gv, vj_R=vj_R,
                                       exxdiv=exxdiv)
        # When energy is computed from the orbitals, we need to account for
        # the different between \int vxc rho and \int exc rho.
        if moe_ks is not None:
            e_scf += moe_ks[0].xcdiff
        return e_scf

    def update_k(self, C_ks, mocc_ks):
        ni = self._numint
        if ni.libxc.is_hybrid_xc(self.xc):
            super().update_k(C_ks, mocc_ks)
        elif "t-ace" not in self.scf_summary:
            self.scf_summary["t-ace"] = np.zeros(2)


if __name__ == "__main__":
    cell = gto.Cell(
        atom = "C 0 0 0; C 0.89169994 0.89169994 0.89169994",
        a = np.asarray([
                [0.       , 1.78339987, 1.78339987],
                [1.78339987, 0.        , 1.78339987],
                [1.78339987, 1.78339987, 0.        ]]),
        basis="gth-szv",
        ke_cutoff=50,
        pseudo="gth-pade",
        #symmorphic=True,
        #space_group_symmetry=True,
    )
    # cell.mesh = [13, 13, 13]
    # cell.mesh = [29, 29, 29]
    cell.build()
    cell.verbose = 6

    # kmesh = [4, 4, 4]
    kmesh = [2, 2, 2]
    kpts = cell.make_kpts(
        kmesh,
        #time_reversal_symmetry=True,
        #space_group_symmetry=True,
    )

    from pyscf.pbc.pwscf import kpt_symm

    ens = []
    ecuts = [18.38235294, 22.05882353, 25.73529412, 29.41176471, 33.08823529,
             36.76470588, 44.11764706, 55.14705882, 73.52941176, 91.91176471]
    for ecut in ecuts:
        print("\n\n\n")
        print("ECUT", ecut)
        mf = PWKRKS(cell, kpts, xc="PBE", ekincut=ecut)
        # mf = kpt_symm.KsymAdaptedPWKRKS(cell, kpts, xc="PBE", ekincut=ecut)
        
        # nxc = 49
        # mf.set_meshes(xc_mesh=[nxc, nxc, nxc])
        
        mf.damp_type = "simple"
        mf.damp_factor = 0.7
        mf.nvir = 4 # converge first 4 virtual bands
        mf.kernel()
        mf.dump_scf_summary()
        ens.append(mf.e_tot)
    print(ens)

    print(ecuts[:-1])
    print(27.2 * (np.array(ens[:-1]) - ens[-1]))

    assert(abs(mf.e_tot - -10.673452914596) < 1.e-5)

