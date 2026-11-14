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

import copy
from functools import reduce

import numpy as np

from pyscf import lib
from pyscf.lib import logger
from pyscf.data import nist
from pyscf.fci import direct_spin1
from pyscf.mcscf import mc1step, newton_casscf
from pyscf.mcscf.df import _DFCASSCF
from pyscf.mcpdft import _dms
from pyscf.mcpdft.lpdft import _LPDFT
from pyscf.grad import sacasscf as sacasscf_grad
from pyscf.prop.dip_moment import lpdft as lpdft_dip
from pyscf.prop.dip_moment.mcpdft import get_guage_origin


def _unpack_state(state):
    if hasattr(state, "__len__"):
        return state[0], state[1]
    return state, state


def _symm_trans_casdm12(mc, ci, bra, ket):
    ncas, nelecas = mc.ncas, mc.nelecas
    dm1a, dm1b = direct_spin1.trans_rdm1s(ci[bra], ci[ket], ncas, nelecas)

    dm1a = 0.5 * (dm1a + dm1a.conj().T)
    dm1b = 0.5 * (dm1b + dm1b.conj().T)
    casdm1s = np.stack((dm1a, dm1b), axis=0)

    _, dm2 = direct_spin1.trans_rdm12(ci[bra], ci[ket], ncas, nelecas)
    casdm2 = 0.5 * (dm2 + dm2.transpose(1, 0, 3, 2))
    return casdm1s, casdm2


def lpdft_trans_HellmanFeynman_dipole(mc, mo_coeff, ci, bra, ket, origin="Coord_Center"):
    ncore, ncas = mc.ncore, mc.ncas
    nocc = ncore + ncas
    mo_cas = mo_coeff[:, ncore:nocc]

    casdm1s, _ = _symm_trans_casdm12(mc, ci, bra, ket)
    casdm1 = casdm1s[0] + casdm1s[1]
    dm_trans = reduce(np.dot, (mo_cas, casdm1, mo_cas.conj().T))

    center = get_guage_origin(mc.mol, origin)
    with mc.mol.with_common_orig(center):
        ao_dip = mc.mol.intor_symmetric("int1e_r", comp=3)
    return -np.tensordot(ao_dip, dm_trans).real


class TransitionDipole(lpdft_dip.ElectricDipole):
    # Hellmann-Feynman term
    def get_ham_response(self, state=None, verbose=None, mo=None, ci=None,
                         origin="Coord_Center", **kwargs):
        if state is None:
            state = self.state
        if mo is None:
            mo = self.base.mo_coeff
        if ci is None:
            ci = self.base.ci
        ket, bra = _unpack_state(state)
        return lpdft_trans_HellmanFeynman_dipole(self.base, mo, ci, bra, ket, origin=origin)

    def get_otp_gradient_response(self, mo=None, ci=None, state=None):
        if mo is None:
            mo = self.base.mo_coeff
        if ci is None:
            ci = self.base.ci
        if state is None:
            state = self.state
        ket, bra = _unpack_state(state)
        mc = self.base

        casdm1s_0, casdm2_0 = mc.get_casdm12_0()

        t_casdm1s, t_casdm2 = _symm_trans_casdm12(mc, ci, bra, ket)
        t_cascm2 = _dms.dm2_cumulant(t_casdm2, t_casdm1s)

        zero_casdm1s = np.zeros_like(t_casdm1s)
        zero_cascm2 = np.zeros_like(t_cascm2)

        dm1s_gt = _dms.casdm1s_to_dm1s(mc, t_casdm1s, mo_coeff=mo)  # core and transitionn density
        dm1s_c = _dms.casdm1s_to_dm1s(mc, zero_casdm1s, mo_coeff=mo)  # core only density

        common = dict(mo=mo, ci=ci, casdm1s=casdm1s_0, casdm2=casdm2_0,
                      jk_pc=True, paaa_only=True, incl_coul=True, delta=False)

        feff1_gt, feff2_gt = mc.get_pdft_feff(c_dm1s=dm1s_gt, c_cascm2=t_cascm2, **common)
        feff1_c, feff2_c = mc.get_pdft_feff(c_dm1s=dm1s_c, c_cascm2=zero_cascm2, **common)

        feff2 = copy.copy(feff2_gt)
        for attr in ("vhf_c", "papa", "ppaa", "j_pc", "k_pc"):
            va = getattr(feff2_gt, attr, None)
            vb = getattr(feff2_c, attr, None)
            if isinstance(va, np.ndarray) and isinstance(vb, np.ndarray):
                try:
                    setattr(feff2, attr, va - vb)
                except AttributeError:
                    pass

        return feff1_gt - feff1_c, feff2


    def make_fcasscf_lpdft_trans(self, state=None, casscf_attr=None,
                                 fcisolver_attr=None):
        if state is None:
            state = self.state
        if casscf_attr is None:
            casscf_attr = {}
        if fcisolver_attr is None:
            fcisolver_attr = {}
        ket, bra = _unpack_state(state)

        casdm1s, casdm2 = _symm_trans_casdm12(self.base, self.base.ci, bra, ket)
        casdm1 = casdm1s[0] + casdm1s[1]
        fcisolver_attr["make_rdm12"] = lambda *a, **k: (casdm1, casdm2)
        fcisolver_attr["make_rdm1"] = lambda *a, **k: casdm1
        fcisolver_attr["make_rdm2"] = lambda *a, **k: casdm2
        return sacasscf_grad.Gradients.make_fcasscf(self, state=ket, casscf_attr=casscf_attr,
                                                    fcisolver_attr=fcisolver_attr)

    def _lpdft_h2eff(self, mo=None, ci=None):   #effective active space LPDFT ham
        mc = self.base
        if mo is None:
            mo = mc.mo_coeff
        if ci is None:
            ci = mc.ci
        ot = mc.otfnal
        spin = abs(mc.nelecas[0] - mc.nelecas[1])
        _, _, hyb = ot._numint.rsh_and_hybrid_coeff(ot.otxc, spin=spin)
        cas_hyb = hyb[0]

        casdm1s_0, casdm2_0 = mc.get_casdm12_0(ci=ci)
        h1, _ = mc.get_h1lpdft(0.0, casdm1s_0, casdm2_0, hyb=1.0 - cas_hyb,
                               mo_coeff=mo)
        h2 = mc.get_h2lpdft()
        return direct_spin1.absorb_h1e(h1, h2, mc.ncas, mc.nelecas, 0.5)

    def get_wfn_response(self, state=None, verbose=None, mo=None, ci=None, feff1=None, feff2=None, **kwargs):
        if state is None:
            state = self.state
        if verbose is None:
            verbose = self.verbose
        if mo is None:
            mo = self.base.mo_coeff
        if ci is None:
            ci = self.base.ci
        if (feff1 is None) or (feff2 is None):
            feff1, feff2 = self.get_otp_gradient_response(mo, ci, state)

        log = logger.new_logger(self, verbose)
        ket, bra = _unpack_state(state)
        ncore = self.base.ncore

        fcasscf = self.make_fcasscf_lpdft_trans(state)
        fcasscf.mo_coeff = mo
        fcasscf.ci = ci[ket]
        fcasscf.get_hcore = self.base.get_lpdft_hcore

        moH = mo.conj().T
        vnocore = self.base.veff2.vhf_c.copy()
        vnocore[:, :ncore] = -moH @ fcasscf.get_hcore() @ mo[:, :ncore]
        with lib.temporary_env(self.base.veff2, vhf_c=vnocore):
            g_explicit = newton_casscf.gen_g_hop(
                fcasscf, mo, ci[ket], self.base.veff2, verbose)[0]

        gorb_explicit = g_explicit[: self.ngorb]

        # explicit CI response
        h2eff = self._lpdft_h2eff(mo=mo, ci=ci)
        hc_ket = direct_spin1.contract_2e(h2eff, ci[ket], self.base.ncas, self.base.nelecas)
        hc_bra = direct_spin1.contract_2e(h2eff, ci[bra], self.base.ncas, self.base.nelecas)

        # implicit response
        fcasscf_sa = self.make_fcasscf_sa()
        fcasscf_sa.get_hcore = lambda: feff1
        g_implicit = newton_casscf.gen_g_hop(fcasscf_sa, mo, ci, feff2, verbose)[0]

        gmo_implicit, gci_implicit = self.unpack_uniq_var(g_implicit)

        g_all = self.pack_uniq_var(gmo_implicit, gci_implicit)
        g_all[: self.ngorb] += gorb_explicit

        gorb, gci = self.unpack_uniq_var(g_all)
        gci[bra] = np.asarray(gci[bra]).reshape(hc_ket.shape) + hc_ket
        gci[ket] = np.asarray(gci[ket]).reshape(hc_bra.shape) + hc_bra

        spin_states = np.asarray(self.spin_states)
        for root in range(self.nroots):
            idx = np.where(spin_states == spin_states[root])[0]
            gci_root = np.asarray(gci[root]).ravel()
            assert root in idx
            ci_proj = np.asarray([ci[i].ravel() for i in idx])
            gci_root -= np.dot(np.dot(ci_proj.conj(), gci_root), ci_proj)
            gci[root] = gci_root.reshape(np.asarray(gci[root]).shape)

        g_all = self.pack_uniq_var(gorb, gci)

        log.debug("L-PDFT TDM g_all orb:\n{}".format(g_all[: self.ngorb]))
        log.debug("L-PDFT TDM g_all ci:\n{}".format(g_all[self.ngorb:]))
        return g_all

    def get_LdotJnuc(self, Lvec, state=None, verbose=None, mo=None, ci=None,
                     origin="Coord_Center", **kwargs):
        mc = self.base
        if mo is None:
            mo = mc.mo_coeff
        if ci is None:
            ci = mc.ci
        ncore, ncas, nelecas = mc.ncore, mc.ncas, mc.nelecas
        nocc = ncore + ncas

        Lorb, Lci = self.unpack_uniq_var(Lvec)  #orbital part

        mo_core = mo[:, :ncore]
        mo_cas = mo[:, ncore:nocc]
        moL = np.dot(mo, Lorb)
        moL_core = moL[:, :ncore]
        moL_cas = moL[:, ncore:nocc]

        casdm1s_0 = mc.get_casdm12_0(ci=ci)[0]
        casdm1_0 = casdm1s_0[0] + casdm1s_0[1]

        dmL_core = np.dot(moL_core, mo_core.conj().T) * 2
        dmL_cas = reduce(np.dot, (moL_cas, casdm1_0, mo_cas.conj().T))
        dmL_core += dmL_core.T
        dmL_cas += dmL_cas.T

        casdm1_transit = np.zeros((ncas, ncas))
        for w, Lc, c in zip(mc.weights, Lci, ci):
            d = direct_spin1.trans_rdm1(np.asarray(Lc).reshape(np.asarray(c).shape),   #ci part
                                        c, ncas, nelecas)
            casdm1_transit += w * (d + d.conj().T)
        dm_cas_transit = reduce(np.dot, (mo_cas, casdm1_transit,
                                         mo_cas.conj().T))

        dm = dmL_core + dmL_cas + dm_cas_transit
        center = get_guage_origin(mc.mol, origin)
        with mc.mol.with_common_orig(center):
            ao_dip = mc.mol.intor_symmetric("int1e_r", comp=3)
        return -np.tensordot(ao_dip, dm).real

    def convert_dipole(self, ham_response, LdotJnuc, mol_dip, unit="Debye"):
        ket, bra = _unpack_state(self.state)
        val = np.linalg.norm(mol_dip)
        dif = abs(self.e_states[bra] - self.e_states[ket])
        osc = 2.0 / 3.0 * dif * val**2
        if unit.upper() == "DEBYE":
            for x in [ham_response, LdotJnuc, mol_dip]:
                x *= nist.AU2DEBYE
        log = lib.logger.new_logger(self, self.verbose)
        log.note("L-PDFT TDM <{}|mu|{}>            {:>10} {:>10} {:>10}".format(
            bra, ket, "X", "Y", "Z"))
        log.note("Hamiltonian Contribution (%s) : %9.5f, %9.5f, %9.5f",
                 unit, *ham_response)
        log.note("Lagrange Contribution    (%s) : %9.5f, %9.5f, %9.5f",
                 unit, *LdotJnuc)
        log.note("Transition Dipole Moment (%s) : %9.5f, %9.5f, %9.5f",
                 unit, *mol_dip)
        log.note("Oscillator strength  : %9.5f", osc)
        return mol_dip


class _LPDFTTransDipole(_LPDFT):
    def trans_moment(self, unit="Debye", origin="Coord_Center", state=None):
        if not isinstance(self, mc1step.CASSCF):
            raise NotImplementedError("CASCI-based L-PDFT transition dipoles")
        elif getattr(self, "frozen", None) is not None:
            raise NotImplementedError("L-PDFT transition dipoles with frozen orbitals")
        elif isinstance(self, _DFCASSCF):
            raise NotImplementedError("L-PDFT transition dipoles with density-fitting ERIs")
        if not hasattr(state, "__len__") or len(state) != 2:
            raise RuntimeError("Transition dipole requires two states")
        if state[0] == state[1]:
            raise RuntimeError("Transition dipole requires two different states; "
                               "use dip_moment() for a permanent dipole")
        return TransitionDipole(self).kernel(state=state, unit=unit, origin=origin)


_LPDFT.trans_moment = _LPDFTTransDipole.trans_moment
