#!/usr/bin/env python
# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
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
from pyscf.lib import logger
from functools import reduce
import numpy as np
from pyscf.data import nist
from pyscf import lib
from pyscf.grad import mspdft
from pyscf.grad.mspdft import _unpack_state, make_rdm12_heff_offdiag
from pyscf.prop.dip_moment.mcpdft import get_guage_origin, nuclear_dipole

def sipdft_HellmanFeynman_dipole (mc, si_bra=None, si_ket=None,
        state=None, mo_coeff=None, ci=None, si=None, origin='Coord_Center'):
    if state is None: state = mc.state
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    if si is None: si = mc.si
    ket, bra = _unpack_state (state)
    if si_bra is None: si_bra = mc.si[:,bra]
    if si_ket is None: si_ket = mc.si[:,ket]
    si_diag = si_bra * si_ket

    mol     = mc.mol
    ncore   = mc.ncore
    ncas    = mc.ncas
    nelecas = mc.nelecas
    nocc    = ncore + ncas
    mo_core = mo_coeff[:,:ncore]
    mo_cas  = mo_coeff[:,ncore:nocc]

    dm_core = np.dot(mo_core, mo_core.T) * 2

    dm_diag=np.zeros_like(dm_core)
    # Diagonal part
    for i, (amp, c) in enumerate (zip (si_diag, ci)):
        if not amp: continue
        casdm1 = mc.fcisolver.make_rdm1(ci[i], ncas, nelecas)
        dm_cas = reduce(np.dot, (mo_cas, casdm1, mo_cas.T))
        dm_i = dm_cas + dm_core
        dm_diag += amp * dm_i
    # Off-diagonal part
    casdm1, _ = make_rdm12_heff_offdiag (mc, ci, si_bra, si_ket)
    dm_off = reduce(np.dot, (mo_cas, casdm1, mo_cas.T))

    dm = dm_diag + dm_off

    center = get_guage_origin(mol,origin)
    with mol.with_common_orig(center):
        ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    elec_term = -np.tensordot(ao_dip, dm).real
    return elec_term

class ElectricDipole (mspdft.Gradients):

    def kernel (self, state=None, mo=None, ci=None, si=None,
     unit='Debye', origin='Coord_Center', **kwargs):
        ''' Cache the Hamiltonian and effective Hamiltonian terms, and pass
            around the IS hessian

            eris, veff1, veff2, and d2f should be available to all top-level
            functions: get_wfn_response, get_Aop_Adiag, get_ham_repsonse, and
            get_LdotJnuc
        '''
        if state is None: state = self.state
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if si is None: si = self.base.si
        if isinstance (ci, np.ndarray): ci = [ci] # hack hack hack...
        if state is None:
            raise NotImplementedError ('Dipole of PDFT state-average energy')
        self.state = state # Not the best code hygiene maybe
        nroots = self.nroots
        veff1 = []
        veff2 = []
        d2f = self.base.diabatizer (ci=ci)[2]
        for ix in range (nroots):
            v1, v2 = self.base.get_pdft_veff (mo, ci, incl_coul=True,
                paaa_only=True, state=ix)
            veff1.append (v1)
            veff2.append (v2)
        kwargs['veff1'], kwargs['veff2'] = veff1, veff2
        kwargs['d2f'] = d2f

        conv, Lvec, bvec, Aop, Adiag = self.solve_lagrange (**kwargs)

        ham_response = self.get_ham_response (origin=origin, **kwargs)

        LdotJnuc = self.get_LdotJnuc (Lvec, origin=origin, **kwargs)

        mol_dip = ham_response + LdotJnuc

        mol_dip = self.convert_dipole (ham_response, LdotJnuc, mol_dip, unit=unit)
        return mol_dip

    def convert_dipole (self, ham_response, LdotJnuc, mol_dip, unit='Debye'):
        i = self.state
        if unit.upper() == 'DEBYE':
            for x in [ham_response, LdotJnuc, mol_dip]: x *= nist.AU2DEBYE
        log = lib.logger.new_logger(self, self.verbose)
        log.note('CMS-PDFT PDM <{}|mu|{}>          {:>10} {:>10} {:>10}'.format(i,i,'X','Y','Z'))
        log.note('Hamiltonian Contribution (%s) : %9.5f, %9.5f, %9.5f', unit, *ham_response)
        log.note('Lagrange Contribution    (%s) : %9.5f, %9.5f, %9.5f', unit, *LdotJnuc)
        log.note('Permanent Dipole Moment  (%s) : %9.5f, %9.5f, %9.5f', unit, *mol_dip)
        return mol_dip

    def get_ham_response (self, si_bra=None, si_ket=None, state=None, mo=None,
                    ci=None, si=None, origin='Coord_Center', **kwargs):
        if state is None: state = self.state
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if si is None: si = self.base.si
        ket, bra = _unpack_state (state)
        if si_bra is None: si_bra = si[:,bra]
        if si_ket is None: si_ket = si[:,ket]
        fcasscf = self.make_fcasscf (state)
        fcasscf.mo_coeff = mo
        fcasscf.ci = ci
        elec_term = sipdft_HellmanFeynman_dipole (fcasscf, si_bra=si_bra, si_ket=si_ket,
         state=state, mo_coeff=mo, ci=ci, si=si, origin=origin)
        nucl_term = nuclear_dipole(fcasscf, origin=origin)
        total = nucl_term + elec_term
        return total

    def get_LdotJnuc (self, Lvec, atmlst=None, verbose=None, mo=None,
            ci=None, origin='Coord_Center', **kwargs):
        if atmlst is None: atmlst = self.atmlst
        if verbose is None: verbose = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci

        mc = self.base

        ngorb, nci = self.ngorb, self.nci
        Lvec_v = Lvec[:ngorb+nci]
        Lorb, Lci = self.unpack_uniq_var (Lvec_v)

        mol   = mc.mol
        ncore = mc.ncore
        ncas  = mc.ncas
        nocc  = ncore + ncas
        nelecas = mc.nelecas

        mo_core = mo[:,:ncore]
        mo_cas = mo[:,ncore:nocc]

        # Orbital part
        # MO coeff contracted against Lagrange multipliers
        moL_coeff = np.dot (mo, Lorb)
        moL_core = moL_coeff[:,:ncore]
        moL_cas = moL_coeff[:,ncore:nocc]

        casdm1 = mc.fcisolver.make_rdm1(ci, ncas, nelecas)

        dmL_core = np.dot(moL_core, mo_core.T) * 2
        dmL_cas = reduce(np.dot, (moL_cas, casdm1, mo_cas.T))
        dmL_core += dmL_core.T
        dmL_cas += dmL_cas.T

        # CI part
        casdm1_transit, _ = mc.fcisolver.trans_rdm12 (Lci, ci, ncas, nelecas)
        casdm1_transit += casdm1_transit.transpose (1,0)

        dm_cas_transit = reduce(np.dot, (mo_cas, casdm1_transit, mo_cas.T))

        # Expansion coefficients are already in Lagrange multipliers
        dm = dmL_core + dmL_cas + dm_cas_transit

        center = get_guage_origin(mol,origin)
        with mol.with_common_orig(center):
            ao_dip = mol.intor_symmetric('int1e_r', comp=3)
        mol_dip_L = -np.tensordot(ao_dip, dm).real

        return mol_dip_L
