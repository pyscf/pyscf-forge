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
# Author: Helen Clifford <helenclifford@uchicago.edu>

from pyscf.lib import logger
from functools import reduce
from scipy import linalg
import numpy as np
from pyscf.data import nist
from pyscf import lib
from pyscf.grad import lpdft as lpdft_grad
from pyscf.prop.dip_moment.mcpdft import mcpdft_HellmanFeynman_dipole, get_guage_origin, nuclear_dipole
from pyscf.mcscf import mc1step
from pyscf.mcscf.df import _DFCASSCF
from pyscf.mcpdft.lpdft import _LPDFT

class ElectricDipole (lpdft_grad.Gradients):

    def kernel (self, unit='Debye', origin='Coord_Center', **kwargs):
        state = kwargs['state'] if 'state' in kwargs else self.state
        if state is None:
            raise NotImplementedError ('Dipole of PDFT state-average energy')
        self.state = state
        mo = kwargs['mo'] if 'mo' in kwargs else self.base.mo_coeff
        ci = kwargs['ci'] if 'ci' in kwargs else self.base.ci
        if isinstance (ci, np.ndarray): ci = [ci]
        kwargs['ci'] = ci
        if ("feff1" not in kwargs) or ("feff2" not in kwargs):
            kwargs["feff1"], kwargs["feff2"] = self.get_otp_gradient_response(
                mo, ci, state
            )

        conv, Lvec, bvec, Aop, Adiag = self.solve_lagrange (**kwargs)
        self.debug_lagrange (Lvec, bvec, Aop, Adiag, **kwargs)

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
        log.note('Permanent Dipole Moment  (%s) : %9.5f, %9.5f, %9.5f', unit, *ham_response)

        log.note('L-PDFT PDM <{}|mu|{}>           {:>10} {:>10} {:>10}'.format(i,i,'X','Y','Z'))
        log.note('Hamiltonian Contribution (%s) : %9.5f, %9.5f, %9.5f', unit, *ham_response)
        log.note('Lagrange Contribution    (%s) : %9.5f, %9.5f, %9.5f', unit, *LdotJnuc)
        log.note('Permanent Dipole Moment  (%s) : %9.5f, %9.5f, %9.5f', unit, *mol_dip)
        return mol_dip

    def get_ham_response(self, state=None, verbose=None, mo=None,
            ci=None, origin='Coord_Center', **kwargs):
        if state is None: state   = self.state
        if verbose is None: verbose = self.verbose
        if mo is None: mo      = self.base.mo_coeff
        if ci is None: ci      = self.base.ci

        fcasscf = self.make_fcasscf (state)
        fcasscf.mo_coeff = mo
        fcasscf.ci = ci[state]

        elec_term = mcpdft_HellmanFeynman_dipole (fcasscf, mo_coeff=mo, ci=ci[state], origin=origin)
        nucl_term = nuclear_dipole(fcasscf, origin=origin)
        total = nucl_term + elec_term
        return total

    def get_LdotJnuc (self, Lvec, state=None, verbose=None,
            mo=None, ci=None, origin='Coord_Center', **kwargs):
        if state is None: state   = self.state
        if verbose is None: verbose = self.verbose
        if mo is None: mo      = self.base.mo_coeff
        if ci is None: ci      = self.base.ci[state]
        mc = self.base

        Lorb, Lci = self.unpack_uniq_var (Lvec)

        mol = mc.mol
        ncore = mc.ncore
        ncas = mc.ncas
        nocc = ncore + ncas
        nelecas = mc.nelecas

        mo_core = mo[:,:ncore]
        mo_cas  = mo[:,ncore:nocc]

        moL_coeff = np.dot (mo, Lorb)
        moL_core  = moL_coeff[:,:ncore]
        moL_cas   = moL_coeff[:,ncore:nocc]

        casdm1 = mc.fcisolver.make_rdm1(ci, ncas, nelecas)

        dmL_core = np.dot(moL_core, mo_core.T) * 2
        dmL_cas  = reduce(np.dot, (moL_cas, casdm1, mo_cas.T))
        dmL_core += dmL_core.T
        dmL_cas  += dmL_cas.T

        casdm1_transit, _ = mc.fcisolver.trans_rdm12 (Lci, ci, ncas, nelecas)
        casdm1_transit += casdm1_transit.transpose (1,0)

        dm_cas_transit = reduce(np.dot, (mo_cas, casdm1_transit, mo_cas.T))

        dm = dmL_core + dmL_cas + dm_cas_transit

        center = get_guage_origin(mol, origin)
        with mol.with_common_orig(center):
            ao_dip = mol.intor_symmetric('int1e_r', comp=3)
        mol_dip_L = -np.tensordot(ao_dip, dm).real

        return mol_dip_L



class _LPDFTDipole(_LPDFT):
    def dip_moment (self, unit='Debye', origin='Coord_Center', state=None):
        if not isinstance (self, mc1step.CASSCF):
            raise NotImplementedError ("CASCI-based PDFT dipole moments")
        elif getattr (self, 'frozen', None) is not None:
            raise NotImplementedError ("PDFT dipole moments with frozen orbitals")
        elif isinstance (self, _DFCASSCF):
            raise NotImplementedError ("PDFT dipole moments with density-fitting ERIs")
        if not lib.isinteger (state):
            raise RuntimeError ('Permanent dipole requires a single state')
        dip_obj =  ElectricDipole(self)
        mol_dipole = dip_obj.kernel (state=state, unit=unit, origin=origin)
        return mol_dipole
_LPDFT.dip_moment = _LPDFTDipole.dip_moment
