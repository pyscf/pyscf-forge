from pyscf.lib import logger
import numpy as np
from pyscf.data import nist
from pyscf import lib
from pyscf.prop.dip_moment import mspdft
from pyscf.prop.dip_moment.mspdft import sipdft_HellmanFeynman_dipole, get_guage_origin
from pyscf.grad.mspdft import mspdft_heff_response
from pyscf.grad.mspdft import _unpack_state

class TransitionDipole (mspdft.ElectricDipole):

    def convert_dipole (self, ham_response, LdotJnuc, mol_dip, unit='Debye'):
        val = np.linalg.norm(mol_dip)
        i   = self.state[0]
        j   = self.state[1]
        dif = abs(self.e_states[i]-self.e_states[j])
        osc = 2/3*dif*val**2
        if unit.upper() == 'DEBYE':
            for x in [ham_response, LdotJnuc, mol_dip]: x *= nist.AU2DEBYE
        log = lib.logger.new_logger(self, self.verbose)
        log.note('CMS-PDFT TDM <{}|mu|{}>          {:>10} {:>10} {:>10}'.format(i,j,'X','Y','Z'))
        log.note('Hamiltonian Contribution (%s) : %9.5f, %9.5f, %9.5f', unit, *ham_response)
        log.note('Lagrange Contribution    (%s) : %9.5f, %9.5f, %9.5f', unit, *LdotJnuc)
        log.note('Transition Dipole Moment (%s) : %9.5f, %9.5f, %9.5f', unit, *mol_dip)
        log.note('Oscillator strength  : %9.5f', osc)
        return mol_dip

    def get_ham_response (self,  si_bra=None, si_ket=None, state=None, verbose=None, mo=None,
                    ci=None, si=None, **kwargs):
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
         state=state, mo_coeff=mo, ci=ci, si=si)
        return elec_term
