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
import pyscf
import numpy as np

import time
from scipy import linalg
from pyscf import gto, dft, ao2mo, fci, mcscf, lib
from pyscf.lib import logger
from pyscf.mcscf import mc_ao2mo
from pyscf.mcscf.addons import StateAverageMCSCFSolver, state_average_mix, state_average_mix_
from pyscf.mcdcft.convfnal import convfnal

def kernel(mc, dcxc, natorb, occ, root=-1):
    ''' Calculate MC-DCFT total energy

        Args:
            mc : an instance of CASSCF or CASCI class
                Note: this function does not currently run the CASSCF or CASCI calculation itself
                prior to calculating the MC-DCFT energy. Call mc.kernel () before passing to this function!
            dcxc : an instance of on-top density functional class - see otfnal.py

        Kwargs:
            root : int
                If mc describes a state-averaged calculation, select the root (0-indexed)
                Negative number requests state-averaged MC-DCFT results (i.e., using state-averaged density matrices)

        Returns:
            Total MC-DCFT energy including nuclear repulsion energy.
    '''
    t0 = (logger.process_clock(), logger.perf_counter())

    mc_1root = mc
    if isinstance(mc, StateAverageMCSCFSolver) and root >= 0:
        mc_1root = mcscf.CASCI(mc._scf, mc.ncas, mc.nelecas)
        mc_1root.fcisolver = fci.solver(mc._scf.mol, singlet = False, symm = False)
        mc_1root.mo_coeff = mc.mo_coeff
        mc_1root.ci = mc.ci[root]
        mc_1root.e_tot = mc.e_states[root]
    dm1s = np.asarray(mc_1root.make_rdm1s())
    spin = abs(mc.nelecas[0] - mc.nelecas[1])
    t0 = logger.timer(dcxc, 'rdms', *t0)

    Vnn = mc._scf.energy_nuc()
    h = mc._scf.get_hcore()
    dm1 = dm1s[0] + dm1s[1]
    vj, vk = mc._scf.get_jk(dm=dm1s)
    vj = vj[0] + vj[1]
    Te_Vne = np.tensordot(h, dm1)
    # (vj_a + vj_b) * (dm_a + dm_b)
    E_j = np.tensordot(vj, dm1) * 0.5
    # (vk_a * dm_a) + (vk_b * dm_b) Mind the difference!
    E_x = -(np.tensordot(vk[0], dm1s[0]) + np.tensordot(vk[1], dm1s[1])) * 0.5
    t0 = logger.timer(dcxc, 'Vnn, Te, Vne, E_j, E_x', *t0)
    e_tot, E_dc = dcft_energy(dcxc, Vnn, Te_Vne, E_j, E_x, dm1s, natorb, occ)

    chkdata = {'Vnn':Vnn, 'Te_Vne':Te_Vne, 'E_j':E_j, 'E_x':E_x, 'dm1s':dm1s, 'spin':spin}

    return e_tot, E_dc, chkdata

def dcft_energy(dcxc, Vnn, Te_Vne, E_j, E_x, dm1s, natorb, occ):
    t0 = (logger.process_clock(), logger.perf_counter())
    logger.debug(dcxc, 'CAS energy decomposition:')
    logger.debug(dcxc, 'Vnn = %s', Vnn)
    logger.debug(dcxc, 'Te + Vne = %s', Te_Vne)
    logger.debug(dcxc, 'E_j = %s', E_j)
    logger.debug(dcxc, 'E_x = %s', E_x)

    E_dc = get_E_dc(dcxc, dm1s, natorb, occ)
    t0 = logger.timer(dcxc, 'E_dc', *t0)
    e_tot = Vnn + Te_Vne + E_j + (dcxc.hyb_x * E_x) + E_dc
    logger.note(dcxc, 'MC-DCFT E = %s, Edc(%s) = %s', e_tot, dcxc.dcxc, E_dc)
    return e_tot, E_dc

def _recalculate_with_xc(dcxc, chkdata):
    ''' Recalculate MC-DCFT total energy based on intermediate quantities from a previous MC-DCFT calculation

        Args:
            dcxc : str or an instance of on-top density functional class - see otfnal.py
            chkdata : chkdata dict generated by previous calculation

        Returns:
            Total MC-DCFT energy including nuclear repulsion energy.
    '''
    Vnn = chkdata['Vnn']
    Te_Vne = chkdata['Te_Vne']
    E_j = chkdata['E_j']
    E_x = chkdata['E_x']
    dm1s = chkdata['dm1s']
    natorb = chkdata['natorb']
    occ = chkdata['occ']

    logger.debug(dcxc, 'CAS energy restored from chkfile.')

    return dcft_energy(dcxc, Vnn, Te_Vne, E_j, E_x, dm1s, natorb, occ)


def get_E_dc(dcxc, oneCDMs, natorb, occ, max_memory=20000, hermi=1):
    ''' E_MCDCFT = h_pq l_pq + 1/2 v_pqrs l_pq l_rs + E_dc[D]
        or, in other terms,
        E_MCDCFT = T_KS[rho] + E_ext[rho] + E_coul[rho] + E_ot[D]
                 = E_DFT[1rdm] - E_xc[rho] + E_ot[D]
        Args:
            dcxc : an instance of dcfnal class
            oneCDMs : ndarray of shape (2, nao, nao)
                containing spin-separated one-body density matrices

        Kwargs:
            max_memory : int or float
                maximum cache size in MB
                default is 20000
            hermi : int
                1 if 1CDMs are assumed hermitian, 0 otherwise

        Returns : float
            The MC-DCFT on-top exchange-correlation energy

    '''
    ni, xctype, dens_deriv = dcxc._numint, dcxc.xctype, dcxc.dens_deriv
    norbs_ao = oneCDMs.shape[1]

    E_dc = 0.0
    dcxc.ms = 0.0

    t0 = (logger.process_clock(), logger.perf_counter())
    for ao, mask, weight, coords in ni.block_loop(dcxc.mol, dcxc.grids, norbs_ao, dens_deriv, max_memory):
        E_dc += dcxc.get_E_dc(natorb, occ, ao, weight)
        t0 = logger.timer(dcxc, 'on-top exchange-correlation energy calculation', *t0)

    return E_dc

class DCFT_base:
    def __init__(self, mol, dc, hyb_x=0., display_name=None, grids_level=None):
        self.dcfnal = convfnal(mol, dc, hyb_x, display_name, grids_level)
        self.grids_level = grids_level

    def load_mcdcft_chk(self, chkfile):
        self.chkdata = lib.chkfile.load(chkfile, 'mcdcft')

    def recalculate_with_xc(self, dc, display_name=None, hyb_x=0., chkdata=None,
                            load_chk=None, dump_chk=None, grids_level=None, **kwargs):
        ''' Recalculate MC-DCFT total energy based on intermediate quantities from a previous MC-DCFT calculation

            Args:
                ot : str of on-top density matrix functional class. It should
                     follow the convention in the ks module, i.e. do NOT add any prefix
                ot_name : display name of the functional. If None, use ot as the display name
                chkdata : chkdata dict generated by previous calculation
                load_chk : str of chk filename to load chkdata from before the calculation
                dump_chk : str of chk filename to dump newly calculated energies
                grids_level : grids

            Returns:
                Total MC-DCFT energy including nuclear repulsion energy and E_ot
        '''
        if grids_level is None:
            grids_level = self.grids_level
        if load_chk is not None:
            self.load_mcdcft_chk(load_chk)
        if chkdata is None:
            chkdata = self.chkdata
        if isinstance(dc, str):
            self.dcfnal.display_name = 'c' + dc if display_name is None else display_name
            self.dcfnal = convfnal(self.mol, dc, hyb_x, display_name, grids_level, self.verbose)
        else:
            self.dcfnal = dcxc
        n_states = chkdata['n_states']
        if n_states > 1:
            epdft = [_recalculate_with_xc(self.dcfnal, ichkdata) for ichkdata in chkdata]
            self.e_states, self.e_dc = zip(*epdft)
            weights = chkdata['weights']
            self.e_tot = np.dot(self.e_states, weights)
        else:
            self.e_tot, self.e_dc = _recalculate_with_xc(self.dcfnal, chkdata)
        if dump_chk is not None:
            lib.chkfile.dump(dump_chk, 'mcdcft/e_tot/' + self.dcfnal.display_name, self.e_tot)
            lib.chkfile.dump(dump_chk, 'mcdcft/e_dc/' + self.dcfnal.display_name, self.e_dc)
        return self.e_tot, self.e_dc

    def kernel(self, mo_coeff=None, ci=None, skip_scf=False, **kwargs):
        if not skip_scf:
            self.e_mcscf, self.e_cas, self.ci, self.mo_coeff, self.mo_energy = super().kernel(mo_coeff,
                                                                                              ci, **kwargs)
            natorb, _, occ = self.cas_natorb_(sort=False)

        # TODO: State average has not been tested !!!
        if isinstance(self, StateAverageMCSCFSolver):
            epdft = [kernel(self, self.dcfnal, natorb, occ, root=ix) for ix in range(len(self.e_states))]
            self.e_mcscf = self.e_states
            #  self.fcisolver.e_states = [e_tot for e_tot, e_dc in epdft]
            #  self.e_dc = [e_dc for e_tot, e_dc in epdft]
            self.fcisolver.e_states, self.e_dc, self.chkdata = zip(*epdft)
            self.chkdata['n_states'] = len(epdft)
            self.chkdata['weights'] = self.weights
            self.e_tot = np.dot(self.e_states, self.weights)
        else:
            self.e_tot, self.e_dc, self.chkdata = kernel(self, self.dcfnal, natorb, occ)
            self.chkdata['n_states'] = 1
        self.chkdata['e_tot'] = {self.dcfnal.display_name: self.e_tot}
        self.chkdata['natorb'] = natorb
        self.chkdata['occ'] = occ
        return self.e_tot, self.e_dc, self.e_mcscf, self.e_cas, self.ci, self.mo_coeff, self.mo_energy

    def dump_mcdcft_chk(self, chkfile, key='mcdcft', chkdata=None):
        '''Save MC-DCFT calculation results in chkfile.
        '''
        if chkdata is None:
            chkdata = self.chkdata
        lib.chkfile.dump(chkfile, key, chkdata)
        lib.chkfile.dump_mol(self.mol, chkfile)

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose=verbose)
        log = logger.new_logger(self, verbose)
        log.info('density coherence exchange-correlation functional: %s', self.dcfnal.dcxc)

    # TODO: gradient has not been implemented
    def nuc_grad_method(self):
        raise NotImplementedError("MC-DCFT nuclear gradients")

    @property
    def dcxc(self):
        return self.dcfnal.dcxc

def get_mcdcft_child_class(mc, dc, ncas, nelecas, **kwargs):
    class CASDCFT(DCFT_base, mc.__class__):
        def __init__(self, mol, dc, ncas, nelecas, **kwargs):
            try:
                mc.__class__.__init__(self, mol, ncas, nelecas)
            except TypeError:
                mc.__class__.__init__(self)
            DCFT_base.__init__(self, mol, dc, **kwargs)
            keys = set(('e_dc', 'e_mcscf', 'e_states'))
            self._keys = set(self.__dict__.keys()).union(keys)

    dcft = CASDCFT(mc.mol, dc, ncas, nelecas, **kwargs)
    dcft.__dict__.update(mc.__dict__)
    return dcft

def CASSCFDCFT(mf_or_mol, dc, ncas, nelecas, display_name=None, grids_level=None, hyb_x=0., chkfile=None, ncore=None, frozen=None, **kwargs):
    mc = mcscf.CASSCF(mf_or_mol, ncas, nelecas, ncore=ncore, frozen=frozen)
    if chkfile is not None:
        mc.__dict__.update(lib.chkfile.load(chkfile, 'mcscf'))
    mc = get_mcdcft_child_class(mc, dc, ncas, nelecas, display_name=display_name, grids_level=grids_level, hyb_x=hyb_x, **kwargs)
    mc.chkfile = chkfile
    return mc

CASSCF = CASSCFDCFT
