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

def get_unpaired_density(natorb, occ, ao):
    r''' Calculate unpaired density D

        Args:
            natorb : ndarray of shape (nao, nao)
                generated by natorb TRANSPOSED
            occ : ndarray with shape (nao,)
                occupation numbers of natorb
            ao : ndarray of shape (ngrids, nao) for LDA or (4, ngrids, nao) for GGA
                magnitude of atomic basis function [and gradients]

        Returns : ndarray with shape (ngrids,) for LDA or (4, ngrids) for GGA
            unpaired density of each grid
    '''
    if ao.ndim == 3:  # GGA
        ao_magnitude = ao[0]
        ao_magnitude_grad = ao[1:4]
    else:  # LDA
        ao_magnitude = ao

    # Magnitude of natural orbital and gradient
    natorb_magnitude = natorb @ ao_magnitude.T

    c = (occ * (2 - occ))
    D = c @ (natorb_magnitude ** 2)

    if ao.ndim == 3:  # GGA
        natorb_magnitude_grad = natorb @ ao_magnitude_grad.transpose((0,2,1))
        D_grad = (2 * c) @ (natorb_magnitude * natorb_magnitude_grad)
        return np.vstack((D, D_grad))
    else:  # LDA
        return D


def kernel(mc, ot, root=-1):
    ''' Calculate MC-DCFT total energy

        Args:
            mc : an instance of CASSCF or CASCI class
                Note: this function does not currently run the CASSCF or CASCI calculation itself
                prior to calculating the MC-DCFT energy. Call mc.kernel () before passing to this function!
            ot : an instance of on-top density functional class - see otfnal.py

        Kwargs:
            root : int
                If mc describes a state-averaged calculation, select the root (0-indexed)
                Negative number requests state-averaged MC-DCFT results (i.e., using state-averaged density matrices)

        Returns:
            Total MC-DCFT energy including nuclear repulsion energy.
    '''
    t0 = (logger.process_clock (), logger.perf_counter ())

    mc_1root = mc
    if isinstance (mc, StateAverageMCSCFSolver) and root >= 0:
        mc_1root = mcscf.CASCI (mc._scf, mc.ncas, mc.nelecas)
        mc_1root.fcisolver = fci.solver (mc._scf.mol, singlet = False, symm = False)
        mc_1root.mo_coeff = mc.mo_coeff
        mc_1root.ci = mc.ci[root]
        mc_1root.e_tot = mc.e_states[root]
    dm1s = np.asarray (mc_1root.make_rdm1s ())
    spin = abs(mc.nelecas[0] - mc.nelecas[1])
    t0 = logger.timer (ot, 'rdms', *t0)

    Vnn = mc._scf.energy_nuc ()
    h = mc._scf.get_hcore ()
    dm1 = dm1s[0] + dm1s[1]
    vj, vk = mc._scf.get_jk (dm=dm1s)
    vj = vj[0] + vj[1]
    Te_Vne = np.tensordot (h, dm1)
    # (vj_a + vj_b) * (dm_a + dm_b)
    E_j = np.tensordot (vj, dm1) / 2
    # (vk_a * dm_a) + (vk_b * dm_b) Mind the difference!
    E_x = -(np.tensordot (vk[0], dm1s[0]) + np.tensordot (vk[1], dm1s[1])) / 2
    logger.debug (ot, 'CAS energy decomposition:')
    logger.debug (ot, 'Vnn = %s', Vnn)
    logger.debug (ot, 'Te + Vne = %s', Te_Vne)
    logger.debug (ot, 'E_j = %s', E_j)
    logger.debug (ot, 'E_x = %s', E_x)
    t0 = logger.timer (ot, 'Vnn, Te, Vne, E_j, E_x', *t0)

    E_ot = get_E_ot (ot, dm1s)
    t0 = logger.timer (ot, 'E_ot', *t0)
    e_tot = Vnn + Te_Vne + E_j + (ot.hyb_x * E_x) + E_ot
    logger.note (ot, 'MC-DCFT E = %s, Eot(%s) = %s', e_tot, ot.otxc, E_ot)

    chkdata = {'Vnn':Vnn, 'Te_Vne':Te_Vne, 'E_j':E_j, 'E_x':E_x, 'dm1s':dm1s, 'spin':spin}

    return e_tot, E_ot, chkdata


def _recalculate_with_xc(ot, chkdata):
    ''' Recalculate MC-DCFT total energy based on intermediate quantities from a previous MC-DCFT calculation

        Args:
            ot : str or an instance of on-top density functional class - see otfnal.py
            chkdata : chkdata dict generated by previous calculation

        Returns:
            Total MC-DCFT energy including nuclear repulsion energy.
    '''
    t0 = (logger.process_clock(), logger.perf_counter())
    omega, alpha, hyb = ot._numint.rsh_and_hybrid_coeff(ot.otxc, spin=chkdata['spin'])

    Vnn = chkdata['Vnn']
    Te_Vne = chkdata['Te_Vne']
    E_j = chkdata['E_j']
    E_x = chkdata['E_x']
    dm1s = chkdata['dm1s']

    logger.debug(ot, 'CAS energy decomposition (restored from previous calculation):')
    logger.debug(ot, 'Vnn = %s', Vnn)
    logger.debug(ot, 'Te + Vne = %s', Te_Vne)
    logger.debug(ot, 'E_j = %s', E_j)
    logger.debug(ot, 'E_x = %s', E_x)
    t0 = logger.timer(ot, 'Vnn, Te, Vne, E_j, E_x', *t0)

    E_ot = get_E_ot(ot, dm1s)

    t0 = logger.timer (ot, 'E_ot', *t0)
    e_tot = Vnn + Te_Vne + E_j + (ot.hyb_x * E_x) + E_ot
    logger.note(ot, 'MC-DCFT E = %s, Eot(%s) = %s', e_tot, ot.ot_name, E_ot)

    return e_tot, E_ot


def get_E_ot (ot, oneCDMs, max_memory=20000, hermi=1):
    ''' E_MCDCFT = h_pq l_pq + 1/2 v_pqrs l_pq l_rs + E_ot[D]
        or, in other terms,
        E_MCDCFT = T_KS[rho] + E_ext[rho] + E_coul[rho] + E_ot[D]
                 = E_DFT[1rdm] - E_xc[rho] + E_ot[D]
        Args:
            ot : an instance of otfnal class
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
    ni, xctype, dens_deriv = ot._numint, ot.xctype, ot.dens_deriv
    norbs_ao = oneCDMs.shape[1]

    E_ot = 0.0
    ot.ms = 0.0

    t0 = (logger.process_clock (), logger.perf_counter ())
    make_rho = tuple (ni._gen_rho_evaluator (ot.mol, oneCDMs[i,:,:], hermi) for i in range(2))
    for ao, mask, weight, coords in ni.block_loop (ot.mol, ot.grids, norbs_ao, dens_deriv, max_memory):
        rho = np.asarray ([m[0] (0, ao, mask, xctype) for m in make_rho])
        if ot.verbose > logger.DEBUG and dens_deriv > 0:
            for ideriv in range (1,4):
                rho_test  = np.einsum ('ijk,aj,ak->ia', oneCDMs, ao[ideriv], ao[0])
                rho_test += np.einsum ('ijk,ak,aj->ia', oneCDMs, ao[ideriv], ao[0])
                logger.debug (ot, "Spin-density derivatives, |PySCF-einsum| = %s",
                              linalg.norm (rho[:,ideriv,:]-rho_test))
        t0 = logger.timer (ot, 'untransformed density', *t0)
        D = get_unpaired_density(ot.natorb, ot.occ, ao)
        if ot.scaleD is not None:
            D = ot.scaleD(D)
        E_ot += ot.get_E_ot(rho, D, weight)
        t0 = logger.timer (ot, 'on-top exchange-correlation energy calculation', *t0)

    return E_ot

def get_mcdcft_child_class (mc, ot, **kwargs):

    class DCFT (mc.__class__):

        def __init__(self, scf, ncas, nelecas, my_ot=None, ot_name=None, grids_level=None, **kwargs):
            # Keep the same initialization pattern for backwards-compatibility.
            # Use a separate intializer for the ot functional
            try:
                super().__init__(scf, ncas, nelecas)
            except TypeError:
                # I think this is the same DFCASSCF problem as with the DF-SACASSCF gradients earlier
                super().__init__()
            keys = set(('e_ot', 'e_mcscf', 'e_states'))
            self._keys = set (self.__dict__.keys()).union(keys)
            self.grids_level = grids_level
            if my_ot is not None:
                self._init_ot_grids(my_ot, ot_name, grids_level=grids_level)

        def _init_ot_grids (self, my_ot, ot_name, grids_level=None):
            if isinstance (my_ot, (str, np.bytes_)):
                ks = dft.RKS(self.mol)
                ks.xc = my_ot
                self.otfnal = convfnal(self.mol, my_ot)
                self.otfnal.scaleD = None
            else:
                self.otfnal = my_ot
            self.grids = self.otfnal.grids
            self.ot_name = self.otfnal.ot_name = my_ot if ot_name is None else ot_name
            if grids_level is not None:
                self.grids.level = grids_level
                assert (self.grids.level == self.otfnal.grids.level)
            # Make sure verbose and stdout don't accidentally change (i.e., in scanner mode)
            self.otfnal.verbose = self.verbose
            self.otfnal.stdout = self.stdout

        def load_mcdcft_chk(self, chkfile):
            self.chkdata = lib.chkfile.load(chkfile, 'mcdcft')

        def recalculate_with_xc(self, ot, ot_name=None, scaleD=None, chkdata=None,
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
            self._init_ot_grids(ot, ot_name, grids_level=grids_level)
            self.otfnal.scaleD = scaleD
            if load_chk is not None:
                self.load_mcdcft_chk(load_chk)
            if chkdata is None:
                chkdata = self.chkdata
            natorb = chkdata['natorb']
            occ = chkdata['occ']
            self.otfnal._set_natorb(natorb, occ)
            self.otfnal.ot_name = ot if ot_name is None else ot_name
            n_states = chkdata['n_states']
            if n_states > 1:
                epdft = [_recalculate_with_xc(self.otfnal, ichkdata) for ichkdata in chkdata]
                self.e_states, self.e_ot = zip(*epdft)
                weights = chkdata['weights']
                self.e_tot = np.dot(self.e_states, weights)
            else:
                self.e_tot, self.e_ot = _recalculate_with_xc(self.otfnal, chkdata)
            if dump_chk is not None:
                lib.chkfile.dump(dump_chk, 'mcdcft/e_tot/' + self.otfnal.ot_name, self.e_tot)
                lib.chkfile.dump(dump_chk, 'mcdcft/e_ot/' + self.otfnal.ot_name, self.e_ot)
            return self.e_tot, self.e_ot

        def kernel(self, mo_coeff=None, ci=None, skip_scf=False, **kwargs):
            # Hafta reset the grids so that geometry optimization works!
            ot_name = self.ot_name
            self._init_ot_grids(self.otfnal.otxc, ot_name, grids_level=self.grids.level)
            if not skip_scf:
                self.e_mcscf, self.e_cas, self.ci, self.mo_coeff, self.mo_energy = super().kernel(mo_coeff,
                                                                                                  ci, **kwargs)
                natorb, _, occ = self.cas_natorb_(sort=False)
                self.otfnal._set_natorb(natorb.T, occ)

            # TODO: State average has not been tested !!!
            if isinstance (self, StateAverageMCSCFSolver):
                epdft = [kernel(self, self.otfnal, root=ix) for ix in range(len(self.e_states))]
                self.e_mcscf = self.e_states
                #  self.fcisolver.e_states = [e_tot for e_tot, e_ot in epdft]
                #  self.e_ot = [e_ot for e_tot, e_ot in epdft]
                self.fcisolver.e_states, self.e_ot, self.chkdata = zip(*epdft)
                self.chkdata['n_states'] = len(epdft)
                self.chkdata['weights'] = self.weights
                self.e_tot = np.dot(self.e_states, self.weights)
            else:
                self.e_tot, self.e_ot, self.chkdata = kernel(self, self.otfnal)
                self.chkdata['n_states'] = 1
            self.chkdata['e_tot'] = {self.otfnal.ot_name: self.e_tot}
            self.chkdata['natorb'] = self.otfnal.natorb
            self.chkdata['occ'] = self.otfnal.occ
            return self.e_tot, self.e_ot, self.e_mcscf, self.e_cas, self.ci, self.mo_coeff, self.mo_energy

        def dump_mcdcft_chk(self, chkfile, key='mcdcft', chkdata=None):
            '''Save MC-DCFT calculation results in chkfile.
            '''
            if chkdata is None:
                chkdata = self.chkdata
            lib.chkfile.dump(chkfile, key, chkdata)
            lib.chkfile.dump_mol(self.mol, chkfile)

        def dump_flags (self, verbose=None):
            super().dump_flags (verbose=verbose)
            log = logger.new_logger(self, verbose)
            log.info ('on-top pair density exchange-correlation functional: %s', self.otfnal.otxc)

        # TODO: gradient has not been implemented
        def nuc_grad_method (self):
            raise NotImplementedError ("MC-DCFT nuclear gradients")

        @property
        def otxc (self):
            return self.otfnal.otxc

    pdft = DCFT (mc._scf, mc.ncas, mc.nelecas, my_ot=ot, **kwargs)
    pdft.__dict__.update (mc.__dict__)
    return pdft

def CASSCFDCFT (mf_or_mol, ot, ncas, nelecas, chkfile=None, ncore=None, frozen=None, **kwargs):
    mc = mcscf.CASSCF(mf_or_mol, ncas, nelecas, ncore=ncore, frozen=frozen)
    if chkfile is not None:
        mc.__dict__.update(lib.chkfile.load(chkfile, 'mcscf'))
    return get_mcdcft_child_class(mc, ot, **kwargs)

CASSCF = CASSCFDCFT
