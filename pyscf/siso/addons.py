#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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

# Author: Bhavnesh Jangid

# Helper Functions for the analysis of the SOC States.

import numpy as np
from pyscf import lib
from pyscf.siso import anisoaddons

# Taken from pyscf
ge = 2.00231930436182

def _compute_angmom_in_so_basis(aniso_data):
    r"""
    Get the SO-basis orbital angular-momentum matrices.
    The ``generate_aniso_data`` dict, does have the spin
    and magnetic-moment matrices in the spin-orbit eigenbasis.
    Therefore, the orbital angular-momentum matrices can be
    recovered using the relation

        \mu = -ge * S - L,

    Note the factor ``1j`` converting the real antisymmetric ``angmom``
    representation into the physical Hermitian L matrix is already included
    in ``mu``.
    """

    spin_mat = np.array([
        aniso_data['spin_xr'] + 1j * aniso_data['spin_xi'],
        aniso_data['spin_yr'] + 1j * aniso_data['spin_yi'],
        aniso_data['spin_zr'] + 1j * aniso_data['spin_zi'],
    ])

    mu_mat = np.array([
        aniso_data['magn_xr'] + 1j * aniso_data['magn_xi'],
        aniso_data['magn_yr'] + 1j * aniso_data['magn_yi'],
        aniso_data['magn_zr'] + 1j * aniso_data['magn_zi'],
    ])

    return -mu_mat - ge * spin_mat


def _compute_total_angmom_in_so_basis(aniso_data):
    """Return the total angular-momentum matrices in the SOC basis."""
    spin_mat = np.array([
        aniso_data['spin_xr'] + 1j * aniso_data['spin_xi'],
        aniso_data['spin_yr'] + 1j * aniso_data['spin_yi'],
        aniso_data['spin_zr'] + 1j * aniso_data['spin_zi'],
    ])
    return _compute_angmom_in_so_basis(aniso_data) + spin_mat

def _compute_spin_free_l_s_values(aniso_data, log):
    """
    Compute the L and S values for each spin-free state
    """
    # The spin-free angmom matrices are stored in their real antisymmetric
    # representation, so multiply by 1j to obtain physical Hermitian L
    Lx_mat = 1j * aniso_data['angmom_x']
    Ly_mat = 1j * aniso_data['angmom_y']
    Lz_mat = 1j * aniso_data['angmom_z']

    l2_mat = Lx_mat @ Lx_mat + Ly_mat @ Ly_mat + Lz_mat @ Lz_mat
    l2_expectation = np.real_if_close(np.diag(l2_mat)).real
    l_values = (-1.0 + np.sqrt(1.0 + 4.0 * l2_expectation)) / 2.0

    spin_mult = np.asarray(aniso_data['multiplicity'], dtype=np.int32)
    spin_values = (spin_mult - 1.0) / 2.0
    s2_values = spin_values * (spin_values + 1.0)
    sof_energy = np.asarray(aniso_data['esfs'], dtype=np.float64)

    log.note(" ")
    log.info('******** %s ********',
             "Spin-Orbit Free Energies, L, and S values")
    log.note('  State       Energy (au)    L-value       Spin (S)')
    for state, (energy, l_value, s_value) in enumerate(
            zip(sof_energy, l_values, spin_values)):
        log.note(" {:<10} {:>20.10f} {:>20.4f} {:>20.4f}".format
                 (state, energy, l_value, s_value))

    return (Lx_mat, Ly_mat, Lz_mat), l_values, spin_values, s2_values

def _compute_soc_j_values(aniso_data, log):
    r"""
    Compute the effective J values for each SOC state.
    The total angular momentum is given by
        ``J = L + S``
    where ``L`` is the orbital angular momentum and ``S`` is the
    spin angular momentum.

    Note: J is well defined quantum number only for the atoms.
    """

    tot_ang_mom = _compute_total_angmom_in_so_basis(aniso_data)

    j_mat = (tot_ang_mom[0] @ tot_ang_mom[0] +
             tot_ang_mom[1] @ tot_ang_mom[1] +
             tot_ang_mom[2] @ tot_ang_mom[2])

    j2_exp = np.real_if_close(np.diag(j_mat)).real
    j_values = (-1.0 + np.sqrt(1.0 + 4.0 * j2_exp)) / 2.0

    so_energies = np.asarray(aniso_data['eso'], dtype=np.float64)

    log.note(" ")
    log.info('******** %s ********', "SOC Energies and effective J values")
    log.note('  State      Energy (au)       J-values')
    for state, (energy, j_value) in enumerate(
            zip(so_energies, j_values)):
        log.note(" {:<10} {:>20.10f} {:>20.4f} ".format(
            state, energy, j_value))

    return j_mat, j_values

def _compute_soc_omega_values(aniso_data, log, axis='z'):
    r"""
    Compute the effective Omega values for each SOC state.
    Omega is the absolute projection of the total angular momentum along
    a principal axis.  The z axis (``axis=2``) is used by default, consistent
    with the molecular-axis convention for linear molecules.
    """

    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if isinstance(axis, str):
        axis = axis_map.get(axis.lower())

    if axis not in (0, 1, 2):
        raise ValueError('axis must be 0 (x), 1 (y), or 2 (z)')

    total_ang_mom = _compute_total_angmom_in_so_basis(aniso_data)
    omega_matrix = total_ang_mom[axis]
    omega_values = np.abs(np.real_if_close(np.diag(omega_matrix)).real)
    energies = np.asarray(aniso_data['eso'], dtype=np.float64)

    log.note(" ")
    log.info('******** %s ********',
             "SOC Energies and effective Omega values")
    log.note('  State      Energy (au)       Omega-values')
    for state, (energy, omega) in enumerate(zip(energies, omega_values)):
        log.note(" {:<10} {:>20.10f} {:>20.4f} ".format(
            state, energy, omega))

    return omega_matrix, omega_values

def soc_state_analysis(aniso_data, log, state=(0,), threshold=5e-3):
    """
    The decomposition of selected SOC states in the spin-free basis.
    args:
        aniso_data: dict
            Containing the SOC analysis data.
        log: The PySCF logger instance.
            For printing the analysis results.
        state: int or list
            The SOC-state(s) to analyze.
        threshold: float
            Minimum coefficient weight to report.
    """
    coefficients = (np.asarray(aniso_data['eigenr'])
                    + 1j * np.asarray(aniso_data['eigeni']))
    energies = np.asarray(aniso_data['eso'])
    root_weights = np.zeros((sum(aniso_data['nroot']), 
                             coefficients.shape[1]))

    states = np.atleast_1d(state)
    if not np.issubdtype(states.dtype, np.integer):
        raise TypeError('state must be an integer or a list of integers')
    
    states = states.tolist()
    nstates = energies.size

    # Sanity check:
    invalid = [state for state in states 
               if state < 0 or state >= nstates]
    
    if invalid: 
        raise IndexError(f'SOC state index {invalid} is outside'\
                                 f'the range [0, {nstates})')

    log.info('\nSOC-state decomposition in the spin-free basis')
    offset = 0
    root_number = 0
    for nroots, multiplicity in zip(aniso_data['nroot'], aniso_data['imult']):
        for root in range(nroots):
            block = slice(offset + root * multiplicity,
                          offset + (root + 1) * multiplicity)
            root_weights[root_number, :] = np.sum(
                np.abs(coefficients[block, :])**2, axis=0)
            root_number += 1

        offset += nroots * multiplicity

    for state_idx in states:
        energy = energies[state_idx]
        log.note('\nSOC state %d, energy = %.10f au', state_idx, energy)
        for root, weight in enumerate(root_weights[:, state_idx]):
            if weight > threshold:
                log.info('  spin-free root %d: total weight = %.8f',
                         root, weight)

        offset = 0
        root_number = 0
        for nroots, multiplicity in zip(aniso_data['nroot'],
                                        aniso_data['imult']):
            spin = (multiplicity - 1) / 2.0
            ms_values = np.arange(-spin, spin + 1.0, 1.0)
            for root in range(nroots):
                block_start = offset + root * multiplicity
                for ms_index, ms in enumerate(ms_values):
                    coefficient = coefficients[block_start + ms_index, state_idx]
                    if np.abs(coefficient)**2 > threshold:
                        log.note('    root %d, m_s = %+.1f: coefficient = %s, '
                                 'weight = %.8f', root_number, ms,
                                 coefficient, np.abs(coefficient)**2)
                root_number += 1
            offset += nroots * multiplicity

    return coefficients, root_weights


class soc_analysis:
    """
    Class for analyzing the SOC states of a converged SISO object.

    args:
        mysiso: converged SISO object.  
            It must have ``si_energies`` and
            ``si_vecs`` when ``aniso_data`` is not supplied.
    kwargs:
        aniso_data: dict
            Containing the SOC analysis data generated by
            anisoaddons.generate_aniso_data or by the SISO object.
        modelspace: list, optional
            The SISO model space used to generate the ``aniso_data``.
        origin: str, optional
            Gauge origin used when generating ``aniso_data``.
        ham: str, optional 
            SOC Hamiltonian used when generating ``aniso_data``.
    """

    def __init__(self, mysiso, aniso_data=None, modelspace=None,
                 origin='CHARGE_CENTER', ham=None):
        self.mysiso = mysiso
        self.mc = mysiso.mc
        self.log = lib.logger.Logger(self.mc.stdout, self.mc.verbose)
        self.origin = origin
        self.ham = mysiso.ham if ham is None else ham
        
        if self.aniso_data is None:
            assert modelspace is not None, \
                "modelspace must be provided and probably same as that of the SISO object " \
                "if aniso_data is not supplied"
            self.modelspace = list(modelspace)

        if aniso_data is None:
            mol = self.mc._scf.mol
            assert self.modelspace == list(mysiso.modelspace), \
                "modelspace must be the same as that of the SISO object " \
                "if aniso_data is not supplied"
            aniso_data = anisoaddons.generate_aniso_data(
                mol, self.mc, self.modelspace, mysiso,
                origin=origin, ham=self.ham)
        self.aniso_data = aniso_data

    def _compute_angmom_in_so_basis(self):
        return _compute_angmom_in_so_basis(self.aniso_data)

    def compute_L_and_S_values(self):
        return _compute_spin_free_l_s_values(self.aniso_data, self.log)

    def compute_J_values(self):
        return _compute_soc_j_values(self.aniso_data, self.log)

    def _compute_soc_omega_values(self, axis='z'):
        return _compute_soc_omega_values(self.aniso_data, self.log, axis=axis)

    def compute_omega(self, axis='z'):
        """
        Compute SOC-state Omega values along a principal axis.
        """
        # Only use this for the linear molecules.
        return self._compute_soc_omega_values(axis=axis)

    def soc_state_analysis(self, state=(0,), threshold=5e-3):
        return soc_state_analysis(self.aniso_data, self.log, state=state,
                                  threshold=threshold)
