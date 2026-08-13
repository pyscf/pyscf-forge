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
from pyscf.siso import anisoaddons, socaddons

# Register this function as an alias for generate_aniso_data, to
# avoid confusion with the name "aniso" in the context of SISO calculations.
generate_siso_data = anisoaddons.generate_aniso_data

# Taken from pyscf
ge = 2.00231930436182

def _compute_angmom_in_so_basis(siso_data):
    r"""
    Get the SO-basis orbital angular-momentum matrices.
    The ``generate_siso_data`` dict, does have the spin
    and magnetic-moment matrices in the spin-orbit eigenbasis.
    Therefore, the orbital angular-momentum matrices can be
    recovered using the relation

        \mu = -ge * S - L,

    Note the factor ``1j`` converting the real antisymmetric ``angmom``
    representation into the physical Hermitian L matrix is already included
    in ``mu``.
    """

    spin_mat = np.array([
        siso_data['spin_xr'] + 1j * siso_data['spin_xi'],
        siso_data['spin_yr'] + 1j * siso_data['spin_yi'],
        siso_data['spin_zr'] + 1j * siso_data['spin_zi'],
    ])

    mu_mat = np.array([
        siso_data['magn_xr'] + 1j * siso_data['magn_xi'],
        siso_data['magn_yr'] + 1j * siso_data['magn_yi'],
        siso_data['magn_zr'] + 1j * siso_data['magn_zi'],
    ])

    return -mu_mat - ge * spin_mat

def _compute_total_angmom_in_so_basis(siso_data):
    """
    Total angular-momentum matrices in the SOC basis.
    """
    spin_mat = np.array([
        siso_data['spin_xr'] + 1j * siso_data['spin_xi'],
        siso_data['spin_yr'] + 1j * siso_data['spin_yi'],
        siso_data['spin_zr'] + 1j * siso_data['spin_zi'],
    ])
    return _compute_angmom_in_so_basis(siso_data) + spin_mat

def _select_degenerate_subblocks(energies, degeneracy_tol=1e-6):
    """
    Generate energy-degenerate state subblocks in original-state order.
    """
    energies = np.asarray(energies, dtype=np.float64).ravel()
    energy_order = np.argsort(energies, kind='stable')
    blocks = []
    current_block = [int(energy_order[0])]
    block_reference_energy = energies[current_block[0]]

    for index in energy_order[1:]:
        index = int(index)
        if abs(energies[index] - block_reference_energy) <= degeneracy_tol:
            current_block.append(index)
        else:
            blocks.append(np.asarray(current_block, dtype=np.int32))
            current_block = [index]
            block_reference_energy = energies[index]
    blocks.append(np.asarray(current_block, dtype=np.int32))
    return blocks

def _diagonalize_degenerate_subblocks(operator, energies, degeneracy_tol=1e-6):
    """
    Diagonalize an operator independently within energy-degenerate blocks.
    """
    operator = np.asarray(operator, dtype=np.complex128)
    energies = np.asarray(energies, dtype=np.float64).ravel()
    nstate = energies.size
    if operator.shape != (nstate, nstate):
        raise ValueError('Operator has shape {}, but {} energies were provided'
                         .format(operator.shape, nstate))

    # Making it hermitian and removing numerical residuals.
    operator = 0.5 * (operator + operator.conj().T)

    blocks = _select_degenerate_subblocks(energies, degeneracy_tol)
    rotation = np.eye(nstate, dtype=np.complex128)
    for block_indices in blocks:
        if block_indices.size == 1:
            continue
        block = operator[np.ix_(block_indices, block_indices)]
        _, block_vectors = np.linalg.eigh(block)
        rotation[np.ix_(block_indices, block_indices)] = block_vectors

    rotated_operator = rotation.conj().T @ operator @ rotation

    diagonal = np.real_if_close(np.diag(rotated_operator), tol=1000)

    if np.iscomplexobj(diagonal):
        max_imaginary = np.max(np.abs(diagonal.imag))
        if max_imaginary > 1e-10:
            raise ValueError('Diagonalized operator has imaginary diagonal component '
                '{:.3e}'.format(max_imaginary))
        diagonal = diagonal.real
    return rotated_operator, np.asarray(diagonal, dtype=np.float64), rotation, blocks

def _compute_spin_free_l_s_values(siso_data, log, degeneracy_tol=1e-6):
    """
    Compute the L and S values for each spin-free state
    """
    # Note: the spin-free angmom matrices are stored in their real antisymmetric
    # representation, so multiply by 1j to obtain physical Hermitian L
    Lx_mat = 1j * siso_data['angmom_x']
    Ly_mat = 1j * siso_data['angmom_y']
    Lz_mat = 1j * siso_data['angmom_z']

    l2_mat = Lx_mat @ Lx_mat + Ly_mat @ Ly_mat + Lz_mat @ Lz_mat
    _, l2_expectation, rotation, _ = _diagonalize_degenerate_subblocks(
        l2_mat, siso_data['esfs'], degeneracy_tol)

    Lx_mat = rotation.conj().T @ Lx_mat @ rotation
    Ly_mat = rotation.conj().T @ Ly_mat @ rotation
    Lz_mat = rotation.conj().T @ Lz_mat @ rotation
    l_values = (-1.0 + np.sqrt(1.0 + 4.0 * l2_expectation)) / 2.0

    spin_mult = np.asarray(siso_data['multiplicity'], dtype=np.int32)
    spin_values = (spin_mult - 1.0) / 2.0
    s2_values = spin_values * (spin_values + 1.0)
    sof_energy = np.asarray(siso_data['esfs'], dtype=np.float64)
    sort_idx = np.argsort(np.asarray(sof_energy), kind='stable')

    log.note(" ")
    log.info('******** %s ********',
             "Spin-Orbit Free Energies, L, and S values")
    log.note('{:^7s} {:^22s} {:^14s} {:^14s}'.format(
        'State', 'Energy (au)', 'L-value', 'Spin (S)'))
    log.note('-' * 60)
    for i, state in enumerate(sort_idx):
        log.note('{:>7d} {:>22.10f} {:>10.4f} {:>10.4f}'.format(
                i,
                sof_energy[state],
                l_values[state],
                spin_values[state]))

    return (Lx_mat, Ly_mat, Lz_mat), l_values, spin_values, s2_values

def _compute_soc_j_values(siso_data, log, degeneracy_tol=1e-6):
    r"""
    Compute the effective J values for each SOC state.
    The total angular momentum is given by
        ``J = L + S``
    where ``L`` is the orbital angular momentum and ``S`` is the
    spin angular momentum.

    Note: J is well defined quantum number only for the atoms.
    """

    tot_ang_mom = _compute_total_angmom_in_so_basis(siso_data)

    j_mat = (tot_ang_mom[0] @ tot_ang_mom[0] +
             tot_ang_mom[1] @ tot_ang_mom[1] +
             tot_ang_mom[2] @ tot_ang_mom[2])

    j_mat, j2_exp, _, _ = _diagonalize_degenerate_subblocks(
        j_mat, siso_data['eso'], degeneracy_tol)
    j_values = (-1.0 + np.sqrt(1.0 + 4.0 * j2_exp)) / 2.0

    so_energies = np.asarray(siso_data['eso'], dtype=np.float64)

    log.note(" ")
    log.info('******** %s ********', "SOC Energies and effective J values")
    log.note('{:^10s} {:^22s} {:^16s}'.format(
        'State', 'Energy (au)', 'J-values'))
    log.note('-' * 60)
    for state, (energy, j_value) in enumerate(zip(so_energies, j_values)):
        log.note('{:>7d} {:>22.10f} {:>14.4f}'.format(state, energy, j_value))

    return j_mat, j_values

def _compute_soc_omega_values(siso_data, log, axis='z', degeneracy_tol=1e-6):
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

    energies = np.asarray(siso_data['eso'], dtype=np.float64)
    nstate = energies.size

    total_ang_mom = _compute_total_angmom_in_so_basis(siso_data)
    omega_matrix = np.asarray(total_ang_mom[axis], dtype=np.complex128)

    if omega_matrix.shape != (nstate, nstate):
        raise ValueError('Angular-momentum matrix has shape {}, but {} SOC energies '
            'were provided'.format(omega_matrix.shape, nstate))

    omega_matrix, omega_diagonal, _, degenerate_blocks = (
        _diagonalize_degenerate_subblocks(
            omega_matrix, energies, degeneracy_tol))

    # Removing small numerical residuals
    omega_values = np.abs(omega_diagonal)

    # Clean up insignificant numerical residuals.
    omega_values[omega_values < 1.0e-12] = 0.0

    axis_label = ('x', 'y', 'z')[axis]

    block_number_by_state = np.empty(nstate, dtype=np.int32)
    for block_number, block_indices in enumerate(degenerate_blocks):
        block_number_by_state[block_indices] = block_number

    log.note(' ')
    log.info('******** %s ********','SOC energies and effective Omega values',)
    log.note('Projection axis: J_%s', axis_label)
    log.note('Degeneracy tolerance: %.3e Hartree',degeneracy_tol,)
    log.note('{:^10s} {:^22s} {:^16s} {:^8s}'.format(
        'State', 'Energy (au)', 'Omega-value', 'Block'))
    log.note('-' * 60)
    for state, (energy, omega) in enumerate(
            zip(energies, omega_values)):
        log.note('{:>3d} {:>22.10f} {:>16.4f} {:>8d}'.format(
                state,
                energy,
                omega,
                int(block_number_by_state[state])))
    return omega_matrix, omega_values

def _compute_L_for_diatomics(siso_data, log, axis='z', degeneracy_tol=1e-6):
    r"""
    Compute the effective orbital-angular-momentum projection for each
    spin-orbit state of a diatomic molecule.

    For a linear molecule, the orbital projection is

        ``Lambda = abs(<L_axis>)``

    where the molecular axis is conventionally the z axis.  This function
    projects the orbital angular momentum only; it does not include spin and
    therefore does not compute Omega or J.
    """

    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if isinstance(axis, str):
        axis = axis_map.get(axis.lower())

    if axis not in (0, 1, 2):
        raise ValueError('axis must be 0 (x), 1 (y), or 2 (z)')

    energies = np.asarray(siso_data['eso'], dtype=np.float64)
    nstate = energies.size

    angmom = _compute_angmom_in_so_basis(siso_data)
    projection_matrix = np.asarray(angmom[axis], dtype=np.complex128)

    if projection_matrix.shape != (nstate, nstate):
        raise ValueError('Angular-momentum matrix has shape {}, but {} SOC energies '
                         'were provided'.format(projection_matrix.shape, nstate))

    projection_matrix, projection_diagonal, _, degenerate_blocks = (
        _diagonalize_degenerate_subblocks(
            projection_matrix, energies, degeneracy_tol))

    lambda_values = np.abs(projection_diagonal)
    lambda_values[lambda_values < 1.0e-12] = 0.0

    axis_label = ('x', 'y', 'z')[axis]
    block_number_by_state = np.empty(nstate, dtype=np.int32)
    for block_number, block_indices in enumerate(degenerate_blocks):
        block_number_by_state[block_indices] = block_number

    log.note(' ')
    log.info('******** %s ********',
             'SOC energies and effective orbital L projections')
    log.note('Projection axis: L_%s', axis_label)
    log.note('Degeneracy tolerance: %.3e Hartree', degeneracy_tol)
    log.note('{:^10s} {:^22s} {:^16s} {:^8s}'.format(
        'State', 'Energy (au)', 'Lambda-value', 'Block'))
    log.note('-' * 60)
    for state, (energy, lambda_value) in enumerate(
            zip(energies, lambda_values)):
        log.note('{:>3d} {:>22.10f} {:>16.4f} {:>8d}'.format(
            state,
            energy,
            lambda_value,
            int(block_number_by_state[state])))

    return projection_matrix, lambda_values

def soc_state_analysis(siso_data, log, state=(0,), threshold=5e-3):
    """
    The decomposition of selected SOC states in the spin-free basis.
    args:
        siso_data: dict
            Containing the SOC analysis data.
        log: The PySCF logger instance.
            For printing the analysis results.
        state: int or list
            The SOC-state(s) to analyze.
        threshold: float
            Minimum coefficient weight to report.
    """
    coefficients = (np.asarray(siso_data['eigenr'])
                    + 1j * np.asarray(siso_data['eigeni']))
    energies = np.asarray(siso_data['eso'])
    root_weights = np.zeros((sum(siso_data['nroot']),
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
        raise IndexError(f'SOC state index {invalid} is outside'
                                 f'the range [0, {nstates})')

    # Printing the Spin-Orbit Free states and energies: to avoid
    # confusion for below analysis.
    sf_energies = np.asarray(siso_data['esfs'], dtype=np.float64)
    sm_values = np.asarray(siso_data['multiplicity'], dtype=np.float64)
    s2_values = (sm_values - 1.0) / 2.0 * ((sm_values - 1.0) / 2.0 + 1.0)

    log.note(" ")
    log.info('******** %s ********',
             'Spin-Orbit Free States and Energies')
    for state_idx in range(len(sf_energies)):
        log.note(' state %d, energy = %.10f au, S^2 = %.6f',
                 state_idx, sf_energies[state_idx], s2_values[state_idx])

    log.note(" ")
    log.info('******** %s ********',
             'SOC-state decomposition in the spin-free basis')
    log.note('-' * 60)
    offset = 0
    root_number = 0
    for nroots, multiplicity in zip(siso_data['nroot'], siso_data['imult']):
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
                log.info('  spin-free state %d: total weight = %.8f',
                         root, weight)

        offset = 0
        root_number = 0
        for nroots, multiplicity in zip(siso_data['nroot'],
                                        siso_data['imult']):
            spin = (multiplicity - 1) / 2.0
            ms_values = np.arange(-spin, spin + 1.0, 1.0)
            for root in range(nroots):
                block_start = offset + root * multiplicity
                for ms_index, ms in enumerate(ms_values):
                    coefficient = coefficients[block_start + ms_index, state_idx]
                    if np.abs(coefficient)**2 > threshold:
                        log.note('    state %d, m_s = %+.1f: '
                                 'weight = %.8f', root_number, ms,
                                  np.abs(coefficient)**2)
                root_number += 1
            offset += nroots * multiplicity

    return coefficients, root_weights


class soc_analysis:
    """
    Class for analyzing the SOC states of a converged SISO object.

    args:
        mysiso: converged SISO object.
            It must have ``si_energies`` and
            ``si_vecs`` when ``siso_data`` is not supplied.
    kwargs:
        siso_data: dict
            Containing the SOC analysis data generated by
            anisoaddons.generate_siso_data or by the SISO object.
        modelspace: list, optional
            The SISO model space used to generate the ``siso_data``.
        origin: str, optional
            Gauge origin used when generating ``siso_data``.
        ham: str, optional
            SOC Hamiltonian used when generating ``siso_data``.
    """

    def __init__(self, mysiso, siso_data=None, modelspace=None,
                 origin='CHARGE_CENTER', ham=None):
        self.mysiso = mysiso
        self.mc = mysiso.mc
        self.log = lib.logger.Logger(self.mc.stdout, self.mc.verbose)
        self.origin = origin
        self.ham = mysiso.ham if ham is None else ham

        if siso_data is None:
            mol = self.mc._scf.mol
            if modelspace is None:
                self.modelspace = list(mysiso.modelspace)
            else:
                self.modelspace = socaddons._validate_modelspace(
                    modelspace, mol=mol, ncas=self.mc.ncas,
                    nelecas=self.mc.nelecas)
            if self.modelspace != list(mysiso.modelspace):
                raise ValueError(
                    "modelspace must match the SISO model space when "
                    "siso_data is not supplied")
            siso_data = anisoaddons.generate_siso_data(
                mol, self.mc, self.modelspace, mysiso,
                origin=origin, ham=self.ham)
        self.siso_data = siso_data

    def _compute_angmom_in_so_basis(self):
        return _compute_angmom_in_so_basis(self.siso_data)

    def compute_L_values(self, degeneracy_tol=1e-6):
        return _compute_spin_free_l_s_values(
            self.siso_data, self.log, degeneracy_tol=degeneracy_tol)

    def compute_J_values(self, degeneracy_tol=1e-6):
        return _compute_soc_j_values(
            self.siso_data, self.log, degeneracy_tol=degeneracy_tol)

    def _compute_soc_omega_values(self, axis='z', degeneracy_tol=1e-6):
        return _compute_soc_omega_values(
            self.siso_data, self.log, axis=axis,
            degeneracy_tol=degeneracy_tol)

    def _compute_L_for_diatomics(self, axis='z', degeneracy_tol=1e-6):
        return _compute_L_for_diatomics(
            self.siso_data, self.log, axis=axis,
            degeneracy_tol=degeneracy_tol)

    def compute_omega_values(self, axis='z', degeneracy_tol=1e-6):
        """
        Compute SOC-state Omega values along a principal axis.
        """
        # Only use this for the linear molecules.
        return self._compute_soc_omega_values(
            axis=axis, degeneracy_tol=degeneracy_tol)

    def compute_L_values_for_diatomics(self, axis='z', degeneracy_tol=1e-6):
        """
        Compute orbital L-projection (Lambda) values for a diatomic molecule.
        """
        return self._compute_L_for_diatomics(
            axis=axis, degeneracy_tol=degeneracy_tol)

    def soc_state_analysis(self, state=(0,), threshold=5e-3):
        return soc_state_analysis(self.siso_data, self.log, state=state,
                                  threshold=threshold)
