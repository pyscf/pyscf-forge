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
from Development.mrh.tests.lasucc.test_uop import spin
from pyscf import lib, mcscf

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

def _compute_spin_free_l_s_values(aniso_data, logger):
    """
    Print spin-free energies with effective l and spin values.
    """
    # The spin-free angmom matrices are stored in their real antisymmetric
    # representation, so multiply by 1j to obtain physical Hermitian L
    Lx_mat = 1j * aniso_data['angmom_x']
    Ly_mat = 1j * aniso_data['angmom_y']
    Lz_mat = 1j * aniso_data['angmom_z']

    l2_mat = Lx_mat @ Lx_mat + Ly_mat @ Ly_mat + Lz_mat @ Lz_mat
    l2_expectation = np.real_if_close(np.diag(l2_mat)).real
    l_values = (-1.0 + np.sqrt(1.0 + 4.0 * l2_expectation)) / 2.0

    spin_mult = np.asarray(aniso_data['multiplicity'], dtype=float)
    spin_values = (spin_mult - 1.0) / 2.0
    s2_values = spin_values * (spin_values + 1.0)

    logger.info('\nSpin-free energies, L values, and S values')
    logger.info('  State                 Energy (au)       <L^2>     L       S')
    for state, (energy, l2, l_value, s_value) in enumerate(
            zip(aniso_data['esfs'], l2_expectation, l_values, spin_values)):
        logger.info(f'  {state:5d}   {energy:20.12f}   {l2:10.6f} '
                    f'{l_value:8.5f} {s_value:8.5f}')

    return (Lx_mat, Ly_mat, Lz_mat), l_values, spin_values, s2_values

def _compute_soc_j_values(aniso_data):
    r"""
    Compute the effective J values for each SOC state.
    The total angular momentum is given by
        ``J = L + S``
    where ``L`` is the orbital angular momentum and ``S`` is the 
    spin angular momentum.
    """
    spin_mat = np.array([
        aniso_data['spin_xr'] + 1j * aniso_data['spin_xi'],
        aniso_data['spin_yr'] + 1j * aniso_data['spin_yi'],
        aniso_data['spin_zr'] + 1j * aniso_data['spin_zi'],
    ])
    angmom_mat = _compute_angmom_in_so_basis(aniso_data)[0]

    tot_ang_mom = angmom_mat + spin_mat

    j_mat = tot_ang_mom[0] @ tot_ang_mom[0] + \
            tot_ang_mom[1] @ tot_ang_mom[1] + \
            tot_ang_mom[2] @ tot_ang_mom[2]
    
    j2_exp = np.real_if_close(np.diag(j_mat)).real
    j_values = (-1.0 + np.sqrt(1.0 + 4.0 * j2_exp)) / 2.0

    print('\nSOC energies and effective J values')
    print('  State                 Energy (au)       <J^2>           J')
    for state, (energy, j2, j_value) in enumerate(
            zip(aniso_data['eso'], j_mat, j_values)):
        print(f'  {state:5d}   {energy:20.12f}   {j2:12.8f}   {j_value:10.8f}')

    return j_mat, j_values

def soc_state_analysis(aniso_data, state=[0,], threshold=1e-8):
    """
    Print the decomposition of each SOC state in the spin-free basis.
    """
    coefficients = (np.asarray(aniso_data['eigenr'])
                    + 1j * np.asarray(aniso_data['eigeni']))
    energies = np.asarray(aniso_data['eso'])
    root_weights = np.zeros((sum(aniso_data['nroot']), coefficients.shape[1]))

    print('\nSOC-state decomposition in the spin-free basis')
    offset = 0
    root_number = 0
    for nroots, multiplicity in zip(aniso_data['nroot'], aniso_data['imult']):
        spin = (multiplicity - 1) / 2.0
        ms_values = np.arange(-spin, spin + 1.0, 1.0)

        for root in range(nroots):
            block = slice(offset + root * multiplicity,
                          offset + (root + 1) * multiplicity)
            root_weights[root_number, :] = np.sum(
                np.abs(coefficients[block, :])**2, axis=0)
            root_number += 1

        offset += nroots * multiplicity

    for state, energy in enumerate(energies):
        print(f'\nSOC state {state}, energy = {energy:.12f} au')
        for root, weight in enumerate(root_weights[:, state]):
            if weight > threshold:
                print(f'  spin-free root {root}: total weight = {weight:.8f}')

        offset = 0
        root_number = 0
        for nroots, multiplicity in zip(aniso_data['nroot'],
                                        aniso_data['imult']):
            spin = (multiplicity - 1) / 2.0
            ms_values = np.arange(-spin, spin + 1.0, 1.0)
            for root in range(nroots):
                block_start = offset + root * multiplicity
                for ms_index, ms in enumerate(ms_values):
                    coefficient = coefficients[block_start + ms_index, state]
                    if abs(coefficient)**2 > threshold:
                        print(f'    root {root_number}, m_s = {ms:+.1f}: '
                              f'coefficient = {coefficient:.8f}, '
                              f'weight = {abs(coefficient)**2:.8f}')
                root_number += 1
            offset += nroots * multiplicity

    return coefficients, root_weights

