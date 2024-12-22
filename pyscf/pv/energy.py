# Copyright 2014-2024 The PySCF Developers. All Rights Reserved.
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

'''

@author Juan Jose Aucar

'''

import numpy
from pyscf.scf import dhf
from pyscf import lib, gto


def fc_integrals(mol, mf, atom, **kwargs):
    """
    Compute the Fermi Contact (FC) integrals for a given atom.

    This function calculates the Fermi Contact integrals in
    relativistic form for a specified atom in the molecule using
    four-component Dirac-Coulomb spinor wavefunctions.

    The FC integrals consist of four distinct blocks:
    - LL: Large-Large component
    - LS: Large-Small component
    - SL: Small-Large component
    - SS: Small-Small component

    :param mol: Molecule object containing information about the molecular system.
    :type mol: pyscf.gto.Mole
    :param mf: Mean-field object, obtained from a Dirac-Hartree-Fock (DHF) calculation.
    :type mf: pyscf.scf.hf.SCF
    :param atom: Index of the atom for which the FC integrals are computed.
    :type atom: int
    :return: Matrix of Fermi Contact integrals (4-component spinor form) for the specified atom.
    :rtype: numpy.ndarray (complex128), shape (n4c, n4c)

    Example:
        mol = gto.Mole()
        # Setup mol with geometry and basis set...
        mf = scf.DHF(mol)
        mf.kernel()
        atom_index = 0
        fc_matrix = fc_integrals(mol, mf, atom_index)
    """
    c = lib.param.LIGHT_SPEED
    n4c = mf.mo_coeff.shape[0]
    n2c = n4c // 2
    coordinates = mf.mol.atom_coords()[[atom]]

    # Obtaining AO integrals in spinor form
    ao_spinor = gto.eval_gto(mf.mol, "GTOval_spinor", coordinates, comp=1)[:, 0, :]
    ao_spinor_S = gto.eval_gto(mf.mol, "GTOval_sp_spinor", coordinates, comp=1)[:, 0, :]

    # Forming the FC integrals matrix (LL, LS, SL, SS blocks)
    fc_integrals = numpy.zeros((n4c, n4c), dtype=numpy.complex128)
    fc_integrals[:n2c, :n2c] = numpy.einsum("ip,iq->pq", ao_spinor.conjugate(), ao_spinor)
    fc_integrals[:n2c, n2c:] = numpy.einsum("ip,iq->pq", ao_spinor.conjugate(), ao_spinor_S) * (0.5 / c)
    fc_integrals[n2c:, :n2c] = numpy.einsum("ip,iq->pq", ao_spinor_S.conjugate(), ao_spinor) * (0.5 / c)
    fc_integrals[n2c:, n2c:] = numpy.einsum("ip,iq->pq", ao_spinor_S.conjugate(), ao_spinor_S) * ((0.5 / c) ** 2)

    return fc_integrals

def fc_expval(mol, mf, atom):
    """
    Calculate the Fermi Contact (FC) expectation values for each occupied orbital in a molecule,
    focusing on a specific atom.

    The expectation values are calculated using the four-component Dirac-Coulomb spinor
    wavefunctions. The function computes the contributions from the large-large (LL) and
    small-small (SS) components of the spinor for each occupied orbital.

    :param mol: Molecule object containing information about the molecular system.
    :type mol: pyscf.gto.Mole
    :param mf: Mean-field object, obtained from a Dirac-Hartree-Fock (DHF) calculation.
    :type mf: pyscf.scf.hf.SCF
    :param atom: Index of the atom for which the FC expectation values are computed.
    :type atom: int
    :return: Array of FC expectation values for each occupied orbital.
    :rtype: numpy.ndarray (real), shape (nocc,)

    Example:
        mol = gto.Mole()
        # Setup mol with geometry and basis set...
        mf = scf.DHF(mol)
        mf.kernel()
        atom_index = 0
        fc_expval_per_orbital = fc_expval(mol, mf, atom_index)
    """
    n4c, nmo = mf.mo_coeff.shape
    n2c = n4c // 2
    nocc = mf.mol.nelectron
    expval_perorb = numpy.zeros(nocc)

    mo_pos_l = mf.mo_coeff[:n2c, nmo//2:]
    mo_pos_s = mf.mo_coeff[n2c:, nmo//2:]

    Lo = mo_pos_l[:, :nocc]
    So = mo_pos_s[:, :nocc]

    fac = 8 * numpy.pi / 3
    fc_ao = fc_integrals(mol, mf, atom)

    # Split the fc_integrals matrix into LL and SS blocks
    fc_ao_LL = fc_ao[:n2c, :n2c]
    fc_ao_SS = fc_ao[n2c:, n2c:]

    for k in range(nocc):
        expval_LL_k = numpy.einsum('pi,ij,qj->pq', Lo[:, k].conjugate(), fc_ao_LL, Lo[:, k])
        expval_SS_k = numpy.einsum('pi,ij,qj->pq', So[:, k].conjugate(), fc_ao_SS, Lo[:, k])
        expval_perorb[k] = expval_LL_k + expval_SS_k

    return fac * expval_perorb.real



def Epv_atom(mol, mf, atom_index):
    """
    Calculate the parity-violating (PV) contribution to energy for a given atom in a molecule.

    It then returns the contributions from each occupied orbital.
    (First term of Eq. 4 -  https://doi.org/10.1002/wcms.1396)

    :param mol: Molecule object containing information about the molecular system.
    :type mol: pyscf.gto.Mole
    :param mf: Mean-field object, obtained from a Dirac-Hartree-Fock (DHF) calculation.
    :type mf: pyscf.scf.hf.SCF
    :param atom_index: Index of the atom for which the PV contribution is computed.
    :type atom_index: int
    :return: Array of PV expectation values for each occupied orbital.
    :rtype: numpy.ndarray (real), shape (nocc,)

    Example:
        mol = gto.Mole()
        # Setup mol with geometry and basis set...
        mf = scf.DHF(mol)
        mf.kernel()
        atom_index = 0
        pv_values = Epv_atom(mol, mf, atom_index)
    """
    n4c, nmo = mf.mo_coeff.shape
    n2c = n4c // 2
    nNeg = nmo // 2  # Molecular orbitals of negative energy
    nocc = mf.mol.nelectron


    # Get the positive components of the MO spinors
    Lo = mf.mo_coeff[:n2c, nNeg:nNeg + nocc]
    So = mf.mo_coeff[n2c:, nNeg:nNeg + nocc]

    # Atom masses and atomic numbers
    masses = mf.mol.atom_mass_list(isotope_avg=False)
    atomic_numbers = mf.mol.atom_charges()

    # Neutrons per atom
    neutrons = masses - atomic_numbers

    S2THETAW = 0.23122  # AS DIRAC24 (CODATA 2018)

    # Weak charge of the nucleus of the selected atom
    QW = (1 - 4 * S2THETAW) * atomic_numbers[atom_index] - neutrons[atom_index]

    # Get the Fermi Contact integrals for the selected atom
    fc_ao = fc_integrals(mol, mf, atom_index)

    # Expectation values
    expval_LS = numpy.einsum('ij,ji->i', Lo.conjugate().T @ fc_ao[:n2c, n2c:], So)
    expval_SL = numpy.einsum('ij,ji->i', So.conjugate().T @ fc_ao[n2c:, :n2c], Lo)

    # Sum LS and SL terms
    expval_perorb = expval_LS + expval_SL

    # Prefactor
    fac = (2.2225 * 10 ** (-14) * QW) / (2 * numpy.sqrt(2))

    return fac * expval_perorb.real



def Epv_molecule(mol, mf):
    """
    Calculate the weak charge parity-violating (PV) contributions for all atoms in the molecule
    within a punctual nuclear charge distribution model.

    This function iterates over all atoms in the molecule and computes the PV contributions
    for each atom.

    :param mol: Molecule object containing information about the molecular system.
    :type mol: pyscf.gto.Mole
    :param mf: Mean-field object, typically obtained from a Dirac-Hartree-Fock (DHF) calculation.
    :type mf: pyscf.scf.hf.SCF
    :return: A 2D array where each row corresponds to an atom and each column contains the PV
             contributions for the occupied orbitals of that atom.
    :rtype: numpy.ndarray (real), shape (n_atoms, n_occ)
    """

    nocc = mf.mol.nelectron
    result = numpy.zeros((mf.mol.natm, nocc))
    for i in range(mf.mol.natm):
        result[i, :] = Epv_atom(mol, mf, i)
    return result

