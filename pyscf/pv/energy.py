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

from pyscf import gto, lib
from pyscf.scf import dhf


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



def gamma5_fc_integrals(mol, mf, atom, **kwargs):
    """
    Compute the integrals for PV contributions to energy for a given atom.
    The integrals are calculated in the four-component spinor basis.
    Point nuclear charge distribution is assumed.

    :param mol: Molecule object containing information about the molecular system.
    :type mol: pyscf.gto.Mole
    :param mf: Mean-field object, obtained from a Dirac-Hartree-Fock (DHF) calculation.
    :type mf: pyscf.scf.hf.SCF
    :param atom: Index of the atom for which the integrals are computed.
    :type atom: int
    :return: Matrix of integrals (4-component spinor form) for the specified atom.
    :rtype: numpy.ndarray (complex128), shape (n4c, n4c)

    Example:
        mol = gto.Mole()
        # Setup mol with geometry and basis set...
        mf = scf.DHF(mol)
        mf.kernel()
        atom_index = 0
        fc_matrix = PV_integrals(mol, mf, atom_index)
    """
    n4c = mf.mo_coeff.shape[0]
    n2c = n4c // 2


    # Get the Fermi Contact integrals for the selected atom
    fc_ao = fc_integrals(mol, mf, atom)

    # Prepare the empty matrix for the FC*Gamma5 integrals (LL, LS, SL, SS blocks)
    gamma5_fc_integrals = numpy.zeros((n4c, n4c), dtype=numpy.complex128)

    # LS and SL blocks
    gamma5_fc_integrals[:n2c, n2c:] = fc_ao[:n2c, n2c:].copy()
    gamma5_fc_integrals[n2c:, :n2c] = fc_ao[n2c:, :n2c:].copy()




    return gamma5_fc_integrals

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
        expval_LL_k = numpy.einsum('i,ij,j->', Lo[:,k].conj(), fc_ao_LL, Lo[:,k])
        expval_SS_k = numpy.einsum('i,ij,j->', So[:,k].conj(), fc_ao_SS, So[:,k])

        expval_perorb[k] = expval_LL_k + expval_SS_k


    return fac * expval_perorb.real



def Epv_atom(mol, mf, atom_index, dm=None):
    """
    Calculate the parity-violating (PV) contribution to energy for a given atom in a molecule.

    The function supports:
    - orbital expectation values from DHF orbitals
    - density matrix contraction in AO or MO basis

    For a one-electron operator:
        E_PV = Tr(D * h_PV)

    If dm is provided:
        - AO density matrices are transformed to MO basis
        - MO density matrices (e.g. MP2/CCSD PySCF densities) are used directly


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

    # Molecular orbitals of negative energy
    nNeg = nmo // 2

    # Number of occupied electrons
    nocc = mf.mol.nelectron

    # ------------------------------------------------------------
    # Nuclear weak charge prefactor
    # ------------------------------------------------------------

    # Atom masses and atomic numbers
    masses = mf.mol.atom_mass_list(isotope_avg=False)
    atomic_numbers = mf.mol.atom_charges()

    # Neutrons per atom
    neutrons = masses - atomic_numbers

    S2THETAW = 0.23122  # (CODATA 2018)

    # Weak charge of the nucleus of the selected atom
    QW = (1 - 4 * S2THETAW) * atomic_numbers[atom_index] - neutrons[atom_index]

    # Prefactor
    fac = (2.2225 * 10 ** (-14) * QW) / (2 * numpy.sqrt(2))

    if dm is None:
        # ------------------------------------------------------------
        # PV operator in AO basis
        # ------------------------------------------------------------

        # Get the Fermi Contact integrals for the selected atom
        fc_ao = fc_integrals(mol, mf, atom_index)

        # ------------------------------------------------------------
        # Orbital expectation values (4c-DHF or 4c-DFT reference)
        # ------------------------------------------------------------

        # Get the positive components of the MO spinors
        Lo = mf.mo_coeff[:n2c, nNeg:nNeg + nocc]
        So = mf.mo_coeff[n2c:, nNeg:nNeg + nocc]

        # Expectation values
        expval_LS = numpy.einsum('ij,ji->i', Lo.conjugate().T @ fc_ao[:n2c, n2c:], So)
        expval_SL = numpy.einsum('ij,ji->i', So.conjugate().T @ fc_ao[n2c:, :n2c], Lo)
        expval_perorb = expval_LS + expval_SL

    # ------------------------------------------------------------
    # Density matrix contraction
    # ------------------------------------------------------------

    if dm is not None:

        # Positive-energy MO coefficient matrix
        C = mf.mo_coeff[:, nNeg:]

        nmo_pos = C.shape[1]

        # Get the Gamma5 Fermi-Contact integrals for the selected atom (in AO basis)
        gamma5_fc_ao = gamma5_fc_integrals(mol, mf, atom_index)

        # Transform operator AO -> MO
        # Operator transformation:
        #
        # hᴹᴼ = C† hᴬᴼ C
        #
        # where:
        #   C   : MO coefficient matrix
        #   hᴬᴼ : operator matrix in AO basis
        #   hᴹᴼ : operator matrix in MO basis
        gamma5_fc_mo = C.conj().T @ gamma5_fc_ao @ C

        # --------------------------------------------------------
        # Determine density matrix representation
        # --------------------------------------------------------

        if dm.shape[0] == n4c:
            # Transform AO density matrix to MO
            # 𝑫ᴹᴼ = C† S 𝑫ᴬᴼ S C
            #
            # where:
            #   C  : MO coefficient matrix
            #   S  : AO overlap matrix
            #   𝑫  : one-particle density matrix
            S = mf.get_ovlp()
            dm_mo = C.conj().T @ S @ dm @ S @ C

        elif dm.shape[0] == nmo_pos:
            # Density already in MO basis
            # (MP2/CCSD densities)
            dm_mo = dm
        else:
            raise ValueError(
                "Density matrix dimension does not match AO or positive-energy MO space"
            )

        # Total expectation value
        # E = Tr(𝑫 h)
        #expval_dm = numpy.trace(dm_mo @ gamma5_fc_mo)

        # Orbital decomposition in MO basis
        # For correlated densities (MP2/CCSD), the orbital decomposition
        # is representation-dependent. Only the total trace is invariant.
        expval_perorb = numpy.einsum(
            'ij,ji->i',
            gamma5_fc_mo,
            dm_mo
        )


    # ------------------------------------------------------------
    # Return
    # ------------------------------------------------------------

    return fac * expval_perorb.real







def Epv_molecule(mol, mf, dm=None):
    """
    Calculate the weak charge parity-violating (PV) contributions for all atoms in the molecule
    within a punctual nuclear charge distribution model.

    This function iterates over all atoms in the molecule and computes the PV contributions
    for each atom.

    If dm is None:
        returns orbital contributions from the reference DHF/DFT orbitals.

    If dm is provided:
        returns contributions obtained by contracting the PV operator with
        the supplied density matrix (AO or MO representation).


    :param mol: Molecule object containing information about the molecular system.
    :type mol: pyscf.gto.Mole
    :param mf: Mean-field object, typically obtained from a Dirac-Hartree-Fock (DHF) calculation.
    :type mf: pyscf.scf.hf.SCF
    :return: A 2D array where each row corresponds to an atom and each column contains the PV
             contributions for the occupied orbitals of that atom.
    :rtype: numpy.ndarray (real), shape (n_atoms, n_occ)
    """

    natm = mf.mol.natm

    # Determine output size from the calculation type
    if dm is None:
        n4c, nmo = mf.mo_coeff.shape
        nocc = mf.mol.nelectron
        ncol = nocc

    else:
        n4c, nmo = mf.mo_coeff.shape
        nNeg = nmo // 2
        nmo_pos = nmo - nNeg

        # AO or MO density
        if dm.shape[0] == n4c or dm.shape[0] == nmo_pos:
            ncol = nmo_pos

        else:
            raise ValueError(
                "Density matrix dimension does not match AO "
                "or positive-energy MO space"
            )


    result = numpy.zeros((natm, ncol))

    for i in range(natm):
        result[i, :] = Epv_atom(mol, mf, i, dm)

    return result

