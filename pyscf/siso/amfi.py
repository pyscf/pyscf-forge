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

# Author: Bhavnesh Jangid <jangidbhavnesh@uchicago.edu>

'''
SOC Integral Generation Based on Atomic Mean-field Approximation (AMFI)
and One-Center Approximation for both 1e and 2e integrals.
'''

import copy
import scipy
import numpy as np
from functools import reduce
from pyscf import lib, ao2mo
from pyscf.scf import atom_hf
from pyscf.data import elements

logger = lib.logger

def _diagonalize(mat):
    eigval, eigvec = np.linalg.eigh(mat)
    return eigval, eigvec

def _sqrt(mat, tol=1e-15):
    e, v = _diagonalize(mat)
    idx = e > tol
    return np.dot(v[:,idx]*np.sqrt(e[idx]), v[:,idx].T.conj())

def _inv(mat):
    return np.linalg.inv(mat)

def _invsqrt(mat, tol=1e-15):
    e, v = np.linalg.eigh(mat)
    idx = e > tol
    return np.dot(v[:,idx]/np.sqrt(e[idx]), v[:,idx].T.conj())

def compute_amfi_dm(mol, atomic_configuration=elements.CONFIGURATION):
    '''
    Source: mrh

    Generate AMFI density matrix, which is exactly like the
    "init_guess_by_atom" density matrix except that the orbitals
    of the atom hf's aren't optimized.
    args:
        mol : pyscf.gto.Mole
            Molecule object
        atomic_configuration : dict
            Atomic configuration for the atoms in the molecule.
            Default is elements.CONFIGURATION, which is a dictionary
            containing the atomic configurations for all elements.
    returns:
        dm : np.array of shape (nao, nao)
            AMFI density matrix
    '''

    with lib.temporary_env(mol, verbose=0):
        atm_scf = atom_hf.get_atm_nrhf(mol, atomic_configuration=atomic_configuration)

    aoslice = mol.aoslice_by_atom()
    atm_dms = []
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        if symb not in atm_scf:
            symb = mol.atom_pure_symbol(ia)
        if symb in atm_scf:
            occ = atm_scf[symb][3]
            dm = np.diag (occ)
        else:  # symb's basis is not specified in the input
            nao_atm = aoslice[ia,3] - aoslice[ia,2]
            dm = np.zeros((nao_atm, nao_atm))

        atm_dms.append(dm)

    dm = scipy.linalg.block_diag(*atm_dms)

    if mol.cart:
        cart2sph = mol.cart2sph_coeff(normalized='sp')
        dm = reduce(np.dot, (cart2sph, dm, cart2sph.T))
    for k, v in atm_scf.items():
        lib.logger.debug1(mol, 'Atom %s, E = %.12g', k, v[0])
    return dm

def compute_kinematic_factors(pmol, contr_coeff, ham='DKH'):
    '''
    Computing the kinematical factors.
    E1:  sqrt((2*p**2)*c**2 + c**4)
    R1:  sqrt((E1+c**2)/2*E1)
    R2:  c*R1/(E1+c**2)
    args:
        pmol: pyscf.gto.Mole
            molecule object in pGTO basis
        contr_coeff: np.array (ncGTO, npGTO)
            uncontracted coefficients of the basis functions
        ham: str
            Hamiltonian type, 'bp' or 'dkh'
            bp: Breit-Pauli
            dkh: Douglas-Kroll-Hess up to first order
    returns
        contr_coeff_1:  np.array (npGTO, ncGTO)
            contracted coefficient decorated with kinematical factor of type-1
        contr_coeff_2: np.array (ncGTO, npGTO)
            contracted coefficient decorated with kinematical factor of type-2
    '''
    # Some constants
    c = lib.param.LIGHT_SPEED # in atomic units
    c2 = c * c
    c4 = c2 * c2

    # Kinetic energy and overlap integrals
    t = pmol.intor('int1e_kin')
    s = pmol.intor('int1e_ovlp')

    # Inverse and square root of overlap integrals
    sinv = _inv(s)
    ssqrt = _sqrt(s)
    sinvsq = _invsqrt(s)

    # Orthogonalizing the kinetic energy integral and diagonalizing it
    # to obatin the p2-basis
    t = reduce(np.dot, (sinvsq, t, sinvsq))
    teigval, teigvec = _diagonalize(t)

    # If the Hamiltonian is of type BP, we set the eigenvalues to zero
    if ham=='BP':
        teigval = np.zeros(teigvec.shape[0], dtype=teigvec.dtype)

    # Generate the kinematical factors
    e1 = np.sqrt((2. * teigval)*c2 + c4)
    r1 = np.sqrt((e1+c2)/(2.*e1))
    r2 = c*r1/(e1+c2)

    # Transform the kinematical factors to the real-space basis
    r1_pos = np.einsum('i,ji->ji', r1, teigvec)
    r1_pos = np.dot(r1_pos, teigvec.T)
    r2_pos = np.einsum('i,ji->ji', r2, teigvec)
    r2_pos = np.dot(r2_pos, teigvec.T)

    # Decorate the contraction coefficients with the kinematical factors
    r1_pos = reduce(np.dot,(ssqrt, r1_pos, ssqrt))
    r1_pos = np.dot(sinv, r1_pos)
    contr_coeff1 = np.dot(r1_pos, contr_coeff)

    r2_pos = reduce(np.dot,(ssqrt, r2_pos, ssqrt))
    r2_pos = np.dot(sinv, r2_pos)
    contr_coeff2 = np.dot(r2_pos, contr_coeff)

    del t, s, sinv, ssqrt, sinvsq, teigval, teigvec
    del e1, r1, r2, r1_pos, r2_pos

    return contr_coeff1, contr_coeff2

def compute_soc2e_jk(pmol, dm0, mo1, mo3):
    """
    Compute the 2e SOC Integrals in pGTO basis, transform them to cGTO basis using
    contraction coefficients, and compute the mean-field like terms, i.e.
    J and K matrices based on the density matrix.

    args:
        pmol: pyscf.gto.Mole
            molecule object in pGTO basis
            nao=ncGTO: int
                number of contracted GTOs
            npGTO: int
                number of primitive GTOs
        dm0: np.array (nao, nao)
            density matrix
        mo1: np.array (npGTO, ncGTO)
            contraction coefficients of type-1
        mo3: np.array (npGTO, ncGTO)
            contraction coefficients of type-2
    Return:
        vj: np.array (nao, nao)
            J-like matrix
        vk: np.array (nao, nao)
            K-like matrix
    """

    # The way it is implemented in OpenMolcas
    # hso2e_ = ao2mo.general(pmol, (mo3, mo1, mo3, mo1), \
    #                       intor='int2e_p1vxp1',aosym='s1',comp=3,compact=True)
    # hso2e = np.asarray([ao2mo.restore(1, hso2e_[i], nao) for i in range(3)])
    # vj = np.einsum('yijkl,lk->yij',hso2e,dm0)
    # hso2e_=ao2mo.general(pmol, (mo3, mo1, mo1, mo3),\
    #                       intor='int2e_p1vxp1',aosym='s1',comp=3,compact=True)
    # hso2e=0.25* np.asarray([ao2mo.restore(1, hso2e_[i], nao) for i in range(3)])
    # hso2e_=ao2mo.general(pmol, (mo1, mo1, mo3, mo3), \
    #                       intor='int2e_p1vxp1',aosym='s1',comp=3,compact=True)
    # hso2e+=0.25*np.asarray([ao2mo.restore(1, hso2e_[i], nao) for i in range(3)])
    # hso2e_=ao2mo.general(pmol, (mo3, mo3, mo1, mo1), \
    #                       intor='int2e_p1vxp1',aosym='s1',comp=3,compact=True)
    # hso2e+=0.25*np.asarray([ao2mo.restore(1, hso2e_[i], nao) for i in range(3)])
    # hso2e_=ao2mo.general(pmol, (mo1, mo3, mo3, mo1), \
    #                       intor='int2e_p1vxp1',aosym='s1',comp=3,compact=True)
    # hso2e+=0.25*np.asarray([ao2mo.restore(1, hso2e_[i], nao) for i in range(3)])
    # vk=np.einsum('yijkl,jk->yil', hso2e, dm0)
    # vk+=np.einsum('yijkl,li->ykj', hso2e, dm0)

    nao = dm0.shape[0]

    # Directly using the ao2mo.general to get the 2e integrals in the cGTO basis

    # Columb like matrix
    vj = np.zeros((3, nao, nao))
    hso2e_ = ao2mo.general(pmol, (mo3, mo1, mo3, mo1),
                           intor='int2e_p1vxp1',aosym='s1',comp=3,compact=True)
    vj = np.asarray([np.einsum('ijkl,lk->ij', ao2mo.restore(1, hso2e_[i], nao),
                                   dm0) for i in range(3)])

    # Exchange like matrix
    hso2e_ = ao2mo.general(pmol, (mo3, mo1, mo1, mo3),
                         intor='int2e_p1vxp1',aosym='s1',comp=3,compact=True)

    vk = np.zeros_like(vj)
    for i in range(3):
        hso2etemp = ao2mo.restore(1, hso2e_[i], nao)
        vk[i] = np.einsum('ijkl,jk->il', hso2etemp, dm0)
        vk[i] += np.einsum('ijkl,li->kj', hso2etemp, dm0)

    del hso2e_

    return vj, vk

def compute_hso1(mol, ham='DKH'):
    '''
    Compute the 1e SOC integrals in pGTO basis and transform them to cGTO basis
    using decorated contraction coefficients.
    args:
        mol: pyscf.gto.Mole
            molecule object
            nao=ncGTO: int
                number of contracted GTOs
            npGTO: int
                number of primitive GTOs
        ham: str
            Hamiltonian type, 'bp' or 'dkh'
            bp: Breit-Pauli
            dkh: Douglas-Kroll-Hess up to first order
    returns:
        hso1: np.array (3, nao, nao)
            1e SOC integrals in cGTO basis
    '''
    nao = mol.nao_nr()
    hso1e = np.zeros((3,nao,nao))
    aoslice = mol.aoslice_by_atom()
    atoms = copy.copy(mol)

    for i in range(mol.natm):
        b0, b1, p0, p1 = aoslice[i]
        atoms._bas = mol._bas[b0:b1]
        pmol, ctr_coeff = atoms.decontract_basis()
        contr_coeff = scipy.linalg.block_diag(*ctr_coeff)
        contr_coeff2 = compute_kinematic_factors(pmol,contr_coeff, ham=ham)[1]
        pmol.set_rinv_orig(mol.atom_coord(i))
        atom_1e = pmol.intor('int1e_prinvxp', comp=3)
        hso1etemp = -1. * atom_1e * (mol.atom_charge(i))
        hso1e[:,p0:p1,p0:p1] += np.einsum('ij,yjl,lm->yim',
                                          contr_coeff2.T, hso1etemp, contr_coeff2)
    return hso1e

def compute_hso2(mol, dm0, ham='DKH'):
    '''
    Compute the 2e SOC integrals in pGTO basis and transform them to cGTO basis
    using decorated contraction coefficients.
    args:
        mol: pyscf.gto.Mole
            molecule object
            nao=ncGTO: int
                number of contracted GTOs
            npGTO: int
                number of primitive GTOs
        dm0: np.array (nao, nao)
            density matrix
        ham: str
            Hamiltonian type, 'bp' or 'dkh'
            bp: Breit-Pauli
            dkh: Douglas-Kroll-Hess up to first order
    returns:
        vj: np.array (3, nao, nao)
            J-like matrix in cGTO basis
        vk: np.array (3, nao, nao)
            K-like matrix in cGTO basis
    '''
    nao = mol.nao_nr()
    aoslice = mol.aoslice_by_atom()
    vj = np.zeros((3, nao, nao))
    vk = np.zeros((3, nao, nao))
    atom = copy.copy(mol)

    for ia in range(mol.natm):
        b0, b1, p0, p1 = aoslice[ia]
        atom._bas = mol._bas[b0:b1]
        pmol, ctr_coeff = atom.decontract_basis()
        ctr_coeff = scipy.linalg.block_diag(*ctr_coeff)
        contr_coeff1, contr_coeff2 = compute_kinematic_factors(pmol,ctr_coeff, ham=ham)
        vj1, vk1 = compute_soc2e_jk(pmol, dm0[p0:p1, p0:p1], contr_coeff1, contr_coeff2)
        vj[:, p0:p1, p0:p1] = vj1
        vk[:, p0:p1, p0:p1] = vk1

    # SOC 2e integrals
    hso2e = vj - vk*1.5
    del vj, vk
    return hso2e

def compute_soc_integrals(mol, dm, ham='DKH'):
    '''
    Computing the 1e + 2e SOC integrals.
    args:
        mol: pyscf.gto.Mole
            molecule object
        dm: np.array (nao, nao)
            density matrix of parent wavefunction.
        ham: str
            Hamiltonian type, 'bp' or 'dkh'
            bp: Breit-Pauli
            dkh: Douglas-Kroll-Hess up to first order
    return:
        hso: tuple ((3, nao, nao), (3, nao, nao))
            1e and 2e SOC integrals in cGTO basis
    '''
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mol, mol.verbose)
    hso1e = compute_hso1(mol, ham=ham)
    log.timer("1e SOC integrals generation took:", *cput0)
    hso2e = compute_hso2(mol, dm, ham=ham)
    log.timer("2e integrals generation took: ", *cput0)
    return 1j * 2. * hso1e, 1j * 2. * hso2e
