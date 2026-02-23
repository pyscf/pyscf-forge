# Copyright 2024 The PySCF Developers. All Rights Reserved.
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
# Author: Zehao Zhou  zehaozhoucase@gmail.com


# import
import numpy as np
import gc, sys, os, h5py, scipy

from pyscf import gto, lib, scf
from pyscf.tdscf import parameter, math_helper, spectralib, _krylov_tools
from pyscf.tdscf.math_helper import get_avail_cpumem, get_mem_info
from pyscf.data.nist import HARTREE2EV
from pyscf.lib import logger, einsum

CITATION_INFO = """
Please cite the TDDFT-ris method:

    1.  Zhou, Zehao, Fabio Della Sala, and Shane M. Parker.
        Minimal auxiliary basis set approach for the electronic excitation spectra
        of organic molecules. The Journal of Physical Chemistry Letters
        14, no. 7 (2023): 1968-1976.
        (must cite)

    2.  Zhou, Zehao, and Shane M. Parker.
        Converging Time-Dependent Density Functional Theory Calculations in Five Iterations
        with Minimal Auxiliary Preconditioning. Journal of Chemical Theory and Computation
        20, no. 15 (2024): 6738-6746.
        (for efficient orbital truncation technique)

    3.  Giannone, Giulia, and Fabio Della Sala.
        Minimal auxiliary basis set for time-dependent density functional theory and
        comparison with tight-binding approximations: Application to silver nanoparticles.
        The Journal of Chemical Physics 153, no. 8 (2020).
        (TDDFT-ris is for hybrid functionals, originates from TDDFT-as with pure functional)
"""

LINEAR_EPSILON = 1e-8


def get_minimal_auxbasis(auxmol_basis_keys, theta, fitting_basis, excludeHs=False):
    '''
    Args:
        auxmol_basis_keys: (['C1', 'H2', 'O3', 'H4', 'H5', 'H6'])
        theta: float 0.2
        fitting_basis: str ('s','sp','spd')

    return:
        aux_basis:
        C1 [[0, [0.1320292535005648, 1.0]]]
        H2 [[0, [0.1999828038466018, 1.0]]]
        O3 [[0, [0.2587932305664396, 1.0]]]
        H4 [[0, [0.1999828038466018, 1.0]]]
        H5 [[0, [0.1999828038466018, 1.0]]]
        H6 [[0, [0.1999828038466018, 1.0]]]
    '''
    aux_basis = {}

    for atom_index in auxmol_basis_keys:
        atom = ''.join([char for char in atom_index if char.isalpha()])

        if excludeHs:
            if atom == 'H':
                continue
        '''
        exponent_alpha = theta/R^2
        '''
        exp_alpha = parameter.ris_exp[atom] * theta

        if 's' in fitting_basis:
            aux_basis[atom_index] = [[0, [exp_alpha, 1.0]]]

        if atom != 'H':
            if 'p' in fitting_basis:
                aux_basis[atom_index].append([1, [exp_alpha, 1.0]])
            if 'd' in fitting_basis:
                aux_basis[atom_index].append([2, [exp_alpha, 1.0]])

    return aux_basis

def get_auxmol(mol, theta=0.2, fitting_basis='s', excludeHs=False):
    """
    Assigns a minimal auxiliary basis set to the molecule.

    Args:
        mol: The input molecule object.
        theta: The scaling factor for the exponents.
        fitting_basis: Basis set type ('s', 'sp', 'spd').

    Returns:
        auxmol: The molecule object with assigned auxiliary basis.
    """


    '''
    parse_arg = False
    turns off PySCF built-in parsing function
    '''
    auxmol = mol.copy()
    auxmol.verbose=0
    auxmol_basis_keys = mol._basis.keys()
    auxmol.basis = get_minimal_auxbasis(auxmol_basis_keys, theta, fitting_basis,excludeHs=excludeHs)
    auxmol.build(dump_input=False, parse_arg=False)
    return auxmol


'''
            n_occ          n_vir
       -|-------------||-------------|
        |             ||             |
  n_occ |   3c2e_ij   ||  3c2e_ia    |
        |             ||             |
        |             ||             |
       =|=============||=============|
        |             ||             |
  n_vir |             ||  3c2e_ab    |
        |             ||             |
        |             ||             |
       -|-------------||-------------|
'''

def get_PuvCupCvq_to_Ppq(eri3c: np.ndarray, C_p: np.ndarray, C_q: np.ndarray, in_ram: bool = False):
    '''
    eri3c : (P|pq) , P = auxnao or 3
    C_p and C_q:  C[:, :n_occ] or C[:, n_occ:], can be both

    Ppq = einsum("Puv,up,vq->Ppq", eri3c, Cp, C_q)
    '''
    P, u, v = eri3c.shape
    eri3c = eri3c.reshape(P * v, u)
    tmp = eri3c.dot(C_p)
    tmp = tmp.reshape(P, v, C_p.shape[1])
    Ppq = einsum('Pvp,vq->Ppq', tmp, C_q)
    del tmp
    return Ppq

def get_uvPCupCvq_to_Ppq(eri3c: np.ndarray, C_pT: np.ndarray, C_q: np.ndarray):
    '''
    eri3c : (uv|P) , P = nauxao
    C_p and C_q:  C[:, :n_occ] or C[:, n_occ:], can be both

    Ppq = einsum("uvP,up,vq->Ppq", eri3c, Cp, C_q)
    '''
    nao, nao, nauxao = eri3c.shape
    size_p,nao = C_pT.shape
    nao, size_q = C_q.shape
    # eri3c = eri3c.reshape(nao,  nao * nauxao)  # (u, vP)
    tmp = C_pT.dot(eri3c)  #(p,u) (u, vP) -> p,vP
    Ppq = einsum('pvP,vq->Ppq', tmp, C_q)

    del tmp
    return Ppq


def einsum2dot(a, b):
    P, Q = a.shape
    # assert P == Q
    Q, p, q = b.shape
    b_2d = b.reshape(Q, p*q)

    ab = a.dot(b_2d)

    del b_2d

    result = ab.reshape(P, p, q)

    del ab
    gc.collect()

    # result = einsum('PQ,Qpq->Ppq', a, b)

    return result


def calculate_batches_by_basis_count(mol, max_nbf_per_batch):
    """
    slice the mol into batches by the number of basis functions
    Args:
        mol: pyscf.gto.Mole
        max_nbf_per_batch: the maximum number of basis functions per batch
    Returns:
        a list of tuples, each tuple is (start_shell, end_shell), representing the shell range of each batch
    """
    ao_loc = mol.ao_loc  # mapping from shell to basis functions
    nbas = mol.nbas      # number of shells

    batches = []
    current_batch_start = 0
    current_nbf = 0  # the cumulative number of basis functions in the current batch

    for i in range(nbas):
        # the number of basis functions in the current shell
        nbf_in_shell = ao_loc[i+1] - ao_loc[i]
        if current_nbf + nbf_in_shell > max_nbf_per_batch:
            # if the current batch exceeds the basis function limit, end the current batch
            batches.append((current_batch_start, i))
            current_batch_start = i
            current_nbf = 0

        # accumulate the number of basis functions in the current shell
        current_nbf += nbf_in_shell

    # handle the last batch
    if current_batch_start < nbas:
        batches.append((current_batch_start, nbas))

    return batches


# def get_Tpq1(mol, auxmol, lower_inv_eri2c, C_p, C_q,
#            calc='JK',omega=None, alpha=None, beta=None,
#            log=None, in_ram=True, single=True):
#     """
#     (3c2e_{Puv}, C_{up}, C_{vq} -> Ppq)。

#     Parameters:
#         mol: pyscf.gto.Mole
#         auxmol: pyscf.gto.Mole
#         C_p: np.ndarray (nao, p)
#         C_q: np.ndarray  (nao, q)

#         lower_inv_eri2c is the inverse of the lower part of the 2-center Coulomb integral
#         in the case of RSH, lower_inv_eri2c already includes the RSH factor when parsed into this function
#         thus lower_inv_eri2c do not need specific processing

#     Returns:
#         Tpq: np.ndarray (naux, nao, nao)
#     """

#     nao = mol.nao
#     naux = auxmol.nao

#     pmol = mol + auxmol
#     pmol.cart = mol.cart

#     tag = '_cart' if mol.cart else '_sph'

#     siz_p = C_p.shape[1]
#     siz_q = C_q.shape[1]

#     int3c_dtype = np.dtype(np.float32 if single else np.float64)
#     log.info(f'int3c_dtype: {int3c_dtype}')

#     if 'J' in calc:
#         Pia = np.empty((naux, siz_p, siz_q), dtype=int3c_dtype)

#     if 'K' in calc:
#         '''only store lower triangle of Tij and Tab'''
#         n_tri_p = (siz_p * (siz_p + 1)) // 2
#         n_tri_q = (siz_q * (siz_q + 1)) // 2
#         Pij = np.empty((naux, n_tri_p), dtype=int3c_dtype)
#         Pab = np.empty((naux, n_tri_q), dtype=int3c_dtype)

#         tril_indices_p = np.tril_indices(siz_p)
#         tril_indices_q = np.tril_indices(siz_q)

#     byte_eri3c = nao * nao * int3c_dtype.itemsize

#     n_eri3c_per_aux = nao * nao * 1
#     n_Ppq_per_aux = siz_p * nao  + siz_p * siz_q * 1.5

#     bytes_per_aux = (  n_eri3c_per_aux + n_Ppq_per_aux) * int3c_dtype.itemsize
#     batch_size = min(naux, max(16, int(get_avail_cpumem() * 0.5 // bytes_per_aux)) )

#     DEBUG = False
#     if DEBUG:
#         batch_size = 2

#     log.info(f'eri3c per aux dimension will take {byte_eri3c / 1024**2:.0f} MB memory')
#     log.info(f'eri3c per aux batch will take {byte_eri3c * batch_size / 1024**2:.0f} MB memory')
#     log.info(get_mem_info('before generate int3c2e'))

#     upper_inv_eri2c = lower_inv_eri2c.T

#     if 'K' in calc:
#         eri2c_inv = lower_inv_eri2c.dot(upper_inv_eri2c)
#         eri2c_inv = eri2c_inv.astype(int3c_dtype, copy=False)


#     if omega and omega != 0:

#         log.info(f'omega {omega}')
#         mol_omega = mol.copy()
#         auxmol_omega = auxmol.copy()

#         mol_omega.omega = omega
#         auxmol_omega.omega = omega

#         pmol_omega = mol_omega + auxmol_omega
#         pmol_omega.cart = mol.cart


#     batches = calculate_batches_by_basis_count(auxmol, batch_size)

#     p1 = 0
#     for start_shell, end_shell in batches:
#         shls_slice = (
#             0, mol.nbas,            # First dimension (mol)
#             0, mol.nbas,            # Second dimension (mol)
#             mol.nbas + start_shell,   # Start of aux basis
#             mol.nbas + end_shell      # End of aux basis (exclusive)
#         )

#         eri3c_batch = pmol.intor('int3c2e' + tag, shls_slice=shls_slice)

#         aux_batch_size = eri3c_batch.shape[2]

#         p0, p1 = p1, p1 + aux_batch_size

#         if omega and omega != 0:
#             eri3c_batch_omega = pmol_omega.intor('int3c2e' + tag, shls_slice=shls_slice)
#             ''' eri3c_batch_tmp = alpha * eri3c_batch_tmp + beta * eri3c_batch_omega_tmp '''
#             eri3c_batch       *= alpha
#             eri3c_batch_omega *= beta
#             eri3c_batch       += eri3c_batch_omega
#             del eri3c_batch_omega

#         DEBUG = False
#         if DEBUG:
#             ''' generate full eri3c_batch and compare with incore.aux_e2(mol, auxmol) '''
#             from pyscf.df import incore
#             ref = incore.aux_e2(mol, auxmol)
#             log.info(f'eri3c_batch.shape {eri3c_batch.shape}')
#             if omega and omega != 0:
#                 mol_omega = mol.copy()
#                 auxmol_omega = auxmol.copy()
#                 mol_omega.omega = omega
#                 ref_omega = incore.aux_e2(mol_omega, auxmol_omega)
#                 ref = alpha * ref + beta * ref_omega
#                 log.info(f'eref.shape {ref.shape}')
#             log.info(f'-------------eri3c DEBUG: out vs .incore.aux_e2(mol, auxmol) {abs(eri3c_batch-ref).max()}')
#             assert abs(eri3c_batch-ref).max() < 1e-10

#         eri3c_batch = eri3c_batch.astype(int3c_dtype, copy=False)
#         eri3c_batch = eri3c_batch.transpose(2, 1, 0)

#         '''Puv -> Ppq, AO->MO transform '''
#         if 'J' in calc:
#             Pia[p0:p1,:,:] = get_PuvCupCvq_to_Ppq(eri3c_batch,C_p,C_q)

#         if 'K' in calc:
#             Pij_tmp = get_PuvCupCvq_to_Ppq(eri3c_batch,C_p,C_p)
#             Pij_lower = Pij_tmp[:, tril_indices_p[0], tril_indices_p[1]].reshape(Pij_tmp.shape[0], -1)
#             del Pij_tmp
#             gc.collect()

#             Pij[p0:p1,:] = Pij_lower
#             del Pij_lower
#             gc.collect()

#             Pab_tmp = get_PuvCupCvq_to_Ppq(eri3c_batch,C_q,C_q)
#             Pab_lower = Pab_tmp[:, tril_indices_q[0], tril_indices_q[1]].reshape(Pab_tmp.shape[0], -1)
#             del Pab_tmp
#             gc.collect()

#             Pab[p0:p1,:] = Pab_lower
#             del Pab_lower
#             gc.collect()

#         last_reported = 0
#         progress = int(100.0 * p1 / naux)

#         if progress % 20 == 0 and progress != last_reported:
#             log.last_reported = progress
#             log.info(f'get_Tpq batch {p1} / {naux} done ({progress} percent). aux_batch_size: {aux_batch_size}')


#     log.info(f' get_Tpq {calc} all batches processed')
#     log.info(get_mem_info('after generate Ppq'))

#     if calc == 'J':
#         Tia = einsum2dot(upper_inv_eri2c, Pia)
#         return Tia

#     if calc == 'K':
#         Tij = eri2c_inv.dot(Pij, out=Pij)
#         Tab = Pab
#         return Tij, Tab

#     if calc == 'JK':
#         Tia = einsum2dot(upper_inv_eri2c, Pia)
#         Tij = eri2c_inv.dot(Pij, out=Pij)
#         Tab = Pab
#         return Tia, Tij, Tab


def get_Tpq(mol, auxmol, lower_inv_eri2c, C_p_a=None, C_q_a=None, C_p_b=None, C_q_b=None, RKS=False, UKS=False,
           calc='JK',omega=None, alpha=None, beta=None,
           log=None, in_ram=True, single=True):
    """
    (3c2e_{Puv}, C_{up}, C_{vq} -> Ppq)。

    Parameters:
        mol: pyscf.gto.Mole
        auxmol: pyscf.gto.Mole
        C_p: np.ndarray (nao, p)
        C_q: np.ndarray  (nao, q)

        lower_inv_eri2c is the inverse of the lower part of the 2-center Coulomb integral
        in the case of RSH, lower_inv_eri2c already includes the RSH factor when parsed into this function
        thus lower_inv_eri2c do not need specific processing

    Returns:
        Tpq: np.ndarray (naux, nao, nao)
    """
    if RKS and UKS:
        raise ValueError("RKS and UKS cannot be both True, only one can be selected")
    if not RKS and not UKS:
        raise ValueError("RKS and UKS must have at least one True")

    nao = mol.nao
    naux = auxmol.nao

    pmol = mol + auxmol
    pmol.cart = mol.cart

    tag = '_cart' if mol.cart else '_sph'

    int3c_dtype = np.dtype(np.float32 if single else np.float64)
    log.info(f'int3c_dtype: {int3c_dtype}')

    upper_inv_eri2c = lower_inv_eri2c.T
    if 'K' in calc:
        eri2c_inv = lower_inv_eri2c.dot(upper_inv_eri2c)
        eri2c_inv = eri2c_inv.astype(int3c_dtype, copy=False)


    if RKS:
        C_p = C_p_a
        C_q = C_q_a
        siz_p = C_p.shape[1]
        siz_q = C_q.shape[1]

        n_Ppq_per_aux = siz_p * nao  + siz_p * siz_q * 1.5 # 1.5 for lower triangle in the case of K

        if 'J' in calc:
            Pia = np.empty((naux, siz_p, siz_q), dtype=int3c_dtype)

        if 'K' in calc:
            '''only store lower triangle of Tij and Tab'''
            n_tri_p = (siz_p * (siz_p + 1)) // 2
            n_tri_q = (siz_q * (siz_q + 1)) // 2
            Pij = np.empty((naux, n_tri_p), dtype=int3c_dtype)
            Pab = np.empty((naux, n_tri_q), dtype=int3c_dtype)

            tril_indices_p = np.tril_indices(siz_p)
            tril_indices_q = np.tril_indices(siz_q)

    elif UKS:
        siz_p_a = C_p_a.shape[1]
        siz_q_a = C_q_a.shape[1]
        siz_p_b = C_p_b.shape[1]
        siz_q_b = C_q_b.shape[1]

        n_Ppq_per_aux = siz_p_a * nao  + siz_p_a * siz_q_a * 1.5 + siz_p_b * nao  + siz_p_b * siz_q_b * 1.5

        if 'J' in calc:
            Pia_a = np.empty((naux, siz_p_a, siz_q_a), dtype=int3c_dtype)
            Pia_b = np.empty((naux, siz_p_b, siz_q_b), dtype=int3c_dtype)
        if 'K' in calc:
            '''only store lower triangle of Tij and Tab'''
            n_tri_p_a = (siz_p_a * (siz_p_a + 1)) // 2
            n_tri_q_a = (siz_q_a * (siz_q_a + 1)) // 2
            n_tri_p_b = (siz_p_b * (siz_p_b + 1)) // 2
            n_tri_q_b = (siz_q_b * (siz_q_b + 1)) // 2
            Pij_a = np.empty((naux, n_tri_p_a), dtype=int3c_dtype)
            Pab_a = np.empty((naux, n_tri_q_a), dtype=int3c_dtype)
            Pij_b = np.empty((naux, n_tri_p_b), dtype=int3c_dtype)
            Pab_b = np.empty((naux, n_tri_q_b), dtype=int3c_dtype)

            tril_indices_p_a = np.tril_indices(siz_p_a)
            tril_indices_q_a = np.tril_indices(siz_q_a)
            tril_indices_p_b = np.tril_indices(siz_p_b)
            tril_indices_q_b = np.tril_indices(siz_q_b)

    byte_eri3c = nao * nao * int3c_dtype.itemsize
    n_eri3c_per_aux = nao * nao * 1
    bytes_per_aux = (  n_eri3c_per_aux + n_Ppq_per_aux) * int3c_dtype.itemsize
    batch_size = min(naux, max(1, int(get_avail_cpumem() * 0.5 // bytes_per_aux)) )
    log.info(f'batch_size: {batch_size}')

    DEBUG = False
    if DEBUG:
        batch_size = 2

    log.info(f'eri3c per aux dimension will take {byte_eri3c / 1024**2:.0f} MB memory')
    log.info(f'eri3c per aux batch will take {byte_eri3c * batch_size / 1024**2:.0f} MB memory')
    log.info(get_mem_info('before generate int3c2e'))

    if omega and omega != 0:

        log.info(f'omega {omega}')
        mol_omega = mol.copy()
        auxmol_omega = auxmol.copy()

        mol_omega.omega = omega
        auxmol_omega.omega = omega

        pmol_omega = mol_omega + auxmol_omega
        pmol_omega.cart = mol.cart


    batches = calculate_batches_by_basis_count(auxmol, batch_size)

    p1 = 0
    for start_shell, end_shell in batches:
        shls_slice = (
            0, mol.nbas,            # First dimension (mol)
            0, mol.nbas,            # Second dimension (mol)
            mol.nbas + start_shell,   # Start of aux basis
            mol.nbas + end_shell      # End of aux basis (exclusive)
        )

        eri3c_batch = pmol.intor('int3c2e' + tag, shls_slice=shls_slice)

        aux_batch_size = eri3c_batch.shape[2]

        p0, p1 = p1, p1 + aux_batch_size

        if omega and omega != 0:
            eri3c_batch_omega = pmol_omega.intor('int3c2e' + tag, shls_slice=shls_slice)
            ''' eri3c_batch_tmp = alpha * eri3c_batch_tmp + beta * eri3c_batch_omega_tmp '''
            eri3c_batch       *= alpha
            eri3c_batch_omega *= beta
            eri3c_batch       += eri3c_batch_omega
            del eri3c_batch_omega

        DEBUG = False
        if DEBUG:
            ''' generate full eri3c_batch and compare with incore.aux_e2(mol, auxmol) '''
            from pyscf.df import incore
            ref = incore.aux_e2(mol, auxmol)
            log.info(f'eri3c_batch.shape {eri3c_batch.shape}')
            if omega and omega != 0:
                mol_omega = mol.copy()
                auxmol_omega = auxmol.copy()
                mol_omega.omega = omega
                ref_omega = incore.aux_e2(mol_omega, auxmol_omega)
                ref = alpha * ref + beta * ref_omega
                log.info(f'eref.shape {ref.shape}')
            log.info(f'-------------eri3c DEBUG: out vs .incore.aux_e2(mol, auxmol) {abs(eri3c_batch-ref).max()}')
            assert abs(eri3c_batch-ref).max() < 1e-10

        tmp_astype = eri3c_batch.astype(int3c_dtype, copy=False)
        del eri3c_batch
        eri3c_batch = tmp_astype.transpose(2, 1, 0)
        del tmp_astype


        '''Puv -> Ppq, AO->MO transform '''
        if RKS:
            if 'J' in calc:
                Pia_tmp = get_PuvCupCvq_to_Ppq(eri3c_batch,C_p,C_q)
                Pia[p0:p1,:,:] = Pia_tmp
                del Pia_tmp

            if 'K' in calc:
                Pij_tmp = get_PuvCupCvq_to_Ppq(eri3c_batch,C_p,C_p)
                Pij_lower = Pij_tmp[:, tril_indices_p[0], tril_indices_p[1]].reshape(Pij_tmp.shape[0], -1)
                del Pij_tmp

                Pij[p0:p1,:] = Pij_lower
                del Pij_lower

                Pab_tmp = get_PuvCupCvq_to_Ppq(eri3c_batch,C_q,C_q)
                Pab_lower = Pab_tmp[:, tril_indices_q[0], tril_indices_q[1]].reshape(Pab_tmp.shape[0], -1)
                del Pab_tmp

                Pab[p0:p1,:] = Pab_lower
                del Pab_lower

        elif UKS:
            if 'J' in calc:
                Pia_a[p0:p1,:,:] = get_PuvCupCvq_to_Ppq(eri3c_batch,C_p_a,C_q_a)
                Pia_b[p0:p1,:,:] = get_PuvCupCvq_to_Ppq(eri3c_batch,C_p_b,C_q_b)
            if 'K' in calc:
                Pij_a_tmp = get_PuvCupCvq_to_Ppq(eri3c_batch,C_p_a,C_p_a)
                Pij_a_lower = Pij_a_tmp[:, tril_indices_p_a[0], tril_indices_p_a[1]].reshape(Pij_a_tmp.shape[0], -1)
                del Pij_a_tmp
                Pij_a[p0:p1,:] = Pij_a_lower
                del Pij_a_lower

                Pij_b_tmp = get_PuvCupCvq_to_Ppq(eri3c_batch,C_p_b,C_p_b)
                Pij_b_lower = Pij_b_tmp[:, tril_indices_p_b[0], tril_indices_p_b[1]].reshape(Pij_b_tmp.shape[0], -1)
                del Pij_b_tmp
                Pij_b[p0:p1,:] = Pij_b_lower
                del Pij_b_lower

                Pab_a_tmp = get_PuvCupCvq_to_Ppq(eri3c_batch,C_q_a,C_q_a)
                Pab_a_lower = Pab_a_tmp[:, tril_indices_q_a[0], tril_indices_q_a[1]].reshape(Pab_a_tmp.shape[0], -1)
                del Pab_a_tmp
                Pab_a[p0:p1,:] = Pab_a_lower
                del Pab_a_lower

                Pab_b_tmp = get_PuvCupCvq_to_Ppq(eri3c_batch,C_q_b,C_q_b)
                Pab_b_lower = Pab_b_tmp[:, tril_indices_q_b[0], tril_indices_q_b[1]].reshape(Pab_b_tmp.shape[0], -1)
                del Pab_b_tmp
                Pab_b[p0:p1,:] = Pab_b_lower
                del Pab_b_lower
        del eri3c_batch
        gc.collect()

        last_reported = 0
        progress = int(100.0 * p1 / naux)
        if progress % 20 == 0 and progress != last_reported:
            log.info(f'get_Tpq batch {p1} / {naux} done ({progress} percent). aux_batch_size: {aux_batch_size}')


    log.info(f' get_Tpq {calc} all batches processed')
    log.info(get_mem_info('after generate Ppq'))

    if RKS:
        if calc == 'J':
            Tia = einsum2dot(upper_inv_eri2c, Pia)
            # Tia = einsum('PQ,Qpq->Ppq', upper_inv_eri2c, Pia)
            del Pia
            gc.collect()
            return Tia

        if calc == 'K':
            Tij = eri2c_inv.dot(Pij, out=Pij)
            Tab = Pab
            del Pij
            gc.collect()
            return Tij, Tab

        if calc == 'JK':
            Tia = einsum2dot(upper_inv_eri2c, Pia)
            Tij = eri2c_inv.dot(Pij, out=Pij)
            Tab = Pab
            del Pia, Pij
            gc.collect()
            return Tia, Tij, Tab

    elif UKS:
        if calc == 'J':
            Tia_a = einsum2dot(upper_inv_eri2c, Pia_a)
            Tia_b = einsum2dot(upper_inv_eri2c, Pia_b)
            del Pia_a, Pia_b
            gc.collect()
            return Tia_a, Tia_b

        if calc == 'K':
            Tij_a = eri2c_inv.dot(Pij_a, out=Pij_a)
            Tij_b = eri2c_inv.dot(Pij_b, out=Pij_b)
            Tab_a = Pab_a
            Tab_b = Pab_b
            del Pij_a, Pij_b
            gc.collect()
            return Tij_a, Tij_b, Tab_a, Tab_b

        if calc == 'JK':
            Tia_a = einsum2dot(upper_inv_eri2c, Pia_a)
            Tia_b = einsum2dot(upper_inv_eri2c, Pia_b)
            Tij_a = eri2c_inv.dot(Pij_a, out=Pij_a)
            Tij_b = eri2c_inv.dot(Pij_b, out=Pij_b)
            Tab_a = Pab_a
            Tab_b = Pab_b
            del Pia_a, Pia_b
            gc.collect()
            return Tia_a, Tia_b, Tij_a, Tij_b, Tab_a, Tab_b


def get_eri2c_inv_lower(auxmol, omega=0, alpha=None, beta=None, dtype=np.float64):

    tag = '_cart' if auxmol.cart else '_sph'

    eri2c = auxmol.intor('int2c2e'+ tag)

    if omega and omega != 0:

        with auxmol.with_range_coulomb(omega):
            eri2c_erf = auxmol.intor('int2c2e'+ tag)

        eri2c = alpha * eri2c + beta * eri2c_erf

    try:
        ''' we want lower_inv_eri2c = X
                X X.T = eri2c^-1
                (X X.T)^-1 = eri2c
                (X.T)^-1 X^-1 = eri2c = L L.T
                (X.T)^-1 = L
        need to solve  L_inv = L^-1
                X = L_inv.T

        '''
        L = np.linalg.cholesky(eri2c)
        L_inv = scipy.linalg.solve_triangular(L, np.eye(L.shape[0]), lower=True)
        lower_inv_eri2c = L_inv.T

    except scipy.linalg.LinAlgError:
        ''' lower_inv_eri2c = eri2c ** -0.5
            LINEAR_EPSILON = 1e-8 to remove the linear dependency, sometimes the aux eri2c is not full rank.
        '''
        lower_inv_eri2c = math_helper.matrix_power(eri2c,-0.5,epsilon=LINEAR_EPSILON)

    lower_inv_eri2c = np.asarray(lower_inv_eri2c, dtype=dtype, order='C')
    return lower_inv_eri2c


def gen_hdiag_MVP(hdiag, n_occ, n_vir):
    def hdiag_MVP(V):
        m = V.shape[0]
        V = np.asarray(V)
        V = V.reshape(m, n_occ*n_vir)

        hdiag_v = hdiag[None,:] * V
        hdiag_v = hdiag_v.reshape(m, n_occ, n_vir)
        return hdiag_v

    return hdiag_MVP

def gen_iajb_MVP_Tpq(T_ia, T_jb=None, log=None):
    '''
    (ia|jb)V = Σ_Pjb (T_ia^P T_jb^P V_jb^m)
             = Σ_P [ T_ia^P Σ_jb(T_jb^P V_jb^m) ]

    if T_ia == T_jb, then it is either
        (1) (ia|jb) in RKS
        or
        (2)(ia_α|jb_α) or (ia_β|jb_β) in UKS,

    elif T_ia != T_jb
        it is (ia_α|jb_β) or (ia_β|jb_α) in UKS

    V in shape (m, n_occ * n_vir)
    '''
    if T_jb is None:
        T_jb = T_ia

    def iajb_MVP1(V, factor, out=None):
        '''for debugging'''
        T_right_jb_V = einsum("Pjb,mjb->Pm", T_jb, V)
        iajb_V = einsum("Pia,Pm->mia", T_ia, T_right_jb_V)
        if out is None:
            out = np.zeros_like(V)
        iajb_V *= factor
        out += iajb_V
        return out

    def iajb_MVP(V, factor, out=None):
        '''
        Optimized calculation of (ia|jb) = Σ_Pjb (T_left_ia^P T_right_jb^P V_jb^m)
        by chunking along the auxao dimension to reduce memory usage.

        Parameters:
            V (np.ndarray): Input tensor of shape (m, n_occ * n_vir).
            results are accumulated in out if provided.

        Returns:
            iajb_V (np.ndarray): Result tensor of shape (m, n_occ, n_vir).
        '''
        # Get the shape of the tensors
        cpu0 = (logger.process_clock(), logger.perf_counter())
        nauxao, n_occ, n_vir = T_ia.shape
        n_state, n_occ, n_vir = V.shape
        # Initialize result tensor
        if out is None:
            out = np.zeros_like(V)

        # 1 denotes one auxao, we are slucing the auxao dimension.
        n_Tia_chunk = 1 * n_occ * n_vir
        n_TjbVjb_chunk = 1 * n_state
        n_iajb_V_chunk = n_state * n_occ * n_vir

        estimated_chunk_size_bytes = (n_Tia_chunk + n_TjbVjb_chunk + n_iajb_V_chunk) * T_ia.itemsize

        # Estimate the optimal chunk size based on available GPU memory
        aux_batch_size = int(get_avail_cpumem() * 0.8 // estimated_chunk_size_bytes)

        # Ensure the chunk size is at least 1 and doesn't exceed the total number of auxao
        aux_batch_size = max(1, min(nauxao, aux_batch_size))

        # Iterate over chunks of the auxao dimension
        for aux_start in range(0, nauxao, aux_batch_size):
            aux_end = min(aux_start + aux_batch_size, nauxao)
            Tjb_chunk = T_jb[aux_start:aux_end, :, :]

            Tjb_Vjb_chunk = einsum("Pjb,mjb->Pm", Tjb_chunk, V)

            if T_jb is T_ia:
                Tia_chunk = Tjb_chunk                     # Shape: (aux_range, n_occ, n_vir)
            else:
                Tia_chunk = T_ia[aux_start:aux_end, :, :] # Shape: (aux_range, n_occ, n_vir)
            tmp = einsum("Pia,Pm->mia", Tia_chunk, Tjb_Vjb_chunk)
            del Tia_chunk, Tjb_Vjb_chunk, Tjb_chunk
            tmp *= factor
            out += tmp
            del tmp
            gc.collect()

        log.timer(' iajb_MVP time', *cpu0)
        return out

    return iajb_MVP

def gen_ijab_MVP_Tpq(T_ij, T_ab, log=None):
    '''
    (ij|ab)V = Σ_Pjb (T_ij^P T_ab^P V_jb^m)
             = Σ_P [T_ij^P Σ_jb(T_ab^P V_jb^m)]
    V in shape (m, n_occ * n_vir)
    '''

    nauxao, n_tri_ij = T_ij.shape
    nauxao, n_tri_ab = T_ab.shape

    n_occ = int((-1 + (1 + 8 * n_tri_ij)**0.5) / 2)
    n_vir = int((-1 + (1 + 8 * n_tri_ab)**0.5) / 2)

    tril_indices_occ = np.tril_indices(n_occ)
    tril_indices_vir = np.tril_indices(n_vir)

    def ijab_MVP_simple(X, a_x, out=None):
        '''for debugging'''
        T_ab_full = np.empty((nauxao, n_vir, n_vir),dtype=T_ab.dtype)
        T_ab_full[:, tril_indices_vir[0], tril_indices_vir[1]] = T_ab
        T_ab_full[:, tril_indices_vir[1], tril_indices_vir[0]] = T_ab

        T_ab_X = einsum("Pab,mjb->Pamj", T_ab_full, X)
        del T_ab_full

        T_ij_full = np.empty((nauxao, n_occ, n_occ),dtype=T_ij.dtype)
        T_ij_full[:, tril_indices_occ[0], tril_indices_occ[1]] = T_ij
        T_ij_full[:, tril_indices_occ[1], tril_indices_occ[0]] = T_ij

        tmp = einsum("Pij,Pamj->mia", T_ij_full, T_ab_X)
        del T_ij_full, T_ab_X

        tmp *= -a_x
        out += tmp
        del tmp
        gc.collect()
        return out


    def ijab_MVP(X, a_x, out=None):
        '''
        Optimized calculation of (ij|ab) = Σ_Pjb (T_ij^P T_ab^P X_jb^m)
        by chunking along the P (nauxao) dimension for both T_ij and T_ab,
        slice chunks to reduce memory usage.

        Parameters:
            X (np.ndarray): Input tensor of shape (n_state, n_occ, n_vir).
            a_x (float): Scaling factor.
            out (np.ndarray, optional): Output tensor of shape (n_state, n_occ, n_vir).

        Returns:
            ijab_X (np.ndarray): Result tensor of shape (n_state, n_occ, n_vir).
        '''

        cpu0 = (logger.process_clock(), logger.perf_counter())
        n_state, n_occ, n_vir = X.shape    # Dimensions of X

        # Initialize result tensor
        if out is None:
            out = np.zeros_like(X)

        # Get free memory and dynamically calculate chunk size

        log.info(get_mem_info('          ijab_MVP start'))

        # Memory estimation for one P index
        n_T_ab_chunk = 1 * n_vir * n_vir * 1.5  # T_ab chunk: (1, n_vir, n_vir)
        n_T_ij_chunk = 1 * n_occ * n_occ * 1.5 # T_ij chunk: (1, n_occ, n_occ)
        n_T_ab_chunk_X = 1 * n_state * n_occ * n_vir  # T_ab_X chunk: (1, n_state, n_occ)
        n_ijab_chunk_X = n_state * n_occ * n_vir  # Full output size (accumulated)

        bytes_per_P = max(n_T_ab_chunk + n_T_ab_chunk_X,  n_T_ij_chunk + n_T_ab_chunk_X ) * T_ab.itemsize
        # log.info(f'bytes_per_P {bytes_per_P}')
        mem_alloc = int((get_avail_cpumem() * 0.7 - n_ijab_chunk_X * T_ab.itemsize))
        P_chunk_size = min(nauxao, max(1, mem_alloc // bytes_per_P))
        log.info(f'    ijab with Tij Tab, P_chunk_size = {P_chunk_size}')
        # Iterate over chunks of the P (nauxao) dimension
        for P_start in range(0, nauxao, P_chunk_size):
            P_end = min(P_start + P_chunk_size, nauxao)

            # log.info(get_mem_info(f'  ijab {P_start,P_end}'))
            # Extract corresponding chunks of T_ab and T_ij
            T_ab_chunk_lower = T_ab[P_start:P_end, :]  # Shape: (P_chunk_size, (n_vir*n_vir+1)//2 )

            # Compute T_ab_X for the current chunk
            T_ab_chunk = np.empty((T_ab_chunk_lower.shape[0], n_vir, n_vir),dtype=T_ab_chunk_lower.dtype)
            T_ab_chunk[:, tril_indices_vir[0], tril_indices_vir[1]] = T_ab_chunk_lower
            T_ab_chunk[:, tril_indices_vir[1], tril_indices_vir[0]] = T_ab_chunk_lower
            del T_ab_chunk_lower
            gc.collect()
            T_ab_chunk_X = einsum("Pab,mjb->Pamj", T_ab_chunk, X)
            del T_ab_chunk
            gc.collect()

            T_ij_chunk_lower =T_ij[P_start:P_end, :]  # Shape: (P_chunk_size, (n_occ*n_occ+1)//2 )
            T_ij_chunk = np.empty((T_ij_chunk_lower.shape[0], n_occ, n_occ),dtype=T_ij_chunk_lower.dtype)
            T_ij_chunk[:, tril_indices_occ[0], tril_indices_occ[1]] = T_ij_chunk_lower
            T_ij_chunk[:, tril_indices_occ[1], tril_indices_occ[0]] = T_ij_chunk_lower
            del T_ij_chunk_lower
            gc.collect()

            # out = einsum("Pij,Pamj->mia", T_ij_chunk, T_ab_chunk_X, -a_x, 1, out=out)
            tmp = einsum("Pij,Pamj->mia", T_ij_chunk, T_ab_chunk_X)
            del T_ij_chunk, T_ab_chunk_X
            tmp *= -a_x
            out += tmp
            del tmp

            gc.collect()
            # Release intermediate variables and clean up memory
        log.timer(' ijab_MVP time', *cpu0)
        log.info(get_mem_info('          ijab_MVP done'))
        return out

    return ijab_MVP

def gen_ibja_MVP_Tpq(T_ia, log=None):
    '''
    the exchange (ib|ja) in B matrix
    (ib|ja) = Σ_Pjb (T_ib^P T_ja^P V_jb^m)
            = Σ_P [T_ja^P Σ_jb(T_ib^P V_jb^m)]
    '''
    # def ibja_MVP(V):
    #     T_ib_V = einsum("Pib,mjb->Pimj", T_ia, V)
    #     ibja_V = einsum("Pja,Pimj->mia", T_ia, T_ib_V)
    #     return ibja_V

    def ibja_MVP(V, a_x, out=None):
        '''
        Optimized calculation of (ib|ja) = Σ_Pjb (T_ib^P T_ja^P V_jb^m)
        by chunking along the n_occ dimension to reduce memory usage.

        Parameters:
            V (np.ndarray): Input tensor of shape (n_state, n_occ, n_vir).
            occ_chunk_size (int): Chunk size for splitting the n_occ dimension.

        Returns:
            ibja_V (np.ndarray): Result tensor of shape (n_state, n_occ, n_vir).
        '''
        cpu0 = (logger.process_clock(), logger.perf_counter())
        nauxao, n_occ, n_vir = T_ia.shape
        n_state, n_occ, n_vir = V.shape

        bytes_per_aux = (n_occ * n_vir * 1 + n_state * n_occ * n_vir ) * T_ia.itemsize

        batch_size = max(1, int(get_avail_cpumem() * 0.8 // bytes_per_aux))

        # Initialize result tensor
        # ibja_V = np.empty((n_state, n_occ, n_vir), dtype=T_ia.dtype)
        if out is None:
            out = np.zeros_like(V)
        # Iterate over chunks of the n_occ dimension
        for p0 in range(0, nauxao, batch_size):
            p1 = min(p0+batch_size, nauxao)

            # Extract the corresponding chunk of T_ia
            T_ib_chunk = T_ia[p0:p1, :, :]  # Shape: (batch_size, n_occ, n_vir)
            T_jb_chunk = T_ib_chunk

            T_ib_V_chunk = einsum("Pib,mjb->mPij", T_ib_chunk, V)

            # out = einsum("Pja,mPij->mia", T_jb_chunk, T_ib_V_chunk, -a_x, 1, out=out)
            tmp = einsum("Pja,mPij->mia", T_jb_chunk, T_ib_V_chunk)
            del T_jb_chunk, T_ib_V_chunk, T_ib_chunk
            tmp *= -a_x
            out += tmp
            del tmp

            gc.collect()

        log.timer(' ibja_MVP time', *cpu0)
        return out

    return ibja_MVP

def get_ab(td, mf, J_fit, K_fit, theta, mo_energy=None, mo_coeff=None, mo_occ=None, singlet=True):
    r'''A and B matrices for TDDFT response function.

    A[i,a,j,b] = \delta_{ab}\delta_{ij}(E_a - E_i) + (ai||jb)
    B[i,a,j,b] = (ai||bj)

    Ref: Chem Phys Lett, 256, 454
    '''
    from pyscf.df import incore

    if mo_energy is None:
        mo_energy = mf.mo_energy
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_occ is None:
        mo_occ = mf.mo_occ

    mo_energy = np.asarray(mo_energy)
    mo_coeff = np.asarray(mo_coeff)
    mo_occ = np.asarray(mo_occ)
    mol = mf.mol
    nao, nmo = mo_coeff.shape
    occidx = np.where(mo_occ==2)[0]
    viridx = np.where(mo_occ==0)[0]
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]
    nvir = orbv.shape[1]
    nocc = orbo.shape[1]
    mo = np.hstack((orbo,orbv))

    e_ia = mo_energy[viridx] - mo_energy[occidx,None]
    a = np.diag(e_ia.ravel()).reshape(nocc,nvir,nocc,nvir)
    b = np.zeros_like(a)
    ni = mf._numint
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    auxmol_J = get_auxmol(mol=mol, theta=theta, fitting_basis=J_fit)
    if K_fit == J_fit and (omega == 0 or omega is None):
        auxmol_K = auxmol_J
    else:
        auxmol_K = get_auxmol(mol=mol, theta=theta, fitting_basis=K_fit)

    def get_erimo(auxmol_i):
        naux = auxmol_i.nao
        int3c = incore.aux_e2(mol, auxmol_i)
        int2c2e = auxmol_i.intor('int2c2e')

        df_coef = np.linalg.solve(int2c2e, int3c.reshape(nao*nao, naux).T)
        df_coef = df_coef.reshape(naux, nao, nao)
        eri = einsum('ijP,Pkl->ijkl', int3c, df_coef)
        eri_mo = einsum('pjkl,pi->ijkl', eri, orbo.conj())
        eri_mo = einsum('ipkl,pj->ijkl', eri_mo, mo)
        eri_mo = einsum('ijpl,pk->ijkl', eri_mo, mo.conj())
        eri_mo = einsum('ijkp,pl->ijkl', eri_mo, mo)
        eri_mo = eri_mo.reshape(nocc,nmo,nmo,nmo)
        return eri_mo
    def get_erimo_omega(auxmol_i, omega):
        naux = auxmol_i.nao
        with mol.with_range_coulomb(omega):
            int3c = incore.aux_e2(mol, auxmol_i)
        with auxmol_i.with_range_coulomb(omega):
            int2c2e = auxmol_i.intor('int2c2e')

        df_coef = np.linalg.solve(int2c2e, int3c.reshape(nao*nao, naux).T)
        df_coef = df_coef.reshape(naux, nao, nao)
        eri = einsum('ijP,Pkl->ijkl', int3c, df_coef)
        eri_mo = einsum('pjkl,pi->ijkl', eri, orbo.conj())
        eri_mo = einsum('ipkl,pj->ijkl', eri_mo, mo)
        eri_mo = einsum('ijpl,pk->ijkl', eri_mo, mo.conj())
        eri_mo = einsum('ijkp,pl->ijkl', eri_mo, mo)
        eri_mo = eri_mo.reshape(nocc,nmo,nmo,nmo)
        return eri_mo
    def add_hf_(a, b, hyb=1):
        eri_mo_J = get_erimo(auxmol_J)
        eri_mo_K = get_erimo(auxmol_K)
        if singlet:
            a += np.einsum('iabj->iajb', eri_mo_J[:nocc,nocc:,nocc:,:nocc]) * 2
            a -= np.einsum('ijba->iajb', eri_mo_K[:nocc,:nocc,nocc:,nocc:]) * hyb
            b += np.einsum('iajb->iajb', eri_mo_J[:nocc,nocc:,:nocc,nocc:]) * 2
            b -= np.einsum('jaib->iajb', eri_mo_K[:nocc,nocc:,:nocc,nocc:]) * hyb
        else:
            a -= np.einsum('ijba->iajb', eri_mo_K[:nocc,:nocc,nocc:,nocc:]) * hyb
            b -= np.einsum('jaib->iajb', eri_mo_K[:nocc,nocc:,:nocc,nocc:]) * hyb

    if getattr(td, 'with_solvent', None):
        raise NotImplementedError("PCM TDDFT RIS is not supported")

    if isinstance(mf, scf.hf.KohnShamDFT):
        add_hf_(a, b, hyb)
        if omega != 0:  # For RSH
            eri_mo_K = get_erimo_omega(auxmol_K, omega)
            k_fac = alpha - hyb
            a -= np.einsum('ijba->iajb', eri_mo_K[:nocc,:nocc,nocc:,nocc:]) * k_fac
            b -= np.einsum('jaib->iajb', eri_mo_K[:nocc,nocc:,:nocc,nocc:]) * k_fac

        if mf.do_nlc():
            raise NotImplementedError('vv10 nlc not implemented in get_ab(). '
                                      'However the nlc contribution is small in TDDFT, '
                                      'so feel free to take the risk and comment out this line.')
    else:
        add_hf_(a, b)

    return a, b

def rescale_spin_free_amplitudes(xy, state_id):
    '''
    Rescales spin-free excitation amplitudes in TDDFT-ris to the normalization
    convention used in standard RKS-TDDFT.

    The original RKS-TDDFT formulation uses excitation amplitudes corresponding to
    the spin-up components only. The TDDFT-RIS implementation employs spin-free
    amplitudes that are not equivalent to the spin-up components and are
    normalized to 1.
    '''
    x, y = xy
    x = x[state_id] * .5**.5
    if y is not None: # TDDFT
        y = y[state_id] * .5**.5
    else: # TDA
        y = np.zeros_like(x)
    return x, y

def as_scanner(td):
    if isinstance(td, lib.SinglePointScanner):
        return td

    logger.info(td, 'Set %s as a scanner', td.__class__)
    name = td.__class__.__name__ + TD_Scanner.__name_mixin__
    return lib.set_class(TD_Scanner(td), (TD_Scanner, td.__class__), name)


class TD_Scanner(lib.SinglePointScanner):
    def __init__(self, td):
        self.__dict__.update(td.__dict__)
        self._scf = td._scf.as_scanner()

    def __call__(self, mol_or_geom, **kwargs):
        assert self.device == 'gpu'
        if isinstance(mol_or_geom, gto.MoleBase):
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)

        self.reset(mol)
        mf_scanner = self._scf
        mf_e = mf_scanner(mol)
        self.n_occ = None
        self.n_vir = None
        self.rest_occ = None
        self.rest_vir = None
        self.C_occ_notrunc = None
        self.C_vir_notrunc = None
        self.C_occ_Ktrunc = None
        self.C_vir_Ktrunc = None
        self.delta_hdiag = None
        self.hdiag = None
        # self.eri_tag = None
        self.auxmol_J = None
        self.auxmol_K = None
        self.lower_inv_eri2c_J = None
        self.lower_inv_eri2c_K = None
        self.RKS = True
        self.UKS = False
        self.mo_coeff = np.asarray(self._scf.mo_coeff, dtype=self.dtype)
        self.build()
        self.kernel()
        return mf_e + self.energies/HARTREE2EV

class RisBase(lib.StreamObject):
    def __init__(self, mf,
                theta: float = 0.2, J_fit: str = 'sp', K_fit: str = 's', excludeHs=False,
                Ktrunc: float = 40.0, full_K_diag: bool = False, a_x: float = None, omega: float = None,
                alpha: float = None, beta: float = None, conv_tol: float = 1e-3,
                nstates: int = 5, max_iter: int = 25, extra_init=8, restart_iter=None, spectra: bool = False,
                out_name: str = '', print_threshold: float = 0.05, gram_schmidt: bool = True,
                single: bool = True,
                verbose=None, citation=True):
        """
        Args:
            mf (object): Mean field object, typically obtained from a ground - state calculation.
            theta (float, optional): Global scaling factor for the fitting basis exponent.
                                The relationship is defined as `alpha = theta/R_A^2`,
                                    where `alpha` is the Gaussian exponent
                                and `R_A` is tabulated semi-empirical radii for element A. Defaults to 0.2.
            J_fit (str, optional): Fitting basis for the J matrix (`iajb` integrals).
                                   's' means only one s orbital per atom,
                                   'sp' means adding one extra p orbital per atom.
                                   Defaults to 'sp', becasue more accurate than s.
            K_fit (str, optional): Fitting basis for the K matrix (`ijab` and `ibja` integrals).
                                  's' means only one s orbital per atom,
                                  'sp' means adding one extra p orbital per atom.
                                   Defaults to 's', becasue 'sp' has no accuracy improvement.
            Ktrunc (float, optional): Truncation threshold for the K matrix. Orbitals are discarded if:
                                    - Occupied orbitals with energies < e_LUMO - Ktrunc
                                    - Virtual orbitals with energies > e_HOMO + Ktrunc. Defaults to 40.0.
            a_x (float, optional): Hartree-Fock component. By default, it will be assigned according
                                    to the `mf.xc` attribute.
                                    Will override the default value if provided.
            omega (float, optional): Range-separated hybrid functional parameter. By default, it will be
                                    assigned according to the `mf.xc` attribute.
                                    Will override the default value if provided.
            alpha (float, optional): Range-separated hybrid functional parameter. By default, it will be
                                    assigned according to the `mf.xc` attribute.
                                    Will override the default value if provided.
            beta (float, optional): Range-separated hybrid functional parameter. By default, it will be
                                    assigned according to the `mf.xc` attribute.
            conv_tol (float, optional): Convergence tolerance for the Davidson iteration. Defaults to 1e-3.
            nstates (int, optional): Number of excited states to be calculated. Defaults to 5.
            max_iter (int, optional): Maximum number of iterations for the Davidson iteration. Defaults to 25.
            spectra (bool, optional): Whether to calculate and dump the excitation spectra in G16 & Multiwfn style.
                                     Defaults to False.
            out_name (str, optional): Output file name for the excitation spectra. Defaults to ''.
            print_threshold (float, optional): Threshold for printing the transition coefficients. Defaults to 0.05.
            gram_schmidt (bool, optional): Whether to calculate the ground state. Defaults to False.
            single (bool, optional): Whether to use single precision. Defaults to True.
            tensor_in_ram (bool, optional): Whether to store Tpq tensors in RAM. Defaults to False.
            krylov_in_ram (bool, optional): Whether to store Krylov vectors in RAM. Defaults to False.
            verbose (optional): Verbosity level of the logger. If None, it will use the verbosity of `mf`.
        """
        self.single = single

        if single:
            self.dtype = np.dtype(np.float32)
        else:
            self.dtype = np.dtype(np.float64)

        self._scf = mf
        # self.chkfile = mf.chkfile
        self.singlet = True # TODO: add R-T excitation.
        self.exclude_nlc = False # TODO: exclude nlc functional
        self.xy = None

        self.theta = theta
        self.J_fit = J_fit
        self.K_fit = K_fit

        self.Ktrunc = Ktrunc
        self._excludeHs = excludeHs
        self._full_K_diag = full_K_diag
        self.a_x = a_x
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.conv_tol = conv_tol
        self.nstates = nstates
        self.max_iter = max_iter
        self.extra_init = extra_init
        self.restart_iter = restart_iter
        self.mol = mf.mol
        # self.mo_coeff = np.asarray(mf.mo_coeff, dtype=self.dtype)
        self.spectra = spectra
        self.out_name = out_name
        self.print_threshold = print_threshold
        self.gram_schmidt = gram_schmidt

        self.verbose = verbose if verbose else mf.verbose

        # self.device = mf.device
        self.converged = None
        # self._store_Tpq_J = store_Tpq_J
        # self._store_Tpq_K = store_Tpq_K

        # self._tensor_in_ram = tensor_in_ram
        # self._krylov_in_ram = krylov_in_ram

        self.log = logger.new_logger(verbose=self.verbose)

        ''' following attributes will be initialized in self.build() '''
        self.n_occ = None
        self.n_vir = None
        self.rest_occ = None
        self.rest_vir = None

        self.C_occ_notrunc = None
        self.C_vir_notrunc = None
        self.C_occ_Ktrunc = None
        self.C_vir_Ktrunc = None

        self.delta_hdiag = None
        self.hdiag = None

        self.eri_tag = '_cart' if self.mol.cart else '_sph'

        self.auxmol_J = None
        self.auxmol_K = None
        self.lower_inv_eri2c_J = None
        self.lower_inv_eri2c_K = None

        self.RKS = True
        self.UKS = False
        self._citation = citation

    def transition_dipole(self):
        '''
        transition dipole u
        xuv -> xia AO to MO
        reshape(3,-1) for xyz three directions, helpful when calculating oscillator strength and polarizability.
        '''
        int_r = self.mol.intor_symmetric('int1e_r' + self.eri_tag)
        int_r = np.asarray(int_r, dtype=self.dtype)
        if self.RKS:
            P = get_PuvCupCvq_to_Ppq(int_r, self.C_occ_notrunc, self.C_vir_notrunc).reshape(3,-1)
        else:
            P_alpha = get_PuvCupCvq_to_Ppq(int_r, self.C_occ_a_notrunc, self.C_vir_a_notrunc).reshape(3,-1)
            P_beta = get_PuvCupCvq_to_Ppq(int_r, self.C_occ_b_notrunc, self.C_vir_b_notrunc).reshape(3,-1)
            P = np.hstack((P_alpha, P_beta))
        return P

    def transition_magnetic_dipole(self):
        '''
        magnatic dipole m
        xuv -> xia AO to MO
        reshape(3,-1) for xyz three directions, helpful when calculating oscillator strength and polarizability.
        '''
        int_rxp = self.mol.intor('int1e_cg_irxp' + self.eri_tag, comp=3, hermi=2)
        int_rxp = np.asarray(int_rxp, dtype=self.dtype)

        if self.RKS:
            mdpol = get_PuvCupCvq_to_Ppq(int_rxp, self.C_occ_notrunc, self.C_vir_notrunc).reshape(3,-1)
        else:
            ''' TODO '''
            mdpol_alpha = get_PuvCupCvq_to_Ppq(int_rxp, self.C_occ_a_notrunc, self.C_vir_a_notrunc).reshape(3,-1)
            mdpol_beta = get_PuvCupCvq_to_Ppq(int_rxp, self.C_occ_b_notrunc, self.C_vir_b_notrunc).reshape(3,-1)
            mdpol = np.hstack((mdpol_alpha, mdpol_beta))
        return mdpol

    @property
    def e_tot(self):
        '''Excited state energies'''
        return self._scf.e_tot + self.energies/HARTREE2EV

    def get_ab(self, mf=None):
        if mf is None:
            mf = self._scf
        J_fit = self.J_fit
        K_fit = self.K_fit
        theta = self.theta
        return get_ab(self, mf, J_fit, K_fit, theta, singlet=True)

    def build(self):
        log = self.log
        log.info("TDA&TDDFT-ris is still in the experimental stage, APIs may subject to change in future releases.")

        log.info(f'nstates: {self.nstates}')
        log.info(f'N atoms:{self.mol.natm}')
        log.info(f'conv_tol: {self.conv_tol}')
        log.info(f'max_iter: {self.max_iter}')
        log.info(f'Ktrunc: {self.Ktrunc}')
        log.info(f'calculate and print UV-vis spectra info: {self.spectra}')
        log.info(get_mem_info('  after init of RisBase'))

        if self.spectra:
            log.info(f'spectra files will be written and their name start with: {self.out_name}')

        if self.a_x or self.omega or self.alpha or self.beta:
            ''' user wants to define some XC parameters '''
            if self.a_x:
                if self.a_x == 0:
                    log.info('use pure XC functional, a_x = 0')
                elif self.a_x > 0 and self.a_x < 1:
                    log.info(f'use hybrid XC functional, a_x = {self.a_x}')
                    if self.single:
                        self.a_x = np.float32(self.a_x)
                elif self.a_x == 1:
                    log.info('use HF, a_x = 1')
                else:
                    log.info('a_x > 1, weird')

            elif self.omega and self.alpha and self.beta:
                log.info('use range-separated hybrid XC functional')
            else:
                raise ValueError('Please dounble check the XC functional parameters')
        else:
            ''' use default XC parameters
                note: the definition of a_x, α and β is kind of weird in pyscf/libxc
            '''
            log.info(f'auto detect functional: {self._scf.xc}')

            omega, alpha_libxc, hyb_libxc = self._scf._numint.rsh_and_hybrid_coeff(
                                                 self._scf.xc, spin=self._scf.mol.spin)
            log.info(f'omega, alpha_libxc, hyb_libxc: {omega}, {alpha_libxc}, {hyb_libxc}')

            if omega > 0:
                log.info('use range-separated hybrid XC functional')
                self.a_x = 1
                self.omega = omega
                self.alpha = hyb_libxc
                self.beta = alpha_libxc - hyb_libxc

            elif omega == 0:
                self.a_x = alpha_libxc
                if self.a_x == 0:
                    log.info('use pure XC functional, a_x = 0')
                elif self.a_x > 0 and self.a_x < 1:
                    log.info(f'use hybrid XC functional, a_x = {self.a_x}')
                elif self.a_x == 1:
                    log.info('use HF, a_x = 1')
                else:
                    log.info('a_x > 1, weird')

        log.info(f'omega: {self.omega}')
        log.info(f'alpha: {self.alpha}')
        log.info(f'beta: {self.beta}')
        log.info(f'a_x: {self.a_x}')
        log.info(f'gram_schmidt: {self.gram_schmidt}')
        log.info(f'single: {self.single}')

        if self.J_fit == self.K_fit:
            log.info(f'use same J and K fitting basis: {self.J_fit}')
        else:
            log.info(f'use different J and K fitting basis: J with {self.J_fit} and K with {self.K_fit}')


        log.info(f'cartesian or spherical electron integral: {self.eri_tag}')

        log.info(get_mem_info('  before process mo_coeff'))

        auxmol_J = get_auxmol(mol=self.mol, theta=self.theta, fitting_basis=self.J_fit)
        log.info(f'n_bf in auxmol_J = {auxmol_J.nao_nr()}')
        self.auxmol_J = auxmol_J

        if self._scf.mo_coeff.ndim == 2:
            self.RKS = True
            self.UKS = False
            n_occ = int(sum(self._scf.mo_occ>0))
            n_vir = int(sum(self._scf.mo_occ==0))
            self.n_occ = n_occ
            self.n_vir = n_vir

            self.C_occ_notrunc = np.asarray(self._scf.mo_coeff[:,:n_occ], dtype=self.dtype, order='F')
            self.C_vir_notrunc = np.asarray(self._scf.mo_coeff[:,n_occ:], dtype=self.dtype, order='F')
            mo_energy = self._scf.mo_energy

            occ_ene = mo_energy[:n_occ].reshape(n_occ,1)
            vir_ene = mo_energy[n_occ:].reshape(1,n_vir)

            delta_hdiag = np.repeat(vir_ene, n_occ, axis=0) - np.repeat(occ_ene, n_vir, axis=1)
            if self.single:
                delta_hdiag = np.asarray(delta_hdiag, dtype=np.float32)

            self.delta_hdiag = delta_hdiag
            self.hdiag = delta_hdiag.reshape(-1)

            log.info(f'n_occ = {n_occ}, E_HOMO ={occ_ene[-1,0]}')
            log.info(f'n_vir = {n_vir}, E_LOMO ={vir_ene[0,0]}')
            log.info(f'H-L gap = {(vir_ene[0,0] - occ_ene[-1,0])*HARTREE2EV:.2f} eV')

            if self.Ktrunc > 0:
                log.info(f' MO truncation in K with threshold {self.Ktrunc} eV above HOMO and below LUMO')

                trunc_tol_au = self.Ktrunc/HARTREE2EV

                homo_vir_delta_ene = delta_hdiag[-1,:]
                occ_lumo_delta_ene = delta_hdiag[:,0]

                rest_occ = int(np.sum(occ_lumo_delta_ene <= trunc_tol_au))
                rest_vir = int(np.sum(homo_vir_delta_ene <= trunc_tol_au))

                assert rest_occ > 0
                assert rest_vir > 0

            elif self.Ktrunc == 0:
                log.info('no MO truncation in K')
                rest_occ = n_occ
                rest_vir = n_vir

            log.info(f'rest_occ = {rest_occ}')
            log.info(f'rest_vir = {rest_vir}')

            self.C_occ_Ktrunc = np.asarray(self._scf.mo_coeff[:,n_occ-rest_occ:n_occ], dtype=self.dtype, order='F')
            self.C_vir_Ktrunc = np.asarray(self._scf.mo_coeff[:,n_occ:n_occ+rest_vir], dtype=self.dtype, order='F')

            self.rest_occ = rest_occ
            self.rest_vir = rest_vir

            byte_T_ia_J = self.auxmol_J.nao_nr() * n_occ * n_vir * self.dtype.itemsize
            log.info(f'FYI, storing T_ia_J will take {byte_T_ia_J / 1024**2:.0f} MB memory')

        elif self._scf.mo_coeff.ndim == 3:
            ''' UKS method '''
            self.RKS = False
            self.UKS = True
            mo_occ_a = self._scf.mo_occ[0]
            mo_occ_b = self._scf.mo_occ[1]
            mo_energy_a = self._scf.mo_energy[0]
            mo_energy_b = self._scf.mo_energy[1]

            n_occ_a = sum(mo_occ_a>0)
            n_vir_a = sum(mo_occ_a==0)

            n_occ_b = sum(mo_occ_b>0)
            n_vir_b = sum(mo_occ_b==0)

            self.n_occ_a = n_occ_a
            self.n_vir_a = n_vir_a

            self.n_occ_b = n_occ_b
            self.n_vir_b = n_vir_b

            mo_coeff_a = self._scf.mo_coeff[0]
            mo_coeff_b = self._scf.mo_coeff[1]

            self.C_occ_a_notrunc = np.asarray(mo_coeff_a[:,:n_occ_a], dtype=self.dtype, order='F')
            self.C_vir_a_notrunc = np.asarray(mo_coeff_a[:,n_occ_a:], dtype=self.dtype, order='F')
            self.C_occ_b_notrunc = np.asarray(mo_coeff_b[:,:n_occ_b], dtype=self.dtype, order='F')
            self.C_vir_b_notrunc = np.asarray(mo_coeff_b[:,n_occ_b:], dtype=self.dtype, order='F')

            occ_ene_a = mo_energy_a[:n_occ_a].reshape(n_occ_a,1)
            vir_ene_a = mo_energy_a[n_occ_a:].reshape(1,n_vir_a)
            occ_ene_b = mo_energy_b[:n_occ_b].reshape(n_occ_b,1)
            vir_ene_b = mo_energy_b[n_occ_b:].reshape(1,n_vir_b)

            delta_hdiag_a = np.repeat(vir_ene_a, n_occ_a, axis=0) - np.repeat(occ_ene_a, n_vir_a, axis=1)
            delta_hdiag_b = np.repeat(vir_ene_b, n_occ_b, axis=0) - np.repeat(occ_ene_b, n_vir_b, axis=1)

            if self.single:
                delta_hdiag_a = np.asarray(delta_hdiag_a, dtype=np.float32)
                delta_hdiag_b = np.asarray(delta_hdiag_b, dtype=np.float32)

            self.delta_hdiag_a = delta_hdiag_a
            self.delta_hdiag_b = delta_hdiag_b
            self.hdiag_a = delta_hdiag_a.reshape(-1)
            self.hdiag_b = delta_hdiag_b.reshape(-1)
            self.hdiag = np.concatenate([self.hdiag_a, self.hdiag_b], axis=0) # total hdiag, for preconditioning

            log.info(f'n_occ_a = {n_occ_a}, E_HOMO_a ={occ_ene_a[-1,0]}')
            log.info(f'n_vir_a = {n_vir_a}, E_LOMO_a ={vir_ene_a[0,0]}')
            log.info(f'n_occ_b = {n_occ_b}, E_HOMO_b ={occ_ene_b[-1,0]}')
            log.info(f'n_vir_b = {n_vir_b}, E_LOMO_b ={vir_ene_b[0,0]}')
            log.info(f'alpha spin H-L gap = {(vir_ene_a[0,0] - occ_ene_a[-1,0])*HARTREE2EV:.2f} eV')
            log.info(f'beta  spin H-L gap = {(vir_ene_b[0,0] - occ_ene_b[-1,0])*HARTREE2EV:.2f} eV')

            if self.Ktrunc > 0:
                log.info(f' MO truncation in K with threshold {self.Ktrunc} eV above HOMO and below LUMO')

                trunc_tol_au = self.Ktrunc/HARTREE2EV

                homo_vir_delta_ene_a = delta_hdiag_a[-1,:]
                occ_lumo_delta_ene_a = delta_hdiag_a[:,0]
                homo_vir_delta_ene_b = delta_hdiag_b[-1,:]
                occ_lumo_delta_ene_b = delta_hdiag_b[:,0]

                rest_occ_a = int(np.sum(occ_lumo_delta_ene_a <= trunc_tol_au))
                rest_vir_a = int(np.sum(homo_vir_delta_ene_a <= trunc_tol_au))
                rest_occ_b = int(np.sum(occ_lumo_delta_ene_b <= trunc_tol_au))
                rest_vir_b = int(np.sum(homo_vir_delta_ene_b <= trunc_tol_au))

                assert rest_occ_a > 0
                assert rest_vir_a > 0
                assert rest_occ_b > 0
                assert rest_vir_b > 0

            elif self.Ktrunc == 0:
                log.info('no MO truncation in K')
                rest_occ_a = n_occ_a
                rest_vir_a = n_vir_a
                rest_occ_b = n_occ_b
                rest_vir_b = n_vir_b

            log.info(f'rest_occ_a = {rest_occ_a}')
            log.info(f'rest_vir_a = {rest_vir_a}')
            log.info(f'rest_occ_b = {rest_occ_b}')
            log.info(f'rest_vir_b = {rest_vir_b}')

            self.C_occ_a_Ktrunc = np.asarray(mo_coeff_a[:,n_occ_a-rest_occ_a:n_occ_a], dtype=self.dtype, order='F')
            self.C_vir_a_Ktrunc = np.asarray(mo_coeff_a[:,n_occ_a:n_occ_a+rest_vir_a], dtype=self.dtype, order='F')
            self.C_occ_b_Ktrunc = np.asarray(mo_coeff_b[:,n_occ_b-rest_occ_b:n_occ_b], dtype=self.dtype, order='F')
            self.C_vir_b_Ktrunc = np.asarray(mo_coeff_b[:,n_occ_b:n_occ_b+rest_vir_b], dtype=self.dtype, order='F')

            self.rest_occ_a = rest_occ_a
            self.rest_vir_a = rest_vir_a
            self.rest_occ_b = rest_occ_b
            self.rest_vir_b = rest_vir_b

            byte_T_ia_J = self.auxmol_J.nao_nr() * (n_occ_a * n_vir_a + n_occ_b * n_vir_b) * self.dtype.itemsize
            log.info(f'storing T_ia_J_a and T_ia_J_b will take {byte_T_ia_J / 1024**2:.0f} MB memory')


        self.lower_inv_eri2c_J = get_eri2c_inv_lower(self.auxmol_J, omega=0)



        if self.a_x != 0:

            auxmol_K = get_auxmol(mol=self.mol, theta=self.theta, fitting_basis=self.K_fit, excludeHs=self._excludeHs)

            log.info(f'n_bf in auxmol_K = {auxmol_K.nao_nr()}')
            self.auxmol_K = auxmol_K

            self.lower_inv_eri2c_K = get_eri2c_inv_lower(auxmol_K, omega=self.omega, alpha=self.alpha, beta=self.beta)


            if self.RKS:
                byte_T_ij_K = auxmol_K.nao_nr() * (self.rest_occ * (self.rest_occ +1) //2 )* self.dtype.itemsize
                byte_T_ab_K = auxmol_K.nao_nr() * (self.rest_vir * (self.rest_vir +1) //2 )* self.dtype.itemsize
                log.info(f'T_ij_K will take {byte_T_ij_K / 1024**2:.0f} MB memory')
                log.info(f'T_ab_K will take {byte_T_ab_K / 1024**2:.0f} MB memory')

                byte_T_ia_K = auxmol_K.nao_nr() * self.rest_occ * self.rest_vir * self.dtype.itemsize
                log.info(f'(if full TDDFT) T_ia_K will take {byte_T_ia_K / 1024**2:.0f} MB memory')

            if self.UKS:
                byte_T_ij_K_a = auxmol_K.nao_nr() * (self.rest_occ_a * (self.rest_vir_a+1)//2) * self.dtype.itemsize
                byte_T_ab_K_a = auxmol_K.nao_nr() * (self.rest_vir_a * (self.rest_vir_a+1)//2 )* self.dtype.itemsize
                byte_T_ij_K_b = auxmol_K.nao_nr() * (self.rest_occ_b * (self.rest_vir_b+1)//2 )* self.dtype.itemsize
                byte_T_ab_K_b = auxmol_K.nao_nr() * (self.rest_vir_b * (self.rest_vir_b+1)//2 )* self.dtype.itemsize

                log.info(f'T_ij_K_a will take {byte_T_ij_K_a / 1024**2:.0f} MB memory')
                log.info(f'T_ab_K_a will take {byte_T_ab_K_a / 1024**2:.0f} MB memory')
                log.info(f'T_ij_K_b will take {byte_T_ij_K_b / 1024**2:.0f} MB memory')
                log.info(f'T_ab_K_b will take {byte_T_ab_K_b / 1024**2:.0f} MB memory')

                byte_T_ia_K_a = auxmol_K.nao_nr() * self.rest_occ_a * self.rest_vir_a * self.dtype.itemsize
                byte_T_ia_K_b = auxmol_K.nao_nr() * self.rest_occ_b * self.rest_vir_b * self.dtype.itemsize
                log.info(f'(if full TDDFT) T_ia_K_a will take {byte_T_ia_K_a / 1024**2:.0f} MB memory')
                log.info(f'(if full TDDFT) T_ia_K_b will take {byte_T_ia_K_b / 1024**2:.0f} MB memory')

        log.info(get_mem_info('  built ris obj'))
        self.log = log

    ''' RKS '''
    def get_T_J_RKS(self):
        log = self.log
        log.info('==================== RIJ RKS ====================')
        cpu0 = (logger.process_clock(), logger.perf_counter())

        T_ia_J = get_Tpq(mol=self.mol, auxmol=self.auxmol_J, lower_inv_eri2c=self.lower_inv_eri2c_J,
                        C_p_a=self.C_occ_notrunc, C_q_a=self.C_vir_notrunc,
                        calc="J", omega=0, RKS=True, UKS=False,
                        single=self.single, log=log)

        log.timer('build T_ia_J', *cpu0)
        log.info(get_mem_info('after T_ia_J_RKS'))
        return T_ia_J

    def get_2T_K_RKS(self):
        log = self.log
        log.info('==================== RIK ====================')
        cpu1 = (logger.process_clock(), logger.perf_counter())

        T_ij_K, T_ab_K = get_Tpq(mol=self.mol, auxmol=self.auxmol_K, lower_inv_eri2c=self.lower_inv_eri2c_K,
                                C_p_a=self.C_occ_Ktrunc, C_q_a=self.C_vir_Ktrunc,
                                calc='K', RKS=True, UKS=False,
                                omega=self.omega, alpha=self.alpha,beta=self.beta,
                                single=self.single,log=log)

        log.timer('T_ij_K T_ab_K', *cpu1)
        log.info(get_mem_info('after get_2T_K_RKS'))
        return T_ij_K, T_ab_K

    def get_3T_K_RKS(self):
        log = self.log
        log.info('==================== RIK ====================')
        cpu1 = (logger.process_clock(), logger.perf_counter())
        T_ia_K, T_ij_K, T_ab_K = get_Tpq(mol=self.mol, auxmol=self.auxmol_K, lower_inv_eri2c=self.lower_inv_eri2c_K,
                                        C_p_a=self.C_occ_Ktrunc, C_q_a=self.C_vir_Ktrunc,
                                        calc='JK', RKS=True, UKS=False,
                                        omega=self.omega, alpha=self.alpha,beta=self.beta,
                                        single=self.single,log=log)

        log.timer('T_ia_K T_ij_K T_ab_K', *cpu1)
        log.info(get_mem_info('after get_3T_K_RKS'))
        return T_ia_K, T_ij_K, T_ab_K

    ''' UKS '''
    def get_T_J_UKS(self):
        log = self.log
        log.info('==================== RIJ UKS ====================')
        cpu0 = (logger.process_clock(), logger.perf_counter())

        T_ia_J_a, T_ia_J_b = get_Tpq(mol=self.mol, auxmol=self.auxmol_J, lower_inv_eri2c=self.lower_inv_eri2c_J,
                        C_p_a=self.C_occ_a_notrunc, C_q_a=self.C_vir_a_notrunc,
                        C_p_b=self.C_occ_b_notrunc, C_q_b=self.C_vir_b_notrunc,
                        calc="J", omega=0, RKS=False, UKS=True,
                        single=self.single, log=log)

        log.timer('build T_ia_J_a, T_ia_J_b', *cpu0)
        log.info(get_mem_info('after get_T_J_UKS'))
        return T_ia_J_a, T_ia_J_b

    def get_2T_K_UKS(self):
        log = self.log
        log.info('==================== RIK ====================')
        cpu1 = (logger.process_clock(), logger.perf_counter())

        T_ij_K_a, T_ij_K_b, T_ab_K_a, T_ab_K_b = get_Tpq(
                                        mol=self.mol, auxmol=self.auxmol_K, lower_inv_eri2c=self.lower_inv_eri2c_K,
                                        C_p_a=self.C_occ_a_Ktrunc, C_q_a=self.C_vir_a_Ktrunc,
                                        C_p_b=self.C_occ_b_Ktrunc, C_q_b=self.C_vir_b_Ktrunc,
                                        calc='K', RKS=False, UKS=True,
                                        omega=self.omega, alpha=self.alpha,beta=self.beta,
                                        single=self.single,log=log)
        log.timer('T_ij_K_a, T_ij_K_b, T_ab_K_a, T_ab_K_b', *cpu1)
        log.info(get_mem_info('after get_2T_K_UKS'))
        return T_ij_K_a, T_ij_K_b, T_ab_K_a, T_ab_K_b

    def get_3T_K_UKS(self):
        log = self.log
        log.info('==================== RIK ====================')
        cpu1 = (logger.process_clock(), logger.perf_counter())
        T_ia_K_a, T_ia_K_b, T_ij_K_a, T_ij_K_b, T_ab_K_a, T_ab_K_b = get_Tpq(
                                        mol=self.mol, auxmol=self.auxmol_K, lower_inv_eri2c=self.lower_inv_eri2c_K,
                                        C_p_a=self.C_occ_a_Ktrunc, C_q_a=self.C_vir_a_Ktrunc,
                                        C_p_b=self.C_occ_b_Ktrunc, C_q_b=self.C_vir_b_Ktrunc,
                                        calc='JK', RKS=False, UKS=True,
                                        omega=self.omega, alpha=self.alpha,beta=self.beta,
                                        single=self.single,log=log)

        log.timer('T_ia_K_a, T_ia_K_b, T_ij_K_a, T_ij_K_b, T_ab_K_a, T_ab_K_b', *cpu1)
        log.info(get_mem_info('after get_3T_K_UKS'))
        return T_ia_K_a, T_ia_K_b, T_ij_K_a, T_ij_K_b, T_ab_K_a, T_ab_K_b

    def Gradients(self):
        raise NotImplementedError

    def nuc_grad_method(self):
        return self.Gradients()

    def NAC(self):
        raise NotImplementedError

    def nac_method(self):
        return self.NAC()

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        self._scf.reset(mol)
        return self

    as_scanner = as_scanner

    def get_nto(self,state_id, save_fch=False):

        ''' dump NTO coeff'''
        orbo = self.C_occ_notrunc
        orbv = self.C_vir_notrunc
        nocc = self.n_occ
        nvir = self.n_vir

        log = self.log
        # X
        cis_t1 = self.xy[0][state_id-1, :].copy()
        log.info(f'state_id {state_id}')
        log.info(f'X norm {np.linalg.norm(cis_t1):.3f}')
        # TDDFT (X,Y) has X^2-Y^2=1.
        # Renormalizing X (X^2=1) to map it to CIS coefficients
        # cis_t1 *= 1. / np.linalg.norm(cis_t1)
        cis_t1 = cis_t1.reshape(nocc, nvir)

        nto_o, w, nto_vT = np.linalg.svd(cis_t1)
        '''each column of nto_o and nto_v corresponds to one NTO pair
        usually the first (few) NTO pair have significant weights
        '''

        w_squared = w**2
        dominant_weight = float(w_squared[0]) # usually ~1.0
        log.info(f"Dominant NTO weight: {dominant_weight:.4f} (should be close to 1.0)")

        hole_nto = nto_o[:, 0]      # shape: (nocc,)
        particle_nto = nto_vT[0, :].T  # shape: (nvir,)

        # Phase convention: max abs coeff positive, and consistent phase between hole/particle
        if hole_nto[np.argmax(np.abs(hole_nto))] < 0:
            hole_nto = -hole_nto
            particle_nto = -particle_nto

        occupied_nto_ao = orbo.dot(hole_nto)    # shape: (nao,)
        virtual_nto_ao = orbv.dot(particle_nto) # shape: (nao,)

        nto_coeff = np.hstack((occupied_nto_ao[:,None], virtual_nto_ao[:,None]))


        if save_fch:
            '''save nto_coeff to fch file'''
            try:
                from mokit.lib.py2fch_direct import fchk
                from mokit.lib.rwwfn import del_dm_in_fch
            except ImportError:
                info = 'mokit is not installed. Please install mokit to save nto_coeff to fch file'
                info += 'https://gitlab.com/jxzou/mokit'
                raise ImportError(info)
            nto_mf = self._scf.copy().to_cpu()
            nto_mf.mo_coeff = nto_coeff
            nto_mf.mo_energy = np.asarray([dominant_weight, dominant_weight])

            fchfilename = f'ntopair_{state_id}.fch'
            if os.path.exists(fchfilename):
                os.remove(fchfilename)
            fchk(nto_mf, fchfilename)
            del_dm_in_fch(fchname=fchfilename,itype=1)
            log.info(f'nto_coeff saved to {fchfilename}')
            log.info('Please cite MOKIT: https://gitlab.com/jxzou/mokit')

        else:
            '''save nto_coeff to h5 file'''
            with h5py.File(f'nto_coeff_{state_id}.h5', 'w') as f:
                f.create_dataset('nto_coeff', data=nto_coeff, dtype='f4')
                f.create_dataset('dominant_weight', data=dominant_weight, dtype='f4')
                f.create_dataset('state_id', data=state_id, dtype='i4')
            log.info(f'nto_coeff saved to {fchfilename}')

        return dominant_weight, nto_coeff

'''
        With MO truncation, most the occ and vir orbitals (transition pair) are neglected in the exchange part

        As shown below, * denotes the included transition pair
                     -------------------
                   /                  /
    original X =  /                  /  nstates
                 -------------------
                |******************|
         n_occ  |******************|
                |******************|
                |******************|
                |------------------|
                    n_vir
becomes:
                     -------------------
                   /                  /
            X' =  /                  /  nstates
                 -------------------
                |                  |
 n_occ-rest_occ |                  |
                |-----|------------|
                |*****|            |
    rest_occ    |*****|            |
                |-----|------------|
                rest_vir

            (If no MO truncation, then n_occ-rest_occ=0 and rest_vir=n_vir)
'''
class TDA(RisBase):
    def __init__(self, mf, **kwargs):
        super().__init__(mf, **kwargs)
        log = self.log
        log.info('TDA-ris initialized')

    ''' ===========  RKS hybrid =========== '''
    def get_RKS_TDA_hybrid_MVP(self):
        ''' TDA RKS hybrid '''
        log = self.log
        n_occ = self.n_occ
        n_vir = self.n_vir
        rest_occ = self.rest_occ
        rest_vir = self.rest_vir
        a_x = self.a_x

        # J
        T_ia_J = self.get_T_J_RKS()
        iajb_MVP = gen_iajb_MVP_Tpq(T_ia=T_ia_J, T_jb=T_ia_J, log=log)

        # K
        T_ij_K, T_ab_K = self.get_2T_K_RKS()
        ijab_MVP = gen_ijab_MVP_Tpq(T_ij=T_ij_K, T_ab=T_ab_K, log=log)

        hdiag_MVP = gen_hdiag_MVP(hdiag=self.hdiag, n_occ=n_occ, n_vir=n_vir)

        def RKS_TDA_hybrid_MVP(X):
            ''' hybrid or range-sparated hybrid, a_x > 0
                return AX
                AX = hdiag_MVP(X) + 2*iajb_MVP(X) - a_x*ijab_MVP(X)
                for RSH, a_x = 1
            '''
            nstates = X.shape[0]
            X = X.reshape(nstates, n_occ, n_vir)

            cpu0 = (logger.process_clock(), logger.perf_counter())
            log.info(get_mem_info('       TDA MVP before hdiag_MVP'))
            out = hdiag_MVP(X)
            log.timer('--hdiag_MVP', *cpu0)

            cpu0 = (logger.process_clock(), logger.perf_counter())
            X_trunc = X[:,n_occ-rest_occ:,:rest_vir]
            ijab_MVP(X_trunc, a_x=a_x, out=out[:,n_occ-rest_occ:,:rest_vir])
            del X_trunc

            log.timer('--ijab_MVP', *cpu0)


            cpu0 = (logger.process_clock(), logger.perf_counter())

            iajb_MVP(X, factor=2,out=out)


            log.timer('--iajb_MVP', *cpu0)
            log.info(get_mem_info('       TDA MVP after iajb'))


            out = out.reshape(nstates, n_occ*n_vir)
            return out

        return RKS_TDA_hybrid_MVP, self.hdiag

    ''' ===========  RKS pure =========== '''
    def get_RKS_TDA_pure_MVP(self):
        '''hybrid RKS TDA'''
        log = self.log
        hdiag_MVP = gen_hdiag_MVP(hdiag=self.hdiag, n_occ=self.n_occ, n_vir=self.n_vir)

        T_ia_J = self.get_T_J_RKS()
        iajb_MVP = gen_iajb_MVP_Tpq(T_ia=T_ia_J, log=log)

        def RKS_TDA_pure_MVP(X):
            ''' pure functional, a_x = 0
                return AX
                AV = hdiag_MVP(V) + 2*iajb_MVP(V)
            '''
            nstates = X.shape[0]
            X = X.reshape(nstates, self.n_occ, self.n_vir)
            out = hdiag_MVP(X)
            cpu0 = (logger.process_clock(), logger.perf_counter())
            # AX += 2 * iajb_MVP(X)
            iajb_MVP(X, factor=2,out=out)
            log.timer('--iajb_MVP', *cpu0)
            out = out.reshape(nstates, self.n_occ*self.n_vir)
            return out

        return RKS_TDA_pure_MVP, self.hdiag

    ''' ===========  UKS hybrid =========== '''
    def get_UKS_TDA_hybrid_MVP(self):

        log = self.log
        n_occ_a = self.n_occ_a
        n_vir_a = self.n_vir_a
        n_occ_b = self.n_occ_b
        n_vir_b = self.n_vir_b
        rest_occ_a = self.rest_occ_a
        rest_vir_a = self.rest_vir_a
        rest_occ_b = self.rest_occ_b
        rest_vir_b = self.rest_vir_b
        a_x = self.a_x

        # J
        T_ia_J_a, T_ia_J_b = self.get_T_J_UKS()

        iajb_MVP_aa = gen_iajb_MVP_Tpq(T_ia=T_ia_J_a, T_jb=T_ia_J_a, log=log)
        iajb_MVP_bb = gen_iajb_MVP_Tpq(T_ia=T_ia_J_b, T_jb=T_ia_J_b, log=log)
        iajb_MVP_ab = gen_iajb_MVP_Tpq(T_ia=T_ia_J_a, T_jb=T_ia_J_b, log=log)
        iajb_MVP_ba = gen_iajb_MVP_Tpq(T_ia=T_ia_J_b, T_jb=T_ia_J_a, log=log)

        # K
        T_ij_K_a, T_ij_K_b, T_ab_K_a, T_ab_K_b = self.get_2T_K_UKS()
        ijab_MVP_aa = gen_ijab_MVP_Tpq(T_ij=T_ij_K_a, T_ab=T_ab_K_a, log=log)
        ijab_MVP_bb = gen_ijab_MVP_Tpq(T_ij=T_ij_K_b, T_ab=T_ab_K_b, log=log)

        hdiag_MVP_a = gen_hdiag_MVP(hdiag=self.hdiag_a, n_occ=n_occ_a, n_vir=n_vir_a)
        hdiag_MVP_b = gen_hdiag_MVP(hdiag=self.hdiag_b, n_occ=n_occ_b, n_vir=n_vir_b)

        A_aa_size = n_occ_a * n_vir_a
        A_bb_size = n_occ_b * n_vir_b

        def UKS_TDA_hybrid_MVP(X):
            ''' hybrid UKS TDA
                return AX
                A have 4 blocks, αα, αβ, βα, ββ
                A = [ Aαα Aαβ ]    X = [ Xα ]
                    [ Aβα Aββ ]        [ Xβ ]

                AX = [ Aαα Xα + Aαβ Xβ ] = [out_a]
                     [ Aβα Xα + Aββ Xβ ]   [out_b]

                Aαα Xα = hdiag_MVP(Xα) + iajb_aa_MVP(Xα) - a_x * ijab_aa_MVP(Xα)
                Aββ Xβ = hdiag_MVP(Xβ) + iajb_bb_MVP(Xβ) - a_x * ijab_bb_MVP(Xβ)
                Aαβ Xβ = iajb_ab_MVP(Xβ)
                Aβα Xα = iajb_ba_MVP(Xα)


                out_a = hdiag_MVP(Xα) + iajb_aa_MVP(Xα) + iajb_ab_MVP(Xβ) - a_x * ijab_aa_MVP(Xα)
                out_b = hdiag_MVP(Xβ) + iajb_bb_MVP(Xβ) + iajb_ba_MVP(Xα) - a_x * ijab_bb_MVP(Xβ)
            '''
            nstates = X.shape[0]

            X_a = X[:,:A_aa_size].reshape(nstates, n_occ_a, n_vir_a)
            X_b = X[:,A_aa_size:].reshape(nstates, n_occ_b, n_vir_b)

            cpu0 = (logger.process_clock(), logger.perf_counter())
            out_a = hdiag_MVP_a(X_a)
            out_b = hdiag_MVP_b(X_b)
            log.timer('--hdiag_MVP', *cpu0)

            cpu0 = (logger.process_clock(), logger.perf_counter())
            iajb_MVP_aa(X_a, factor=1, out=out_a)
            iajb_MVP_ab(X_b, factor=1, out=out_a)
            iajb_MVP_bb(X_b, factor=1, out=out_b)
            iajb_MVP_ba(X_a, factor=1, out=out_b)
            log.timer('--iajb_MVP', *cpu0)

            cpu0 = (logger.process_clock(), logger.perf_counter())
            X_a_trunc = X_a[:,n_occ_a-rest_occ_a:,:rest_vir_a]
            out_a_trunc = out_a[:,n_occ_a-rest_occ_a:,:rest_vir_a]
            ijab_MVP_aa(X_a_trunc, a_x=a_x, out=out_a_trunc)
            del X_a_trunc, out_a_trunc

            X_b_trunc = X_b[:,n_occ_b-rest_occ_b:,:rest_vir_b]
            out_b_trunc = out_b[:,n_occ_b-rest_occ_b:,:rest_vir_b]
            ijab_MVP_bb(X_b_trunc, a_x=a_x, out=out_b_trunc)
            del X_b_trunc, out_b_trunc
            log.timer('--ijab_MVP', *cpu0)

            out_a = out_a.reshape(nstates, A_aa_size)
            out_b = out_b.reshape(nstates, A_bb_size)
            out = np.hstack((out_a, out_b))
            del out_a, out_b

            return out

        return UKS_TDA_hybrid_MVP, self.hdiag

    ''' ===========  UKS pure =========== '''
    def get_UKS_TDA_pure_MVP(self):

        log = self.log
        n_occ_a = self.n_occ_a
        n_vir_a = self.n_vir_a
        n_occ_b = self.n_occ_b
        n_vir_b = self.n_vir_b

        # J
        T_ia_J_a, T_ia_J_b = self.get_T_J_UKS()

        iajb_MVP_aa = gen_iajb_MVP_Tpq(T_ia=T_ia_J_a, T_jb=T_ia_J_a, log=log)
        iajb_MVP_bb = gen_iajb_MVP_Tpq(T_ia=T_ia_J_b, T_jb=T_ia_J_b, log=log)
        iajb_MVP_ab = gen_iajb_MVP_Tpq(T_ia=T_ia_J_a, T_jb=T_ia_J_b, log=log)
        iajb_MVP_ba = gen_iajb_MVP_Tpq(T_ia=T_ia_J_b, T_jb=T_ia_J_a, log=log)

        hdiag_MVP_a = gen_hdiag_MVP(hdiag=self.hdiag_a, n_occ=n_occ_a, n_vir=n_vir_a)
        hdiag_MVP_b = gen_hdiag_MVP(hdiag=self.hdiag_b, n_occ=n_occ_b, n_vir=n_vir_b)

        A_aa_size = n_occ_a * n_vir_a
        A_bb_size = n_occ_b * n_vir_b

        def UKS_TDA_pure_MVP(X):
            ''' pure UKS TDA
                return AX
                A have 4 blocks, αα, αβ, βα, ββ
                A = [ Aαα Aαβ ]    X = [ Xα ]
                    [ Aβα Aββ ]        [ Xβ ]

                AX = [ Aαα Xα + Aαβ Xβ ] = [out_a]
                     [ Aβα Xα + Aββ Xβ ]   [out_b]

                Aαα Xα = hdiag_MVP(Xα) + iajb_aa_MVP(Xα)
                Aββ Xβ = hdiag_MVP(Xβ) + iajb_bb_MVP(Xβ)
                Aαβ Xβ = iajb_ab_MVP(Xβ)
                Aβα Xα = iajb_ba_MVP(Xα)


                out_a = hdiag_MVP(Xα) + iajb_aa_MVP(Xα) + iajb_ab_MVP(Xβ)
                out_b = hdiag_MVP(Xβ) + iajb_bb_MVP(Xβ) + iajb_ba_MVP(Xα)
            '''
            nstates = X.shape[0]

            X_a = X[:,:A_aa_size].reshape(nstates, n_occ_a, n_vir_a)
            X_b = X[:,A_aa_size:].reshape(nstates, n_occ_b, n_vir_b)

            cpu0 = (logger.process_clock(), logger.perf_counter())
            out_a = hdiag_MVP_a(X_a)
            out_b = hdiag_MVP_b(X_b)
            log.timer('--hdiag_MVP', *cpu0)

            cpu0 = (logger.process_clock(), logger.perf_counter())
            iajb_MVP_aa(X_a, out=out_a)
            iajb_MVP_ab(X_b, out=out_a)
            iajb_MVP_bb(X_b, out=out_b)
            iajb_MVP_ba(X_a, out=out_b)
            log.timer('--iajb_MVP', *cpu0)

            out_a = out_a.reshape(nstates, A_aa_size)
            out_b = out_b.reshape(nstates, A_bb_size)
            out = np.hstack((out_a, out_b))
            del out_a, out_b

            return out

        return UKS_TDA_pure_MVP, self.hdiag

    def gen_vind(self):
        self.build()
        if self.RKS:
            if self.a_x != 0:
                TDA_MVP, hdiag = self.get_RKS_TDA_hybrid_MVP()

            elif self.a_x == 0:
                TDA_MVP, hdiag = self.get_RKS_TDA_pure_MVP()
        else:
            if self.a_x != 0:
                TDA_MVP, hdiag = self.get_UKS_TDA_hybrid_MVP()

            elif self.a_x == 0:
                TDA_MVP, hdiag = self.get_UKS_TDA_pure_MVP()
        return TDA_MVP, hdiag

    def kernel(self):

        '''for TDA, pure and hybrid share the same form of
                     AX = Xw
            always use the Davidson solver
            Unlike pure TDDFT, pure TDA is not using MZ=Zw^2 form
        '''
        log = self.log

        TDA_MVP, hdiag = self.gen_vind()

        converged, energies, X = _krylov_tools.krylov_solver(matrix_vector_product=TDA_MVP,hdiag=hdiag,
                                            n_states=self.nstates, problem_type='eigenvalue',
                                            conv_tol=self.conv_tol, max_iter=self.max_iter,
                                            extra_init=self.extra_init, restart_iter=self.restart_iter,
                                            gs_initial=False, gram_schmidt=self.gram_schmidt,
                                            print_eigeneV_along=True, single=self.single,  verbose=log)

        self.converged = converged
        log.debug(f'check orthonormality of X: {np.linalg.norm(np.dot(X, X.T) - np.eye(X.shape[0])):.2e}')

        cpu0 = (logger.process_clock(), logger.perf_counter())
        P = self.transition_dipole()
        log.timer('transition_dipole', *cpu0)
        cpu0 = (logger.process_clock(), logger.perf_counter())
        mdpol = self.transition_magnetic_dipole()
        log.timer('transition_magnetic_dipole', *cpu0)

        if self.RKS:
            n_occ_a, n_vir_a, n_occ_b, n_vir_b = self.n_occ, self.n_vir, None, None
        else:
            n_occ_a, n_vir_a, n_occ_b, n_vir_b = self.n_occ_a, self.n_vir_a, self.n_occ_b, self.n_vir_b

        oscillator_strength, rotatory_strength = spectralib.get_spectra(energies=energies, X=X, Y=None,
                                                P=P, mdpol=mdpol,
                                                name=self.out_name+'_TDA_ris' if self.out_name else 'TDA_ris',
                                                RKS=self.RKS, spectra=self.spectra,
                                                print_threshold = self.print_threshold,
                                                n_occ_a=n_occ_a, n_vir_a=n_vir_a, n_occ_b=n_occ_b, n_vir_b=n_vir_b,
                                                verbose=self.verbose)

        energies = energies*HARTREE2EV

        self.energies = energies
        self.xy = (X, None)
        self.oscillator_strength = oscillator_strength
        self.rotatory_strength = rotatory_strength

        if self._citation:
            log.info(CITATION_INFO)

        return energies, X, oscillator_strength, rotatory_strength

    def Gradients(self):
        if getattr(self._scf, 'with_df', None) is not None:
            from gpu4pyscf.df.grad import tdrks_ris
            return tdrks_ris.Gradients(self)
        else:
            from gpu4pyscf.grad import tdrks_ris
            return tdrks_ris.Gradients(self)

    def NAC(self):
        if getattr(self._scf, 'with_df', None) is not None:
            from gpu4pyscf.df.nac.tdrks_ris import NAC
            return NAC(self)
        else:
            from gpu4pyscf.nac.tdrks_ris import NAC
            return NAC(self)

class TDDFT(RisBase):
    def __init__(self, mf, **kwargs):
        super().__init__(mf, **kwargs)
        log = self.log
        log.info('TDDFT-ris is initialized')

    ''' ===========  RKS hybrid =========== '''
    def gen_RKS_TDDFT_hybrid_MVP(self):
        '''hybrid RKS TDDFT'''
        log = self.log
        n_occ = self.n_occ
        n_vir = self.n_vir
        rest_occ = self.rest_occ
        rest_vir = self.rest_vir
        a_x = self.a_x

        # J
        T_ia_J = self.get_T_J_RKS()
        iajb_MVP = gen_iajb_MVP_Tpq(T_ia=T_ia_J, T_jb=T_ia_J, log=log)

        # K
        T_ia_K, T_ij_K, T_ab_K = self.get_3T_K_RKS()
        ijab_MVP = gen_ijab_MVP_Tpq(T_ij=T_ij_K, T_ab=T_ab_K, log=log)
        ibja_MVP = gen_ibja_MVP_Tpq(T_ia=T_ia_K, log=log)

        hdiag_MVP = gen_hdiag_MVP(hdiag=self.hdiag, n_occ=n_occ, n_vir=n_vir)

        def RKS_TDDFT_hybrid_MVP(X, Y):
            '''
            RKS
            [A B][X] = [AX+BY] = [U1]
            [B A][Y]   [AY+BX]   [U2]
            we want AX+BY and AY+BX
            instead of directly computing AX+BY and AY+BX
            we compute (A+B)(X+Y) and (A-B)(X-Y)
            it can save one (ia|jb)V tensor einsumion compared to directly computing AX+BY and AY+BX

            (A+B)V = hdiag_MVP(V) + 4*iajb_MVP(V) - a_x * [ ijab_MVP(V) + ibja_MVP(V) ]
            (A-B)V = hdiag_MVP(V) - a_x * [ ijab_MVP(V) - ibja_MVP(V) ]
            for RSH, a_x = 1, because the exchange component is defined by alpha+beta (alpha+beta not awlways == 1)

            # X Y in shape (m, n_occ*n_vir)
            '''
            nstates = X.shape[0]

            X = X.reshape(nstates, n_occ, n_vir)
            Y = Y.reshape(nstates, n_occ, n_vir)

            XpY = X + Y
            XmY = X - Y
            ApB_XpY = hdiag_MVP(XpY)

            iajb_MVP(XpY, factor=4, out=ApB_XpY)

            ijab_MVP(XpY[:,n_occ-rest_occ:,:rest_vir], a_x=a_x, out=ApB_XpY[:,n_occ-rest_occ:,:rest_vir])

            ibja_MVP(XpY[:,n_occ-rest_occ:,:rest_vir], a_x=a_x, out=ApB_XpY[:,n_occ-rest_occ:,:rest_vir])

            AmB_XmY = hdiag_MVP(XmY)

            ijab_MVP(XmY[:,n_occ-rest_occ:,:rest_vir], a_x=a_x, out=AmB_XmY[:,n_occ-rest_occ:,:rest_vir])

            ibja_MVP(XmY[:,n_occ-rest_occ:,:rest_vir], a_x=-a_x, out=AmB_XmY[:,n_occ-rest_occ:,:rest_vir])


            ''' (A+B)(X+Y) = AX + BY + AY + BX   (1)
                (A-B)(X-Y) = AX + BY - AY - BX   (2)
                (1) + (1) /2 = AX + BY = U1
                (1) - (2) /2 = AY + BX = U2
            '''
            U1 = (ApB_XpY + AmB_XmY)/2
            U2 = (ApB_XpY - AmB_XmY)/2

            U1 = U1.reshape(nstates, n_occ*n_vir)
            U2 = U2.reshape(nstates, n_occ*n_vir)

            return U1, U2
        return RKS_TDDFT_hybrid_MVP, self.hdiag

    ''' ===========  RKS pure =========== '''
    def gen_RKS_TDDFT_pure_MVP(self):
        log = self.log
        n_occ = self.n_occ
        n_vir = self.n_vir

        log.info(get_mem_info('before T_ia_J'))

        hdiag_sq = self.hdiag**2
        hdiag_sqrt_MVP = gen_hdiag_MVP(hdiag=self.hdiag**0.5, n_occ=self.n_occ, n_vir=self.n_vir)
        hdiag_MVP = gen_hdiag_MVP(hdiag=self.hdiag, n_occ=self.n_occ, n_vir=self.n_vir)

        T_ia_J = self.get_T_J_RKS()
        iajb_MVP = gen_iajb_MVP_Tpq(T_ia=T_ia_J, T_jb=T_ia_J, log=log)

        def RKS_TDDFT_pure_MVP(Z):
            '''(A-B)^1/2(A+B)(A-B)^1/2 Z = Z w^2
                                    MZ = Z w^2
                M = (A-B)^1/2 (A+B) (A-B)^1/2
                X+Y = (A-B)^1/2 Z

                (A+B)(V) = hdiag_MVP(V) + 4*iajb_MVP(V)
                (A-B)^1/2(V) = hdiag_sqrt_MVP(V)
            '''
            nstates = Z.shape[0]
            Z = Z.reshape(nstates, n_occ, n_vir)
            AmB_sqrt_V = hdiag_sqrt_MVP(Z)
            ApB_AmB_sqrt_V = hdiag_MVP(AmB_sqrt_V)
            iajb_MVP(AmB_sqrt_V, factor=4, out=ApB_AmB_sqrt_V)

            MZ = hdiag_sqrt_MVP(ApB_AmB_sqrt_V)
            MZ = MZ.reshape(nstates, n_occ*n_vir)
            return MZ

        return RKS_TDDFT_pure_MVP, hdiag_sq


    ''' ===========  UKS hybrid =========== '''
    def gen_UKS_TDDFT_hybrid_MVP(self):
        '''hybrid RKS TDDFT'''
        log = self.log
        n_occ_a = self.n_occ_a
        n_vir_a = self.n_vir_a
        n_occ_b = self.n_occ_b
        n_vir_b = self.n_vir_b
        rest_occ_a = self.rest_occ_a
        rest_vir_a = self.rest_vir_a
        rest_occ_b = self.rest_occ_b
        rest_vir_b = self.rest_vir_b
        a_x = self.a_x

        # J
        T_ia_J_a, T_ia_J_b = self.get_T_J_UKS()

        iajb_MVP_aa = gen_iajb_MVP_Tpq(T_ia=T_ia_J_a, T_jb=T_ia_J_a, log=log)
        iajb_MVP_bb = gen_iajb_MVP_Tpq(T_ia=T_ia_J_b, T_jb=T_ia_J_b, log=log)
        iajb_MVP_ab = gen_iajb_MVP_Tpq(T_ia=T_ia_J_a, T_jb=T_ia_J_b, log=log)
        iajb_MVP_ba = gen_iajb_MVP_Tpq(T_ia=T_ia_J_b, T_jb=T_ia_J_a, log=log)

        # K
        T_ia_K_a, T_ia_K_b, T_ij_K_a, T_ij_K_b, T_ab_K_a, T_ab_K_b = self.get_3T_K_UKS()

        ijab_MVP_aa = gen_ijab_MVP_Tpq(T_ij=T_ij_K_a, T_ab=T_ab_K_a, log=log)
        ijab_MVP_bb = gen_ijab_MVP_Tpq(T_ij=T_ij_K_b, T_ab=T_ab_K_b, log=log)
        ibja_MVP_aa = gen_ibja_MVP_Tpq(T_ia=T_ia_K_a, log=log)
        ibja_MVP_bb = gen_ibja_MVP_Tpq(T_ia=T_ia_K_b, log=log)

        hdiag_MVP_a = gen_hdiag_MVP(hdiag=self.hdiag_a, n_occ=n_occ_a, n_vir=n_vir_a)
        hdiag_MVP_b = gen_hdiag_MVP(hdiag=self.hdiag_b, n_occ=n_occ_b, n_vir=n_vir_b)

        A_aa_size = n_occ_a * n_vir_a
        A_bb_size = n_occ_b * n_vir_b

        def UKS_TDDFT_hybrid_MVP(X, Y):
            '''
            UKS
            [A B][X] = [AX+BY] = [U1]
            [B A][Y]   [AY+BX]   [U2]
            A B have 4 blocks, αα, αβ, βα, ββ
            A = [ Aαα Aαβ ]   B = [ Bαα Bαβ ]
                [ Aβα Aββ ]       [ Bβα Bββ ]

            X = [ Xα ]        Y = [ Yα ]
                [ Xβ ]            [ Yβ ]

            (A+B)αα, (A+B)αβ is shown below

            βα, ββ can be obtained by change α to β
            we compute (A+B)(X+Y) and (A-B)(X-Y)

            V:= X+Y
            (A+B)αα Vα = hdiag_MVP(Vα) + 2*iaαjbα_MVP(Vα) - a_x*[ijαabα_MVP(Vα) + ibαjaα_MVP(Vα)]
                      (A+B)αβ Vβ = 2*iaαjbβ_MVP(Vβ)



            A+B = [ (A+B)αα Cαβ ]   x+y = [ Vα ]
                  [ (A+B)βα Cββ ]         [ Vβ ]
            (A+B)(x+y) =   [ (A+B)αα Vα + (A+B)αβ Vβ ]  = [ ApB_XpY_α ]
                           [ (A+B)βα Vα + (A+B)ββ Vβ ]  = [ ApB_XpY_β ]

            V:= X-Y
            (A-B)αα Vα = hdiag_MVP(Vα) - a_x*[ijαabα_MVP(Vα) - ibαjaα_MVP(Vα)]
                  (A-B)αβ Vβ = 0


            A-B = [ (A-B)αα  0  ]   x-y = [ Vα ]
                  [  0  (A-B)ββ ]         [ Vβ ]
            (A-B)(x-y) =   [ (A-B)αα Vα ] = [ AmB_XmY_α ]
                           [ (A-B)ββ Vβ ]   [ AmB_XmY_β ]
            '''
            nstates = X.shape[0]

            X_a = X[:,:A_aa_size].reshape(nstates, n_occ_a, n_vir_a)
            X_b = X[:,A_aa_size:].reshape(nstates, n_occ_b, n_vir_b)
            Y_a = Y[:,:A_aa_size].reshape(nstates, n_occ_a, n_vir_a)
            Y_b = Y[:,A_aa_size:].reshape(nstates, n_occ_b, n_vir_b)

            XpY_a = X_a + Y_a
            XpY_b = X_b + Y_b
            XmY_a = X_a - Y_a
            XmY_b = X_b - Y_b

            XpY_a_trunc = XpY_a[:,n_occ_a-rest_occ_a:,:rest_vir_a]
            XpY_b_trunc = XpY_b[:,n_occ_b-rest_occ_b:,:rest_vir_b]
            XmY_a_trunc = XmY_a[:,n_occ_a-rest_occ_a:,:rest_vir_a]
            XmY_b_trunc = XmY_b[:,n_occ_b-rest_occ_b:,:rest_vir_b]

            '''============== (A+B) (X+Y) ================'''
            ''' alpha part '''
            ApB_XpY_a = hdiag_MVP_a(XpY_a)
            ApB_XpY_a_trunc = ApB_XpY_a[:,n_occ_a-rest_occ_a:,:rest_vir_a]

            iajb_MVP_aa(XpY_a, factor=2, out=ApB_XpY_a)
            iajb_MVP_ab(XpY_b, factor=2, out=ApB_XpY_a)

            ijab_MVP_aa(XpY_a_trunc, a_x=a_x, out=ApB_XpY_a_trunc)
            ibja_MVP_aa(XpY_a_trunc, a_x=a_x, out=ApB_XpY_a_trunc)


            ''' beta part (simply change above α to β)'''

            ApB_XpY_b = hdiag_MVP_b(XpY_b)
            ApB_XpY_b_trunc = ApB_XpY_b[:,n_occ_b-rest_occ_b:,:rest_vir_b]

            iajb_MVP_bb(XpY_b, factor=2, out=ApB_XpY_b)
            iajb_MVP_ba(XpY_a, factor=2, out=ApB_XpY_b)

            ijab_MVP_bb(XpY_b_trunc, a_x=a_x, out=ApB_XpY_b_trunc)
            ibja_MVP_bb(XpY_b_trunc, a_x=a_x, out=ApB_XpY_b_trunc)


            '''============== (A-B) (X-Y) ================'''
            ''' alpha part '''

            AmB_XmY_a = hdiag_MVP_a(XmY_a)
            AmB_XmY_a_trunc = AmB_XmY_a[:,n_occ_a-rest_occ_a:,:rest_vir_a]

            ijab_MVP_aa(XmY_a_trunc, a_x=a_x, out=AmB_XmY_a_trunc)
            ibja_MVP_aa(XmY_a_trunc, a_x=-a_x, out=AmB_XmY_a_trunc)

            ''' beta part (simply change above α to β)'''

            AmB_XmY_b = hdiag_MVP_b(XmY_b)
            AmB_XmY_b_trunc = AmB_XmY_b[:,n_occ_b-rest_occ_b:,:rest_vir_b]

            ijab_MVP_bb(XmY_b_trunc, a_x=a_x, out=AmB_XmY_b_trunc)
            ibja_MVP_bb(XmY_b_trunc, a_x=-a_x, out=AmB_XmY_b_trunc)

            '''============== U1 = AX+BY and U2 = AY+BX ================'''
            ''' (A+B)(X+Y) = AX + BY + AY + BX   (1) ApB_XpY
                (A-B)(X-Y) = AX + BY - AY - BX   (2) AmB_XmY
                    (1) + (1) /2 = AX + BY = U1
                    (1) - (2) /2 = AY + BX = U2
            '''
            ApB_XpY_a = ApB_XpY_a.reshape(nstates, A_aa_size)
            ApB_XpY_b = ApB_XpY_b.reshape(nstates, A_bb_size)
            AmB_XmY_a = AmB_XmY_a.reshape(nstates, A_aa_size)
            AmB_XmY_b = AmB_XmY_b.reshape(nstates, A_bb_size)

            ApB_XpY = np.concatenate([ApB_XpY_a, ApB_XpY_b], axis=1)
            AmB_XmY = np.concatenate([AmB_XmY_a, AmB_XmY_b], axis=1)

            U1 = (ApB_XpY + AmB_XmY)/2
            U2 = (ApB_XpY - AmB_XmY)/2

            return U1, U2
        return UKS_TDDFT_hybrid_MVP, self.hdiag


    ''' ===========  UKS pure =========== '''
    def gen_UKS_TDDFT_pure_MVP(self):
        '''pure TDDFT'''
        log = self.log
        n_occ_a = self.n_occ_a
        n_vir_a = self.n_vir_a
        n_occ_b = self.n_occ_b
        n_vir_b = self.n_vir_b

        T_ia_J_a, T_ia_J_b = self.get_T_J_UKS()

        iajb_MVP_aa = gen_iajb_MVP_Tpq(T_ia=T_ia_J_a, T_jb=T_ia_J_a, log=log)
        iajb_MVP_bb = gen_iajb_MVP_Tpq(T_ia=T_ia_J_b, T_jb=T_ia_J_b, log=log)
        iajb_MVP_ab = gen_iajb_MVP_Tpq(T_ia=T_ia_J_a, T_jb=T_ia_J_b, log=log)
        iajb_MVP_ba = gen_iajb_MVP_Tpq(T_ia=T_ia_J_b, T_jb=T_ia_J_a, log=log)


        hdiag_a_sqrt_MVP = gen_hdiag_MVP(hdiag=self.hdiag_a**0.5, n_occ=n_occ_a, n_vir=n_vir_a)
        hdiag_b_sqrt_MVP = gen_hdiag_MVP(hdiag=self.hdiag_b**0.5, n_occ=n_occ_b, n_vir=n_vir_b)


        hdiag_a_MVP = gen_hdiag_MVP(hdiag=self.hdiag_a, n_occ=n_occ_a, n_vir=n_vir_a)
        hdiag_b_MVP = gen_hdiag_MVP(hdiag=self.hdiag_b, n_occ=n_occ_b, n_vir=n_vir_b)

        hdiag_a_sq = self.hdiag_a**2
        hdiag_b_sq = self.hdiag_b**2
        '''hdiag_sq: preconditioner'''
        hdiag_sq = np.hstack((hdiag_a_sq, hdiag_b_sq))

        A_aa_size = n_occ_a * n_vir_a
        A_bb_size = n_occ_b * n_vir_b

        def UKS_TDDFT_pure_MVP(Z):
            '''       MZ = Z w^2
                M = (A-B)^1/2(A+B)(A-B)^1/2
                Z = (A-B)^1/2(X-Y)

                X+Y = (A-B)^1/2 Z * 1/w
                A+B = hdiag_MVP(V) + 4*iajb_MVP(V)
                (A-B)^1/2 = hdiag_sqrt_MVP(V)
                preconditioner = hdiag_sq

                M =  [ (A-B)^1/2αα    0   ] [ (A+B)αα (A+B)αβ ] [ (A-B)^1/2αα    0   ]            Z = [ Zα ]
                     [    0   (A-B)^1/2ββ ] [ (A+B)βα (A+B)ββ ] [    0   (A-B)^1/2ββ ]                [ Zβ ]
            '''
            nstates = Z.shape[0]
            Z_a = Z[:, :A_aa_size].reshape(nstates, n_occ_a, n_vir_a)
            Z_b = Z[:, A_aa_size:].reshape(nstates, n_occ_b, n_vir_b)

            AmB_aa_sqrt_Z_a = hdiag_a_sqrt_MVP(Z_a)
            AmB_bb_sqrt_Z_b = hdiag_b_sqrt_MVP(Z_b)

            MZ_a = hdiag_a_MVP(AmB_aa_sqrt_Z_a)
            iajb_MVP_aa(AmB_aa_sqrt_Z_a, factor=2, out=MZ_a)
            iajb_MVP_ab(AmB_bb_sqrt_Z_b, factor=2, out=MZ_a)

            MZ_b = hdiag_b_MVP(AmB_bb_sqrt_Z_b)
            iajb_MVP_bb(AmB_bb_sqrt_Z_b, factor=2, out=MZ_b)
            iajb_MVP_ba(AmB_aa_sqrt_Z_a, factor=2, out=MZ_b)


            MZ_a = hdiag_a_sqrt_MVP(MZ_a)
            MZ_b = hdiag_b_sqrt_MVP(MZ_b)

            MZ_a = MZ_a.reshape(nstates, A_aa_size)
            MZ_b = MZ_b.reshape(nstates, A_bb_size)
            MZ = np.concatenate([MZ_a, MZ_b], axis=1)

            return MZ

        return UKS_TDDFT_pure_MVP, hdiag_sq

    def gen_vind(self):
        self.build()
        if self.RKS:
            if self.a_x != 0:
                TDDFT_MVP, hdiag = self.gen_RKS_TDDFT_hybrid_MVP()

            elif self.a_x == 0:
                TDDFT_MVP, hdiag = self.gen_RKS_TDDFT_pure_MVP()
        else:
            if self.a_x != 0:
                TDDFT_MVP, hdiag = self.gen_UKS_TDDFT_hybrid_MVP()

            elif self.a_x == 0:
                TDDFT_MVP, hdiag = self.gen_UKS_TDDFT_pure_MVP()
        return TDDFT_MVP, hdiag


    def kernel(self):
        log = self.log
        TDDFT_MVP, hdiag = self.gen_vind()
        if self.a_x != 0:
            '''hybrid TDDFT'''
            converged, energies, X, Y = _krylov_tools.ABBA_krylov_solver(matrix_vector_product=TDDFT_MVP,
                                                    hdiag=hdiag, n_states=self.nstates, conv_tol=self.conv_tol,
                                                    max_iter=self.max_iter, gram_schmidt=self.gram_schmidt,
                                                    single=self.single, verbose=self.verbose)
            self.converged = converged
            if not all(self.converged):
                log.info('TD-SCF states %s not converged.',
                            [i for i, x in enumerate(self.converged) if not x])
        elif self.a_x == 0:
            '''pure TDDFT'''
            hdiag_sq = hdiag
            converged, energies_sq, Z = _krylov_tools.krylov_solver(matrix_vector_product=TDDFT_MVP, hdiag=hdiag_sq,
                                            n_states=self.nstates, conv_tol=self.conv_tol, max_iter=self.max_iter,
                                            gram_schmidt=self.gram_schmidt, single=self.single, verbose=self.verbose)
            self.converged = converged
            if not all(self.converged):
                log.info('TD-SCF states %s not converged.',
                            [i for i, x in enumerate(self.converged) if not x])

            energies = energies_sq**0.5
            Z = (energies**0.5).reshape(-1,1) * Z

            X, Y = math_helper.XmY_2_XY(Z=Z, AmB_sq=hdiag_sq, omega=energies)

        XYnorm = np.linalg.norm( (np.dot(X, X.T) - np.dot(Y, Y.T)) - np.eye(self.nstates))
        log.debug(f'check normality of X^TX - Y^YY - I = {XYnorm:.2e}')

        cpu0 = (logger.process_clock(), logger.perf_counter())
        P = self.transition_dipole()
        log.timer('transition_dipole', *cpu0)
        cpu0 = (logger.process_clock(), logger.perf_counter())
        mdpol = self.transition_magnetic_dipole()
        log.timer('transition_magnetic_dipole', *cpu0)

        if self.RKS:
            n_occ_a, n_vir_a, n_occ_b, n_vir_b = self.n_occ, self.n_vir, None, None
        else:
            n_occ_a, n_vir_a, n_occ_b, n_vir_b = self.n_occ_a, self.n_vir_a, self.n_occ_b, self.n_vir_b

        oscillator_strength, rotatory_strength = spectralib.get_spectra(energies=energies, X=X, Y=Y,
                                                P=P, mdpol=mdpol,
                                                name=self.out_name+'_TDDFT_ris' if self.out_name else 'TDDFT_ris',
                                                spectra=self.spectra, RKS=self.RKS,
                                                print_threshold = self.print_threshold,
                                                n_occ_a=n_occ_a, n_vir_a=n_vir_a, n_occ_b=n_occ_b, n_vir_b=n_vir_b,
                                                verbose=self.verbose)
        energies = energies*HARTREE2EV
        if self._citation:
            log.info(CITATION_INFO)
        self.energies = energies
        self.xy = X, Y
        self.oscillator_strength = oscillator_strength
        self.rotatory_strength = rotatory_strength

        return energies, X, Y, oscillator_strength, rotatory_strength

    Gradients = TDA.Gradients
    NAC = TDA.NAC



class StaticPolarizability(RisBase):
    def __init__(self, mf, **kwargs):
        super().__init__(mf, **kwargs)
        log = self.log
        log.info('Static Polarizability-ris initialized')

    ''' ===========  RKS hybrid =========== '''
    def get_ApB_hybrid_MVP(self):
        ''' RKS hybrid '''
        log = self.log

        T_ia_J = self.get_T_J()

        T_ia_K, T_ij_K, T_ab_K = self.get_3T_K()

        hdiag_MVP = gen_hdiag_MVP(hdiag=self.hdiag, n_occ=self.n_occ, n_vir=self.n_vir)

        iajb_MVP = gen_iajb_MVP_Tpq(T_ia=T_ia_J)
        ijab_MVP = gen_ijab_MVP_Tpq(T_ij=T_ij_K, T_ab=T_ab_K)
        ibja_MVP = gen_ibja_MVP_Tpq(T_ia=T_ia_K)

        def RKS_ApB_hybrid_MVP(X):
            ''' hybrid or range-sparated hybrid, a_x > 0
                return AX
                (A+B)X = hdiag_MVP(X) + 4*iajb_MVP(X) - a_x*[ijab_MVP(X) + ibja_MVP(X)]
                for RSH, a_x = 1

                if not MO truncation, then n_occ-rest_occ=0 and rest_vir=n_vir
            '''
            nstates = X.shape[0]
            X = X.reshape(nstates, self.n_occ, self.n_vir)
            cpu0 = (logger.process_clock(), logger.perf_counter())
            ApBX = hdiag_MVP(X)
            ApBX += 4 * iajb_MVP(X)
            log.timer('--iajb_MVP', *cpu0)

            cpu1 = (logger.process_clock(), logger.perf_counter())
            exchange =  ijab_MVP(X[:,self.n_occ-self.rest_occ:,:self.rest_vir])
            exchange += ibja_MVP(X[:,self.n_occ-self.rest_occ:,:self.rest_vir])
            log.timer('--ijab_MVP & ibja_MVP', *cpu1)

            ApBX[:,self.n_occ-self.rest_occ:,:self.rest_vir] -= self.a_x * exchange
            ApBX = ApBX.reshape(nstates, self.n_occ*self.n_vir)

            return ApBX

        return RKS_ApB_hybrid_MVP, self.hdiag

    def gen_vind(self):
        self.build()
        if self.RKS:
            if self.a_x != 0:
                TDA_MVP, hdiag = self.get_ApB_hybrid_MVP()

            elif self.a_x == 0:
                TDA_MVP, hdiag = self.get_ApB_pure_MVP()
        else:
            raise NotImplementedError('Does not support UKS method yet')
        return TDA_MVP, hdiag


    def kernel(self):
        '''for static polarizability, the problem is to solve
            (A+B)(X+Y) = -(P+Q)
            Q=P
        '''

        log = self.log

        TDA_MVP, hdiag = self.gen_vind()
        transition_dipole = self.transition_dipole()

        _, XpY = _krylov_tools.krylov_solver(matrix_vector_product=TDA_MVP,hdiag=hdiag, problem_type='linear',
                                        rhs=-transition_dipole, conv_tol=self.conv_tol, max_iter=self.max_iter,
                                        gram_schmidt=self.gram_schmidt, single=self.single, verbose=log)

        alpha = np.dot(XpY, transition_dipole.T)*4

        self.xy = XpY
        self.alpha = alpha

        if self._citation:
            log.info(CITATION_INFO)
        return XpY

