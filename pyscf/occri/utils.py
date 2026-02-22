"""
Utility functions for OCCRI

This module contains pure mathematical operations and algorithms that are
reused across different contexts.
"""

import numpy
import scipy.linalg
from pyscf import lib


def make_natural_orbitals(cell, kpts, dms):
    """
    Construct natural orbitals from density matrix.

    This is a pure mathematical operation that performs eigenvalue decomposition
    of density matrices to obtain natural orbitals and their occupations.

    Parameters:
    -----------
    cell : Cell
        PySCF cell object
    kpts : ndarray
        K-point coordinates
    dms : ndarray
        Density matrices with shape (nset, nk, nao, nao)

    Returns:
    --------
    ndarray
        Tagged density matrix array with mo_coeff and mo_occ attributes
    """
    nk = kpts.shape[0]
    nao = cell.nao
    nset = dms.shape[0]

    # Compute k-point dependent overlap matrices
    sk = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)
    if abs(dms.imag).max() < 1.0e-6:
        sk = [s.real.astype(numpy.float64) for s in sk]

    mo_coeff = numpy.zeros_like(dms)
    mo_occ = numpy.zeros((nset, nk, nao), numpy.float64)

    for i, dm in enumerate(dms):
        for k, s in enumerate(sk):
            # Diagonalize the DM in AO basis: S^{1/2} * DM * S^{1/2}
            A = numpy.dot (numpy.dot (s, dm[k]), s)
            w, v = scipy.linalg.eigh(A, b=s)

            # Sort eigenvalues/eigenvectors in descending order
            mo_occ[i][k] = numpy.flip(w)
            mo_coeff[i][k] = numpy.flip(v, axis=1)

    return lib.tag_array(dms, mo_coeff=mo_coeff, mo_occ=mo_occ)


def build_full_exchange(S, Kao, mo_coeff):
    """
    Build full exchange matrix from occupied orbital components.

    This implements the exchange matrix construction:
    K_uv = Sa @ Kao.T + (Sa @ Kao.T).T - Sa @ (mo_coeff @ Kao) @ Sa.T
    where Sa = S @ mo_coeff.T

    Parameters:
    -----------
    S : ndarray
        Overlap matrix or similar transformation matrix
    Kao : ndarray
        AO values for occupied orbitals
    mo_coeff : ndarray
        Molecular orbital coefficients

    Returns:
    --------
    ndarray
        Full exchange matrix
    """
    # Compute Sa = S @ mo_coeff.T once and reuse
    Sa = S @ mo_coeff.T

    # First and second terms: Sa @ Kao.T + (Sa @ Kao.T).T
    Sa_Kao = numpy.matmul(Sa, Kao.T.conj(), order='C')
    Kuv = Sa_Kao + Sa_Kao.T.conj()

    # Third term: -Sa @ (mo_coeff @ Kao) @ Sa.T
    Koo = mo_coeff.conj() @ Kao
    Sa_Koo = numpy.matmul(Sa, Koo)
    Kuv -= numpy.matmul(Sa_Koo, Sa.T.conj(), order='C')

    return Kuv
