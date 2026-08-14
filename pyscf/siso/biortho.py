from dataclasses import dataclass
from functools import reduce

import numpy as np
from scipy import linalg

from pyscf.fci import direct_nosym

LU_THRESH = 1e-6

# Author: Bhavnesh Jangid

'''
Malmqvist biorthogonal orbital and CI transformations
'''


def lu_pp_decomposition(cxa, cyb, block_sizes, threshold=1e-6):
    '''
    Simultaneous LU decomposition with unitary pseudo-pivoting.
    As per as I know, it's not available in NumPy or SciPy, so I have
    implemented it here.
    
    In this factorization, at every elimination step, the same block-unitary
    right transformation is applied implicitly to both input matrices.  The
    transformation maximizes the usable pivots of both matrices before their
    coordinated LU updates.
    args:
        cxa, cyb: np.array
            mo_coeffs
        block_sizes: tuple of ints
            size of cas subspaces
        threshold: float
            Minimum squared norm allowed for either remaining pivot row,
            matching the OpenMolcas ``LU2`` criterion.
    returns:
        cxa_t, cyb_t: np.array
            compact LU factors of the transformed matrices
    '''
    cxa = np.asarray(cxa, dtype=np.float64, order="C")
    cyb = np.asarray(cyb, dtype=np.float64, order="C")
    
    matrices = np.stack((cxa, cyb),)
    ndim = matrices.shape[1]
    assert matrices.shape == (2, ndim, ndim), \
        "Input matrices must be square and have equal shapes"
    assert sum(block_sizes) == ndim, \
        "Block sizes do not span the MO space"

    start = 0
    for size in block_sizes:
        end = start + size
        for i in range(start, end):
            xa = matrices[0, i, i:end]
            yb = matrices[1, i, i:end]
            s1 = np.dot(xa, xa)
            s2 = np.dot(yb, yb)
            s3 = np.dot(xa, yb)
            if s1 < threshold or s2 < threshold:
                raise linalg.LinAlgError("The two orbital spaces are too dissimilar for the"
                                         "LU partitioning: squared pivot-tail norms are "
                                         f"{s1:.3e} and {s2:.3e} at orbital {i}")
            x1 = 1.0 / np.sqrt(s1)
            x2 = 1.0 / np.copysign(np.sqrt(s2), s3)
            work = x1 * xa + x2 * yb
            scale = 2.0 * (1.0 + x1 * x2 * s3)
            work *= 1.0 / np.copysign(np.sqrt(scale), work[0])
            alpha = 1.0 / (1.0 + work[0])

            # Householder-like right transformation

            rows = matrices[:, :end, i:end]
            tail_dot = rows[:, :, 1:] @ work[1:]
            first = rows[:, :, 0].copy()
            rows[:, :, 0] = first * work[0] + tail_dot
            rows[:, :, 1:] -= ((first + alpha * tail_dot)[:, :, None] 
                               * work[None, None, 1:])

            if i + 1 < end:
                # Coordinated rank-one LU updates.  Columns beyond the current
                # orbital block must also be updated, as in OpenMolcas LU2.
                multipliers = matrices[:, i + 1:end, i]
                multipliers /= matrices[:, i, i, None]
                matrices[:, i + 1:end, i + 1:] -= (
                    multipliers[:, :, None]
                    * matrices[:, i, None, i + 1:]
                )
        start = end

    cxa_t = matrices[0]
    cyb_t = matrices[1]
    return cxa_t, cyb_t

def _compute_trans_mat(mat):
    '''
    Computing single-orbital transformation coefficients.
    '''
    mat = np.array(mat, copy=True)
    lower = -np.tril(mat, k=-1)
    upper_inverse = linalg.solve_triangular(
        np.triu(mat), np.eye(mat.shape[0]), lower=False)
    return upper_inverse + lower

def compute_trans_mat(ovlp, block_sizes, lu_threshold=LU_THRESH):
    '''
    Computing sequential transformations for two MO sets.

    args:
        ovlp: np.array
            overlap of two mo_coeffs over (inactive + active orbitals).
            ovlp = mo_coeff1.T @ S_AO @ mo_coeff2
        block_sizes: tuple of ints
            sizes of cas subspaces
            generally (ncore, ncas)
        lu_threshold: float (default 1e-6)
            Minimum squared norm allowed for either remaining pivot row.
    returns:
        cxa_t, cyb_t: np.array
            These are sequential single-orbital transformation coefficients,
            not ordinary orbital rotations.
    '''

    sxy = np.asarray(ovlp, dtype=np.float64, order="C")
    nocc = sxy.shape[0]
    block_sizes = tuple(int(x) for x in block_sizes if x)
    assert sxy.shape == (nocc, nocc), \
            "The occupied-space cross-overlap must be square"
    assert sum(block_sizes) == nocc, \
        "block_sizes must sum to the cross-overlap dimension"

    # Try the inverse the cross overlap matrix.
    try:
        amat = linalg.inv(sxy)
    except linalg.LinAlgError as err:
        raise linalg.LinAlgError("The inact+act orbital " \
        "cross-overlap is singular") from err
    
    bmat = np.eye(nocc)

    # Part-1: loops backwards over all blocks except the first.
    offsets = np.cumsum((0,) + block_sizes)
    for iblock in range(len(block_sizes) - 1, 0, -1):
        left_end = offsets[iblock]
        block = slice(offsets[iblock], offsets[iblock + 1])
        abb = amat[block, block]
        abl = amat[block, :left_end]
        bmat[block, :left_end] = linalg.solve(abb, abl)
        amat[block, :left_end] = 0.0
        amat[:left_end, :left_end] -= (
            amat[:left_end, block] @ bmat[block, :left_end])
        
    # Take the transpose of bmat
    bmat = bmat.T

    # Part-2: LU decomposition with pseudo-pivoting.
    bmat, amat = lu_pp_decomposition(bmat, amat, block_sizes, threshold=lu_threshold)

    # Part-3: compute the sequential single-orbital transformation coefficients.
    bmat = _compute_trans_mat(bmat)
    amat = _compute_trans_mat(amat)

    return bmat, amat


def orbital_transformation(tra):
    '''
    Constructing the genuine orbital transformation matrix from the
    sequential single-orbital transformation coefficients.

    args:
        tra: np.array
            sequential single-orbital transformation coefficients
    returns:
        cmat: np.array
            orbital transformation matrix
    '''
    tra = np.asarray(tra, dtype=np.float64, order="C")
    ndim = tra.shape[0]
    assert tra.shape == (ndim, ndim), \
        "The sequential transformation must be square"

    cmat = np.eye(ndim)
    for k in range(ndim):
        diag = tra[k, k]
        if abs(diag) < np.finfo(float).eps:
            raise linalg.LinAlgError("Zero diagonal in sequential "
                                     f"transformation at orbital {k}")

        if k:
            coupling = cmat[:, :k] @ tra[:k, k]
        else:
            coupling = np.zeros(ndim)

        cmat[:k, k] = -coupling[:k] / diag
        rhs = coupling[k:] + tra[k:, k]
        rhs[0] = coupling[k] - 1.0
        cmat[k:, k] = -rhs / diag

    return cmat


def transform_ci(ci, tra, ncore, ncas, nelec):
    '''
    Non-unitary transforming a CI vector.

    args:
        ci: np.array
            determinant-basis CI vector
        tra: np.array
            sequential single-orbital transformation coefficients
        ncore: int
            number of inactive orbitals
        ncas: int
            number of active orbitals
        nelec: tuple of ints
            numbers of active alpha and beta electrons
    returns:
        ci: np.array
            transformed determinant-basis CI vector
    '''
    ci = np.array(ci, dtype=np.float64, copy=True, order="C")
    tra = np.asarray(tra, dtype=np.float64, order="C")
    nocc = ncore + ncas
    assert tra.shape == (nocc, nocc), \
        f"Expected a {(nocc, nocc)} transformation, got {tra.shape}"

    # Part-1: transform the inactive orbitals.
    if ncore:
        ci *= np.prod(np.diag(tra)[:ncore]) ** 2

    # Part-2: transform the active orbitals.
    active_tra = tra[ncore:nocc, ncore:nocc]

    link_index = direct_nosym._unpack(ncas, nelec, None)

    for k in range(ncas):
        delta = np.zeros((ncas, ncas))
        delta[:, k] = active_tra[:, k]
        delta[k, k] -= 1.0

        # Part-1: Apply the transformation.
        # TMP = 1/2 A_k CI
        # CI' = CI + (3-T_kk) TMP + A_k TMP
        tmp = 0.5 * direct_nosym.contract_1e(
            delta, ci, ncas, nelec, link_index=link_index)
        
        # Part-2: Update the CI vector.
        ci = (
            ci
            + (3.0 - active_tra[k, k]) * tmp
            + direct_nosym.contract_1e(
                delta, tmp, ncas, nelec, link_index=link_index))

    return ci

@dataclass
class BiorthogonalPair:
    '''
    Pair-specific biorthogonal intermediates.
    '''
    left: int
    right: int
    tra_left: np.ndarray
    tra_right: np.ndarray
    mo_left: np.ndarray
    mo_right: np.ndarray
    ci_left: np.ndarray
    ci_right: np.ndarray
    overlap: complex = 0.0
    hamiltonian: complex = 0.0
    tdm1: np.ndarray | None = None
    tdm2: np.ndarray | None = None
    zsoc: np.ndarray | None = None

def biorthogonalize(
    mo_left,
    mo_right,
    ci_left,
    ci_right,
    ao_overlap,
    ncore,
    ncas,
    nelec_left,
    nelec_right,
    lu_threshold=LU_THRESH,
    check_tol=1e-9,
):
    '''
    Biorthogonalizing one wave-function pair and counter-transforming the
    corresponding CI vectors.

    args:
        mo_left, mo_right: np.array
            MO coefficients for the left and right states
        ci_left, ci_right: np.array
            CI vectors for the left and right states
        ao_overlap: np.array
            AO overlap matrix
        ncore: int
            number of inactive orbitals
        ncas: int
            number of active orbitals
        nelec_left, nelec_right: tuple of ints
            active alpha and beta electrons for the two states
        lu_threshold: float (default 1e-6)
            pseudo-pivoting threshold
        check_tol: float (default 1e-9)
            tolerance for the biorthonormality check
    returns:
        tra_left, tra_right: np.array
            sequential single-orbital transformation coefficients
        mo_left_bi, mo_right_bi: np.array
            biorthonormal MO coefficients
        ci_left_bi, ci_right_bi: np.array
            counter-transformed CI vectors
    '''
    mo_left = np.asarray(mo_left, dtype=np.float64, order="C")
    mo_right = np.asarray(mo_right, dtype=np.float64, order="C")
    ao_overlap = np.asarray(ao_overlap, dtype=np.float64, order="C")
    nocc = ncore + ncas
    assert mo_left.shape == mo_right.shape, \
        "All model states must use the same AO and MO dimensions"
    assert mo_left.shape[0] == ao_overlap.shape[0], \
        "MO coefficients and AO overlap dimensions are inconsistent"
    assert mo_left.shape[1] >= nocc, \
        "MO coefficient arrays do not contain the full CAS space"

    # Part-1: compute the sequential transformations.
    left_occ = mo_left[:, :nocc]
    right_occ = mo_right[:, :nocc]
    sxy = reduce(np.dot, (left_occ.T, ao_overlap, right_occ))
    singular_values = linalg.svdvals(sxy)
    if singular_values[-1] < np.sqrt(lu_threshold) * 1e-4:
        raise linalg.LinAlgError(
            "The inactive+active orbital spaces have a nearly singular "
            f"cross-overlap (smallest singular value "
            f"{singular_values[-1]:.3e})")

    tra_left, tra_right = compute_trans_mat(
        sxy, (ncore, ncas), lu_threshold=lu_threshold)

    # Part-2: construct and apply the orbital transformation matrices.
    c_left = orbital_transformation(tra_left)
    c_right = orbital_transformation(tra_right)

    mo_left_bi = mo_left.copy()
    mo_right_bi = mo_right.copy()
    mo_left_bi[:, :nocc] = left_occ @ c_left
    mo_right_bi[:, :nocc] = right_occ @ c_right
    cross = reduce(np.dot, (mo_left_bi[:, :nocc].T,
                            ao_overlap,
                            mo_right_bi[:, :nocc]))
    if not np.allclose(cross, np.eye(nocc), atol=check_tol, rtol=check_tol):
        raise RuntimeError(
            "Malmqvist orbital transformation failed the biorthonormality "
            f"check; maximum error is "
            f"{np.max(np.abs(cross - np.eye(nocc))):.3e}")

    # Part-3: counter-transform the CI vectors.
    ci_left_bi = transform_ci(ci_left, tra_left, ncore, ncas, nelec_left)
    ci_right_bi = transform_ci(ci_right, tra_right, ncore, ncas, nelec_right)

    return (tra_left, tra_right, mo_left_bi, mo_right_bi,
            ci_left_bi, ci_right_bi)

