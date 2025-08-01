"""
OCCRI (Occupied Orbital Coulomb Resolution of Identity) Interface for PySCF

This module provides an efficient implementation of the resolution-of-identity (RI)
approximation for computing exact exchange in periodic systems using PySCF. The
method is particularly effective for systems requiring many k-points and uses
FFT-based techniques for optimal performance.

The OCCRI approach exploits the fact that only occupied orbitals contribute to
the exchange interaction, significantly reducing computational cost compared to
traditional methods while maintaining chemical accuracy.

Key Features:
    - Efficient exchange matrix evaluation using occupied orbital RI
    - FFT-based Coulomb potential evaluation in real space
    - OpenMP parallelization through C extension
    - Support for periodic boundary conditions (3D systems)
    - Compatible with RHF, UHF, RKS, UKS, and their k-point variants

Theory:
    The OCCRI method approximates the exchange matrix elements using:
    K_μν ≈ Σ_P C_μP W_PP' C_νP'

    where C_μP are fitting coefficients related to occupied orbitals and
    W_PP' represents the Coulomb interaction in the auxiliary basis.

Dependencies:
    Required:
        - PySCF >= 2.0
        - NumPy >= 1.17
        - SciPy >= 1.5

    Optional (for optimal performance):
        - FFTW3 (for optimized FFT operations)
        - BLAS (for optimized linear algebra)
        - OpenMP (for parallelization)

Build Configuration:
    The module automatically detects available dependencies and falls back to
    pure Python implementation if the optimized C extension cannot be built.

    To control the build process:
        - Set BUILD_OCCRI=OFF to disable C extension build entirely
        - Set CMAKE_CONFIGURE_ARGS for additional CMake options

    Example:
        BUILD_OCCRI=OFF pip install .  # Force Python-only build
        pip install .                  # Auto-detect dependencies

"""

import ctypes
import time

import numpy
import scipy

import pyscf
from pyscf import lib
from pyscf.lib import logger
from pyscf.occri import occri_k_kpts

# Attempt to load the optimized C extension with FFTW, BLAS, and OpenMP
# Fall back to pure Python implementation if unavailable
_OCCRI_C_AVAILABLE = False
liboccri = None
occri_vR = None

try:
    # Load the shared library
    liboccri = lib.load_library("liboccri")

    # Bind functions
    ndpointer = numpy.ctypeslib.ndpointer

    occri_vR = liboccri.occri_vR
    occri_vR.restype = None
    occri_vR.argtypes = [
        ndpointer(
            ctypes.c_double, flags="C_CONTIGUOUS"
        ),  # vk_out - output exchange matrix (nao x nao)
        ndpointer(
            ctypes.c_double, flags="C_CONTIGUOUS"
        ),  # mo_coeff - MO coefficients (nmo x nao)
        ndpointer(
            ctypes.c_double, flags="C_CONTIGUOUS"
        ),  # mo_occ - occupation numbers (nmo,)
        ndpointer(
            ctypes.c_double, flags="C_CONTIGUOUS"
        ),  # aovals - AO values on grid (nao x ngrids)
        ndpointer(
            ctypes.c_double, flags="C_CONTIGUOUS"
        ),  # coulG - Coulomb kernel (ncomplex,)
        ndpointer(
            ctypes.c_double, flags="C_CONTIGUOUS"
        ),  # overlap - overlap matrix (nao x nao)
        ndpointer(
            ctypes.c_int, flags="C_CONTIGUOUS"
        ),  # mesh - FFT mesh dimensions [nx,ny,nz]
        ctypes.c_int,  # nmo - number of MOs
        ctypes.c_int,  # nao - number of AOs
        ctypes.c_int,  # ngrids - number of grid points
        ctypes.c_double,  # weight
    ]

    # k-point exchange function with complex FFT support
    occri_vR_kpts = liboccri.occri_vR_kpts
    occri_vR_kpts.restype = None
    occri_vR_kpts.argtypes = [
        ndpointer(
            ctypes.c_double, flags="C_CONTIGUOUS"
        ),  # vR_dm_real - output exchange potential (real part)
        ndpointer(
            ctypes.c_double, flags="C_CONTIGUOUS"
        ),  # vR_dm_imag - output exchange potential (imag part)
        ndpointer(
            ctypes.c_double, flags="C_CONTIGUOUS"
        ),  # mo_occ - occupation numbers (all k-points, padded)
        ndpointer(
            ctypes.c_double, flags="C_CONTIGUOUS"
        ),  # coulG_all - Coulomb kernels for all k-point differences
        ndpointer(
            ctypes.c_int, flags="C_CONTIGUOUS"
        ),  # mesh - FFT mesh dimensions [nx,ny,nz]
        ndpointer(
            ctypes.c_double, flags="C_CONTIGUOUS"
        ),  # expmikr_all_real - phase factors exp(-i(k-k')·r) real
        ndpointer(
            ctypes.c_double, flags="C_CONTIGUOUS"
        ),  # expmikr_all_imag - phase factors exp(-i(k-k')·r) imag
        ndpointer(
            ctypes.c_double, flags="C_CONTIGUOUS"
        ),  # kpts - k-point coordinates (currently unused)
        ndpointer(
            ctypes.c_double, flags="C_CONTIGUOUS"
        ),  # ao_mos_real - orbital data all k-points (real, padded)
        ndpointer(
            ctypes.c_double, flags="C_CONTIGUOUS"
        ),  # ao_mos_imag - orbital data all k-points (imag, padded)
        ndpointer(
            ctypes.c_int, flags="C_CONTIGUOUS"
        ),  # nmo - number of orbitals per k-point array
        ctypes.c_int,  # ngrids - total number of real-space grid points
        ctypes.c_int,  # nk - total number of k-points
        ctypes.c_int,  # k_idx - target k-point index for computation
    ]
    _OCCRI_C_AVAILABLE = True

except (OSError, ImportError, AttributeError) as e:
    print(
        f"OCCRI: C extension not available ({str(e)}), falling back to pure Python implementation"
    )
    print(
        "OCCRI: Performance will be reduced. Consider installing FFTW, BLAS, and OpenMP for optimal speed"
    )
    _OCCRI_C_AVAILABLE = False


def log_mem(mydf):
    cell = mydf.cell
    nao = cell.nao
    ngrids = numpy.prod(cell.mesh)
    mem_now = lib.current_memory()[0]
    max_memory = mydf.max_memory - mem_now
    blksize = int(
        min(nao, max(1, (max_memory - mem_now) * 1e6 / 16 / 4 / ngrids / nao))
    )
    logger.debug1(
        mydf, "fft_jk: get_k_kpts max_memory %s  blksize %d", max_memory, blksize
    )


def make_natural_orbitals(mydf, dms):
    """
    Parameters:
    -----------
    cell : pyscf.pbc.gto.Cell
        Unit cell object containing atomic and basis set information
    dms : ndarray
        Density matrix or matrices in AO basis, shape (..., nao, nao)
    kpts : ndarray, optional
        k-point coordinates. If None, assumes Gamma point calculation

    Returns:
    --------
    tuple of ndarray
        (mo_coeff, mo_occ) where:
        - mo_coeff: Natural orbital coefficients, same shape as dms
                  mo_occ[n, k, i] = occupation of orbital i at k-point k, spin n
                  Real values ordered from highest to lowest occupation
    """
    # print("Building Orbitals")
    cell = mydf.cell
    kpts = mydf.kpts
    nk = kpts.shape[0]
    nao = cell.nao
    nset = dms.shape[0]

    # Compute k-point dependent overlap matrices
    sk = cell.pbc_intor("int1e_ovlp", hermi=1, kpts=kpts)
    if abs(dms.imag).max() < 1.0e-6:
        sk = [s.real.astype(numpy.float64) for s in sk]

    mo_coeff = numpy.zeros_like(dms)
    mo_occ = numpy.zeros((nset, nk, nao), numpy.float64)
    for i, dm in enumerate(dms):
        for k, s in enumerate(sk):
            # Diagonalize the DM in AO
            A = lib.reduce(numpy.dot, (s, dm[k], s))
            w, v = scipy.linalg.eigh(A, b=s)

            # Flip since they're in increasing order
            mo_occ[i][k] = numpy.flip(w)
            mo_coeff[i][k] = numpy.flip(v, axis=1)

    return lib.tag_array(dms, mo_coeff=mo_coeff, mo_occ=mo_occ)


class OCCRI(pyscf.pbc.df.fft.FFTDF):
    """
    Occupied Orbital Coulomb Resolution of Identity (OCCRI) density fitting class.

    This class implements the OCCRI method for efficient evaluation of exact exchange
    in periodic systems. It extends PySCF's FFTDF class to provide optimized exchange
    matrix construction using the resolution of identity approximation restricted to
    occupied orbitals.

    The method is particularly efficient for systems with many k-points and provides
    significant speedup over traditional exact exchange implementations while
    maintaining chemical accuracy.

    Attributes:
        method (str): The type of mean-field method being used (e.g., 'rhf', 'uhf', 'rks', 'uks')
        cell: The unit cell object
        kmesh (list): k-point mesh dimensions [nkx, nky, nkz]
        kpts (ndarray): Array of k-point coordinates

    Example:
        >>> from pyscf.pbc import gto, scf
        >>> from pyscf.occri import OCCRI
        >>>
        >>> cell = gto.Cell()
        >>> # ... set up cell ...
        >>> mf = scf.RHF(cell)
        >>> mf.with_df = OCCRI(mf)
        >>> energy = mf.kernel()
    """

    def __init__(
        self,
        mydf,
        kmesh=[1, 1, 1],
        disable_c=False,
        **kwargs,
    ):
        """
        Initialize the OCCRI density fitting object.

        Parameters:
        -----------
        mydf : SCF object
            The self-consistent field object (RHF, UHF, RKS, UKS, or k-point variants)
        kmesh : list of int, optional
            k-point mesh dimensions [nkx, nky, nkz]. Default is [1,1,1] (Gamma point)
        **kwargs : dict
            Additional keyword arguments passed to parent FFTDF class

        Raises:
        -------
        AssertionError
            If the method type is not supported (must be one of:
            'hf', 'uhf', 'khf', 'kuhf', 'rks', 'uks', 'krks', 'kuks')
        """

        mydf._is_mem_enough = lambda: False
        # NOTE: PySCF calls get_jk -> get_jk passing dm.reshape(...), dropping the mo_coeff.
        # As a work around, overwrite get_jk so orbitals aren't dropped
        mydf.get_jk = self.get_jk
        # BUG: Some PySCF methods have bugs when tagging initial guess dm.
        # For example, RKRS can modify dm without modifying mo_occ in the same way.
        # As a work around, always diagonalize dm on first iteration. See 02-kpoint...
        self.scf_iter = 0

        self.method = mydf.__module__.rsplit(".", 1)[-1]
        assert self.method in ["hf", "uhf", "khf", "kuhf", "rks", "uks", "krks", "kuks"]

        self.StartTime = time.time()
        self.cell = mydf.cell
        self.kmesh = kmesh
        self.kpts = self.cell.make_kpts(
            self.kmesh,
            space_group_symmetry=False,
            time_reversal_symmetry=False,
            wrap_around=True,
        )
        super().__init__(
            cell=self.cell, kpts=self.kpts
        )  # Need this for pyscf's eval_ao function
        self.exxdiv = "ewald"

        self.get_j = pyscf.pbc.df.fft_jk.get_j_kpts

        if _OCCRI_C_AVAILABLE and not disable_c and self.cell.verbose > 3:
            print("OCCRI: Using optimized C implementation with FFTW, BLAS, and OpenMP")

        if _OCCRI_C_AVAILABLE and not disable_c:
            self.get_k = (
                occri_k_kpts.occri_get_k_kpts_opt
            )  # Optimized C k-point implementation with complex FFT
        else:
            self.get_k = (
                occri_k_kpts.occri_get_k_kpts
            )  # Pure Python k-point fallback with complex arithmetic

        # # Print all attributes
        # print()
        # print("******** <class 'OCCRI'> ********", flush=True)
        # for key, value in vars(self).items():
        #     print(f"{key}: {value}", flush=True)

    def get_jk(
        self,
        cell=None,
        dm=None,
        hermi=1,
        kpt=None,
        kpts_band=None,
        with_j=True,
        with_k=True,
        omega=None,
        exxdiv="ewald",
        **kwargs,
    ):
        """
        Compute Coulomb (J) and exchange (K) matrices using OCCRI method.

        This method combines the efficient J matrix evaluation from PySCF's FFTDF
        with the optimized K matrix evaluation from OCCRI. The exchange part uses
        the occupied orbital resolution of identity approach for improved performance.

        Parameters:
        -----------
        dm : ndarray
            Density matrix or matrices in AO basis
        hermi : int, optional
            Hermiticity flag (1 for Hermitian, 0 for non-Hermitian). Default is 1
        kpt : ndarray, optional
            Single k-point (not used in current implementation)
        kpts_band : ndarray, optional
            k-points for band structure (not used in current implementation)
        with_j : bool, optional
            Whether to compute Coulomb matrix. Default inferred from context
        with_k : bool, optional
            Whether to compute exchange matrix. Default inferred from context
        omega : float, optional
            Range separation parameter (not used in current implementation)
        exxdiv : str, optional
            Treatment of exchange divergence. 'ewald' adds Ewald correction
        **kwargs : dict
            Additional keyword arguments

        Returns:
        --------
        tuple of ndarray
            (vj, vk) where:
            - vj: Coulomb matrix (or None if with_j=False)
            - vk: Exchange matrix (or None if with_k=False)
        """
        if cell is None:
            cell = self.cell
        if dm is None:
            AttributeError(
                "Overwriting get_jk. "
                "Pass dm to get_jk as keyword: get_jk(dm=dm, ...)"
            )

        dm_shape = dm.shape
        nk = self.kpts.shape[0]
        nao = cell.nao
        if with_k:
            if getattr(dm, "mo_coeff", None) is None or self.scf_iter == 0:
                dm = make_natural_orbitals(self, dm.reshape(-1, nk, nao, nao))
            else:
                mo_coeff = numpy.asarray(dm.mo_coeff).reshape(-1, nk, nao, nao)
                mo_occ = numpy.asarray(dm.mo_occ).reshape(-1, nk, nao)
                dm = lib.tag_array(
                    dm.reshape(-1, nk, nao, nao), mo_coeff=mo_coeff, mo_occ=mo_occ
                )

        if with_j:
            vj = self.get_j(self, dm, kpts=self.kpts)
            if abs(dm.imag).max() < 1.0e-6:
                vj = vj.real
            vj = numpy.asarray(vj, dtype=dm.dtype).reshape(dm_shape)
        else:
            vj = None

        if with_k:

            mo_coeff = dm.mo_coeff
            mo_occ = dm.mo_occ
            tol = 1.0e-6
            is_occ = mo_occ > tol
            nset = dm.shape[0]
            mo_coeff = [
                [
                    numpy.ascontiguousarray(mo_coeff[n][k][:, is_occ[n][k]].T)
                    for k in range(nk)
                ]
                for n in range(nset)
            ]
            mo_occ = [
                [numpy.ascontiguousarray(mo_occ[n][k][is_occ[n][k]]) for k in range(nk)]
                for n in range(nset)
            ]
            dm = lib.tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)

            vk = self.get_k(self, dm, exxdiv)
            if abs(dm.imag).max() < 1.0e-6:
                vk = vk.real
            vk = numpy.asarray(vk, dtype=dm.dtype).reshape(dm_shape)
        else:
            vk = None

        self.scf_iter += 1

        return vj, vk

    def __del__(self):
        return

    def copy(self):
        """
        Create a shallow copy of the OCCRI object.
        """
        return self.view(self.__class__)

    def get_keyword_arguments(self):
        """
        Retrieve all keyword arguments for the OCCRI object.
        """
        return {key: value for key, value in self.__dict__.items() if key != "cell"}
