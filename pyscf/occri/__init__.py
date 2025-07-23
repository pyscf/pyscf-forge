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

Basic Usage:
    >>> from pyscf.pbc import gto, scf
    >>> from pyscf.occri import OCCRI
    >>> 
    >>> cell = gto.Cell()
    >>> cell.atom = 'C 0 0 0; C 1 1 1'
    >>> cell.a = numpy.eye(3) * 4
    >>> cell.basis = 'gth-dzvp'
    >>> cell.build()
    >>> 
    >>> mf = scf.RHF(cell)
    >>> mf.with_df = OCCRI(mf)
    >>> energy = mf.kernel()

Advanced Usage with k-points:
    >>> from pyscf.pbc import scf
    >>> kpts = cell.make_kpts([2,2,2])  # 2x2x2 k-point mesh
    >>> mf = scf.KRHF(cell, kpts)
    >>> mf.with_df = OCCRI(mf, kmesh=[2,2,2])  # k-point OCCRI
    >>> energy = mf.kernel()
    
k-point Features:
    - Handles complex Bloch functions and k-point phase factors
    - Supports arbitrary k-point meshes with proper momentum conservation
    - Optimized C implementation for complex FFTs and phase factor handling
    - Automatic fallback to Python implementation if C extension unavailable
    - Compatible with both collinear and non-collinear spin systems

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

Performance:
    - C extension with all dependencies: ~10-20x faster than Python
    - Pure Python fallback: Maintains chemical accuracy but slower performance
    - Recommended: Install FFTW3, BLAS, and OpenMP for best performance

References:
    [1] Original OCCRI method development and implementation
    [2] PySCF: the Python‐based simulations of chemistry framework
        WIREs Comput Mol Sci. 2018;8:e1340
"""

import pyscf
import numpy
import time
import ctypes
from pyscf import lib
from pyscf.occri import occri_k, occri_k_kpts
from pyscf.lib import logger

# Attempt to load the optimized C extension with FFTW, BLAS, and OpenMP
# Fall back to pure Python implementation if unavailable
_OCCRI_C_AVAILABLE = False
liboccri = None
occri_vR = None

try:
    # Load the shared library
    liboccri = lib.load_library('liboccri')
    
    # Bind functions
    ndpointer = numpy.ctypeslib.ndpointer
    
    occri_vR = liboccri.occri_vR
    occri_vR.restype = None
    occri_vR.argtypes = [
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # vk_out - output exchange matrix (nao x nao)
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # mo_coeff - MO coefficients (nmo x nao)
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # mo_occ - occupation numbers (nmo,)
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # aovals - AO values on grid (nao x ngrids)
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # coulG - Coulomb kernel (ncomplex,)
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # overlap - overlap matrix (nao x nao)
        ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),     # mesh - FFT mesh dimensions [nx,ny,nz]
        ctypes.c_int,                                      # nmo - number of MOs
        ctypes.c_int,                                      # nao - number of AOs
        ctypes.c_int,                                      # ngrids - number of grid points
    ]
    
    # k-point exchange function with complex FFT support
    occri_vR_kpts = liboccri.occri_vR_kpts
    occri_vR_kpts.restype = None
    occri_vR_kpts.argtypes = [
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # vR_dm_real - output exchange potential (real part)
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # vR_dm_imag - output exchange potential (imag part) 
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # mo_occ - occupation numbers (all k-points, padded)
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # coulG_all - Coulomb kernels for all k-point differences
        ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),     # mesh - FFT mesh dimensions [nx,ny,nz]
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # expmikr_all_real - phase factors exp(-i(k-k')·r) real
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # expmikr_all_imag - phase factors exp(-i(k-k')·r) imag
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # kpts - k-point coordinates (currently unused)
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # ao_mos_real - orbital data all k-points (real, padded)
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # ao_mos_imag - orbital data all k-points (imag, padded)
        ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),     # nmo - number of orbitals per k-point array
        ctypes.c_int,                                      # ngrids - total number of real-space grid points
        ctypes.c_int,                                      # nk - total number of k-points
        ctypes.c_int,                                      # k_idx - target k-point index for computation
    ]
    _OCCRI_C_AVAILABLE = True
    
except (OSError, ImportError, AttributeError) as e:
    print(f"OCCRI: C extension not available ({str(e)}), falling back to pure Python implementation")
    print("OCCRI: Performance will be reduced. Consider installing FFTW, BLAS, and OpenMP for optimal speed")
    _OCCRI_C_AVAILABLE = False

def log_mem(mydf):
    cell = mydf.cell
    nao = cell.nao
    ngrids = numpy.prod(cell.mesh)
    mem_now = lib.current_memory()[0]
    max_memory = mydf.max_memory - mem_now
    blksize = int(min(nao, max(1, (max_memory-mem_now)*1e6/16/4/ngrids/nao)))
    logger.debug1(mydf, 'fft_jk: get_k_kpts max_memory %s  blksize %d',
                  max_memory, blksize)
    
def build_full_exchange(S, Kao, mo_coeff):
    """
    Construct full exchange matrix from occupied orbital components.
    
    This function builds the complete exchange matrix in the atomic orbital (AO)
    basis from the occupied-occupied (Koo) and occupied-all (Koa) components
    computed using the resolution of identity approximation.
    
    Parameters:
    -----------
    Sa : numpy.ndarray
        Overlap matrix times MO coefficients (nao x nocc)
    Kao : numpy.ndarray
        Occupied-all exchange matrix components (nao x nocc)
    Koo : numpy.ndarray
        Occupied-occupied exchange matrix components (nocc x nocc)
        
    Returns:
    --------
    numpy.ndarray
        Full exchange matrix in AO basis (nao x nao)
        
    Algorithm:
    ----------
    K_μν = Sa_μi * Koa_iν + Sa_νi * Koa_iμ - Sa_μi * Koo_ij * Sa_νj
    
    This corresponds to the resolution of identity expression:
    K_μν ≈ Σ_P C_μP W_PP' C_νP' where C are fitting coefficients
    """

    # Compute Sa = S @ mo_coeff.T once and reuse
    Sa = S @ mo_coeff.T
    
    # First and second terms: Sa @ Kao.T + (Sa @ Kao.T).T
    # This is equivalent to Sa @ Kao.T + Kao @ Sa.T
    # Use symmetric rank-k update (SYRK) when possible
    Sa_Kao = numpy.matmul(Sa, Kao.T.conj(), order='C')
    Kuv = Sa_Kao + Sa_Kao.T.conj()
    
    # Third term: -Sa @ (mo_coeff @ Kao) @ Sa.T
    # Optimize as -Sa @ Koo @ Sa.T using GEMM operations
    Koo = mo_coeff.conj() @ Kao
    Sa_Kao = numpy.matmul(Sa, Koo)
    Kuv -= numpy.matmul(Sa_Kao, Sa.T.conj(), order='C')
    return Kuv    

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
        df_obj: Reference to the original density fitting object
        cell: The unit cell object
        kmesh (list): k-point mesh dimensions [nkx, nky, nkz]
        Nk (int): Total number of k-points
        kpts (ndarray): Array of k-point coordinates
        joblib_njobs (int): Number of OpenMP threads available
    
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

    def __init__(self, mydf, kmesh = [1,1,1], disable_c=False, **kwargs,):
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
        
        self.method = mydf.__module__.rsplit('.', 1)[-1]
        self.df_obj = mydf
        
        assert self.method in ['hf', 'uhf', 'khf', 'kuhf', 'rks', 'uks', 'krks', 'kuks']
        
        self.StartTime = time.time()
        self.cell = mydf.cell
        self.kmesh = kmesh
        self.Nk = numpy.prod(self.kmesh)
        self.kpts = self.cell.make_kpts(
            self.kmesh, space_group_symmetry=False, time_reversal_symmetry=False, wrap_around=True
        )        
        super().__init__(cell=self.cell, kpts=self.kpts)  # Need this for pyscf's eval_ao function
        self.exxdiv = "ewald"
        
        self.get_j = pyscf.pbc.df.fft_jk.get_j_kpts

        if _OCCRI_C_AVAILABLE and not disable_c:
            print("OCCRI: Using optimized C implementation with FFTW, BLAS, and OpenMP")

        if str(self.method[0]) == 'k':
            # Choose k-point implementation based on C extension availability
            # k-point methods handle complex Bloch functions and phase factors
            if _OCCRI_C_AVAILABLE and not disable_c:
                self.get_k = occri_k_kpts.occri_get_k_opt_kpts  # Optimized C k-point implementation with complex FFT
            else:
                self.get_k = occri_k_kpts.occri_get_k_kpts      # Pure Python k-point fallback with complex arithmetic
        else:
            # Choose implementation based on C extension availability
            if _OCCRI_C_AVAILABLE and not disable_c:
                self.get_k = occri_k.occri_get_k_opt  # Optimized C implementation
            else:
                self.get_k = occri_k.occri_get_k      # Pure Python fallback

        # # Print all attributes
        # print()
        # print("******** <class 'OCCRI'> ********", flush=True)
        # for key, value in vars(self).items():
        #     print(f"{key}: {value}", flush=True)


    def get_jk(
        self,
        dm=None,
        hermi=1,
        kpt=None,
        kpts_band=None,
        with_j=None,
        with_k=None,
        omega=None,
        exxdiv=None,
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
            
        Notes:
        ------
        The method automatically reshapes input density matrices to handle
        both single matrices and batches of matrices consistently.
        """
        dm_shape = dm.shape
        nK = self.Nk
        mo_coeff, mo_occ = [None]* 2
        if getattr(dm, "mo_coeff", None) is not None:
            mo_coeff = numpy.asarray(dm.mo_coeff)
            mo_occ = numpy.asarray(dm.mo_occ)
        if str(self.get_k.__name__[-4:]) == 'kpts':
            dm = dm.reshape(-1, nK, dm_shape[-2], dm_shape[-1])
        else:
            dm = dm.reshape(-1, dm_shape[-2], dm_shape[-1])
        
        if mo_coeff is not None:
            dm = lib.tag_array(dm, mo_occ=mo_occ.reshape(dm.shape[0], self.Nk, self.cell.nao), mo_coeff=mo_coeff.reshape(dm.shape))
        
        if with_j:
            vj = self.get_j(self, dm, kpts=self.kpts)
            if abs(dm.imag).max() < 1.e-6:
                vj = vj.real
            vj = numpy.asarray(vj, dtype=dm.dtype).reshape(dm_shape)
        else:
            vj = None

        if with_k:
            vk = self.get_k(self, dm, exxdiv)
            if abs(dm.imag).max() < 1.e-6:
                vk = vk.real 
            vk = numpy.asarray(vk, dtype=dm.dtype).reshape(dm_shape)
        else:
            vk = None

        return vj, vk

    def __del__(self):
        return

    def copy(self):
        """
        Create a shallow copy of the OCCRI object.
        
        Returns:
        --------
        OCCRI
            A shallow copy of the current OCCRI instance
            
        Notes:
        ------
        This creates a new view of the same data rather than duplicating
        the underlying arrays, which is memory efficient for large systems.
        """
        return self.view(self.__class__)

    def get_keyword_arguments(self):
        """
        Retrieve all keyword arguments for the OCCRI object.
        
        This method extracts all object attributes except the cell object,
        which is useful for serialization or creating similar objects.
        
        Returns:
        --------
        dict
            Dictionary of all attributes except 'cell'
            
        Notes:
        ------
        The cell object is excluded because it's typically handled separately
        and contains complex nested data structures.
        """
        return {key: value for key, value in self.__dict__.items() if key != "cell"}
