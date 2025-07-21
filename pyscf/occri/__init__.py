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
    >>> kpts = cell.make_kpts([2,2,2])
    >>> mf = scf.KRHF(cell, kpts)
    >>> mf.with_df = OCCRI(mf, kmesh=[2,2,2])
    >>> energy = mf.kernel()

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
from pyscf.occri import occri_k

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
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        ctypes.c_int,
    ]
    _OCCRI_C_AVAILABLE = True
    print("OCCRI: Using optimized C implementation with FFTW, BLAS, and OpenMP")
    
except (OSError, ImportError, AttributeError) as e:
    print(f"OCCRI: C extension not available ({str(e)}), falling back to pure Python implementation")
    print("OCCRI: Performance will be reduced. Consider installing FFTW, BLAS, and OpenMP for optimal speed")
    _OCCRI_C_AVAILABLE = False


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

    def __init__(self, mydf, kmesh = [1,1,1], **kwargs,):
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
        cell = mydf.cell
        super().__init__(cell=cell)  # Need this for pyscf's eval_ao function
        self.exxdiv = "ewald"
        self.cell = cell
        self.kmesh = kmesh
        
        self.Nk = numpy.prod(self.kmesh)
        self.kpts = self.cell.make_kpts(
            self.kmesh, space_group_symmetry=False, time_reversal_symmetry=False, wrap_around=True
        )

        self.joblib_njobs = lib.numpy_helper._np_helper.get_omp_threads()

        self.get_j = pyscf.pbc.df.fft_jk.get_j_kpts

        if str(self.method[0]) == 'k':
            self.get_jk = self.get_jk_kpts
            self.get_k = occri_k.get_k_occri_kpts
        else:
            # Choose implementation based on C extension availability
            if _OCCRI_C_AVAILABLE:
                self.get_k = occri_k.occri_get_k_opt  # Optimized C implementation
            else:
                self.get_k = occri_k.occri_get_k      # Pure Python fallback

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
        dm = dm.reshape(-1, dm_shape[-2], dm_shape[-1])
        
        if with_j:
            vj = self.get_j(self, dm)

        if with_k:
            vk = self.get_k(self, dm, exxdiv)

        if with_j:
            vj = numpy.asarray(vj, dtype=dm.dtype).reshape(dm_shape)
        else:
            vj = None

        if with_k:
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
