"""
OCCRI (Occupied Orbital Coulomb Resolution of Identity) for efficient
exact exchange evaluation in periodic systems.
"""

import ctypes
import time

import numpy
import pyscf
import scipy
from pyscf import lib
from pyscf.lib import logger
from pyscf.occri import occri_k_kpts
from pyscf.pbc.tools.k2gamma import kpts_to_kmesh

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
    occri_vR.restype = ctypes.c_int
    occri_vR.argtypes = [
        ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # vk_out - output exchange matrix (nao x nao)
        ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # mo_coeff - MO coefficients (nmo x nao)
        ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # mo_occ - occupation numbers (nmo,)
        ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # aovals - AO values on grid (nao x ngrids)
        ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # coulG - Coulomb kernel (ncomplex,)
        ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # overlap - overlap matrix (nao x nao)
        ndpointer(ctypes.c_int, flags='C_CONTIGUOUS'),  # mesh - FFT mesh dimensions [nx,ny,nz]
        ctypes.c_int,  # nmo - number of MOs
        ctypes.c_int,  # nao - number of AOs
        ctypes.c_int,  # ngrids - number of grid points
        ctypes.c_double,  # weight
    ]

    # k-point exchange function with complex FFT support
    occri_vR_kpts = liboccri.occri_vR_kpts
    occri_vR_kpts.restype = ctypes.c_int
    occri_vR_kpts.argtypes = [
        ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # vR_dm_real - output exchange potential (real part)
        ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # vR_dm_imag - output exchange potential (imag part)
        ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # mo_occ - occupation numbers (all k-points, padded)
        ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # coulG_all - Coulomb kernels for all k-point differences
        ndpointer(ctypes.c_int, flags='C_CONTIGUOUS'),  # mesh - FFT mesh dimensions [nx,ny,nz]
        ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # expmikr_all_real - phase factors exp(-i(k-k')·r) real
        ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # expmikr_all_imag - phase factors exp(-i(k-k')·r) imag
        ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # kpts - k-point coordinates (currently unused)
        ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # ao_mos_real - orbital data all k-points (real, padded)
        ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # ao_mos_imag - orbital data all k-points (imag, padded)
        ndpointer(ctypes.c_int, flags='C_CONTIGUOUS'),  # nmo - number of orbitals per k-point array
        ctypes.c_int,  # ngrids - total number of real-space grid points
        ctypes.c_int,  # nk - total number of k-points
        ctypes.c_int,  # k_idx - target k-point index for computation
    ]
    _OCCRI_C_AVAILABLE = True

except (OSError, ImportError, AttributeError) as e:
    print(f'OCCRI: C extension not available ({str(e)}), falling back to pure Python implementation')
    print('OCCRI: Performance will be reduced. Consider installing FFTW, BLAS, and OpenMP for optimal speed')
    _OCCRI_C_AVAILABLE = False


def log_mem(mydf):
    cell = mydf.cell
    nao = cell.nao
    ngrids = numpy.prod(cell.mesh)
    mem_now = lib.current_memory()[0]
    max_memory = mydf.max_memory - mem_now
    blksize = int(min(nao, max(1, (max_memory - mem_now) * 1e6 / 16 / 4 / ngrids / nao)))
    logger.debug1(mydf, 'fft_jk: get_k_kpts max_memory %s  blksize %d', max_memory, blksize)


class OCCRI(pyscf.pbc.df.fft.FFTDF):
    """
    Occupied Orbital Coulomb Resolution of Identity density fitting
    Defaults to incore algorithm if memory is sufficient.
    To require OCCRI method, set mf._is_mem_enough = lambda: False or
    use OCCRI.from_mf(mf) method.
    """

    def __init__(
        self,
        cell,
        kpts=None,
        disable_c=False,
        **kwargs,
    ):
        """Initialize OCCRI density fitting object

        Parameters
        ----------
        cell : pyscf.pbc.gto.Cell
            Unit cell object
        kpts : ndarray, optional
            k-points for periodic boundary conditions
        disable_c : bool, optional
            If True, use pure Python implementation
        """
        # NOTE: Some PySCF methods (see RKRS) do not modify mo_occ like dm on initial guess build.
        # As a work around, always diagonalize dm on first iteration. See 02-kpoint_calculations.py.
        self.scf_iter = 0

        self.StartTime = time.time()
        self.cell = cell
        if kpts is None:
            self.kpts = numpy.zeros(3, numpy.float64)
            self.kmesh = [1, 1, 1]
        else:
            self.kmesh = kpts_to_kmesh(self.cell, kpts, precision=None, rcut=None)
            self.kpts = self.cell.make_kpts(
                self.kmesh,
                space_group_symmetry=False,
                time_reversal_symmetry=False,
                wrap_around=True,
            )
        super().__init__(cell=self.cell, kpts=self.kpts)  # Need this for pyscf's eval_ao function
        self.exxdiv = 'ewald'

        self.get_j = pyscf.pbc.df.fft_jk.get_j_kpts

        if _OCCRI_C_AVAILABLE and not disable_c:
            self.get_k = occri_k_kpts.occri_get_k_kpts_opt  # Optimized C k-point implementation with complex FFT
        else:
            self.get_k = occri_k_kpts.occri_get_k_kpts  # Pure Python k-point fallback with complex arithmetic

        self.dump_flags()

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('******** %s ********', self.__class__)
        log.info('get_k = %s', self.get_k)
        log.info('kmesh = %s', self.kmesh)
        return self

    @classmethod
    def from_mf(cls, mf, disable_c=False, **kwargs):
        """Create OCCRI instance from mean-field object

        Parameters
        ----------
        mf : pyscf mean-field object
            Mean-field instance (RHF, UHF, RKS, UKS, etc.)
        disable_c : bool, optional
            If True, use pure Python implementation
        **kwargs
            Additional arguments passed to OCCRI constructor

        Returns
        -------
        OCCRI
            OCCRI density fitting instance configured for the given mean-field methods


        Example
        -------
        from pyscf.pbc import gto, scf
        from pyscf.occri import OCCRI

        # Set up cell
        cell = gto.Cell()
        cell.atom = 'H 0 0 0; H 0 0 1'
        cell.basis = 'sto3g'
        cell.build()

        # Create mean-field object
        mf = scf.RHF(cell)

        # Use factory method to create OCCRI
        mf.with_df = OCCRI.from_mf(mf)

        # Run calculation
        energy = mf.kernel()

        Alternative direct construction:

        # You can also create OCCRI directly if you have cell and kpts
        # However! It will default to the incore algo if memory is sufficient
        occri = OCCRI(cell, kpts=None)  # Direct construction
        mf.with_df = occri
        """
        # Validate mean-field instance
        mf._is_mem_enough = lambda: False

        # Extract method information
        method = mf.__module__.rsplit('.', 1)[-1]
        assert method in [
            'hf',
            'uhf',
            'khf',
            'kuhf',
            'rks',
            'uks',
            'krks',
            'kuks',
        ], f'Unsupported mean-field method: {method}'

        # Create OCCRI instance
        occri = cls(mf.cell, mf.kpts, disable_c=disable_c, **kwargs)
        occri.method = method

        return occri

    def get_jk(
        self,
        dm=None,
        hermi=1,
        kpt=None,
        kpts_band=None,
        with_j=True,
        with_k=True,
        omega=None,
        exxdiv='ewald',
        **kwargs,
    ):
        """Compute J and K matrices using OCCRI"""
        cell = self.cell
        if isinstance(dm, list):
            dm = numpy.asarray(dm)
        dm_shape = dm.shape
        nk = self.kpts.shape[0]
        nao = cell.nao
        if with_k:
            if self.scf_iter == 0:
                dm = numpy.asarray(dm)
            if getattr(dm, 'mo_coeff', None) is None:
                dm = self.make_natural_orbitals(dm.reshape(-1, nk, nao, nao))
            else:
                mo_coeff = numpy.asarray(dm.mo_coeff).reshape(-1, nk, nao, nao)
                mo_occ = numpy.asarray(dm.mo_occ).reshape(-1, nk, nao)
                dm = lib.tag_array(dm.reshape(-1, nk, nao, nao), mo_coeff=mo_coeff, mo_occ=mo_occ)

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
                [numpy.ascontiguousarray(mo_coeff[n][k][:, is_occ[n][k]].T) for k in range(nk)] for n in range(nset)
            ]
            mo_occ = [[numpy.ascontiguousarray(mo_occ[n][k][is_occ[n][k]]) for k in range(nk)] for n in range(nset)]
            dm = lib.tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)

            vk = self.get_k(self, dm, exxdiv)
            if abs(dm.imag).max() < 1.0e-6:
                vk = vk.real
            vk = numpy.asarray(vk, dtype=dm.dtype).reshape(dm_shape)
        else:
            vk = None

        self.scf_iter += 1

        return vj, vk

    def make_natural_orbitals(self, dms):
        """Construct natural orbitals from density matrix"""
        from .utils import make_natural_orbitals

        return make_natural_orbitals(self.cell, self.kpts, dms)

    def copy(self):
        """Create a shallow copy"""
        return self.view(self.__class__)

    def get_keyword_arguments(self):
        """Get keyword arguments"""
        return {key: value for key, value in self.__dict__.items() if key != 'cell'}
