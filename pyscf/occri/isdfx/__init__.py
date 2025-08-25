"""
ISDFX (Interpolative Separable Density Fitting eXchange)

This module implements the ISDFX method for efficient exchange matrix evaluation
in periodic systems using interpolative density fitting with Voronoi partitioning
and pivoted Cholesky decomposition.

Main classes:
    ISDFX: Main ISDFX-enhanced OCCRI class

Usage:
    from pyscf.occri.isdfx import ISDFX

    mf = scf.KRHF(cell, kpts)
    mf.with_df = ISDFX.from_mf(mf, isdf_thresh=1e-6)
    energy = mf.kernel()
"""

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.occri import OCCRI

# Import submodules for better organization
from pyscf.occri.isdfx import interpolation
from pyscf.occri.isdfx.isdfx_k_kpts import isdfx_get_k_kpts
from scipy.fft import hfftn, ifftn

from .utils import get_fitting_functions


class ISDFX(OCCRI):
    """
    ISDFX-enhanced OCCRI for efficient exact exchange evaluation.

    This class extends the standard OCCRI implementation with Interpolative
    Separable Density Fitting (ISDFX) methods to reduce computational cost
    through adaptive grid point selection and interpolation.

    Defaults to incore algorithm if memory is sufficient.
    To require ISDFX method, set mf._is_mem_enough = lambda: False or
    use ISDFX.from_mf(mf) method.

    Attributes:
    -----------
    isdf_thresh : float
        Threshold for ISDFX interpolation point selection (default: 1e-6)
        Larger values select fewer points but reduce accuracy
    isdf_pts_from_gamma_point : bool
        Whether to use only gamma-point for pivot selection (default: False)
    ao_index_by_atom : list
        AO indices grouped by atom for Voronoi partitioning
    coords_by_atom : list
        Grid point indices assigned to each atom via Voronoi tessellation
    pivots : ndarray
        Selected interpolation point indices on the universal grid
    aovals : list
        AO values evaluated at interpolation points for each k-point
    chi_g : ndarray
        ISDFX fitting coefficients for interpolation
    """

    def __init__(self, cell, kpts=None, isdf_thresh=1e-6, disable_c=False, **kwargs):
        """
        Initialize ISDFX density fitting object.

        Parameters:
        -----------
        cell : pyscf.pbc.gto.Cell
            Unit cell object
        kpts : ndarray, optional
            k-points for periodic boundary conditions
        isdf_thresh : float
            ISDFX interpolation threshold (default: 1e-6)
            Larger values select fewer interpolation points but reduce accuracy
        disable_c : bool
            Disable C extensions (default: False)
        **kwargs : dict
            Additional arguments passed to parent OCCRI class
        """

        self.isdf_thresh = isdf_thresh
        self.isdf_pts_from_gamma_point = True
        super().__init__(cell, kpts, disable_c=disable_c, **kwargs)
        self.build()
        self.get_k = isdfx_get_k_kpts

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('******** %s ********', self.__class__)
        log.info('isdf_threshold = %s', self.isdf_thresh)
        log.info('from_gamma_point = %s', self.isdf_pts_from_gamma_point)
        return self

    @classmethod
    def from_mf(cls, mf, isdf_thresh=1e-6, disable_c=False, **kwargs):
        """Create ISDFX instance from mean-field object

        Parameters
        ----------
        mf : pyscf mean-field object
            Mean-field instance (RHF, UHF, RKS, UKS, etc.)
        isdf_thresh : float, optional
            ISDFX interpolation threshold (default: 1e-6)
            Larger values select fewer interpolation points but reduce accuracy
        disable_c : bool, optional
            If True, use pure Python implementation
        **kwargs
            Additional arguments passed to ISDFX constructor

        Returns
        -------
        ISDFX
            ISDFX density fitting instance configured for the given mean-field method

        Examples
        --------
        >>> from pyscf.pbc import gto, scf
        >>> from pyscf.occri.isdfx import ISDFX
        >>>
        >>> cell = gto.Cell()
        >>> cell.atom = 'H 0 0 0; H 0 0 1'
        >>> cell.basis = 'sto3g'
        >>> cell.build()
        >>>
        >>> mf = scf.KRHF(cell, cell.make_kpts([2,2,2]))
        >>> mf.with_df = ISDFX.from_mf(mf, isdf_thresh=1e-6)
        >>> energy = mf.kernel()
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

        # Create ISDFX instance
        isdfx = cls(mf.cell, mf.kpts, isdf_thresh=isdf_thresh, disable_c=disable_c, **kwargs)
        isdfx.method = method

        return isdfx

    def convolve_with_W(self, U):
        W = self.W
        kmesh = self.kmesh
        n0, n1 = W.shape[-2], W.shape[-1]
        Nk = numpy.prod(kmesh)
        U_fft = hfftn(U.reshape(*kmesh, n0, n1), s=(kmesh), axes=[0, 1, 2], overwrite_x=True).astype(numpy.complex128)
        U_fft *= W
        U[:] = ifftn(U_fft, axes=[0, 1, 2], overwrite_x=True).reshape(Nk, n0, n1)

    def build(self):
        """Build ISDFX interpolation structures."""
        self.dump_flags()
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)

        # Step 1: Get pivot points
        logger.debug(self, 'Selecting ISDFX pivot points')
        pivots, aovals = interpolation.get_pivots(self)
        self.pivots = pivots
        cput0 = log.timer('Pivot selection', *cput0)
        ngrids = self.grids.coords.shape[0]
        logger.info(
            self,
            '  ISDFX selected %d/%d grid points (%.2f%% compression)',
            len(self.pivots),
            ngrids,
            100 * len(self.pivots) / ngrids,
        )

        # Step 2: Build fitting functions
        logger.debug(self, 'Building ISDFX fitting functions')
        fitting_fxns = get_fitting_functions(self, aovals)
        cput0 = log.timer('Build fitting functions', *cput0)

        # Store AOs on pivots
        self.aovals = [numpy.asarray(ao[:, pivots], order='C') for ao in aovals]
        aovals = None

        # Step 3: Calculate THC Potential
        logger.debug(self, 'Calculating THC potential')
        W = interpolation.get_thc_potential(self, fitting_fxns)
        self.W = W
        cput0 = log.timer('Calculate THC potential', *cput0)

        return self
