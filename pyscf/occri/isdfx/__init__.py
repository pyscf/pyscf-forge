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
    mf.with_df = ISDFX(mf, isdf_thresh=1e-6)
    energy = mf.kernel()
"""

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.occri import OCCRI
from scipy.fft import hfftn, ifftn
from pyscf.occri.isdfx.isdfx_k_kpts import isdfx_get_k_kpts

# Import submodules for better organization
from pyscf.occri.isdfx import interpolation

class ISDFX(OCCRI):
    """
    ISDFX-enhanced OCCRI for efficient exact exchange evaluation.
    
    This class extends the standard OCCRI implementation with Interpolative 
    Separable Density Fitting (ISDFX) methods to reduce computational cost
    through adaptive grid point selection and interpolation.
    
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
    
    def __init__(self, mydf, kmesh=[1, 1, 1], isdf_thresh=1e-6, disable_c=False, **kwargs):
        """
        Initialize ISDFX density fitting object.
        
        Parameters:
        -----------
        mydf : SCF object
            Mean field object to attach ISDFX to
        kmesh : list of int
            k-point mesh dimensions (default: [1,1,1])
        isdf_thresh : float
            ISDFX interpolation threshold (default: 1e-6)
            Larger values select fewer interpolation points but reduce accuracy
        disable_c : bool
            Disable C extensions (default: False)
        **kwargs : dict
            Additional arguments passed to parent OCCRI class
        """
        super().__init__(mydf, kmesh=kmesh, disable_c=disable_c, **kwargs)
        self.isdf_thresh = isdf_thresh

        self.isdf_pts_from_gamma_point = False
        self.build()

        self.get_k = isdfx_get_k_kpts

    def convolve_with_W(self, U):
        W = self.W
        kmesh = self.kmesh
        n0, n1 = W.shape[-2], W.shape[-1]
        Nk = numpy.prod(kmesh)
        U_fft = hfftn(
            U.reshape(*kmesh, n0, n1), s=(kmesh), axes=[0, 1, 2], overwrite_x=True
        ).astype(numpy.complex128)
        U_fft *= W
        U[:] = ifftn(U_fft, axes=[0, 1, 2], overwrite_x=True).reshape(Nk, n0, n1)


    def build(self):  
        """Build ISDFX interpolation structures."""
        logger.info(self, 'Doing ISDFX with threshold %.2e', self.isdf_thresh)
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose) 

        # Step 1: Get pivot points  
        logger.debug(self, 'Selecting ISDFX pivot points')
        interpolation.get_pivots(self)
        cput0 = log.timer('Pivot selection', *cput0)
        
        # Step 2: Build fitting functions
        logger.debug(self, 'Building ISDFX fitting functions') 
        fitting_fxns = interpolation.get_fitting_functions(self)
        cput0 = log.timer('Build fitting functions', *cput0)

        # Step 3: Calculate THC Potential
        logger.debug(self, 'Calculating THC potential') 
        interpolation.get_thc_potential(self, fitting_fxns)
        cput0 = log.timer('Calculate THC potential', *cput0)
        
        return self