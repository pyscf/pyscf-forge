"""
ISDFX (Interpolative Separable Density Fitting eXchange)

This module implements the ISDF method for efficient exchange matrix evaluation
in periodic systems using interpolative density fitting with Voronoi partitioning
and pivoted Cholesky decomposition.

Main classes:
    ISDF: Main ISDF-enhanced OCCRI class
    
Usage:
    from pyscf.occri.isdfx import ISDF
    
    mf = scf.KRHF(cell, kpts)
    mf.with_df = ISDF(mf, isdf_thresh=1e-6)
    energy = mf.kernel()
"""

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.occri import OCCRI

# Import submodules for better organization
from pyscf.occri.isdfx import interpolation

class ISDF(OCCRI):
    """
    ISDF-enhanced OCCRI for efficient exact exchange evaluation.
    
    This class extends the standard OCCRI implementation with Interpolative 
    Separable Density Fitting (ISDF) methods to reduce computational cost
    through adaptive grid point selection and interpolation.
    
    Attributes:
    -----------
    isdf_thresh : float
        Threshold for ISDF interpolation point selection (default: 1e-6)
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
        ISDF fitting coefficients for interpolation
    """
    
    def __init__(self, mydf, kmesh=[1, 1, 1], isdf_thresh=1e-6, disable_c=False, **kwargs):
        """
        Initialize ISDF density fitting object.
        
        Parameters:
        -----------
        mydf : SCF object
            Mean field object to attach ISDF to
        kmesh : list of int
            k-point mesh dimensions (default: [1,1,1])
        isdf_thresh : float
            ISDF interpolation threshold (default: 1e-6)
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


    def build(self):
        """Build ISDF interpolation structures."""
        logger.info(self, 'Doing ISDF with threshold %.2e', self.isdf_thresh)
        
        # Step 1: Get pivot points  
        logger.debug(self, 'Selecting ISDF pivot points')
        interpolation.get_pivots(self)
        
        # Step 2: Build fitting functions
        logger.debug(self, 'Building ISDF fitting functions') 
        self.Chi_g = interpolation.get_fitting_functions(self)
        
        return self