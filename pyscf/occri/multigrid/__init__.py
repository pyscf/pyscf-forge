"""
Multigrid OccRI (Occupied Orbital Coulomb Resolution of Identity)

This module provides multigrid extensions to OCCRI for improved convergence
and efficiency in periodic systems, particularly for large basis sets and
dense k-point sampling.

Main classes:
    MultigridOccRI: Main multigrid-enabled OCCRI class
    GridHierarchy: Manages coarse/fine grid relationships
    MGInterpolation: Handles restriction/prolongation operators

Usage:
    from pyscf.occri.multigrid import MultigridOccRI
    
    mf = scf.KRHF(cell, kpts)
    mf.with_df = MultigridOccRI(mf, kmesh=[2,2,2], 
                                mg_levels=3, 
                                coarsening_factor=2)
    energy = mf.kernel()
"""

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.occri import OCCRI

from pyscf.gto.mole import decontract_basis
from pyscf.gto import uncontract
from pyscf.occri.multigrid.mg_grids import build_grids




class MultigridOccRI(OCCRI):
    """
    Multigrid-enhanced OCCRI for efficient exact exchange evaluation
    
    This class extends the standard OCCRI implementation with multigrid
    methods to improve convergence and reduce computational cost for
    large systems and dense k-point sampling.
    """
    
    def __init__(self,
                 mydf,
                 kmesh=[1, 1, 1],
                 alpha_cutoff=2.8,  # Exponents < alpha_cut go on sparse grid.
                 rcut_epsilon=1.0e-6,  # Determines atom-centered grid radii.
                 ke_epsilon=1.0e-7,  # Determines sparse grid mesh.
                 incore=True, # Store AOs or not?
                 disable_c=False,
                 **kwargs):
        """
        Initialize Multigrid OccRI density fitting object
        
        Parameters:
        -----------
        mydf : SCF object
            Mean field object to attach OCCRI to
        kmesh : list of int
            k-point mesh dimensions
        mg_levels : int
            Number of multigrid levels
        coarsening_factor : int
            Grid coarsening factor between levels
        mg_method : str
            Multigrid method ('vcycle', 'fmg', 'adaptive')
        disable_c : bool
            Disable C extensions (use Python implementation)
        """
        # Initialize parent OCCRI class
        super().__init__(mydf, kmesh=kmesh, disable_c=disable_c, **kwargs)
        self.alpha_cutoff=alpha_cutoff
        self.rcut_epsilon=rcut_epsilon
        self.ke_epsilon=ke_epsilon
        self.incore=incore
        self.k_grid_mesh=None

        # Print all attributes
        if self.cell.verbose > 4:
            print()
            print("******** <class 'ISDFX'> ********", flush=True)
            for key, value in vars(self).items():
                if key in  ['method', 'kmesh', 'get_k', 'alpha_cutoff', 'rcut_epsilon', 'ke_epsilon', 'incore']:
                    print(f"{key}: {value}", flush=True)
            print()

        self.to_uncontracted_basis()
        build_grids(self)
        


    def to_uncontracted_basis(myisdf):
        cell = myisdf.cell
        nk = myisdf.kpts.shape[0]
        unc_bs = {}
        for symb, bs in cell._basis.items():
            unc_bs[symb] = uncontract(bs)

        cell_unc = cell.copy(deep=True)
        cell_unc.basis = unc_bs
        cell_unc.build(dump_input=False)
        myisdf.cell_unc = cell_unc
        c = decontract_basis(cell, aggregate=True)[1]
        myisdf.c = c.astype(numpy.complex128) if nk > 1 else c.astype(numpy.float64)

    def primitive_gto_cutoff(self, shell_idx):
        """Cutoff raidus, above which each shell decays to a value less than the
        required precsion"""
        rcut = []
        rcut_epsilon = self.rcut_epsilon
        cell = self.cell_unc
        for ib in shell_idx:
            es = cell.bas_exp(ib)
            r = (-numpy.log(rcut_epsilon) / es) ** 0.5
            rcut.extend(r)
        return numpy.asarray(rcut, numpy.float64)
    
    def primitive_gto_exponent(self, rmin):
        """
        Calculate primitive Gaussian-type orbital exponent based on minimum radius and cutoff precision.
        """
        rcut_epsilon = self.rcut_epsilon
        return -numpy.log(rcut_epsilon) / max(rmin**2, 1e-12)    
    
    def get_jk_kpts(self, dms, exxdiv=None):
        """
        Multigrid exchange matrix evaluation
        
        Parameters:
        -----------
        dms : array_like
            Density matrices with orbital information
        exxdiv : str
            Ewald divergence treatment
            
        Returns:
        --------
        vk : ndarray
            Exchange matrices
        """
        # TODO: Implement multigrid exchange evaluation
        # For now, fallback to standard OCCRI
        return self.get_k(dms, exxdiv)
