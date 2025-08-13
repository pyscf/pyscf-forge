"""
Multigrid OccRI (Occupied Orbital Coulomb Resolution of Identity)

This module provides multigrid extensions to OCCRI for improved convergence
and efficiency in periodic systems, particularly for large basis sets and
dense k-point sampling.

Main classes:
    MultigridOccRI: Main multigrid-enabled OCCRI class

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
    
    def get_jk(        self,
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
        """Compute J and K matrices using OCCRI"""
        if cell is None:
            cell = self.cell
        if dm is None:
            AttributeError(
                "Overwriting get_jk. "
                "Pass dm to get_jk as keyword: get_jk(dm=dm, ...)"
            )

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
        """Compute J and K matrices using OCCRI"""
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
                dm = self.make_natural_orbitals(dm.reshape(-1, nk, nao, nao))
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
            is_contracted_basis = self.cell.nao != self.cell_unc.nao
            mo_coeff = dm.mo_coeff
            mo_occ = dm.mo_occ
            tol = 1.0e-6
            is_occ = mo_occ > tol
            nset = dm.shape[0]
            if is_contracted_basis:
                mo_coeff = [self.c @ coeff for coeff in mo_coeff]
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
            if is_contracted_basis:
                    vk = [[self.c.T @ k_nk @ self.c for k_nk in k_n] for k_n in vk]
            vk = numpy.asarray(vk, dtype=dm.dtype).reshape(dm_shape)
        else:
            vk = None

        self.scf_iter += 1

        return vj, vk
