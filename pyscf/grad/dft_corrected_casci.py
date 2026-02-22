#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analytical and numerical gradients for DFT-corrected CASCI

This module provides nuclear gradients for CASCI calculations with
DFT-corrected core energy, supporting both RHF and UHF references
as well as FOMO (Fractional Occupation Molecular Orbital) wavefunctions.

Author: Arshad Mehmood, IACS, Stony Brook University
Email: arshad.mehmood@stonybrook.edu
"""

import numpy as np
from pyscf import lib, dft, scf
from pyscf.lib import logger
from pyscf.grad import casci as casci_grad


def sanitize_mo_occ_for_cphf(mc):
    """
    Sanitize FOMO fractional occupations for CPHF solver.

    FOMO produces fractional orbital occupations which cause issues
    in the CPHF equations. This function converts fractional occupations
    to binary (0 or 2 for RHF, 0 or 1 for UHF) for CPHF stability.

    Parameters
    ----------
    mc : CASCI object
        CASCI calculation object

    Returns
    -------
    sanitized : ndarray or None
        Sanitized occupations, or None if no changes needed
    """
    mf = mc._scf
    mo_occ = getattr(mf, 'mo_occ', None)
    if mo_occ is None:
        return None

    mo_occ = np.asarray(mo_occ)
    max_occ = float(np.max(mo_occ))

    # Determine threshold based on RHF vs UHF
    if max_occ > 1.5:  # RHF-like (occ = 0 or 2)
        sanitized = np.where(mo_occ > 1.0, 2.0, 0.0)
    else:  # UHF-like (occ = 0 or 1)
        sanitized = np.where(mo_occ > 0.5, 1.0, 0.0)

    # Only return if different from original
    if np.allclose(sanitized, mo_occ, atol=1e-8):
        return None
    return sanitized


class Gradients(casci_grad.Gradients):
    """
    Nuclear gradients for DFT-corrected CASCI (RHF reference).

    Supports both analytical and numerical gradient methods.

    Parameters
    ----------
    mc : CASCI object
        CASCI calculation to compute gradients for
    method : str, optional
        Gradient method: 'analytical', 'numerical', or 'auto' (default: 'analytical')
    step_size : float, optional
        Finite difference step size for numerical gradients (default: 1e-4 Bohr)

    Examples
    --------
    >>> from pyscf import gto, scf
    >>> from pyscf.mcscf import dft_corrected_casci
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='cc-pvdz')
    >>> mf = scf.RHF(mol).run()
    >>> mc = dft_corrected_casci.CASCI(mf, ncas=2, nelecas=2, xc='PBE')
    >>> mc.kernel()
    >>> grad = mc.Gradients(method='analytical').kernel()
    """

    def __init__(self, mc, method='analytical', step_size=1e-4):
        super().__init__(mc)
        self.method = method.lower()
        self.step_size = step_size

        if self.method not in ['analytical', 'numerical', 'auto']:
            raise ValueError(f"method must be 'analytical', 'numerical', or 'auto', got '{method}'")

    def kernel(self, mo_coeff=None, ci=None, atmlst=None, state=None, verbose=None):
        """
        Compute nuclear gradients.

        Parameters
        ----------
        mo_coeff : ndarray, optional
            Molecular orbital coefficients
        ci : ndarray, optional
            CI coefficients
        atmlst : list of int, optional
            List of atom indices for which to compute gradients
        state : int, optional
            Electronic state index
        verbose : int, optional
            Verbosity level

        Returns
        -------
        de : ndarray
            Nuclear gradients, shape (natm, 3)
        """
        log = logger.new_logger(self, verbose)

        if self.method == 'numerical':
            log.info('Using numerical gradients (finite differences)')
            return self.kernel_numerical(mo_coeff, ci, atmlst, state, verbose)

        elif self.method == 'analytical':
            log.info('Using analytical gradients')
            log.note('Expected accuracy: ~1-2e-2 Ha/Bohr')
            return self.kernel_analytical(mo_coeff, ci, atmlst, state, verbose)

        elif self.method == 'auto':
            log.info('Using auto gradient mode (analytical with numerical fallback)')
            try:
                return self.kernel_analytical(mo_coeff, ci, atmlst, state, verbose)
            except Exception as e:
                log.warn(f'Analytical gradient failed: {e}')
                log.warn('Falling back to numerical gradients')
                return self.kernel_numerical(mo_coeff, ci, atmlst, state, verbose)

    def kernel_analytical(self, mo_coeff=None, ci=None, atmlst=None, state=None, verbose=None):
        """
        Analytical gradient implementation.

        Accuracy: ~1-2e-2 Ha/Bohr
        Speed: Fast

        Parameters
        ----------
        mo_coeff : ndarray, optional
            Molecular orbital coefficients
        ci : ndarray, optional
            CI coefficients
        atmlst : list of int, optional
            List of atom indices
        state : int, optional
            Electronic state index
        verbose : int, optional
            Verbosity level

        Returns
        -------
        de : ndarray
            Nuclear gradients, shape (natm, 3)
        """
        mc = self.base
        log = logger.new_logger(self, verbose)

        if atmlst is None:
            atmlst = range(mc.mol.natm)

        # Sanitize FOMO fractional occupations for CPHF solver
        saved_occ = None
        sanitized_occ = sanitize_mo_occ_for_cphf(mc)
        if sanitized_occ is not None:
            log.debug('Sanitizing FOMO fractional occupations for CPHF')
            saved_occ = mc._scf.mo_occ
            mc._scf.mo_occ = sanitized_occ

        try:
            # Compute base CASCI gradient (with HF core)
            de = super().kernel(mo_coeff, ci, atmlst, state, verbose)
        finally:
            # Restore original occupations
            if saved_occ is not None:
                mc._scf.mo_occ = saved_occ

        # Add DFT correction for frozen core
        ncore_frozen = self.get_frozen_core()

        if ncore_frozen > 0:
            if mo_coeff is None:
                mo_coeff = mc.mo_coeff

            mo_core = mo_coeff[:, :ncore_frozen]
            dm_core = 2.0 * np.dot(mo_core, mo_core.T)

            log.info(f'Computing DFT correction for {ncore_frozen} frozen core orbitals')

            try:
                de_corr = self.compute_dft_correction(dm_core, mo_coeff, ncore_frozen, atmlst)
                de = de + de_corr
                log.note('DFT core correction added')
            except Exception as e:
                log.warn(f'DFT correction failed: {e}')
                raise

        self.de = de
        return self.de

    def kernel_numerical(self, mo_coeff=None, ci=None, atmlst=None, state=None, verbose=None):
        """
        Numerical gradient implementation using finite differences.

        Accuracy: ~1e-6 Ha/Bohr
        Speed: Slow (requires 2*natm*3 energy evaluations)

        Parameters
        ----------
        mo_coeff : ndarray, optional
            Molecular orbital coefficients
        ci : ndarray, optional
            CI coefficients
        atmlst : list of int, optional
            List of atom indices
        state : int, optional
            Electronic state index
        verbose : int, optional
            Verbosity level

        Returns
        -------
        de : ndarray
            Nuclear gradients, shape (natm, 3)
        """
        mc = self.base
        mol = mc.mol
        log = logger.new_logger(self, verbose)

        if atmlst is None:
            atmlst = range(mol.natm)

        natm = len(atmlst)
        de = np.zeros((natm, 3))

        # Get reference energy
        e0 = mc.e_tot
        if e0 is None:
            raise RuntimeError('CASCI energy not available. Run mc.kernel() first.')

        coords_ref = mol.atom_coords(unit='Bohr')
        step = self.step_size

        log.info(f'Computing numerical gradients with step size {step:.2e} Bohr')
        log.info(f'This requires {2*natm*3} energy evaluations...')

        for k, ia in enumerate(atmlst):
            for ix in range(3):
                # Forward step
                coords_plus = coords_ref.copy()
                coords_plus[ia, ix] += step
                e_plus = self.compute_energy_at_geometry(coords_plus, log)

                # Backward step
                coords_minus = coords_ref.copy()
                coords_minus[ia, ix] -= step
                e_minus = self.compute_energy_at_geometry(coords_minus, log)

                # Central difference
                de[k, ix] = (e_plus - e_minus) / (2 * step)

        log.note('Numerical gradient computation complete')
        self.de = de
        return self.de

    def compute_energy_at_geometry(self, coords, log):
        """
        Compute DFT-corrected CASCI energy at displaced geometry.

        Parameters
        ----------
        coords : ndarray
            Nuclear coordinates in Bohr, shape (natm, 3)
        log : Logger
            Logger object for output

        Returns
        -------
        energy : float
            Total energy in Hartree
        """
        mc = self.base
        mol_orig = mc.mol

        # Create displaced molecule
        mol_new = mol_orig.copy()
        mol_new.set_geom_(coords, unit='Bohr')
        mol_new.build()

        # Run SCF
        if isinstance(mc._scf, scf.uhf.UHF):
            mf_new = scf.UHF(mol_new)
        else:
            mf_new = scf.RHF(mol_new)

        mf_new.verbose = 0
        mf_new.kernel()

        # Check if FOMO
        if hasattr(mc._scf, 'fomo_temperature'):
            from pyscf.scf import fomoscf
            fomo_kwargs = {}
            for attr in ['fomo_temperature', 'fomo_method', 'fomo_restricted']:
                if hasattr(mc._scf, attr):
                    fomo_kwargs[attr.replace('fomo_', '')] = getattr(mc._scf, attr)
            mf_new = fomoscf.fomo_scf(mf_new, **fomo_kwargs)
            mf_new.verbose = 0
            mf_new.kernel()

        # Run DFT-corrected CASCI
        from pyscf.mcscf import dft_corrected_casci
        mc_new = dft_corrected_casci.CASCI(mf_new, mc.ncas, mc.nelecas, xc=mc.xc)
        mc_new.verbose = 0
        if hasattr(mc, 'grids_level'):
            mc_new.grids_level = mc.grids_level
        mc_new.kernel()

        return mc_new.e_tot

    def get_frozen_core(self):
        """
        Get number of truly frozen core orbitals.

        For FOMO-CASCI with restricted=(ncore, ncas), returns ncore.
        Otherwise returns mc.ncore.

        Returns
        -------
        ncore_frozen : int
            Number of frozen core orbitals receiving DFT correction
        """
        mc = self.base
        mf = mc._scf

        # Check for FOMO restricted setting
        if hasattr(mf, 'fomo_restricted') and mf.fomo_restricted is not None:
            return mf.fomo_restricted[0]

        # Fallback to standard CASCI ncore
        return mc.ncore

    def compute_dft_correction(self, dm_core, mo_coeff, ncore, atmlst):
        """
        Compute DFT vs HF gradient difference for frozen core (analytical).

        Correction = grad(E_DFT[core]) - grad(E_HF[core])

        Parameters
        ----------
        dm_core : ndarray
            Core density matrix, shape (nao, nao)
        mo_coeff : ndarray
            MO coefficients, shape (nao, nmo)
        ncore : int
            Number of core orbitals
        atmlst : list of int
            Atom indices

        Returns
        -------
        de_corr : ndarray
            Gradient correction, shape (natm, 3)
        """
        mc = self.base
        mol = mc.mol

        nmo = mo_coeff.shape[1]
        mo_occ_core = np.zeros(nmo)
        mo_occ_core[:ncore] = 2.0

        # Temporary HF object with core-only occupation
        mf_hf_temp = scf.RHF(mol)
        mf_hf_temp.mo_coeff = mo_coeff
        mf_hf_temp.mo_occ = mo_occ_core
        mf_hf_temp.mo_energy = np.zeros(nmo)
        mf_hf_temp.converged = True

        # Temporary DFT object with core-only occupation
        mf_dft_temp = dft.RKS(mol)
        mf_dft_temp.xc = mc.xc

        # Use same grid as main calculation
        if hasattr(mc, 'grids') and mc.grids is not None:
            mf_dft_temp.grids = mc.grids
        else:
            mf_dft_temp.grids = mc.build_grids(mol)

        if mf_dft_temp.grids.coords is None:
            mf_dft_temp.grids.build()

        mf_dft_temp.mo_coeff = mo_coeff
        mf_dft_temp.mo_occ = mo_occ_core
        mf_dft_temp.mo_energy = np.zeros(nmo)
        mf_dft_temp.converged = True

        # Compute gradients using PySCF machinery
        from pyscf.grad import rhf as rhf_grad, rks as rks_grad

        grad_hf = rhf_grad.Gradients(mf_hf_temp)
        grad_dft = rks_grad.Gradients(mf_dft_temp)

        de_hf = grad_hf.kernel()
        de_dft = grad_dft.kernel()

        # The difference is the DFT correction
        de_diff = de_dft - de_hf

        # Extract only requested atoms
        if len(atmlst) != mol.natm:
            de_diff = de_diff[list(atmlst)]

        return de_diff


class UGradients(lib.StreamObject):
    """
    Nuclear gradients for DFT-corrected UCASCI (UHF reference).

    Currently only numerical gradients are supported for UCASCI.

    Parameters
    ----------
    mc : UCASCI object
        UCASCI calculation to compute gradients for
    method : str, optional
        Gradient method: 'numerical' (default) or 'auto'
    step_size : float, optional
        Finite difference step size (default: 1e-4 Bohr)

    Examples
    --------
    >>> mc = dft_corrected_casci.UCASCI(mf_uhf, ncas=4, nelecas=4, xc='PBE')
    >>> mc.kernel()
    >>> grad = mc.Gradients(method='numerical').kernel()
    """

    def __init__(self, mc, method='numerical', step_size=1e-4):
        self.base = mc
        self.mol = mc.mol
        self.verbose = mc.verbose
        self.stdout = mc.stdout
        self.de = None
        self.method = method.lower()
        self.step_size = step_size

        if self.method not in ['numerical', 'auto']:
            raise ValueError(f"For UCASCI, method must be 'numerical' or 'auto', got '{method}'")

    def kernel(self, mo_coeff=None, ci=None, atmlst=None, state=None, verbose=None):
        """
        Compute numerical nuclear gradients for UCASCI.

        Parameters
        ----------
        mo_coeff : tuple of ndarray, optional
            MO coefficients (alpha, beta)
        ci : ndarray, optional
            CI coefficients
        atmlst : list of int, optional
            Atom indices
        state : int, optional
            Electronic state
        verbose : int, optional
            Verbosity level

        Returns
        -------
        de : ndarray
            Nuclear gradients, shape (natm, 3)
        """
        mc = self.base
        mol = mc.mol
        log = logger.new_logger(self, verbose)

        if atmlst is None:
            atmlst = range(mol.natm)

        natm = len(atmlst)
        de = np.zeros((natm, 3))

        # Get reference energy
        e0 = mc.e_tot
        if e0 is None:
            raise RuntimeError('UCASCI energy not available. Run mc.kernel() first.')

        coords_ref = mol.atom_coords(unit='Bohr')
        step = self.step_size

        log.info(f'Computing numerical gradients with step size {step:.2e} Bohr')
        log.info(f'This requires {2*natm*3} energy evaluations...')

        for k, ia in enumerate(atmlst):
            for ix in range(3):
                # Forward step
                coords_plus = coords_ref.copy()
                coords_plus[ia, ix] += step
                e_plus = self.compute_energy_at_geometry(coords_plus, log)

                # Backward step
                coords_minus = coords_ref.copy()
                coords_minus[ia, ix] -= step
                e_minus = self.compute_energy_at_geometry(coords_minus, log)

                # Central difference
                de[k, ix] = (e_plus - e_minus) / (2 * step)

        log.note('Numerical gradient computation complete')
        self.de = de
        return self.de

    def compute_energy_at_geometry(self, coords, log):
        """
        Compute DFT-corrected UCASCI energy at displaced geometry.

        Parameters
        ----------
        coords : ndarray
            Nuclear coordinates in Bohr, shape (natm, 3)
        log : Logger
            Logger object

        Returns
        -------
        energy : float
            Total energy in Hartree
        """
        mc = self.base
        mol_orig = mc.mol

        # Create displaced molecule
        mol_new = mol_orig.copy()
        mol_new.set_geom_(coords, unit='Bohr')
        mol_new.build()

        # Run UHF
        mf_new = scf.UHF(mol_new)
        mf_new.verbose = 0
        mf_new.kernel()

        # Check if FOMO
        if hasattr(mc._scf, 'fomo_temperature'):
            from pyscf.scf import fomoscf
            fomo_kwargs = {}
            for attr in ['fomo_temperature', 'fomo_method', 'fomo_restricted']:
                if hasattr(mc._scf, attr):
                    fomo_kwargs[attr.replace('fomo_', '')] = getattr(mc._scf, attr)
            mf_new = fomoscf.fomo_scf(mf_new, **fomo_kwargs)
            mf_new.verbose = 0
            mf_new.kernel()

        # Run DFT-corrected UCASCI
        from pyscf.mcscf import dft_corrected_casci
        mc_new = dft_corrected_casci.UCASCI(mf_new, mc.ncas, mc.nelecas, xc=mc.xc)
        mc_new.verbose = 0
        if hasattr(mc, 'grids_level'):
            mc_new.grids_level = mc.grids_level
        mc_new.kernel()

        return mc_new.e_tot
