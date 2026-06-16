#!/usr/bin/env python
# Copyright 2014-2025 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Arshad Mehmood, IACS, Stony Brook University
# Email: arshad.mehmood@stonybrook.edu
#

"""
DFT-Corrected CASCI with FOMO Support (RHF and UHF)
===================================================

This module implements CASCI calculations with DFT-corrected core energy,
supporting both standard CASCI and FOMO-CASCI (Fractional Occupation
Molecular Orbital) wavefunctions. Both RHF and UHF references are supported.

Theory
------
In standard CASCI, the core energy is computed using Hartree-Fock:

    E_core^HF = E_nuc + Tr[D_core * h] + 0.5*Tr[D_core * J] - 0.25*Tr[D_core * K]

In DFT-corrected CASCI, the core energy uses DFT exchange-correlation:

    E_core^DFT = E_nuc + Tr[D_core * h] + 0.5*Tr[D_core * J] + E_xc[core]

The active space embedding still uses HF-like potential (J - 0.5*K) to preserve
the wavefunction topology and CI coefficients.

Classes
-------
CASCI : RHF-based DFT-corrected CASCI
UCASCI : UHF-based DFT-corrected CASCI (unrestricted)

Functions
---------
DFCASCI : Factory function that auto-detects RHF/UHF reference

Examples
--------
RHF-based DFT-corrected CASCI:

>>> from pyscf import gto, scf
>>> from pyscf.mcscf import dft_corrected_casci
>>> mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='cc-pvdz')
>>> mf = scf.RHF(mol).run()
>>> mc = dft_corrected_casci.CASCI(mf, ncas=6, nelecas=6, xc='PBE')
>>> mc.kernel()
>>> g = mc.Gradients(method='analytical').kernel()

UHF-based DFT-corrected CASCI:

>>> mf = scf.UHF(mol).run()
>>> mc = dft_corrected_casci.UCASCI(mf, ncas=6, nelecas=6, xc='PBE')
>>> mc.kernel()
>>> g = mc.Gradients(method='numerical').kernel()

Using factory function (auto-detects RHF/UHF):

>>> mc = dft_corrected_casci.DFCASCI(mf, ncas=6, nelecas=6, xc='PBE')
>>> mc.kernel()

References
----------
DFT Core Embedding:
    S. Pijeau and E. G. Hohenstein,
    J. Chem. Theory Comput. 2017, 13, 1130-1146
    https://doi.org/10.1021/acs.jctc.6b00893

FOMO-CASCI:
    P. Slavicek and T. J. Martinez,
    J. Chem. Phys. 132, 234102 (2010)
    https://doi.org/10.1063/1.3436501
"""

from functools import reduce
import numpy as np

from pyscf import lib, scf, dft
from pyscf.mcscf import casci, ucasci
from pyscf.lib import logger

__all__ = ['CASCI', 'UCASCI', 'DFCASCI', 'DFTCoreCASCI', 'DFTCoreUCASCI']


class CASCI(casci.CASCI):
    """
    CASCI with DFT-evaluated core energy (RHF reference).

    This class modifies the standard CASCI energy calculation to use
    DFT exchange-correlation for the core electrons instead of HF exchange.
    The active space embedding still uses HF-like potential to preserve
    wavefunction topology.

    Parameters
    ----------
    mf : pyscf.scf.hf.RHF
        Converged RHF or ROHF mean-field object
    ncas : int
        Number of active orbitals
    nelecas : int or tuple of int
        Number of active electrons. If a tuple, (nalpha, nbeta)
    xc : str
        Exchange-correlation functional for core energy (default: 'PBE')
    ncore : int, optional
        Number of core orbitals. If not given, determined automatically.

    Attributes
    ----------
    xc : str
        Exchange-correlation functional name
    grids_level : int
        DFT integration grid level (default: 3)

    Examples
    --------
    >>> from pyscf import gto, scf
    >>> from pyscf.mcscf import dft_corrected_casci
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g')
    >>> mf = scf.RHF(mol).run()
    >>> mc = dft_corrected_casci.CASCI(mf, ncas=2, nelecas=2, xc='LDA')
    >>> mc.kernel()
    -1.0831...

    See Also
    --------
    pyscf.mcscf.casci.CASCI : Standard CASCI
    UCASCI : Unrestricted version for UHF reference
    """

    _keys = {'xc', 'grids_level', 'grids', 'ni'}

    def __init__(self, mf, ncas, nelecas, xc='PBE', ncore=None):
        super().__init__(mf, ncas, nelecas, ncore)
        self.xc = xc
        self.grids_level = 3
        self.grids = None
        self.ni = None

    def build_grids(self, mol=None):
        """Build DFT integration grid."""
        if mol is None:
            mol = self.mol
        if self.grids is None or self.grids.mol is not mol:
            self.grids = dft.gen_grid.Grids(mol)
            self.grids.level = self.grids_level
            self.grids.build()
        return self.grids

    def get_ni(self):
        """Get numerical integrator."""
        if self.ni is None:
            self.ni = dft.numint.NumInt()
        return self.ni

    def edft_core(self, dm_core, mol=None):
        """
        Compute DFT energy of core density.

        E_core = E_nuc + Tr[D_core * h] + 0.5*Tr[D_core * J_core] + E_xc[core]

        Parameters
        ----------
        dm_core : ndarray
            Core density matrix in AO basis, shape (nao, nao)
        mol : pyscf.gto.Mole, optional
            Molecule object. If not given, use self.mol

        Returns
        -------
        ecore : float
            DFT core energy in Hartree
        """
        if mol is None:
            mol = self.mol
        grids = self.build_grids(mol)
        ni = self.get_ni()

        h1 = self.get_hcore()
        e1 = np.einsum('ij,ji->', h1, dm_core).real
        vj = scf.hf.get_jk(mol, dm_core, hermi=1, with_j=True, with_k=False)[0]
        ej = 0.5 * np.einsum('ij,ji->', vj, dm_core).real
        _, exc, _ = ni.nr_rks(mol, grids, self.xc, dm_core)

        return self.energy_nuc() + e1 + ej + exc

    def get_h1eff(self, mo_coeff=None, ncas=None, ncore=None):
        """
        Compute effective one-electron Hamiltonian and DFT core energy.

        The effective Hamiltonian for the active space includes the core
        potential using HF-like embedding (J - 0.5*K), but the core energy
        is evaluated using DFT.

        Parameters
        ----------
        mo_coeff : ndarray, optional
            MO coefficients, shape (nao, nmo)
        ncas : int, optional
            Number of active orbitals
        ncore : int, optional
            Number of core orbitals

        Returns
        -------
        h1eff : ndarray
            Effective 1e Hamiltonian in active MO basis, shape (ncas, ncas)
        ecore : float
            DFT core energy in Hartree
        """
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if ncas is None:
            ncas = self.ncas
        if ncore is None:
            ncore = self.ncore

        h1 = self.get_hcore()
        mo_core = mo_coeff[:, :ncore]
        mo_cas = mo_coeff[:, ncore:ncore+ncas]

        if ncore > 0:
            dm_core = np.dot(mo_core, mo_core.T) * 2
            # HF-like embedding for active space (preserves CI topology)
            vhf = self.get_veff(self.mol, dm_core)
            # DFT core energy
            ecore = self.edft_core(dm_core)
        else:
            vhf = 0
            ecore = self.energy_nuc()

        h1eff = reduce(np.dot, (mo_cas.T, h1 + vhf, mo_cas))
        return h1eff, ecore

    def Gradients(self, method='analytical', step_size=1e-4):
        """
        Create gradient object for this CASCI calculation.

        Parameters
        ----------
        method : str, optional
            Gradient method: 'analytical', 'numerical', or 'auto' (default: 'analytical')
        step_size : float, optional
            Step size for numerical gradients in Bohr (default: 1e-4)

        Returns
        -------
        grad : pyscf.grad.dft_corrected_casci.Gradients
            Gradient object

        Examples
        --------
        >>> mc = dft_corrected_casci.CASCI(mf, ncas=4, nelecas=4, xc='PBE')
        >>> mc.kernel()
        >>> g = mc.Gradients(method='analytical').kernel()  # Fast
        >>> g = mc.Gradients(method='numerical').kernel()   # Accurate
        >>> g = mc.Gradients(method='auto').kernel()        # Safe default
        """
        from pyscf.grad import dft_corrected_casci as dft_corrected_casci_grad
        return dft_corrected_casci_grad.Gradients(self, method=method, step_size=step_size)


class UCASCI(ucasci.UCASCI):
    """
    UCASCI with DFT-evaluated core energy (UHF reference).

    This class modifies the standard UCASCI energy calculation to use
    DFT exchange-correlation for the core electrons instead of HF exchange.
    Supports unrestricted (UHF) reference wavefunctions.

    Parameters
    ----------
    mf : pyscf.scf.uhf.UHF
        Converged UHF mean-field object
    ncas : int
        Number of active orbitals
    nelecas : int or tuple of int
        Number of active electrons. If a tuple, (nalpha, nbeta)
    xc : str
        Exchange-correlation functional for core energy (default: 'PBE')
    ncore : int or tuple of int, optional
        Number of core orbitals. Can be (ncore_a, ncore_b) for different
        alpha/beta core sizes. If not given, determined automatically.

    Attributes
    ----------
    xc : str
        Exchange-correlation functional name
    grids_level : int
        DFT integration grid level (default: 3)

    Examples
    --------
    >>> from pyscf import gto, scf
    >>> from pyscf.mcscf import dft_corrected_casci
    >>> mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='cc-pvdz')
    >>> mf = scf.UHF(mol).run()
    >>> mc = dft_corrected_casci.UCASCI(mf, ncas=6, nelecas=6, xc='PBE')
    >>> mc.kernel()

    See Also
    --------
    pyscf.mcscf.ucasci.UCASCI : Standard UCASCI
    CASCI : Restricted version for RHF reference
    """

    _keys = {'xc', 'grids_level', 'grids', 'ni'}

    def __init__(self, mf, ncas, nelecas, xc='PBE', ncore=None):
        super().__init__(mf, ncas, nelecas, ncore)
        self.xc = xc
        self.grids_level = 3
        self.grids = None
        self.ni = None

    def build_grids(self, mol=None):
        """Build DFT integration grid."""
        if mol is None:
            mol = self.mol
        if self.grids is None or self.grids.mol is not mol:
            self.grids = dft.gen_grid.Grids(mol)
            self.grids.level = self.grids_level
            self.grids.build()
        return self.grids

    def get_ni(self):
        """Get numerical integrator."""
        if self.ni is None:
            self.ni = dft.numint.NumInt()
        return self.ni

    def edft_core(self, dm_core, mol=None):
        """
        Compute DFT energy of core density (unrestricted).

        E_core = E_nuc + Tr[D_core * h] + 0.5*Tr[D_core * J_core] + E_xc[core]

        Parameters
        ----------
        dm_core : tuple of ndarray
            Core density matrices (dm_alpha, dm_beta), each shape (nao, nao)
        mol : pyscf.gto.Mole, optional
            Molecule object. If not given, use self.mol

        Returns
        -------
        ecore : float
            DFT core energy in Hartree
        """
        if mol is None:
            mol = self.mol
        grids = self.build_grids(mol)
        ni = self.get_ni()

        dm_core_a, dm_core_b = dm_core
        dm_core_tot = dm_core_a + dm_core_b

        h1 = self.get_hcore()
        # Handle h1 as tuple (for UHF) or single matrix
        if isinstance(h1, (tuple, list)) or (isinstance(h1, np.ndarray) and h1.ndim == 3):
            if isinstance(h1, np.ndarray):
                h1_a, h1_b = h1[0], h1[1]
            else:
                h1_a, h1_b = h1
            e1 = np.einsum('ij,ji->', h1_a, dm_core_a).real + np.einsum('ij,ji->', h1_b, dm_core_b).real
        else:
            e1 = np.einsum('ij,ji->', h1, dm_core_tot).real

        # Coulomb energy
        vj = scf.hf.get_jk(mol, dm_core_tot, hermi=1, with_j=True, with_k=False)[0]
        ej = 0.5 * np.einsum('ij,ji->', vj, dm_core_tot).real

        # XC energy - use unrestricted version
        _, exc, _ = ni.nr_uks(mol, grids, self.xc, (dm_core_a, dm_core_b))

        return self.energy_nuc() + e1 + ej + exc

    def get_h1eff(self, mo_coeff=None, ncas=None, ncore=None):
        """
        Compute effective one-electron Hamiltonian and DFT core energy (unrestricted).

        Parameters
        ----------
        mo_coeff : tuple of ndarray, optional
            MO coefficients (mo_alpha, mo_beta), each shape (nao, nmo)
        ncas : int, optional
            Number of active orbitals
        ncore : int or tuple of int, optional
            Number of core orbitals (ncore_a, ncore_b)

        Returns
        -------
        h1eff : tuple of ndarray
            Effective 1e Hamiltonian (h1eff_a, h1eff_b), each shape (ncas, ncas)
        ecore : float
            DFT core energy in Hartree
        """
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if ncas is None:
            ncas = self.ncas
        if ncore is None:
            ncore = self.ncore

        # Handle ncore as int or tuple
        if isinstance(ncore, (int, np.integer)):
            ncore_a = ncore_b = ncore
        else:
            ncore_a, ncore_b = ncore

        mo_coeff_a, mo_coeff_b = mo_coeff

        mo_core_a = mo_coeff_a[:, :ncore_a]
        mo_core_b = mo_coeff_b[:, :ncore_b]
        mo_cas_a = mo_coeff_a[:, ncore_a:ncore_a+ncas]
        mo_cas_b = mo_coeff_b[:, ncore_b:ncore_b+ncas]

        if ncore_a > 0 or ncore_b > 0:
            dm_core_a = np.dot(mo_core_a, mo_core_a.T)
            dm_core_b = np.dot(mo_core_b, mo_core_b.T)
            dm_core = (dm_core_a, dm_core_b)

            # HF-like embedding for active space
            vhf = self.get_veff(self.mol, dm_core)

            # DFT core energy
            ecore = self.edft_core(dm_core)
        else:
            vhf = (0, 0)
            ecore = self.energy_nuc()

        if isinstance(vhf, np.ndarray) and vhf.ndim == 2:
            vhf = (vhf, vhf)

        # Handle h1 as tuple (for UHF) or single matrix
        h1 = self.get_hcore()
        if isinstance(h1, (tuple, list)) or (isinstance(h1, np.ndarray) and h1.ndim == 3):
            if isinstance(h1, np.ndarray):
                h1_a, h1_b = h1[0], h1[1]
            else:
                h1_a, h1_b = h1
        else:
            h1_a = h1_b = h1

        h1eff_a = reduce(np.dot, (mo_cas_a.T, h1_a + vhf[0], mo_cas_a))
        h1eff_b = reduce(np.dot, (mo_cas_b.T, h1_b + vhf[1], mo_cas_b))

        return (h1eff_a, h1eff_b), ecore

    def Gradients(self, method='numerical', step_size=1e-4):
        """
        Create gradient object for this UCASCI calculation.

        Parameters
        ----------
        method : str, optional
            Gradient method: 'numerical' or 'auto' (default: 'numerical')
            Note: UCASCI only supports numerical gradients
        step_size : float, optional
            Step size for numerical gradients in Bohr (default: 1e-4)

        Returns
        -------
        grad : pyscf.grad.dft_corrected_casci.UGradients
            Gradient object for unrestricted CASCI

        Examples
        --------
        >>> mc = dft_corrected_casci.UCASCI(mf_uhf, ncas=4, nelecas=4, xc='PBE')
        >>> mc.kernel()
        >>> g = mc.Gradients(method='numerical').kernel()
        """
        from pyscf.grad import dft_corrected_casci as dft_corrected_casci_grad
        return dft_corrected_casci_grad.UGradients(self, method=method, step_size=step_size)


# Aliases for backward compatibility
DFTCoreCASCI = CASCI
DFTCoreUCASCI = UCASCI


def DFCASCI(mf, ncas, nelecas, xc='PBE', ncore=None):
    """
    Factory function to create DFT-corrected CASCI object based on SCF reference type.

    Automatically selects CASCI (for RHF/ROHF) or UCASCI (for UHF) based on
    the mean-field object type.

    Parameters
    ----------
    mf : pyscf.scf object
        Converged RHF, ROHF, or UHF mean-field object
    ncas : int
        Number of active orbitals
    nelecas : int or tuple of int
        Number of active electrons
    xc : str
        Exchange-correlation functional (default: 'PBE')
    ncore : int, optional
        Number of core orbitals

    Returns
    -------
    mc : CASCI or UCASCI
        Appropriate DFT-corrected CASCI object based on reference type

    Examples
    --------
    >>> from pyscf import gto, scf
    >>> from pyscf.mcscf import dft_corrected_casci
    >>> mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='cc-pvdz')
    >>> mf_rhf = scf.RHF(mol).run()
    >>> mc = dft_corrected_casci.DFCASCI(mf_rhf, 6, 6, xc='PBE')  # Returns CASCI
    >>> mf_uhf = scf.UHF(mol).run()
    >>> mc = dft_corrected_casci.DFCASCI(mf_uhf, 6, 6, xc='PBE')  # Returns UCASCI
    """
    if isinstance(mf, scf.uhf.UHF):
        return UCASCI(mf, ncas, nelecas, xc, ncore)
    else:
        return CASCI(mf, ncas, nelecas, xc, ncore)


if __name__ == '__main__':
    from pyscf import gto

    mol = gto.M(
        atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
        basis='cc-pvdz',
        unit='Angstrom'
    )

    print("="*60)
    print("DFT-corrected CASCI Energy Test")
    print("="*60)

    # RHF-based
    print("\n1. RHF-based DFT-corrected CASCI:")
    mf_rhf = scf.RHF(mol).run()
    mc_rhf = CASCI(mf_rhf, 6, 6, xc='PBE')
    mc_rhf.kernel()
    print(f"   Energy: {mc_rhf.e_tot:.10f} Ha")

    # UHF-based
    print("\n2. UHF-based DFT-corrected CASCI:")
    mf_uhf = scf.UHF(mol).run()
    mc_uhf = UCASCI(mf_uhf, 6, 6, xc='PBE')
    mc_uhf.kernel()
    print(f"   Energy: {mc_uhf.e_tot:.10f} Ha")

    # Using factory function
    print("\n3. Using DFCASCI factory function:")
    mc_auto = DFCASCI(mf_uhf, 6, 6, xc='PBE')
    mc_auto.kernel()
    print(f"   Energy: {mc_auto.e_tot:.10f} Ha")
    print(f"   Type: {type(mc_auto).__name__}")
