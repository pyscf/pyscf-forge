#!/usr/bin/env python
# Copyright 2014-2024 The PySCF Developers. All Rights Reserved.
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
# Date: 30 December 2025
#

'''
Floating Occupation Molecular Orbital (FOMO) SCF

Refs:
    P. Slavicek, T. J. Martinez, J. Chem. Phys. 132, 234102 (2010)
'''

import numpy
import scipy
from copy import copy
from pyscf import scf

__all__ = ['fomo_scf']


def fomo_scf(mf, temperature, method='gaussian', sigma=None, restricted=None):
    """Return a copy of *mf* with FOMO (fractional) occupations.

    Args:
        mf : pyscf.scf.hf.SCF (RHF/RKS/UHF/UKS)
            The mean-field object to be wrapped.
        temperature : float
            Electronic temperature expressed as kT in Hartree (atomic units).
            This matches PySCF MO energies (Hartree).
        method : str
            Broadening scheme:
                'gaussian'  - Gaussian broadening with integrated occupations.
                'fermi'     - Fermi-Dirac occupations.
        sigma : float or None
            Broadening width in Hartree. Only used for 'gaussian'. If None,
            sigma = kT is used.
        restricted : None or tuple
            If None, fractional occupations may occur over all orbitals.
            If provided, restrict fractional occupations to a subspace:
                restricted = (ncore, ncas)
            where orbitals [ncore : ncore+ncas] are fractionally occupied and
            orbitals outside this window are forced to integer occupations
            (2 or 0 for RHF/RKS; 1 or 0 per spin for UHF/UKS).

    Returns:
        mf_fomo : a (shallow) copy of mf, with mf_fomo.get_occ overridden and
            attributes fomo_temperature, fomo_sigma, fomo_method, fomo_restricted.

    Examples:

    >>> from pyscf import gto, scf
    >>> from pyscf.scf import fomoscf
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='ccpvdz')
    >>> mf = scf.RHF(mol)
    >>> mf = fomoscf.fomo_scf(mf, temperature=0.01, method='gaussian')
    >>> mf.kernel()
    """
    mf_fomo = copy(mf)

    # Reset SCF internal state to allow re-running
    mf_fomo.converged = False
    mf_fomo.mo_energy = None
    mf_fomo.mo_coeff = None
    mf_fomo.mo_occ = None
    mf_fomo.e_tot = 0
    if hasattr(mf_fomo, '_opt'):
        mf_fomo._opt = {}

    kT = float(temperature)

    method_lc = method.lower()
    if method_lc not in ('gaussian', 'fermi', 'fermi-dirac', 'fermi_dirac'):
        raise ValueError('method must be "gaussian" or "fermi"')

    if method_lc.startswith('fermi'):
        sigma_eff = None
    else:
        sigma_eff = (kT if sigma is None else float(sigma))
        if sigma_eff <= 0:
            raise ValueError('Gaussian broadening requires sigma > 0 (Hartree).')

    # Store settings on the object
    mf_fomo.fomo_temperature = float(temperature)
    mf_fomo.fomo_kT = kT
    mf_fomo.fomo_sigma = sigma_eff
    mf_fomo.fomo_method = ('fermi' if method_lc.startswith('fermi') else 'gaussian')
    mf_fomo.fomo_restricted = restricted

    def occ_fermi(eps, mu, kT):
        # n_i = 2 / (1 + exp((eps_i - mu)/kT))
        x = (eps - mu) / max(kT, 1e-20)
        x = numpy.clip(x, -200.0, 200.0)
        return 2.0 / (1.0 + numpy.exp(x))

    def occ_gauss(eps, mu, sigma):
        # n_i = 1 + erf((mu - eps_i)/(sqrt(2)*sigma))
        arg = (mu - eps) / (numpy.sqrt(2.0) * sigma)
        return 1.0 + scipy.special.erf(arg)

    def solve_mu(eps, nelec_target, scheme, width):
        eps = numpy.asarray(eps)
        if nelec_target <= 0:
            return eps.min() - 100.0 * (width if width is not None else 1.0)
        if nelec_target >= 2 * eps.size:
            return eps.max() + 100.0 * (width if width is not None else 1.0)

        if scheme == 'fermi':
            w = max(width, 1e-6)
            f = lambda mu: occ_fermi(eps, mu, w).sum() - nelec_target
            pad = 50.0 * w
        else:
            w = max(width, 1e-12)
            f = lambda mu: occ_gauss(eps, mu, w).sum() - nelec_target
            pad = 50.0 * w

        lo = eps.min() - pad
        hi = eps.max() + pad
        flo = f(lo)
        fhi = f(hi)
        if flo > 0:
            return lo
        if fhi < 0:
            return hi
        return scipy.optimize.brentq(f, lo, hi, maxiter=200)

    def get_occ_rhf(mo_energy, nelec_total, restricted):
        mo_energy = numpy.asarray(mo_energy)
        nmo = mo_energy.size

        if restricted is None:
            idx = numpy.arange(nmo)
            nelec_smear = nelec_total
            base_occ = numpy.zeros(nmo)
        else:
            ncore, ncas = restricted
            ncore = int(ncore)
            ncas = int(ncas)
            if ncore < 0 or ncas <= 0 or (ncore + ncas) > nmo:
                raise ValueError('restricted must be (ncore, ncas) with 0<=ncore and ncore+ncas<=nmo')
            idx = numpy.arange(ncore, ncore+ncas)
            base_occ = numpy.zeros(nmo)
            base_occ[:ncore] = 2.0
            base_occ[ncore+ncas:] = 0.0
            nelec_smear = nelec_total - 2*ncore
            if nelec_smear < -1e-8 or nelec_smear > 2*ncas + 1e-8:
                raise ValueError('restricted (ncore,ncas) incompatible with total electron count')

        if mf_fomo.fomo_method == 'fermi':
            if kT <= 1e-12:
                occ = base_occ.copy()
                order = numpy.argsort(mo_energy[idx])
                nfill = int(round(nelec_smear/2))
                occ_idx = idx[order]
                occ[occ_idx[:nfill]] = 2.0
                return occ
            mu = solve_mu(mo_energy[idx], nelec_smear, 'fermi', kT)
            occ = base_occ.copy()
            occ[idx] = occ_fermi(mo_energy[idx], mu, kT)
            return occ
        else:
            if sigma_eff <= 1e-12:
                occ = base_occ.copy()
                order = numpy.argsort(mo_energy[idx])
                nfill = int(round(nelec_smear/2))
                occ_idx = idx[order]
                occ[occ_idx[:nfill]] = 2.0
                return occ
            mu = solve_mu(mo_energy[idx], nelec_smear, 'gaussian', sigma_eff)
            occ = base_occ.copy()
            occ[idx] = occ_gauss(mo_energy[idx], mu, sigma_eff)
            return occ

    def get_occ_uhf(mo_energy, nelec_ab, restricted):
        mo_energy_a, mo_energy_b = mo_energy
        na, nb = nelec_ab
        mo_energy_a = numpy.asarray(mo_energy_a)
        mo_energy_b = numpy.asarray(mo_energy_b)

        def one_spin(eps, n_elec_spin, restricted_spin):
            if restricted_spin is None:
                idx = numpy.arange(eps.size)
                nelec_smear = n_elec_spin
                base = numpy.zeros(eps.size)
            else:
                ncore, ncas = restricted_spin
                ncore = int(ncore)
                ncas = int(ncas)
                idx = numpy.arange(ncore, ncore+ncas)
                base = numpy.zeros(eps.size)
                base[:ncore] = 1.0
                nelec_smear = n_elec_spin - ncore
            if mf_fomo.fomo_method == 'fermi':
                if kT <= 1e-12:
                    occ = base.copy()
                    order = numpy.argsort(eps[idx])
                    nfill = int(round(nelec_smear))
                    occ[idx[order[:nfill]]] = 1.0
                    return occ
                mu = solve_mu(eps[idx], nelec_smear, 'fermi', kT)
                occ = base.copy()
                x = (eps[idx] - mu) / max(kT, 1e-20)
                x = numpy.clip(x, -200.0, 200.0)
                occ[idx] = 1.0 / (1.0 + numpy.exp(x))
                return occ
            else:
                if sigma_eff <= 1e-12:
                    occ = base.copy()
                    order = numpy.argsort(eps[idx])
                    nfill = int(round(nelec_smear))
                    occ[idx[order[:nfill]]] = 1.0
                    return occ
                mu = solve_mu(eps[idx], nelec_smear, 'gaussian', sigma_eff)
                occ = base.copy()
                arg = (mu - eps[idx]) / (numpy.sqrt(2.0) * sigma_eff)
                occ[idx] = 0.5 * (1.0 + scipy.special.erf(arg))
                return occ

        if restricted is None:
            ra = rb = None
        else:
            if len(restricted) == 2 and isinstance(restricted[0], (int, numpy.integer)):
                ra = rb = restricted
            else:
                ra, rb = restricted

        occa = one_spin(mo_energy_a, na, ra)
        occb = one_spin(mo_energy_b, nb, rb)
        return (occa, occb)

    def get_occ(mf_self, mo_energy=None, mo_coeff=None):
        if mo_energy is None:
            mo_energy = mf_self.mo_energy
        if isinstance(mf_self, scf.uhf.UHF) or isinstance(mo_energy, (tuple, list)):
            nelec_ab = mf_self.nelec
            occ = get_occ_uhf(mo_energy, nelec_ab, restricted)
        else:
            nelec_total = mf_self.mol.nelectron
            occ = get_occ_rhf(mo_energy, nelec_total, restricted)
        return occ

    import types
    mf_fomo.get_occ = types.MethodType(get_occ, mf_fomo)
    return mf_fomo


if __name__ == '__main__':
    from pyscf import gto

    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='ccpvdz', verbose=4)

    mf = scf.RHF(mol)
    mf = fomo_scf(mf, temperature=0.01, method='gaussian')
    mf.kernel()

    print('FOMO-RHF Energy:', mf.e_tot)
    print('FOMO Occupations:', mf.mo_occ)
