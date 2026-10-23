#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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
# Author: Erika Magnusson <ermag@chalmers.se>
#

'''
XP-PCM (eXtreme-Pressure Polarizable Continuum Model) for PySCF.
'''

# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------
# XP, XPCav: main classes for XP-PCM and XP-PCM with cavitation energy
#   - E_rep available via mf.get_e_rep()
#   - E_cav available via mf.with_xp.get_e_cav() (XPCav only)
#   - XPCav gradients available via mf.with_xp.nuc_grad_method()
#   - Built via factory functions ks.XP(...) and ks.XPCav(...), which wrap standard SCF objects
# get_z0, get_z, get_eps: helpers for calculating XP-PCM parameters from solvent properties
# calculate_fitted_pressure: fit E(V) to an equation of state and compute pressure
# calculate_numerical_pressure: compute pressure by numerical differentiation of E(V)
# calculate_analytical_pressure: compute pressure analytically
# get_cavity_volume_atom: compute cavity volume for a single atom
# get_cavity_volume: compute cavity volume

import numpy as np
import warnings
from scipy.optimize import curve_fit
from pyscf import lib, scf, dft, gto, ci, cc, mcscf, fci
from pyscf.lib.parameters import BOHR
from pyscf.data.radii import VDW
from pyscf.dft.gen_grid import MakeAngularGrid, LEBEDEV_ORDER, _default_ang, _default_rad, get_partition
from .pcm_derivatives import get_pcm_obj, pol_derivatives
from .analytical_cavity import union_of_spheres_volume
from .radii import RAHM

# AU pressure conversion factor: 1 Eh/Bohr^3 = 29421.015798 GPa
EH_BOHR3_TO_GPA = 29421.015798

Rahm = RAHM.copy()
Vdw  = VDW.copy()
Mod_Bondi = Vdw.copy()
Mod_Bondi[1] = 1.1 / BOHR

# ---------------------------------------------------------------------------
#  Public helpers
# ---------------------------------------------------------------------------

def get_z0(solvent_num_valence_elec, solvent_density, solvent_molar_mass):
    """
    Calculate the initial barrier height z_0 from solvent parameters.

    Parameters
    ----------
    solvent_num_valence_elec : int
        Number of valence electrons in a solvent molecule.
    solvent_density : float
        Density of the solvent (g/cm^3).
    solvent_molar_mass : float
        Molar mass of the solvent (g/mol).

    Returns
    -------
    float
        Semi-empirical barrier height / implicit electron density in atomic units.
    """
    # n_val * rho / M is the solvent valence-electron density in mol e-/cm^3;
    # 0.063 is the semi-empirical proportionality constant of the XP-PCM
    # parametrisation that maps it onto the barrier height z0 (a.u.).
    return (solvent_num_valence_elec * solvent_density / solvent_molar_mass) * 0.063

def get_z(volume, volume_0, z0, eta=6):
    """
    Calculate the barrier height z at a given cavity volume.

    z = z0 * s^-(3 + eta) with s = (V_c / V_c^0)^(1/3), i.e. eq 9 of
    Cammi, Chen & Rahm (2018): z = z0 * (V_c / V_c^0)^(-(3 + eta)/3).

    Parameters
    ----------
    volume : float or array
        Cavity volume at the current compression (Bohr^3, atomic units).
    volume_0 : float
        Reference (ambient-pressure) cavity volume (Bohr^3, atomic units).
    z0 : float
        Barrier height at volume_0.
    eta : int, default 6
        Repulsive exponent.

    Returns
    -------
    float or array
        Semi-empirical barrier height / implicit electron density in atomic units.
    """
    return z0 * _get_scaling_factor(volume, volume_0) ** -(3 + eta)

def get_eps(volume, volume_0, eps_0):
    """
    Calculate the dielectric constant at a given cavity volume.

    Parameters
    ----------
    volume : float or array
        Cavity volume at the current compression (Bohr^3, atomic units).
    volume_0 : float
        Reference (ambient-pressure) cavity volume (Bohr^3, atomic units).
    eps_0 : float
        Dielectric constant at volume_0 (ambient value of the medium at the
        working temperature).

    Returns
    -------
    float or array
        Dielectric constant.
    """
    return 1 + (eps_0 - 1) * _get_scaling_factor(volume, volume_0) ** -3

def calculate_fitted_pressure(energies, volumes):
    """
    Calculate pressure from a fitted Birch–Murnaghan-type equation of state.
    Cannot capture electronic transitions but works well for smooth E(V) curves.

    Parameters
    ----------
    energies : array
        Total energies (Hartree).
    volumes : array
        Cavity volumes (Bohr^3, atomic units).

    Returns
    -------
    array
        Pressures (GPa) at each volume.
    """
    def _eos(x, a, b, c):
        return (a / b) * (1 / x) ** b + (a - c) * x

    volumes = np.asarray(volumes)
    energies = np.asarray(energies)

    x_data = volumes / volumes[0]
    y_data = energies - energies[0]

    try:
        popt, _ = curve_fit(_eos, x_data, y_data, maxfev=20000)
    except RuntimeError as e:
        raise RuntimeError(
            "calculate_fitted_pressure: the equation-of-state fit did not converge. "
            "Check that energies and volumes are ordered consistently and that E(V) "
            "is smooth (no electronic transition); otherwise use "
            "calculate_numerical_pressure.") from e
    a_fit, b_fit, c_fit = popt

    a = a_fit / volumes[0]
    b = b_fit
    c = c_fit / volumes[0]

    v_ratio = volumes[0] / volumes
    return (a * (v_ratio ** (b + 1) - 1) + c) * EH_BOHR3_TO_GPA

def calculate_numerical_pressure(energies, volumes):
    """
    Calculate pressure by numerical differentiation of E(V).
    Central differences in the interior, one-sided (first-order) differences at the two end points.

    Parameters
    ----------
    energies : array
        Energies (Hartree). Can be total energies or separate contribtions (i.e. polarisation energy from PCM).
    volumes : array
        Cavity volumes (Bohr^3, atomic units).

    Returns
    -------
    array
        Pressures (GPa) at each volume.
    """
    energies = np.asarray(energies, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    pressure = np.zeros_like(energies)

    # Central difference for interior points
    for i in range(1, len(energies) - 1):
        dE_dV = (energies[i + 1] - energies[i - 1]) / (volumes[i + 1] - volumes[i - 1])
        pressure[i] = dE_dV * EH_BOHR3_TO_GPA

    # Forward / backward difference for endpoints
    pressure[0]  = (energies[1] - energies[0]) / (volumes[1] - volumes[0]) * EH_BOHR3_TO_GPA
    pressure[-1] = (energies[-1] - energies[-2]) / (volumes[-1] - volumes[-2]) * EH_BOHR3_TO_GPA

    return -pressure

def calculate_analytical_pressure(method_obj, volume_cavity, eta=6, lebedev_order=47,
                                  polarisation=True, return_terms=False, dm=None):
    """
    Calculate pressure from the XP-PCM SCF analytical expression.
    Cammi, R., Chen, B. & Rahm, M.,
    Journal of Computational Chemistry 39, 2243-2250 (2018).

        P_rep  = (3 + eta)/(3 V_c) * E_rep            from z scaling with volume
        P_surf = z * <rho>_surface                    from the shrinking exterior

    If a PySCF PCM solvent is attached, two further terms are added:

        P_pol  = -dG_pol/dV_c = -(dG_pol/dlnf)/(dV_c/dlnf)  from a relative scaling of the cavity radii
        P_eps  = (eps - 1)/V_c * dG_pol/deps          from the dielectric constant scaling with volume

    Parameters
    ----------
    method_obj : method object
        SCF object with XP attached, or a post-HF object (CISD, CCSD, CASCI,
        CASSCF) whose _scf has XP attached. A PySCF PCM may be attached to
        either with .PCM().
    volume_cavity : float
        Cavity volume (Bohr^3, atomic units).
    eta : float, default 6
        Repulsive exponent.
    lebedev_order : int, default 47
        Lebedev quadrature order for surface grids (47 = 770 points per sphere).
    polarisation : bool, default True
        Include the PCM polarisation terms when a PCM solvent is attached.
        Ignored if no PCM is attached.
    return_terms : bool, default False
        Return a dict of the individual contributions instead of just the total.
    dm : array, optional
        AO density matrix, (nao, nao) or (2, nao, nao), used instead of the one
        from method_obj: e.g. a relaxed post-HF density, or the density of an
        fci.FCI solver (then pass the XP SCF object as method_obj).

    Notes
    -----
    Hellmann-Feynman derivative at fixed wavefunction: exact for variational
    methods (SCF, FCI, CASSCF with self-consistent PCM). For CCSD, CISD, CASCI
    and perturbation theories the unrelaxed density misses the orbital response.

    Returns
    -------
    float or dict
        Pressure in GPa. If return_terms is True, a dict with keys
        'total', 'repulsion', 'surface', 'polarisation', 'dielectric' (all GPa).
    """
    post_hf = hasattr(method_obj, '_scf') # Post HF objects have an _scf attribute that contains the SCF object.
    xp = getattr(method_obj, 'with_xp', None)
    if xp is None and post_hf:
        xp = getattr(method_obj._scf, 'with_xp', None)
    if xp is None:
        raise ValueError(
            "calculate_analytical_pressure: no XP attached to method_obj "
            "(or to its underlying SCF object). Build the reference with mf.XP(...).")
    z = xp.z
    radii_table = xp.radii_table
    mol = method_obj.mol

    if dm is not None:
        dm_tot = _as_total_dm(dm)
    elif post_hf:
        dm_tot = _get_post_hf_dm_tot(method_obj)
    else: # SCF method with XP attached directly
        dm_tot = _as_total_dm(method_obj.make_rdm1())

    # Term 1: Repulsive energy contribution
    e_rep = xp.get_e_rep(dm_tot, mol=mol)
    term1 = ((3.0 + eta) / (3.0 * volume_cavity)) * e_rep

    # Term 2: Surface electron density contribution
    surface_data = _get_cavity_surface_grid(
        mol, radii_table, lebedev_order=lebedev_order)

    numerator_surfint = 0.0   # Σ_A R_A^3 Σ_exposed(w_l * ρ_l)
    denominator_surfint = 0.0 # Σ_A R_A^3 Σ_exposed(w_l)

    for atom in range(mol.natm):
        sd = surface_data[atom]
        w = sd['weights']
        r3 = sd['radius'] ** 3

        if len(sd['coords']) > 0:
            # Evaluate electron density ρ(r) = Σ_μν P_μν φ_μ(r) φ_ν(r)
            ao_surf = dft.numint.eval_ao(mol, sd['coords'], deriv=0)
            rho_surf = np.einsum('gi,ij,gj->g', ao_surf, dm_tot, ao_surf)

            # Σ_A R_A^3 Σ_exposed(w_l * ρ_l)
            numerator_surfint += r3 * np.sum(w * rho_surf)

            # Σ_A R_A^3 Σ_exposed(w_l)
            denominator_surfint += r3 * np.sum(w)

    # Compute weighted surface density
    if denominator_surfint > 1e-14:
        term2 = z * numerator_surfint / denominator_surfint
    else:
        term2 = 0.0

    # dV_c/dlnf = 4 pi sum_A R_A^3 sum_l w_l: the volume derivative with respect to a
    # relative scaling of the current radii (article: dV_c/df = 4 pi f^2 sum r_A^3 w_l,
    # i.e. this divided by f). The polarisation derivatives use the same convention.
    dvc_dlnf = 4.0 * np.pi * denominator_surfint

    # Terms 3 and 4: PCM polarisation contributions
    term3 = term4 = 0.0
    pcmobj = get_pcm_obj(method_obj)
    if pcmobj is not None:
        # polarisation=False is an explicit opt-out, so it passes without comment
        if polarisation:
            term3, term4 = _polarisation_pressure_terms(
                pcmobj, dm_tot, volume_cavity, dvc_dlnf)
    else:
        # A non-PCM solvent model (ddCOSMO, ddPCM) may be attached, on the
        # object itself or (post-HF) on the underlying SCF object; no
        # analytical polarisation derivative is available for it.
        solvent = getattr(method_obj, 'with_solvent', None)
        if solvent is None and post_hf:
            solvent = getattr(method_obj._scf, 'with_solvent', None)
        if solvent is not None:
            warnings.warn(
                f"Analytical pressure calculation does not include polarisation "
                f"effects from the attached "
                f"{solvent.__class__.__name__} solvent model; only "
                f"PySCF's switching/Gaussian PCM is supported.",
                UserWarning,
                stacklevel=2
            )

    if return_terms:
        return {
            'total':        (term1 + term2 + term3 + term4) * EH_BOHR3_TO_GPA,
            'repulsion':    term1 * EH_BOHR3_TO_GPA,
            'surface':      term2 * EH_BOHR3_TO_GPA,
            'polarisation': term3 * EH_BOHR3_TO_GPA,
            'dielectric':   term4 * EH_BOHR3_TO_GPA,
        }

    p_gpa = (term1 + term2 + term3 + term4) * EH_BOHR3_TO_GPA

    return p_gpa

def get_cavity_volume_atom(mol, radii_table):
    """
    Exact volume of the cavity for a single atom.
    Raises ``ValueError`` for multi-atom molecules.

    Parameters
    ----------
    mol : gto.Mole
        Molecule object. Must contain exactly one atom.
    radii_table : array
        Cavity radii in Bohr, indexed by nuclear charge.

    Returns
    -------
    float
        Volume in Bohr^3.
    """
    if mol.natm != 1:
        raise ValueError("get_cavity_volume_atom is only valid for single-atom molecules.")
    R = _get_cavity_radii(mol, radii_table)[0]
    return (4.0 / 3.0) * np.pi * R ** 3

def get_cavity_volume(mol, radii_table):
    """
    Exact analytical volume of the van der Waals cavity (union of the atomic spheres).

    Parameters
    ----------
    mol : gto.Mole
        Molecule object.
    radii_table : array
        Cavity radii in Bohr, indexed by nuclear charge.

    Returns
    -------
    float
        Volume in Bohr^3.
    """
    radii_arr = _get_cavity_radii(mol, radii_table)
    return union_of_spheres_volume(mol.atom_coords(), radii_arr)

# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _get_scaling_factor(volume, volume_0):
    """
    Effective cavity scaling factor s = (V_c / V_c^0)^(1/3).

    Parameters
    ----------
    volume : float or array
        Cavity volume at the current compression (Bohr^3, atomic units).
    volume_0 : float
        Reference (ambient-pressure) cavity volume (Bohr^3, atomic units).

    Returns
    -------
    float or array
        Scaling factor s.
    """
    return (np.asarray(volume, dtype=float) / volume_0) ** (1.0 / 3.0)

def _as_total_dm(dm):
    """
    Collapse a density matrix to the total (alpha+beta) AO density.

    Parameters
    ----------
    dm : array, shape (nao, nao) or (2, nao, nao)

    Returns
    -------
    array, shape (nao, nao)
    """
    dm = np.asarray(dm)
    return dm[0] + dm[1] if dm.ndim == 3 else dm

def _get_cavity_radii(mol, radii_table):
    """
    Resolve cavity radii in Bohr as a per-atom array.

    Parameters
    ----------
    mol : gto.Mole
        Molecule object.
    radii_table : array
        Cavity radii in Bohr, indexed by nuclear charge.

    Returns
    -------
    array, shape (natm,)
        Cavity radius of each atom in Bohr.
    """
    return np.array([
        radii_table[gto.charge(mol.atom_symbol(i))]
        for i in range(mol.natm)
    ], dtype=float)

class _PerAtomKeys:
    """
    Mole proxy for ``dft.gen_grid.get_partition`` whose ``atom_symbol(ia)``
    returns the atom index, so an ``atom_grids_tab`` can hold one atomic grid
    per atom.
    """

    def __init__(self, mol):
        self._mol = mol

    def __getattr__(self, name):
        return getattr(self._mol, name)

    def atom_symbol(self, ia):
        return ia

def _build_grid(mol, radii_table, grid_level=3):
    """
    Build the exterior grid for the XP-PCM integrals: all points outside the
    molecular cavity.

    Each atom's radial shells (Treutler-Ahlrichs) start at that
    atom's cavity radius, and the atoms are joined by Becke weights.
    Points that fall inside another atom's sphere are removed. Angular pruning is not applied.
    This grid is used only for the XP-PCM integrals, not for the DFT part when using a DFT functional.
    # TODO: Radius fitting support in the future!

    Parameters
    ----------
    mol : gto.Mole
        Molecule object.
    radii_table : array
        Cavity radii in Bohr, indexed by nuclear charge.
    grid_level : int, default 3
        PySCF grid level for integration (default: 3).

    Returns
    -------
    coords : array, shape (ngrid_out, 3)
        Grid coordinates in Bohr, exterior points only.
    weights : array, shape (ngrid_out,)
        Grid weights, exterior points only.
    """

    radii = _get_cavity_radii(mol, radii_table)

    atom_grids_tab = {}
    for ia in range(mol.natm):
        chg = gto.charge(mol.atom_symbol(ia))
        n_rad = _default_rad(chg, grid_level)
        n_ang = _default_ang(chg, level=grid_level)
        rad, dr = dft.radi.treutler_ahlrichs(n_rad, chg)
        rad = rad + radii[ia]
        ang = MakeAngularGrid(n_ang)                      # (n_ang, 4): x, y, z, w
        coords = np.einsum('i,jk->ijk', rad, ang[:, :3]).reshape(-1, 3)
        vol = np.einsum('i,j->ij', 4.0 * np.pi * rad**2 * dr, ang[:, 3]).ravel()
        atom_grids_tab[ia] = (coords, vol)
    defaults = dft.gen_grid.Grids(mol)   # PySCF's Becke partition settings
    coords, weights = get_partition(
        _PerAtomKeys(mol), atom_grids_tab,
        defaults.radii_adjust, defaults.atomic_radii, defaults.becke_scheme)

    # Hard cut at the cavity surface. An atom's own shells start at its radius,
    # but shells of other atoms cross this sphere and their points switch in and
    # out as the nuclei move, so the energy is only piecewise smooth in the
    # nuclear coordinates (noise ~1e-6 Eh at grid_level 3 for z ~ 0.3). The
    # analytical gradient is the continuum limit and is unaffected.
    atom_coords = mol.atom_coords()
    inside = np.zeros(len(coords), dtype=bool)
    for ia in range(mol.natm):
        dist = np.linalg.norm(coords - atom_coords[ia], axis=1)
        inside |= (dist <= radii[ia])
    outside = ~inside
    return coords[outside], weights[outside]

def _xp_grid_blksize(mol, max_memory_mb=None):
    """
    Grid points per block for the exterior integrals, such that one (nblk, nao)
    AO array takes about a third of the remaining memory budget.
    """
    if max_memory_mb is None:
        max_memory_mb = max(mol.max_memory - lib.current_memory()[0], 0)
    return int(max(2000, max_memory_mb * 1e6 / 3 / 8 / max(mol.nao, 1)))

def _get_S_out(mol, radii_table, grid_level=3):
    """
    Overlap matrix S_out evaluated on the confined (exterior) grid.

    Parameters
    ----------
    mol : gto.Mole
        Molecule object.
    radii_table : array
        Cavity radii in Bohr, indexed by nuclear charge.
    grid_level : int, default 3
        PySCF grid level for integration (default: 3).

    Returns
    -------
    array, shape (nao, nao)
        Overlap matrix in the AO basis outside the cavity.
    """
    coords, weights = _build_grid(mol, radii_table, grid_level=grid_level)
    S_out = np.zeros((mol.nao, mol.nao))
    for p0, p1 in lib.prange(0, len(weights), _xp_grid_blksize(mol)):
        ao = dft.numint.eval_ao(mol, coords[p0:p1], deriv=0)
        S_out += ao.T @ (ao * weights[p0:p1, np.newaxis])
    return S_out

def _get_S_out_nuc_grad(mol, dm, radii_table, grid_level=3):
    """
    Analytical nuclear gradient of S_out w.r.t. nuclear coordinates.

    Parameters
    ----------
    mol : gto.Mole
        Molecule object.
    dm : array, shape (nao, nao)
        Density matrix in the AO basis.
    radii_table : array
        Cavity radii in Bohr, indexed by nuclear charge.
    grid_level : int, default 3
        PySCF grid level for integration (default: 3).

    Returns
    -------
    array, shape (natm, 3)
        In atomic units (Bohr^-1).
    """
    coords, weights = _build_grid(mol, radii_table, grid_level=grid_level)

    contrib = np.zeros((3, mol.nao))
    for p0, p1 in lib.prange(0, len(weights), _xp_grid_blksize(mol) // 4):   # deriv=1: 4 AO arrays
        ao_all = dft.numint.eval_ao(mol, coords[p0:p1], deriv=1)
        ao  = ao_all[0]        # (nblk, nao)
        dao = ao_all[1:4]      # (3, nblk, nao)
        Vphi = (ao * weights[p0:p1, np.newaxis]) @ dm        # (nblk, nao)
        contrib += np.einsum('kgm,gm->km', dao, Vphi)        # (3, nao)

    grad = np.zeros((mol.natm, 3))
    ao_slices = mol.aoslice_by_atom()
    for A in range(mol.natm):
        p0, p1 = ao_slices[A][2], ao_slices[A][3]
        grad[A, :] = -2.0 * contrib[:, p0:p1].sum(axis=1)
    return grad

def _get_cavity_surface_grid(mol, radii_table, lebedev_order=47):
    """
    Exposed Lebedev surface grid on each atom's cavity sphere.
    Parameters
    ----------
    mol : gto.Mole
        Molecule object.
    radii_table : array
        Cavity radii in Bohr, indexed by nuclear charge.
    lebedev_order : int, default 47
        Order of the Lebedev grid.

    Returns
    -------
    list of dict
        One dict per atom with keys 'coords', 'normals', 'weights', 'radius'.
    """
    atom_coords = mol.atom_coords()                          # (natm, 3) Bohr
    cavity_radii = _get_cavity_radii(mol, radii_table)

    npts  = LEBEDEV_ORDER[lebedev_order]
    grid  = MakeAngularGrid(npts)          # (m, 4): x, y, z, w  (w sums to 1)
    dirs  = grid[:, :3]
    leb_w = grid[:, 3]

    surface_data = []
    for A in range(mol.natm):
        R_A = cavity_radii[A]
        surf_pts = atom_coords[A] + R_A * dirs                # (m, 3)

        exposed = np.ones(len(dirs), dtype=bool)
        for B in range(mol.natm):
            if B == A:
                continue
            dist2 = np.sum((surf_pts - atom_coords[B])**2, axis=1)
            exposed &= dist2 > cavity_radii[B]**2

        surface_data.append({
            'coords':  surf_pts[exposed],
            'normals': dirs[exposed],
            'weights': leb_w[exposed],
            'radius':  R_A,
        })

    return surface_data

def _polarisation_pressure_terms(pcmobj, dm_tot, volume_cavity, dvc_dlnf):
    """
    The two polarisation pressure contributions in Eh/Bohr^3 (not yet in GPa).

    ``dvc_dlnf`` is the exact dV_c/dlnf from the cavity surface grid, the
    derivative with respect to a relative scaling of the current radii.
    pol_derivatives returns dG_pol/dlnf in the same convention, so the ratio
    is dG_pol/dV_c regardless of the absolute scaling factor f.

    Returns
    -------
    tuple of float
        (P_pol, P_eps): the cavity-scaling and dielectric-scaling terms.
    """
    dG_dlnf, dG_deps = pol_derivatives(pcmobj, dm_tot)

    if dvc_dlnf > 1e-14:
        p_pol = -dG_dlnf / dvc_dlnf
    else:
        p_pol = 0.0

    # eps(V_c) = 1 + (eps_0 - 1) V_c^0 / V_c (the get_eps law), so
    # deps/dV_c = -(eps_0 - 1) V_c^0 / V_c^2 = -(eps - 1)/V_c.
    # A Clausius-Mossotti law would give -(eps - 1)(eps + 2)/(3 V_c) instead
    # (More accurate for non-polar media but diverges for polar)
    if pcmobj.eps == float('inf'):
        p_eps = 0.0     # f_epsilon is pinned at 1, so dG_pol/deps vanishes
    else:
        p_eps = (pcmobj.eps - 1.0) / volume_cavity * dG_deps
    return p_pol, p_eps

# ---------------------------------------------------------------------------
#  XP (Pauli repulsion only)
# ---------------------------------------------------------------------------

class _XP:
    """
    Encapsulates Pauli repulsion energy and Hamiltonian calculations.

    This class manages the configuration and computes two types of corrections:
    - Hamiltonian correction: h_rep = z·S_out (added to core Hamiltonian each SCF iteration)
    - Energy correction: E_rep = z·Tr[P·S_out] (for diagnostics/reporting)

    where z is a semi-empirical barrier height and S_out is the overlap matrix
    evaluated on the cavity exterior.

    Intended for internal use by _SCFWithXP; not for direct instantiation.
    Use XP() factory function or mf.XP().

    Parameters
    ----------
    mol : gto.Mole
    z : float
        Semi-empirical barrier height z (Hartree); see get_z0 and get_z.
    radii_table : array
        Pre-scaled cavity radii in Bohr, indexed by nuclear charge.
    grid_level : int, default 3
        PySCF grid level for XP overlap integrals.
    """

    def __init__(self, mol, z, radii_table, grid_level=3):
        self.mol = mol
        self.z = z
        self.radii_table = radii_table
        self.grid_level = int(grid_level)

    def reset(self, mol=None):
        """Reset the object to a new molecule."""
        if mol is not None:
            self.mol = mol
        return self

    def get_h_rep(self, mol=None):
        """h_rep = z · S_out."""
        mol = mol if mol is not None else self.mol
        return _get_S_out(mol, self.radii_table, grid_level=self.grid_level) * self.z

    def get_e_rep(self, dm, mol=None):
        """E_rep = z · Tr[P · S_out]."""
        mol = mol if mol is not None else self.mol
        dm_tot = _as_total_dm(dm)
        S_out = _get_S_out(mol, self.radii_table, grid_level=self.grid_level)
        return np.einsum('ij,ji->', dm_tot, S_out) * self.z

class _SCFWithXP:
    """
    SCF wrapper that includes XP corrections to the Hamiltonian and energy.
    """
    _keys = {'with_xp'}

    def __init__(self, mf, xp_params):
        self.__dict__.update(mf.__dict__)
        self.with_xp = xp_params

    @property
    def radii_table(self):
        """Expose radii_table for PCM compatibility."""
        return self.with_xp.radii_table

    @radii_table.setter
    def radii_table(self, value):
        """Allow setting radii_table for PCM compatibility."""
        self.with_xp.radii_table = value

    def undo_xp(self):
        """Remove XP, returning the underlying SCF object."""
        cls = self.__class__
        obj = lib.view(self, lib.drop_class(cls, _SCFWithXP, 'XP'))
        del obj.with_xp
        return obj

    def reset(self, mol=None):
        """Reset both the SCF object and the XP parameters to a new molecule."""
        self.with_xp.reset(mol)
        return super().reset(mol)

    def get_hcore(self, mol=None):
        return super().get_hcore(mol) + self.with_xp.get_h_rep(mol)

    def get_e_rep(self, dm=None, mol=None):
        """Get Pauli repulsion energy for diagnostics."""
        if dm is None:
            dm = self.make_rdm1()
        return self.with_xp.get_e_rep(dm, mol)

    def nuc_grad_method(self):
        raise NotImplementedError(
            "Nuclear gradients are not defined for XP because "
            "geometry optimisation under XPPCM requires a cavitation energy contribution. (Otherwise cavity is fixed) "
            "Use XPCav(mf, z, radii_table, p_gpa) for geometry optimisation under a fixed pressure.")

    Gradients = nuc_grad_method

    def Hessian(self):
        raise NotImplementedError(
            "Nuclear Hessians are not implemented for XP-PCM. Without this guard the "
            "call would silently fall through to the vacuum SCF Hessian, which omits "
            "every XP derivative term and returns a plausible-looking but wrong result.")

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        xp = self.with_xp
        log = lib.logger.new_logger(self, verbose)
        log.info('')
        log.info('******** XP parameters ********')
        log.info('z (barrier height)    = %.3f', xp.z)
        log.info('grid_level            = %d', xp.grid_level)
        return self

def XP(mf, z, radii_table, grid_level=3):
    """
    Attach XP (Pauli repulsion only) to an SCF method.

    Usage:
        ks = dft.RKS(mol)
        ks_xp = XP(ks, z=0.5, radii_table=Vdw)
        ks_xp.kernel()

    Parameters
    ----------
    mf : pyscf.scf.SCF
        SCF object to wrap.
    z : float
        Semi-empirical barrier height z (Hartree); see get_z0 and get_z.
    radii_table : array
        Pre-scaled cavity radii in Bohr, indexed by nuclear charge.
    grid_level : int, default 3
        PySCF grid level for XP integrals.

    Returns
    -------
    Modified SCF object with XP Hamiltonian correction
    """
    if isinstance(mf, _SCFWithXP):
        raise ValueError(
            "Cannot attach XP to an object that already has XP.")

    with_xp = _XP(mf.mol, z, radii_table, grid_level=grid_level)

    sol_mf = _SCFWithXP(mf, with_xp)
    name = 'XP' + mf.__class__.__name__
    return lib.set_class(sol_mf, (_SCFWithXP, mf.__class__), name)

# ---------------------------------------------------------------------------
#  XPCav (Pauli repulsion + cavitation)
# ---------------------------------------------------------------------------

class _XPCav(_XP):
    """
    Encapsulates Pauli repulsion and approximate cavitation plus gradient calculations.

    Extends _XP with fixed-pressure cavitation: computes E_cav = pV in addition to
    h_rep and E_rep. Also provides analytical nuclear gradients for geometry optimisation
    under constant external pressure.

    Intended for internal use by _SCFWithXPCav; not for direct instantiation.
    Use XPCav() factory function or mf.XPCav().

    Parameters
    ----------
    mol : gto.Mole
    z : float
        Semi-empirical barrier height z (Hartree); see get_z0 and get_z.
    radii_table : array
        Pre-scaled cavity radii in Bohr, indexed by nuclear charge.
    p_gpa : float
        External pressure (GPa).
    grid_level : int, default 3
        PySCF grid level for volume and overlap integrals.
    lebedev_order : int, default 47
        Lebedev quadrature order for surface integrals (→ 770 points).
    """

    def __init__(self, mol, z, radii_table, p_gpa=None, grid_level=3,
                 lebedev_order=47):
        super().__init__(mol, z, radii_table, grid_level=grid_level)
        self.p_gpa = p_gpa
        self.lebedev_order = int(lebedev_order)

    def get_e_cav(self, mol=None):
        """E_cav = pV in Hartree."""
        mol = mol if mol is not None else self.mol
        V = get_cavity_volume(mol, self.radii_table)
        return V * self.p_gpa / EH_BOHR3_TO_GPA

    def grad(self, dm, mol=None):
        """Nuclear gradient of (E_rep + E_cav), shape (natm, 3), Hartree/Bohr.

        Computes three contributions:
        1. Cavitation:  dE_cav/dR_A = p · R_A^2 · 4π · Σ_l w_l n_{l,k}
        2. Boundary:    dE_rep/dR_A|_bnd = −z · R_A^2 · 4π · Σ_l w_l ρ_l n_{l,k}
        3. Volume:      z · d/dR Tr[P · S_out]
        """
        mol = mol if mol is not None else self.mol
        dm = _as_total_dm(dm)
        p_au = self.p_gpa / EH_BOHR3_TO_GPA  # pressure in atomic units

        # Surface integrals (cavitation + boundary)
        surface_data = _get_cavity_surface_grid(
            mol, self.radii_table, lebedev_order=self.lebedev_order)

        grad = np.zeros((mol.natm, 3))
        for atom in range(mol.natm):
            sd = surface_data[atom]
            R  = sd['radius']
            w  = sd['weights']
            n  = sd['normals']
            prefactor = R**2 * 4.0 * np.pi

            # Cavitation: p · dV/dR_A
            wn = np.einsum('l,lk->k', w, n)
            grad[atom] += p_au * prefactor * wn

            # Boundary response of h_rep
            if len(sd['coords']) > 0:
                ao  = dft.numint.eval_ao(mol, sd['coords'], deriv=0)
                rho = np.einsum('lm,lm->l', ao @ dm, ao)
                grad[atom] -= self.z * prefactor * np.einsum('l,l,lk->k', w, rho, n)

        # Volume-integral contribution from h_rep
        grad += self.z * _get_S_out_nuc_grad(
            mol, dm, self.radii_table, grid_level=self.grid_level)

        return grad

class _SCFWithXPCav(_SCFWithXP):
    """
    SCF wrapper that includes both repulsion and cavitation contributions,
    along with analytical gradients for geometry optimisation under fixed pressure.
    """

    def get_e_cav(self, mol=None):
        """Get cavitation energy."""
        return self.with_xp.get_e_cav(mol)

    def grad(self, dm=None, mol=None):
        """Compute nuclear gradients."""
        if dm is None:
            dm = self.make_rdm1()
        return self.with_xp.grad(dm, mol)

    def energy_tot(self, dm=None, h1e=None, vhf=None):
        """Total energy including cavitation energy."""
        e_tot = super().energy_tot(dm, h1e, vhf)
        e_cav = self.with_xp.get_e_cav()
        e_tot += e_cav
        self.scf_summary['e_cav'] = e_cav
        return e_tot

    def nuc_grad_method(self):
        """Return gradient object supporting XPCav gradients."""
        # Unwrap to get underlying gradient object
        inner_grad = self.undo_xp().nuc_grad_method()
        # Store reference for XPCav gradient computation
        inner_grad.base = self
        # Wrap with XPCav gradient layer
        grad_obj = WithXPCavGrad(inner_grad)
        # Create dynamic class with appropriate name
        name = 'XPCav' + inner_grad.__class__.__name__
        return lib.set_class(grad_obj, (WithXPCavGrad, inner_grad.__class__), name)

    Gradients = nuc_grad_method

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        xp = self.with_xp
        log = lib.logger.new_logger(self, verbose)
        log.info('')
        log.info('******** XPCav parameters (geometry optimisation) ********')
        log.info('z (barrier height)    = %.3f', xp.z)
        log.info('p (pressure)          = %.4f GPa', xp.p_gpa)
        log.info('grid_level            = %d', xp.grid_level)
        log.info('lebedev_order         = %d', xp.lebedev_order)
        return self

class WithXPCavGrad:
    """
    Gradient mixin for XPCav — adds cavitation + boundary + volume
    gradient to whatever gradient object is underneath (vanilla SCF, PCM, …).
    """
    _keys = {'de_xp'}

    def __init__(self, grad_method):
        self.__dict__.update(grad_method.__dict__)
        self.de_xp = None

    def undo_xp(self):
        cls = self.__class__
        obj = lib.view(self, lib.drop_class(cls, WithXPCavGrad, 'XPCav'))
        del obj.de_xp
        return obj

    def kernel(self, *args, dm=None, **kwargs):
        if dm is None:
            dm = self.base.make_rdm1()
        dm_tot = _as_total_dm(dm)

        # XPCav gradient: cavitation + boundary + volume, for all atoms
        self.de_xp = self.base.with_xp.grad(dm_tot)

        # Inner gradient (vanilla SCF, and/or PCM). GradientsBase.kernel records
        # the atmlst it used on self.atmlst and returns (len(atmlst), 3).
        de_inner = super().kernel(*args, **kwargs)

        de_xp = self.de_xp
        if self.atmlst is not None:
            de_xp = de_xp[np.asarray(self.atmlst)]
        self.de = de_inner + de_xp
        return self.de

    grad = kernel

def XPCav(mf, z, radii_table, p_gpa=None, grid_level=3, lebedev_order=47):
    """
    Attach XPCav (Pauli repulsion + cavitation) to an SCF method
    for geometry optimisation under fixed pressure.

    Usage:
        ks = dft.RKS(mol)
        ks_xpcav = XPCav(ks, z=0.5, radii_table=Rahm, p_gpa=0.1)
        ks_xpcav.kernel()  # SCF
        grad_solver = ks_xpcav.nuc_grad_method()  # Get gradient solver
        grad_solver.kernel()  # Compute gradients

    Parameters
    ----------
    mf : pyscf.scf.SCF
        SCF object to wrap.
    z : float
        Semi-empirical barrier height z (Hartree); see get_z0 and get_z.
    radii_table : array
        Pre-scaled cavity radii in Bohr, indexed by nuclear charge.
    p_gpa : float
        External pressure in GPa.
    grid_level : int, default 3
        PySCF grid level for volume and overlap integrals.
    lebedev_order : int, default 47
        Lebedev quadrature order for surface integrals (→ 770 points).

    Returns
    -------
    Modified SCF object with XPCav corrections (h_rep + e_cav) and gradients

    Notes
    -----
    The exterior grid (_build_grid) is cut hard at the cavity surface, so the
    repulsion energy is only piecewise smooth in the nuclear coordinates: as
    nuclei move, points of one atom's shells cross another atom's sphere and
    switch in or out of the integral. The analytical gradient is the derivative
    of the underlying continuum expression and is essentially converged at
    grid_level 3, but the energy carries grid noise.
    For geometry optimisations use grid_level >= 5 and do not rely on
    energy-based convergence criteria tighter than ~1e-6 Eh; converge on the
    gradient and the step instead.
    """
    if p_gpa is None:
        raise ValueError("p_gpa (pressure in GPa) is required for XPCav")

    if isinstance(mf, _SCFWithXP):
        raise ValueError(
            "Cannot attach XPCav to an object that already has XP.")

    with_xpcav = _XPCav(
        mf.mol, z, radii_table, p_gpa=p_gpa, grid_level=grid_level,
        lebedev_order=lebedev_order)

    sol_mf = _SCFWithXPCav(mf, with_xpcav)
    name = 'XPCav' + mf.__class__.__name__
    return lib.set_class(sol_mf, (_SCFWithXPCav, mf.__class__), name)

# ---------------------------------------------------------------------------
#  Method-style convenience wrappers on SCF objects
# ---------------------------------------------------------------------------

def _xp_method(self, z, radii_table, grid_level=3):
    """Method-style XP wrapper: ks = ks.XP(...)."""
    return XP(self, z, radii_table, grid_level=grid_level)

def _xpcav_method(self, z, radii_table, p_gpa=None, grid_level=3,
                  lebedev_order=47):
    """Method-style XPCav wrapper: ks = ks.XPCav(...)."""
    return XPCav(self, z, radii_table, p_gpa=p_gpa, grid_level=grid_level,
                 lebedev_order=lebedev_order)

def _patch_scf_factory_methods():
    """Enable ks.XP(...) and ks.XPCav(...) on PySCF SCF/KS objects."""
    from pyscf.scf import hf as scf_hf

    if getattr(scf_hf.SCF, "_xppcm_factory_methods_patched", False):
        return

    scf_hf.SCF.XP = _xp_method
    scf_hf.SCF.XPCav = _xpcav_method
    scf_hf.SCF._xppcm_factory_methods_patched = True

# ---------------------------------------------------------------------------
#  XP for post-HF (CCSD, CISD, CASCI/CASSCF)
# ---------------------------------------------------------------------------

def _get_post_hf_dm_tot(method_obj):
    """Helper to compute the total AO density matrix from a post-HF method object."""
    mf = method_obj._scf
    if isinstance(mf, scf.ghf.GHF):
        raise NotImplementedError(
            "XP for post-HF methods on a GHF reference is not supported.")
    if isinstance(method_obj, mcscf.casci.CASBase):
        # CASCI/CASSCF (and their solvent-decorated subclasses) return the
        # AO-basis density, unlike CISD/CCSD which return the MO basis.
        return _as_total_dm(method_obj.make_rdm1())
    if isinstance(mf, scf.uhf.UHF):   # UKS is a subclass of UHF
        dm_mo_a, dm_mo_b = method_obj.make_rdm1()
        dm_a = np.einsum('pi,ij,qj->pq', method_obj.mo_coeff[0], dm_mo_a, method_obj.mo_coeff[0].conj())
        dm_b = np.einsum('pi,ij,qj->pq', method_obj.mo_coeff[1], dm_mo_b, method_obj.mo_coeff[1].conj())
        return dm_a + dm_b
    else:
        dm_mo = method_obj.make_rdm1()
        return np.einsum('pi,ij,qj->pq', method_obj.mo_coeff, dm_mo, method_obj.mo_coeff.conj())

def _get_e_rep_post_hf(self, dm=None, mol=None):
    """Computes the Pauli repulsion energy for a post-HF method with XP attached."""
    if not hasattr(self, '_scf') or getattr(self._scf, 'with_xp', None) is None:
        raise AttributeError("Underlying SCF object does not use XP-PCM.")

    mf = self._scf
    xp_obj = mf.with_xp
    mol = mol if mol is not None else self.mol
    if not isinstance(self, (ci.cisd.CISD, cc.ccsd.CCSDBase, mcscf.casci.CASBase)):
        raise ValueError(
            f"Post-HF method must be a CISD, CCSD, CASCI or CASSCF object. "
            f"Found: {self.__class__.__name__}")

    if dm is None:
        dm = _get_post_hf_dm_tot(self)

    return xp_obj.get_e_rep(dm, mol=mol)

def _patch_post_scf_kernel_for_xp(cls):
    """Patch a post-SCF class to allow get_e_rep access."""
    if not hasattr(cls, 'with_xp'):
        setattr(cls, 'with_xp', property(lambda self: getattr(self._scf, 'with_xp', None)))
    if not hasattr(cls, 'get_e_rep'):
        setattr(cls, 'get_e_rep', _get_e_rep_post_hf)

def _patch_post_scf_kernels_for_xp():
    """Allow get_e_rep access for supported post-SCF methods."""
    #_patch_post_scf_kernel_for_xp(mp.mp2.MP2) # No MP2 because orbital relaxation too important for pressure.
    #_patch_post_scf_kernel_for_xp(mp.ump2.UMP2)
    _patch_post_scf_kernel_for_xp(ci.cisd.CISD)
    _patch_post_scf_kernel_for_xp(ci.ucisd.UCISD)
    _patch_post_scf_kernel_for_xp(cc.ccsd.CCSD)
    _patch_post_scf_kernel_for_xp(cc.uccsd.UCCSD)
    _patch_post_scf_kernel_for_xp(cc.rccsd.RCCSD)
    _patch_post_scf_kernel_for_xp(mcscf.casci.CASBase)   # CASCI, CASSCF, FCI via CASCI(mf, nao, nelec)
    # CCSDT or higher not natively supported because cannot get DM.

def _patch_fci_factory_for_xp():
    """
    Warn when fci.FCI is built on an XP reference. The factory takes h_rep from
    mf.get_hcore() but only mf.energy_nuc() for the constant, so any PCM is
    silently dropped, and the returned solver has no _scf, so the pressure
    cannot be evaluated from it. CASCI with every orbital active is the same
    FCI and works with .PCM() and calculate_analytical_pressure.
    """
    if getattr(fci, '_xppcm_factory_patched', False):
        return
    _orig_FCI = fci.FCI

    def FCI(mol_or_mf, *args, **kwargs):
        if getattr(mol_or_mf, 'with_xp', None) is not None:
            warnings.warn(
                "fci.FCI(mf) on an XP reference includes the Pauli operator (via "
                "mf.get_hcore()) but drops any PCM solvent, and its solver cannot be "
                "passed to calculate_analytical_pressure. Use "
                "mcscf.CASCI(mf, mf.mol.nao, mf.mol.nelec) instead (the same FCI; "
                ".PCM() for a self-consistent PCM).",
                UserWarning, stacklevel=2)
        return _orig_FCI(mol_or_mf, *args, **kwargs)

    FCI.__doc__ = _orig_FCI.__doc__
    fci.FCI = FCI
    fci._xppcm_factory_patched = True

# ---------------------------------------------------------------------------
# Patch at import time
# ---------------------------------------------------------------------------

_patch_post_scf_kernels_for_xp()
_patch_fci_factory_for_xp()
_patch_scf_factory_methods()
