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
#   - E_cav via mf.get_e_cav() (XPCav only)
#   - XPCav nuclear gradients via mf.nuc_grad_method(); post-HF gradients built on an
#     XPCav reference (MP2, CISD, CCSD, CASCI, CASSCF, TDDFT) include the XPCav terms
#   - Built via factory functions ks.XP(...) and ks.XPCav(...), which wrap standard SCF objects
# get_e_rep: Pauli repulsion energy of an SCF-with-XP or post-HF object
# get_z0, get_z, get_eps: helpers for calculating XP-PCM parameters from solvent properties
# calculate_fitted_pressure: fit E(V) to an equation of state and compute pressure
# calculate_numerical_pressure: compute pressure by numerical differentiation of E(V)
# calculate_analytical_pressure: compute pressure analytically
# get_cavity_volume_atom: compute cavity volume for a single atom
# get_cavity_volume: compute cavity volume

import numpy as np
import warnings
from scipy.optimize import curve_fit
from pyscf import lib, scf, dft, gto, ci, cc, mcscf
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
        Initial barrier height in Hartree.
    """
    # n_val * rho / M is the solvent valence-electron density in mol e-/cm^3;
    # 0.063 is the semi-empirical proportionality constant of the XP-PCM
    # parametrisation that maps it onto the barrier height z0 (Hartree).
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
        Barrier height in Hartree.
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

def get_e_rep(method_obj, dm=None, mol=None):
    """
    Pauli repulsion energy E_rep = z Tr[P S_out].

    Parameters
    ----------
    method_obj : SCF object with XP attached, or a post-HF object (CISD, CCSD,
        CASCI, CASSCF) built on one.
    dm : array, optional
        AO density matrix, (nao, nao) or (2, nao, nao), used instead of the one
        from method_obj (e.g. a relaxed post-HF density).
    mol : gto.Mole, optional

    Returns
    -------
    float
        E_rep in Hartree.

    Notes
    -----
    fci.FCI(mf) on an XP reference also includes h_rep (through mf.get_hcore())
    but drops any PCM and returns a solver without a reference, so it cannot be
    passed here or to calculate_analytical_pressure. mcscf.CASCI(mf, mol.nao,
    mol.nelec) is the same FCI and works with both, and with .PCM().
    """
    xp = getattr(method_obj, 'with_xp', None)
    post_hf = xp is None and hasattr(method_obj, '_scf')
    if post_hf:
        xp = getattr(method_obj._scf, 'with_xp', None)
    if xp is None:
        raise ValueError(
            "get_e_rep: no XP attached to method_obj (or to its underlying SCF "
            "object). Build the reference with XP(mf, ...).")
    if post_hf and not isinstance(method_obj, (ci.cisd.CISD, cc.ccsd.CCSDBase,
                                               mcscf.casci.CASBase)):
        raise ValueError(
            "get_e_rep: post-HF method must be a CISD, CCSD, CASCI or CASSCF "
            f"object. Found: {method_obj.__class__.__name__}")
    if dm is None:
        dm = _get_post_hf_dm_tot(method_obj) if post_hf else method_obj.make_rdm1()
    return xp.get_e_rep(dm, mol=mol if mol is not None else method_obj.mol)

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

def _get_S_out_ip(mol, radii_table, grid_level=3):
    """
    Derivative of _get_S_out, shape (3, nao, nao).
    The exterior-grid analogue of mol.intor('int1e_ipovlp').

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
    array, shape (3, nao, nao)
    """
    coords, weights = _build_grid(mol, radii_table, grid_level=grid_level)
    s_ip = np.zeros((3, mol.nao, mol.nao))
    for p0, p1 in lib.prange(0, len(weights), _xp_grid_blksize(mol) // 4):   # deriv=1: 4 AO arrays
        ao_all = dft.numint.eval_ao(mol, coords[p0:p1], deriv=1)
        aow = ao_all[0] * weights[p0:p1, np.newaxis]                          # (nblk, nao)
        for k in range(3):
            s_ip[k] += ao_all[1 + k].T @ aow
    return s_ip

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
        if mol is None:
            mol = self.mol
        return super().get_hcore(mol) + self.with_xp.get_h_rep(mol)

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
    h_rep and E_rep, and the two derivative terms the nuclear gradient needs,
    d(h_rep)/dR_A (hcore_deriv_generator) and dE_cav/dR_A (grad_cav).

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

    def hcore_deriv_generator(self, mol=None):
        """
        Derivative of h_rep, d(h_rep)/dR_A, shape (3, nao, nao), Hartree/Bohr.

        Two contributions to d(h_rep)/dR_A:
        1. Volume:    the basis functions on A move with it,
                      -z [d_S_out] + transpose
        2. Boundary:  sphere A moves and the exterior region with it,
                      -z R_A^2 4π Σ_l w_l n_{l,k} rho(r_l)
        PySCF's gradient codes contract these with the density, see
        WithXPCavGrad.hcore_generator.
        """
        mol = mol if mol is not None else self.mol
        nao = mol.nao
        aoslices = mol.aoslice_by_atom()
        s_ip = _get_S_out_ip(mol, self.radii_table, grid_level=self.grid_level)
        surface_data = _get_cavity_surface_grid(
            mol, self.radii_table, lebedev_order=self.lebedev_order)

        def hcore_deriv(atm_id):
            p0, p1 = aoslices[atm_id, 2:]
            h = np.zeros((3, nao, nao))
            h[:, p0:p1] -= s_ip[:, p0:p1]
            h = h + h.transpose(0, 2, 1)
            sd = surface_data[atm_id]
            if len(sd['coords']) > 0:
                ao = dft.numint.eval_ao(mol, sd['coords'], deriv=0)
                wn = sd['radius']**2 * 4.0 * np.pi * sd['weights'][:, np.newaxis] * sd['normals']
                for k in range(3):
                    h[k] -= (ao * wn[:, k:k + 1]).T @ ao
            return self.z * h
        return hcore_deriv

    def grad_cav(self, mol=None, atmlst=None):
        """
        Derivative of the cavitation energy, dE_cav/dR_A, shape (len(atmlst), 3), Hartree/Bohr.

        dE_cav/dR_A = p R_A^2 4π Σ_l w_l n_l over the exposed surface of sphere A. Assuming fixed p.
        """
        mol = mol if mol is not None else self.mol
        surface_data = _get_cavity_surface_grid(
            mol, self.radii_table, lebedev_order=self.lebedev_order)
        p_au = self.p_gpa / EH_BOHR3_TO_GPA
        de = np.zeros((mol.natm, 3))
        for A, sd in enumerate(surface_data):
            de[A] = p_au * sd['radius']**2 * 4.0 * np.pi * (sd['weights'] @ sd['normals'])
        if atmlst is not None:
            de = de[np.asarray(atmlst)]
        return de

    def grad(self, dm, mol=None):
        """
        XPCav part of the nuclear gradient for a given density.

        For analysis (e.g. separating the pressure forces from the vacuum ones).
        The SCF and post-HF gradients do not call this; they receive the same
        terms through WithXPCavGrad.
        """
        mol = mol if mol is not None else self.mol
        dm = _as_total_dm(dm)
        hcore_deriv = self.hcore_deriv_generator(mol)
        de = self.grad_cav(mol)
        for A in range(mol.natm):
            de[A] += np.einsum('xij,ij->x', hcore_deriv(A), dm)
        return de

class _SCFWithXPCav(_SCFWithXP):
    """
    SCF wrapper that includes both repulsion and cavitation contributions,
    along with analytical gradients for geometry optimisation under fixed pressure.
    """

    def get_e_cav(self, mol=None):
        """Get cavitation energy."""
        return self.with_xp.get_e_cav(mol)

    def energy_nuc(self):
        """Nuclear repulsion plus the cavitation work E_cav = p V_c."""
        return super().energy_nuc() + self.with_xp.get_e_cav(self.mol)

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
    Gradient decorator for XPCav.

    XPCav adds h_rep = z S_out to the core Hamiltonian and E_cav = p V_c to the
    nuclear energy, so its derivatives enter at the two matching points of
    PySCF's gradient interface: hcore_generator() gains d(h_rep)/dR_A and
    grad_nuc() gains dE_cav/dR_A, as QM/MM does for its terms.
    """

    def __init__(self, grad_method):
        self.__dict__.update(grad_method.__dict__)

    def undo_xp(self):
        """Remove the XPCav terms, returning the underlying gradient object."""
        return lib.view(self, lib.drop_class(self.__class__, WithXPCavGrad, 'XPCav'))

    def hcore_generator(self, mol=None):
        """Derivative of the core Hamiltonian including repulsion contributions."""
        if mol is None:
            mol = self.mol
        hcore_deriv = super().hcore_generator(mol)
        h_rep_deriv = self.base.with_xp.hcore_deriv_generator(mol)

        def hcore_deriv_xp(atm_id):
            return hcore_deriv(atm_id) + h_rep_deriv(atm_id)
        return hcore_deriv_xp

    def grad_nuc(self, mol=None, atmlst=None):
        """Nuclear gradient including cav contributions."""
        if mol is None:
            mol = self.mol
        return (super().grad_nuc(mol, atmlst)
                + self.base.with_xp.grad_cav(mol, atmlst))

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

# ---------------------------------------------------------------------------
# Patch at import time
# ---------------------------------------------------------------------------

scf.hf.SCF.XP = _xp_method
scf.hf.SCF.XPCav = _xpcav_method
