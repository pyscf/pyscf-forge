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
Analytic derivatives of the PCM polarisation free energy, for XP-PCM pressures.

Support module for xppcm.py. Everything here differentiates PySCF's
switching/Gaussian PCM with respect to a uniform scaling of the cavity radii
and with respect to the solvent dielectric constant, at fixed solute density.
'''

import numpy as np
import warnings
from scipy.spatial.distance import cdist
from scipy.special import erf
from pyscf import lib, df, gto
from pyscf.solvent import pcm
from pyscf.solvent.grad.pcm import grad_switch_h
from collections import namedtuple

_SQRT_PI = np.sqrt(np.pi)

# Generic form of the PCM operators:
#     K = S - alpha lambda_eps/(2 pi) bar(DAS),   R = -lambda_eps (I - alpha DA/(2 pi)),
#     lambda_eps = (eps - 1)/(eps + beta),
#     bar(DAS) = DAS, or (DAS + S A D^T)/2 when symmetrised.
_PCMForm = namedtuple('_PCMForm', 'alpha beta symmetrised')
_PCM_FORMS = {
    'C-PCM':   _PCMForm(0, 0.0, False),
    'CPCM':    _PCMForm(0, 0.0, False),
    'COSMO':   _PCMForm(0, 0.5, False),
    'IEF-PCM': _PCMForm(1, 1.0, False),
    'IEFPCM':  _PCMForm(1, 1.0, False),
    'SMD':     _PCMForm(1, 1.0, False),    # electrostatics as IEF-PCM
    'SS(V)PE': _PCMForm(1, 1.0, True),
}
PCM_METHODS = tuple(_PCM_FORMS)


def _form(pcmobj):
    """The _PCMForm for this solvent object; raises for an unsupported model."""
    try:
        return _PCM_FORMS[pcmobj.method.upper()]
    except KeyError:
        raise RuntimeError(
            f"Unknown implicit solvent model: {pcmobj.method}") from None

def get_pcm_obj(method_obj):
    """
    Return the PySCF PCM solvent object attached to ``method_obj``, or None.

    Looks on the object itself and, for post-HF methods, on the underlying SCF
    object. Non-PCM solvent models (ddCOSMO, ddPCM) return None.
    """
    for obj in (method_obj, getattr(method_obj, '_scf', None)):
        with_solvent = getattr(obj, 'with_solvent', None)
        if isinstance(with_solvent, pcm.PCM):
            return with_solvent
    return None

def _psi(x, exp_x2=None):
    """
    psi(x) = erf(x) - x erf'(x) = erf(x) - 2x/sqrt(pi) exp(-x^2): the fraction
    of a unit Gaussian's charge enclosed within radius x/zeta of its centre.
    ``exp_x2`` may pass a precomputed exp(-x^2).
    """
    if exp_x2 is None:
        exp_x2 = np.exp(-x * x)
    return erf(x) - (2.0 / _SQRT_PI) * x * exp_x2

def _dpsi(x, exp_x2=None):
    """psi'(x) = 4 x^2 / sqrt(pi) exp(-x^2)."""
    if exp_x2 is None:
        exp_x2 = np.exp(-x * x)
    return (4.0 / _SQRT_PI) * x * x * exp_x2

def _grid_velocities(surface):
    """
    Grid velocity under scaling of the current radii,
    u_i = dr_i/dlnf = r_i - q_A(i) = R_A n_i.

    Returns
    -------
    array, shape (ngrids, 3)
    """
    grid_coords = surface['grid_coords']
    atom_coords = surface['atom_coords']
    u = np.empty_like(grid_coords)
    for ia, (p0, p1) in enumerate(surface['gslice_by_atom']):
        u[p0:p1] = grid_coords[p0:p1] - atom_coords[ia]
    return u

def _dlogF_dlnf(surface):
    """
    d ln F_i / dlnf for the switching function F_i = prod_J h(d_iJ), with
    d_iJ = (r_iJ - R_in_J)/R_sw_J and R_in_J, R_sw_J both proportional to the
    cavity radius. Then with 1/f -> 1:

        d d_iJ/dlnf = (d r_iJ/dlnf - r_iJ) / R_sw_J

    Returns
    -------
    array, shape (ngrids,)
    """
    atom_coords = surface['atom_coords']
    grid_coords = surface['grid_coords']
    R_in_J = surface['R_in_J']
    R_sw_J = surface['R_sw_J']

    dlogF = np.zeros(grid_coords.shape[0])
    for ia, (p0, p1) in enumerate(surface['gslice_by_atom']):
        if p1 <= p0:
            continue
        coords = grid_coords[p0:p1]
        u = coords - atom_coords[ia]                       # (n, 3)
        ri_rJ = coords[:, None, :] - atom_coords           # (n, natm, 3)
        riJ = np.linalg.norm(ri_rJ, axis=-1)               # (n, natm)
        with np.errstate(divide='ignore', invalid='ignore'):
            diJ = (riJ - R_in_J) / R_sw_J
        diJ[:, ia] = 1.0
        diJ[diJ < 1e-8] = 0.0
        fiJ = pcm.switch_h(diJ)

        # d r_iJ/dlnf = (r_i - q_J) . u_i / r_iJ   (q_J is fixed)
        with np.errstate(divide='ignore', invalid='ignore'):
            driJ = np.einsum('nJx,nx->nJ', ri_rJ, u) / riJ
            ddiJ = (driJ - riJ) / R_sw_J
        ddiJ[:, ia] = 0.0

        safe = fiJ > 0.0
        ratio = np.where(safe, grad_switch_h(diJ) / np.where(safe, fiJ, 1.0), 0.0)
        dlogF[p0:p1] = np.einsum('nJ,nJ->n', ratio, ddiJ)
    return dlogF

def _dD_dS_dlnf(surface, dlogF, with_D=True):
    """
    dD/dlnf and dS/dlnf (full ngrids x ngrids matrices).

    With t = xi_ij r_ij, nu_ij = (r_i - r_j).n_j
    (SI S1.1.2, S1.2.3, with 1/f -> 1):

        S_ij = erf(t)/r_ij            dS_ij = -psi(t) dr_ij / r_ij^2 - 2 xi_ij/sqrt(pi) exp(-t^2)
        S_ii = xi_i sqrt(2/pi)/F_i    dS_ii = -S_ii (1 + dlnF_i)
        D_ij = nu_ij psi(t)/r_ij^3    dD_ij = [psi'(t) dt nu_ij + psi(t) dnu_ij]/r_ij^3 - 3 D_ij dr_ij/r_ij
        D_ii = -xi_i sqrt(2/pi)/(2R_A) dD_ii = -2 D_ii

    where dt = xi_ij dr_ij - t and dnu_ij = (u_i - u_j).n_j.

    Parameters
    ----------
    surface : dict
        Built PCM cavity surface (pcmobj.surface).
    dlogF : array, shape (ngrids,)
        dlnF_i/dlnf from _dlogF_dlnf.
    with_D : bool, default True
        Whether dD is needed (only for the models with alpha = 1).

    Returns
    -------
    dD : array, shape (ngrids, ngrids) or None
    dS : array, shape (ngrids, ngrids)
    """
    grid_coords = surface['grid_coords']
    xi = surface['charge_exp']
    norm_vec = surface['norm_vec']
    switch_fun = surface['switch_fun']
    R_vdw = surface['R_vdw']
    u = _grid_velocities(surface)

    xi2 = xi * xi
    xi_ij = np.outer(xi, xi) / np.sqrt(np.add.outer(xi2, xi2))

    rij = cdist(grid_coords, grid_coords)
    t = xi_ij * rij
    np.fill_diagonal(rij, 1.0)

    # d r_ij/dlnf = (r_i - r_j).(u_i - u_j)/r_ij, expanded so that only
    # (ngrids, ngrids) arrays appear
    gu = np.einsum('ix,ix->i', grid_coords, u)
    drij = -(grid_coords.dot(u.T) + u.dot(grid_coords.T))
    drij += gu[:, None]
    drij += gu[None, :]
    drij /= rij
    np.fill_diagonal(drij, 0.0)

    exp_t2 = np.exp(-t * t)
    psi = _psi(t, exp_t2)

    # dD (built first so that its temporaries can be released)
    dD = None
    if with_D:
        gn = np.einsum('ix,ix->i', grid_coords, norm_vec)
        un = np.einsum('ix,ix->i', u, norm_vec)
        nu_ij = grid_coords.dot(norm_vec.T) - gn[None, :]  # (r_i - r_j).n_j
        dt = xi_ij * drij - t                              # dt/dlnf
        dD = _dpsi(t, exp_t2) * dt                         # psi'(t) dt
        del dt
        dD *= nu_ij
        dD += psi * (u.dot(norm_vec.T) - un[None, :])      # psi(t) dnu_ij
        nu_ij *= psi
        nu_ij *= drij
        dD -= 3.0 * nu_ij / rij                            # -3 D_ij r_ij^3 dr_ij/r_ij
        del nu_ij
        dD /= rij**3
        D_diag = -xi * (2.0 / np.pi)**0.5 / (2.0 * R_vdw)  # as pcm.get_D_S
        np.fill_diagonal(dD, -2.0 * D_diag)

    # dS
    dS = -psi * drij / rij**2 - (2.0 / _SQRT_PI) * exp_t2 * xi_ij
    S_diag = xi * (2.0 / np.pi)**0.5 / switch_fun          # as pcm.get_D_S
    np.fill_diagonal(dS, -S_diag * (1.0 + dlogF))
    return dD, dS

def _dv_dlnf(pcmobj, dm_tot):
    """
    dv_i/dlnf for the solute electrostatic potential v = v_nuc - v_elec at
    the cavity surface points, at fixed solute density.

        dv_i/dlnf = grad_{r_i} v_i . u_i - xi_i dv_i/dxi_i,

    with, for a unit Gaussian, d/dR [erf(xi R)/R] = -psi(xi R)/R^2 and
    d/dxi [erf(xi R)/R] = 2/sqrt(pi) exp(-xi^2 R^2).
    The width response is therefore an overlap integral (int3c1e) rather than
    a Coulomb one.

    Parameters
    ----------
    pcmobj : pyscf.solvent.pcm.PCM
        Built PCM object.
    dm_tot : array, shape (nao, nao)
        Total (alpha+beta) AO density matrix.

    Returns
    -------
    array, shape (ngrids,)
    """
    mol = pcmobj.mol
    surface = pcmobj.surface
    grid_coords = surface['grid_coords']
    xi = surface['charge_exp']
    u = _grid_velocities(surface)
    ngrids = grid_coords.shape[0]
    nao = mol.nao

    # electronic part: numerical derivative of the Coulomb potential
    dv_e = np.zeros(ngrids)
    max_memory = pcmobj.max_memory - lib.current_memory()[0]
    blksize = int(max(max_memory * .9e6 / 8 / nao**2 / 4, 100))
    int3c2e_ip2 = mol._add_suffix('int3c2e_ip2')
    int3c1e = mol._add_suffix('int3c1e')
    opt_ip2 = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ip2)
    opt_1e = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c1e)
    for p0, p1 in lib.prange(0, ngrids, blksize):
        fakemol = gto.fakemol_for_charges(grid_coords[p0:p1], expnt=xi[p0:p1]**2)
        # int3c2e_ip2 returns -d/dr_i of the Coulomb integrals (mu nu | g_i)
        ip2 = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip2, aosym='s1',
                               cintopt=opt_ip2)
        grad_ve = -np.einsum('xijL,ij->Lx', ip2, dm_tot)
        # d/dxi (mu nu | g_i) = 2/sqrt(pi) (pi/xi^2)^{3/2} <mu nu | g_i>
        ovlp = df.incore.aux_e2(mol, fakemol, intor=int3c1e, aosym='s1',
                                cintopt=opt_1e)
        dve_dxi = (2.0 / _SQRT_PI * (np.pi / xi[p0:p1]**2)**1.5
                   * np.einsum('ijL,ij->L', ovlp, dm_tot))
        dv_e[p0:p1] = (np.einsum('Lx,Lx->L', grad_ve, u[p0:p1])
                       - dve_dxi * xi[p0:p1])

    # nuclear part: analytical derivative of the Coulomb potential
    atom_coords = mol.atom_coords(unit='B')
    charges = np.asarray(mol.atom_charges(), dtype=float)
    rvec = grid_coords[:, None, :] - atom_coords           # (ngrids, natm, 3)
    R = np.linalg.norm(rvec, axis=-1)
    xR = xi[:, None] * R
    # grad_{r_i} sum_A Z_A erf(xi R)/R = -sum_A Z_A psi(xi R)/R^2 (r_i - q_A)/R
    grad_vn = np.einsum('iA,iAx->ix', -charges * _psi(xR) / R**3, rvec)
    dvn_dxi = 2.0 / _SQRT_PI * np.einsum('A,iA->i', charges, np.exp(-xR**2))
    dv_n = np.einsum('ix,ix->i', grad_vn, u) - dvn_dxi * xi

    return dv_n - dv_e

def _dKq_dRv_dlnf(pcmobj, dS, dD, dA, v, q):
    """
    The products (dK/dlnf) q and (dR/dlnf) v for the generic form of K and R:

        dR v  = alpha lambda_eps/(2 pi) d(DA) v
        dK q  = dS q - alpha lambda_eps/(2 pi) d(bar(DAS)) q

    with d(DA) x = dD (A x) + D (dA x) and, when symmetrised,
    d(bar(DAS)) q = [d(DAS) q + (d(DAS))^T q]/2, where
    (d(DAS))^T q = S (A dD^T q + dA D^T q) + dS (A D^T q).

    Returns
    -------
    tuple of (array, array or None)
        (dK @ q, dR @ v); dR v is None for the conductor-like models.
    """
    form = _form(pcmobj)
    dSq = dS.dot(q)
    if form.alpha == 0:
        return dSq, None                        # K = S, R = -lambda_eps I

    A = pcmobj._intermediates['A']
    D = pcmobj._intermediates['D']
    S = pcmobj._intermediates['S']
    f_eps = pcmobj._intermediates['f_epsilon']  # the article's lambda_eps
    fac = form.alpha * f_eps / (2.0 * np.pi)

    dDA_dot = lambda x: dD.dot(A * x) + D.dot(dA * x)
    dRv = fac * dDA_dot(v)
    dDAS_q = dDA_dot(S.dot(q)) + D.dot(A * dSq)
    if form.symmetrised:
        DTq = D.T.dot(q)
        dDAS_T_q = S.dot(A * dD.T.dot(q) + dA * DTq) + dS.T.dot(A * DTq)
        dDAS_q = 0.5 * (dDAS_q + dDAS_T_q)
    return dSq - fac * dDAS_q, dRv

def _dE_deps(pcmobj, vK_1, v, q, q_sym):
    """
    Only lambda_eps depends on eps, and KQ = R
    collapses the bracket of eq:E_pol_deriv_theta to (dlambda/lambda) S q,

        dE_pol/deps = (1 + beta)/(2 (eps - 1)(eps + beta)) (K^-T v) . (S q).

    """
    eps = pcmobj.eps
    if eps == float('inf'):
        return 0.0                              # lambda_eps is pinned at 1
    form = _form(pcmobj)
    f_eps = pcmobj._intermediates['f_epsilon']
    dlam_deps = (1.0 + form.beta) / (eps + form.beta)**2
    if f_eps == 0.0:                            # eps = 1
        dKq, dRv = _dKq_dRv_deps(pcmobj, v, q)
        return _energy_derivative(vK_1, q_sym, None, dKq, dRv)
    S = pcmobj._intermediates['S']
    return 0.5 * (dlam_deps / f_eps) * np.dot(vK_1, S.dot(q))

def _dKq_dRv_deps(pcmobj, v, q):
    """
    General-path cross-check for _dE_deps: the products (dK/deps) q and
    (dR/deps) v through the chain rule in lambda_eps,

        dK/deps = (dlambda/deps) (K - S)/lambda,   dR/deps = (dlambda/deps) R/lambda.

    Returns
    -------
    tuple of (array or None, array or None)
        (dK @ q, dR @ v); None where the term vanishes.
    """
    eps = pcmobj.eps
    if eps == float('inf'):
        return None, None
    form = _form(pcmobj)
    dlam_deps = (1.0 + form.beta) / (eps + form.beta)**2
    if form.alpha == 0:                          # K = S, R = -lambda_eps I
        return None, -dlam_deps * v

    A = pcmobj._intermediates['A']
    D = pcmobj._intermediates['D']
    S = pcmobj._intermediates['S']
    fac = form.alpha * dlam_deps / (2.0 * np.pi)
    dRv = fac * D.dot(A * v) - dlam_deps * v
    DAS_q = D.dot(A * S.dot(q))
    if form.symmetrised:
        DAS_q = 0.5 * (DAS_q + S.dot(A * D.T.dot(q)))
    return -fac * DAS_q, dRv

def _energy_derivative(vK_1, q_sym, dv, dKq, dRv):
    """
    eq:E_pol_deriv_theta for one parameter theta:

        dE_pol/dtheta = q_sym . dv + 1/2 (K^-T v) . (dR v - dK q).

    Parameters
    ----------
    vK_1 : array, shape (ngrids,)
        K^-T v, solved once by the caller and reused for every parameter.
    q_sym : array, shape (ngrids,)
        Symmetrised apparent surface charges (PySCF's q_sym).
    dv : array or None
        dv/dtheta; None if v does not depend on theta.
    dKq, dRv : array or None
        The products (dK/dtheta) q and (dR/dtheta) v; None where the term vanishes.

    Returns
    -------
    float
    """
    out = 0.0 if dv is None else np.dot(dv, q_sym)
    if dKq is None and dRv is None:
        return out
    rhs = np.zeros_like(vK_1)
    if dRv is not None:
        rhs += dRv
    if dKq is not None:
        rhs -= dKq
    return out + 0.5 * np.dot(vK_1, rhs)

def pol_derivatives(pcmobj, dm_tot):
    """
    Analytic derivatives of the PCM polarisation free energy at fixed solute
    density: (dE_pol/dlnf, dE_pol/deps).

    dE_pol/dlnf is the derivative with respect to a relative scaling of the
    current cavity radii, R_A -> lambda R_A at lambda = 1. In absolute 
    scaling factor, R_A = f r_A, it equals f dE_pol/df. The volume derivative 
    that the pressure divides by (xppcm) is formed the same way, so the 
    pressure does not depend on this choice.

    Parameters
    ----------
    pcmobj : pyscf.solvent.pcm.PCM
        PCM solvent object; built if it has not been built yet.
    dm_tot : array, shape (nao, nao)
        Total (alpha+beta) AO density matrix.

    Returns
    -------
    tuple of float
        (dE_pol/dlnf, dE_pol/deps) in Hartree (per unit ln f / per unit eps).
    """
    if pcmobj.frozen:
        raise NotImplementedError(
            "The analytical polarisation pressure requires a variational "
            "solvent: with with_solvent.frozen = True the density does not "
            "relax against the solvent charges, so the Hellmann-Feynman "
            "shortcut used here does not hold. Converge the solvent with "
            "(frozen = False)")
    if pcmobj.method.upper() not in PCM_METHODS:
        raise NotImplementedError(
            f"Analytical polarisation pressure is not implemented for "
            f"'{pcmobj.method}'. Supported: C-PCM, COSMO, IEF-PCM, SS(V)PE.")
    if pcmobj.method.upper() == 'SMD':
        warnings.warn(
            "SMD: only the electrostatic (IEF-PCM) part of the polarisation "
            "pressure is included. The CDS surface-tension term also depends on "
            "the cavity size and is not differentiated here.",
            UserWarning, stacklevel=2)

    if not pcmobj._intermediates:
        pcmobj.build()
    # Refresh q and v_grids for this density (post-HF densities differ from the
    # SCF one cached during the last get_veff). Mirrors pyscf.solvent.grad.pcm.
    pcmobj._get_vind(np.asarray(dm_tot))

    surface = pcmobj.surface
    itm = pcmobj._intermediates
    v, q, q_sym, K = itm['v_grids'], itm['q'], itm['q_sym'], itm['K']

    # K^-T v is the same for every parameter, so it is solved once here rather
    # than inside each _energy_derivative call (it is an O(ngrids^3) solve).
    vK_1 = np.linalg.solve(K.T, v)

    # the four derivatives of the SI, in dependency order
    dlogF = _dlogF_dlnf(surface)
    dD, dS = _dD_dS_dlnf(surface, dlogF, with_D=_form(pcmobj).alpha != 0)
    dA = surface['area'] * (2.0 + dlogF)                  # dA_i/dlnf = A_i (2 + dlnF_i)
    dv = _dv_dlnf(pcmobj, dm_tot)

    dKq, dRv = _dKq_dRv_dlnf(pcmobj, dS, dD, dA, v, q)
    dE_dlnf = _energy_derivative(vK_1, q_sym, dv, dKq, dRv)
    dE_deps = _dE_deps(pcmobj, vK_1, v, q, q_sym)
    return dE_dlnf, dE_deps
