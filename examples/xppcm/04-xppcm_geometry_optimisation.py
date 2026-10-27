#!/usr/bin/env python
#
# Author: Erika Magnusson <ermag@chalmers.se>
#

'''
Self-consistent pressure and geometry at a fixed cavity compression.

One point of an XP-PCM compression scan, for methane in cyclohexane:

1. Optimise the geometry in vacuum.
2. Reference cavity volume V_0 at the ambient scale f_0 from that geometry.
3. At the compressed scale f, iterate until the pressure is stable:
     a. V_c, z and eps from the current geometry;
     b. XP + PCM single point and its analytical pressure p;
     c. XPCav + PCM at that fixed p, geometry optimisation with geomeTRIC.

XPCav minimises G + p V_c at constant p, so p has to be known before each
optimisation; the outer loop makes p and the geometry mutually consistent,
and the returned p is always the one evaluated at the returned geometry.
A full scan repeats step 3 for decreasing f, starting each point from the
previous geometry.

Requires geomeTRIC:  pip install geometric
'''

import numpy as np
from pyscf import gto, dft, geomopt
from pyscf import xppcm
from pyscf.xppcm import (Rahm, get_z0, get_z, get_eps, get_cavity_volume,
                         calculate_analytical_pressure)

# Medium: cyclohexane
z0 = get_z0(36, 0.7781, 84.1595)   # valence electrons, density (g/cm^3), molar mass (g/mol)
eps0 = 2.0165                      # dielectric constant at ambient pressure
eta = 6                            # repulsive exponent in z(V_c)

f_0 = 1.30                         # ambient cavity scale, defines V_0
f = 0.90                           # compressed cavity scale for this point
radii = Rahm * f

# Pressure <-> geometry loop
max_outer = 5
p_tol_abs, p_tol_rel = 1e-2, 5e-3  # GPa; converged when |dp| <= max(abs, rel*|p|)

# geomeTRIC convergence. The XPCav energy carries grid noise (see the XPCav
# docstring), so converge on gradient and step rather than on a tight energy.
conv_params = {
    'convergence_energy': 1e-5,    # Eh
    'convergence_grms': 1e-3,      # Eh/Bohr
    'convergence_gmax': 1e-3,      # Eh/Bohr
    'convergence_drms': 5e-2,      # Angstrom
    'convergence_dmax': 5e-2,      # Angstrom
    'maxsteps': 50,
}

mol = gto.M(atom='''
C     -2.5179450512    1.3231302500    0.0000000000
H     -2.1375241435    1.7044693014    0.9245326241
H     -3.2778144672    0.5967039419    0.1995078991
H     -2.9333742079    2.1257041246   -0.5728821029
H     -1.7230673862    0.8656436320   -0.5511584204
''', basis='cc-pvdz', verbose=0)


def make_ks(mol):
    return dft.RKS(mol, xc='pbe0')


def attach_pcm(obj, eps):
    '''IEF-PCM on the same cavity as the Pauli repulsion.'''
    obj = obj.PCM()
    obj.with_solvent.method = 'IEF-PCM'
    obj.with_solvent.eps = eps
    obj.with_solvent.radii_table = radii
    return obj


def bond_length(mol, i=0, j=1):
    return np.linalg.norm(mol.atom_coords()[i] - mol.atom_coords()[j])


#
# 1. Vacuum geometry
#
mol = geomopt.optimize(make_ks(mol), **conv_params)
r_vac = bond_length(mol)

#
# 2. Reference cavity volume at the ambient scale
#
volume_0 = get_cavity_volume(mol, Rahm * f_0)
print('V_0 = %.2f Bohr^3 at f_0 = %.2f' % (volume_0, f_0))

#
# 3. Self-consistent (p, geometry) at scale f
#
dm0 = None
p_prev = None
converged = False
for outer in range(1, max_outer + 1):
    vc = get_cavity_volume(mol, radii)
    z = get_z(vc, volume_0, z0, eta=eta)
    eps = get_eps(vc, volume_0, eps0)

    # a-b. pressure at the current geometry
    ks_xp = attach_pcm(make_ks(mol).XP(z=z, radii_table=radii), eps)
    ks_xp.kernel(dm0=dm0)
    dm0 = ks_xp.make_rdm1()              # SCF guess for the next calculation
    terms = calculate_analytical_pressure(ks_xp, vc, eta=eta, return_terms=True)
    p = terms['total']

    msg = '[f=%.2f pass %d/%d] V_c=%8.2f  z=%.4f  eps=%.4f  p=%8.3f GPa' % (
        f, outer, max_outer, vc, z, eps, p)
    if p_prev is not None:
        msg += '  (dp=%+.3f)' % (p - p_prev)
    print(msg)

    if p_prev is not None and abs(p - p_prev) <= max(p_tol_abs, p_tol_rel * abs(p)):
        converged = True
        break
    p_prev = p
    if outer == max_outer:
        break                            # keep p consistent with the geometry

    # c. relax the geometry at fixed p
    ks_cav = attach_pcm(make_ks(mol).XPCav(z=z, radii_table=radii, p_gpa=p, grid_level=5), eps)
    ks_cav.kernel(dm0=dm0)
    dm0 = ks_cav.make_rdm1()
    mol = geomopt.optimize(ks_cav, **conv_params)

print('\nf = %.2f: p = %.3f GPa after %d pass(es)%s'
      % (f, p, outer, '' if converged else ' (NOT CONVERGED)'))
print('  repulsion %.3f  surface %.3f  polarisation %.3f  dielectric %.3f GPa'
      % (terms['repulsion'], terms['surface'], terms['polarisation'], terms['dielectric']))
print('  C-H bond: %.4f Bohr in vacuum -> %.4f Bohr at %.1f GPa' % (r_vac, bond_length(mol), p))
