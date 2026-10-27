#!/usr/bin/env python
#
# Author: Erika Magnusson <ermag@chalmers.se>
#

'''
Geometry under pressure: XPCav energies and analytical nuclear gradients.

XP alone keeps the cavity fixed, so it has no meaningful nuclear gradient.
XPCav adds the cavitation work E_cav = p V_c at a fixed external pressure p
and provides the analytical gradient of E_rep + E_cav, so that the decorated
SCF object can be used for geometry optimisation at constant pressure.

Note on grids: the exterior integration grid is cut at the cavity surface, so
the energy carries grid noise while the analytical gradient is smooth.  Use
grid_level >= 5 for optimisations and converge on the gradient, not on an
energy criterion tighter than about 1e-6 Eh.
'''

import numpy as np
from pyscf import gto, scf
from pyscf import xppcm
from pyscf.xppcm import Rahm, get_cavity_volume, get_e_rep

radii = Rahm * 0.90
z = 0.30          # barrier height (Hartree)
p_gpa = 10.0      # external pressure (GPa)
# These values are for demonstration only. Really, z and p_gpa should be derived from the cavity volume
# with get_z and calculate_analytical_pressure, see 01-pressure_from_volume_scan.py. Therefore one would
# need an outer loop to converge on the pressure/geometry, see 04-xppcm_geometry_optimisation.py.

mol = gto.M(atom="""
C     -2.5179450512    1.3231302500    0.0000000000
H     -2.1375241435    1.7044693014    0.9245326241
H     -3.2778144672    0.5967039419    0.1995078991
H     -2.9333742079    2.1257041246   -0.5728821029
H     -1.7230673862    0.8656436320   -0.5511584204
""", basis='cc-pvtz', verbose=0)

#
# 1. Energy terms
#
mf = scf.RHF(mol).XPCav(z=z, radii_table=radii, p_gpa=p_gpa, grid_level=5)
mf.conv_tol = 1e-10
mf.kernel()
print('Cavity volume %.3f Bohr^3' % get_cavity_volume(mol, radii))
print('E_tot = %.8f Ha   E_rep = %.6f Ha   E_cav = pV = %.6f Ha'
      % (mf.e_tot, get_e_rep(mf), mf.get_e_cav()))

#
# 2. Analytical gradient, checked against a central finite difference
#
grad = mf.nuc_grad_method()
grad.verbose = 0
de = grad.kernel()
print('\nAnalytical gradient (Ha/Bohr):')
print(np.array2string(de, precision=6, suppress_small=True))

def energy_at(coords_bohr):
    m = mol.copy()
    m.set_geom_(coords_bohr, unit='Bohr')
    m.build()
    mf_ = scf.RHF(m).XPCav(z=z, radii_table=radii, p_gpa=p_gpa, grid_level=5)
    mf_.conv_tol = 1e-12
    mf_.verbose = 0
    return mf_.kernel()

h = 2e-3
coords = mol.atom_coords()
cp, cm = coords.copy(), coords.copy()
cp[1, 1] += h
cm[1, 1] -= h
fd = (energy_at(cp) - energy_at(cm)) / (2 * h)
print('dE/dy(H1): analytical %.6f   finite difference %.6f   (difference %.1e, grid noise)'
      % (de[1, 1], fd, de[1, 1] - fd))

#
# 3. Geometry optimisation at constant pressure (needs geomeTRIC or PyBerny)
#
try:
    from pyscf.geomopt.geometric_solver import optimize
except ImportError:
    optimize = None
if optimize is not None:
    mol_eq = optimize(mf, maxsteps=50)
    r_ch = np.linalg.norm(mol_eq.atom_coords()[1] - mol_eq.atom_coords()[0])
    r_ch0 = np.linalg.norm(coords[1] - coords[0])
    print('\nC-H distance: %.4f Bohr at vacuum geometry -> %.4f Bohr optimised at %.0f GPa'
          % (r_ch0, r_ch, p_gpa))
else:
    print('\n(geomeTRIC not installed: skipping the geometry optimisation; '
          'pip install geometric)')
