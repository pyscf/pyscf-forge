#!/usr/bin/env python
#
# Author: Erika Magnusson <ermag@chalmers.se>
#

'''
Pressure from a cavity-volume scan, three ways.

In XP-PCM the pressure is the derivative of the electronic free energy with
respect to the cavity volume, p = -dG/dV_c.  The cavity is shrunk by scaling
the van der Waals radii with a factor f, and the barrier height z and the
dielectric constant eps follow the volume (get_z, get_eps).

The pressure can then be obtained
  1. by fitting G(V_c) to an equation of state (calculate_fitted_pressure),
  2. by finite differences of G(V_c)        (calculate_numerical_pressure),
  3. analytically at each point              (calculate_analytical_pressure),
     Cammi, Chen & Rahm, J. Comput. Chem. 39, 2243 (2018), extended here with
     the PCM polarisation and dielectric terms.
'''

import numpy as np
from pyscf import gto, dft
from pyscf import xppcm
from pyscf.xppcm import (Rahm, get_z0, get_z, get_eps, get_cavity_volume_atom,
                         calculate_fitted_pressure, calculate_numerical_pressure,
                         calculate_analytical_pressure)

mol = gto.M(atom='Be 0 0 0', basis='cc-pvtz', verbose=0)

# Medium: cyclohexane
solvent_num_valence_elec = 36    # valence electrons per solvent molecule
solvent_density = 0.7781         # g/cm^3
solvent_molar_mass = 84.1595     # g/mol
eps_0 = 2.0165                   # dielectric constant at ambient pressure
z0 = get_z0(solvent_num_valence_elec, solvent_density, solvent_molar_mass)

# Reference (ambient-pressure) cavity: f = 1.30
f_0 = 1.30
volume_0 = get_cavity_volume_atom(mol, Rahm * f_0)

f_values = np.arange(1.20, 0.79, -0.05)

volumes, energies, p_analytical, p_terms = [], [], [], []
for f in f_values:
    radii = Rahm * f
    vc = get_cavity_volume_atom(mol, radii)
    z = get_z(vc, volume_0, z0)
    eps = get_eps(vc, volume_0, eps_0)

    mf = dft.RKS(mol, xc='pbe0').XP(z=z, radii_table=radii).PCM()
    mf.with_solvent.method = 'IEF-PCM'
    mf.with_solvent.eps = eps
    mf.with_solvent.radii_table = radii
    mf.kernel()

    volumes.append(vc)
    energies.append(mf.e_tot)
    terms = calculate_analytical_pressure(mf, vc, return_terms=True)
    p_analytical.append(terms['total'])
    p_terms.append(terms)

p_fitted = calculate_fitted_pressure(energies, volumes)
p_numerical = calculate_numerical_pressure(energies, volumes)

print(' f         V_c (Bohr^3) G (Ha)            p_fit (GPa) p_num (GPa) p_ana (GPa)')
for f, v, e, pf, pn, pa in zip(f_values, volumes, energies, p_fitted, p_numerical, p_analytical):
    print('%5.2f   %10.3f   %14.8f   %9.3f   %9.3f   %9.3f' % (f, v, e, pf, pn, pa))
print('p_num uses central differences with step 0.05, one-sided at the end points.\n'
      ' p_fit assumes a smooth equation of state.')

i = 4   # f = 1.00
t = p_terms[i]
print('\nAnalytical pressure decomposition at f = %.2f (GPa):' % f_values[i])
for key in ('repulsion', 'surface', 'polarisation', 'dielectric', 'total'):
    print('  %-13s %9.3f' % (key, t[key]))
