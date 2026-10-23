#!/usr/bin/env python
#
# Author: Erika Magnusson <ermag@chalmers.se>
#

'''
XP-PCM (eXtreme-Pressure Polarizable Continuum Model) single-point energies.

The XP-PCM model extends the standard PCM model with a Pauli repulsion term
E_rep = z Tr(P S_out), where S_out is the overlap of the basis functions
evaluated outside the cavity and z is a semi-empirical barrier height.

This example attaches the Pauli repulsion to a KS object with .XP() and,
optionally, the polarisation with the standard PySCF .PCM().
Both terms use the same (pre-scaled) cavity radii.
'''

from pyscf import gto, dft
from pyscf import xppcm   # registers mf.XP() and mf.XPCav() on SCF objects
from pyscf.xppcm import Rahm, get_cavity_volume_atom

mol = gto.M(atom='Be 0 0 0', basis='cc-pvtz', verbose=0)

# XP-PCM parameters at a cavity scaling factor f = 0.9 in cyclohexane.
# z and eps are normally derived from the cavity volume with get_z/get_eps,
# see 01-pressure_from_volume_scan.py; here they are given directly.
f_vdw = 0.9
scaled_radii = Rahm * f_vdw   # cavity radii in Bohr, indexed by nuclear charge
z = 0.279                     # barrier height (Hartree)
eps = 3.4025                  # dielectric constant of the compressed medium

# 1. Vacuum reference
mf = dft.RKS(mol, xc='pbe0').run()

# 2. Vacuum + Pauli repulsion
mf_xp = dft.RKS(mol, xc='pbe0').XP(z=z, radii_table=scaled_radii).run()

# 3. Vacuum + Pauli repulsion + PCM polarisation.
#    Setting with_solvent.radii_table makes the PCM use exactly the XP cavity
#    (PySCF's default vdw_scale is only applied when radii_table is None).
mf_xppcm = dft.RKS(mol, xc='pbe0').XP(z=z, radii_table=scaled_radii).PCM()
mf_xppcm.with_solvent.method = 'IEF-PCM'
mf_xppcm.with_solvent.eps = eps
mf_xppcm.with_solvent.radii_table = scaled_radii
mf_xppcm.run()

print('Cavity volume            %10.3f Bohr^3' % get_cavity_volume_atom(mol, scaled_radii))
print('E(vacuum)                %14.8f Ha' % mf.e_tot)
print('E(vacuum + E_rep)        %14.8f Ha   E_rep = %.6f Ha' % (mf_xp.e_tot, mf_xp.get_e_rep()))
print('E(vacuum + E_rep + E_pol)%14.8f Ha   E_rep = %.6f Ha   E_pol = %.6f Ha'
      % (mf_xppcm.e_tot, mf_xppcm.get_e_rep(), mf_xppcm.with_solvent.e))

# xppcm.XP(mf, ...) is equivalent to mf.XP(...), and undo_xp() strips it again.
mf_plain = mf_xppcm.undo_xp()
print('Class with XP+PCM: %s;  after undo_xp(): %s'
      % (mf_xppcm.__class__.__name__, mf_plain.__class__.__name__))
