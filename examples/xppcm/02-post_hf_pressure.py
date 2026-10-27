#!/usr/bin/env python
#
# Author: Erika Magnusson <ermag@chalmers.se>
#

'''
XP-PCM pressure with correlated wavefunctions.

The Pauli repulsion enters the one-electron Hamiltonian of the reference SCF,
so any post-HF method built on an XP reference includes it in its energy.
calculate_analytical_pressure() accepts CISD, CCSD, CASCI and CASSCF objects
and evaluates the pressure with their one-particle density:
exact for variational methods (SCF, FCI, CASSCF), approximate for CCSD/CISD
whose unrelaxed density misses some orbital response. The missed orbital
response is small for CCSD and CISD, but would be large for MP2 (not implemented).
Nuclear gradients need an XPCav reference (03-xpcav_gradient_and_geometry.py);
post-HF gradients built on one include the XPCav terms.
'''

from pyscf import gto, scf, cc, ci, mcscf
from pyscf import xppcm
from pyscf.xppcm import (Rahm, get_z0, get_z, get_eps, get_cavity_volume_atom,
                         get_e_rep, calculate_analytical_pressure)

mol = gto.M(atom='Be 0 0 0', basis='cc-pvtz', verbose=0)

# Cyclohexane, cavity scaled to f = 0.90 relative to the f_0 = 1.30 reference
z0 = get_z0(36, 0.7781, 84.1595)
volume_0 = get_cavity_volume_atom(mol, Rahm * 1.30)
radii = Rahm * 0.90
vc = get_cavity_volume_atom(mol, radii)
z = get_z(vc, volume_0, z0)
eps = get_eps(vc, volume_0, 2.0165)

#
# 1. Pauli repulsion only
#
mf = scf.RHF(mol).XP(z=z, radii_table=radii)
mf.kernel()

mycc = cc.CCSD(mf).run()
myci = ci.CISD(mf).run()
# CASCI with all orbitals active is FCI; its density is variational.
# Use this rather than fci.FCI(mf): the FCI factory takes h_rep from
# mf.get_hcore() but drops any PCM, and its solver carries no reference, so
# it cannot be passed to get_e_rep or calculate_analytical_pressure.
myfci = mcscf.CASCI(mf, mol.nao, mol.nelec).run()

print('Pauli repulsion only, f = 0.90, V_c = %.3f Bohr^3' % vc)
print('  method     E_tot (Ha)        E_rep (Ha)    p (GPa)')
for name, obj in (('RHF', mf), ('CISD', myci), ('CCSD', mycc), ('FCI', myfci)):
    print('  %-8s %16.8f   %10.6f   %8.3f'
          % (name, obj.e_tot, get_e_rep(obj), calculate_analytical_pressure(obj, vc)))

#
# 2. Pauli repulsion + PCM polarisation.  Post-HF methods take the solvent
#    from the reference with the usual PySCF .PCM() decorator; the PCM terms
#    of the pressure are evaluated with the post-HF density.
#
mf_pcm = scf.RHF(mol).XP(z=z, radii_table=radii).PCM()
mf_pcm.with_solvent.method = 'IEF-PCM'
mf_pcm.with_solvent.eps = eps
mf_pcm.with_solvent.radii_table = radii
mf_pcm.kernel()

mycc_pcm = cc.CCSD(mf_pcm).PCM().run()
mycas_pcm = mcscf.CASSCF(mf_pcm, 8, 2).PCM().run()

print('\nPauli repulsion + IEF-PCM (eps = %.4f)' % eps)
print('  method     E_tot (Ha)        p (GPa)   [repulsion, surface, polarisation, dielectric]')
for name, obj in (('RHF', mf_pcm), ('CCSD', mycc_pcm), ('CASSCF', mycas_pcm)):
    t = calculate_analytical_pressure(obj, vc, return_terms=True)
    print('  %-8s %16.8f   %8.3f   [%7.3f, %7.3f, %7.3f, %7.3f]'
          % (name, obj.e_tot, t['total'], t['repulsion'], t['surface'],
             t['polarisation'], t['dielectric']))
