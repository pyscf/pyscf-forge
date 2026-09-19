#!/usr/bin/env python
#
# Author: Yi Deng <yideng@uchicago.edu>
#

"""Singlet N2 GASSCF energy scanner near the equilibrium bond length."""

from pathlib import Path

import numpy

from pyscf import gto, lib, scf
from pyscf.mcscf import gasscf


# OpenMolcas reference input:
#
# &GATEWAY
# ... (Geometry and basis are stored in the .npz file)
# noCD
#
# &SEWARD
#
# &RASSCF
#
# LUMORB
# FILEORB = n2_guessorb.INPORB
#
# Spin = 1
# Symmetry = 1
#
# Nactel = 10 0 0
# Inactive = 2
#
# GASSCF
# 3
# 2
# 2 4
# 4
# 7 9
# 2
# 10 10
#
# CIRoot = 1 1 1
# ITERations = 200 100
# CIMX = 200
#
# OpenMolcas output:
#
# :: RASSCF root number 1 Total energy: -109.13534823


openmolcas_energy = -109.13534823

here = Path(__file__).resolve().parent
with numpy.load(here / "data/n2_scanner_mo.npz", allow_pickle=False) as data:
    mol = gto.loads(data["mol"].item())
    mo_coeff = data["mo_coeff"]

mol.max_memory = 8000  # MB
mol.verbose = 4

# One-cycle RHF for mf generation.
mf = scf.RHF(mol)
mf.max_cycle = 1
mf.kernel()
# Allow the RHF reference to converge at each scanner geometry.
mf.max_cycle = 50
mf.mo_coeff = mo_coeff

mc = gasscf.GASSCF(
    mf,
    ncas=8,
    nelecas=(5,5),
    ncore=2,
    gas_orbs=(2,4,2),
    gas_restr=((2,4),(7,9),(10,10)),
    gas_restr_type="cumulative-occ",
)

mc.verbose = 4
mc.max_cycle_macro = 80
mc.conv_tol = 1e-9
mc.conv_tol_grad = 1e-5
mc.fcisolver.max_cycle = 200
mc.fcisolver.max_space = 30
mc.fcisolver.conv_tol = 1e-11
mc.fcisolver.conv_tol_residual = 1e-7

mc.kernel(mo_coeff)

ss, _ = mc.spin_square()
energy = float(mc.e_tot)

coords = mol.atom_coords(unit="Bohr")
bond = coords[1] - coords[0]
r0 = numpy.linalg.norm(bond) * lib.param.BOHR
direction = bond / numpy.linalg.norm(bond)
rows = [(r0, energy, ss, bool(mc.converged))]

print()
print("OpenMolcas/PySCF GASSCF comparison at the starting geometry")
print(f"R(N-N) / Angstrom      : {r0:.7f}")
print(f"E(OpenMolcas) / Eh     : {openmolcas_energy:.12f}")
print(f"E(PySCF, physical) / Eh: {energy:.12f}")
print(f"Difference / Eh       : {energy - openmolcas_energy:+.3e}")

# The scanner projects the previous orbitals and reuses the previous CI guess.
# Keep the atom order, basis, electron counts, GAS definition and spin fixed.
# SA weights, root count, spin penalty and frozen orbitals must also stay fixed.
# Root ordering follows each GASCI solve; state identity is not tracked.
scanner = mc.as_scanner()
for distance in (r0 - 0.001, r0 + 0.001):
    new_coords = coords.copy()
    new_coords[1] = coords[0] + (distance / lib.param.BOHR) * direction
    new_mol = mol.set_geom_(new_coords, unit="Bohr", inplace=False)
    scanner(new_mol)
    energy = float(scanner.e_tot)
    ss, _ = scanner.spin_square()
    rows.append((distance, energy, ss, bool(scanner.converged)))

print()
print("N2 singlet GASSCF energy scan")
print("R / Angstrom       E(physical) / Eh        <S^2>     converged")
for distance, energy, ss, converged in sorted(rows):
    print(f"{distance:12.7f}  {energy:22.12f}  {ss:12.8f}  {converged}")

# Example output (last digits may depend on platform):
#
# R / Angstrom       E(physical) / Eh        <S^2>     converged
#    1.0967000       -109.135311779472   -0.00000000  True
#    1.0977000       -109.135348228725   -0.00000000  True
#    1.0987000       -109.135379223650   -0.00000000  True

