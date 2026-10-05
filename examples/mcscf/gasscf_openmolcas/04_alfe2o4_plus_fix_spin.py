#!/usr/bin/env python
#
# Author: Yi Deng <yideng@uchicago.edu>
#

"""Single-root GASSCF(10e,20o) for nonet AlFe2O4+."""

from pathlib import Path

import numpy

from pyscf import gto, scf
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
# Title = AlFe2O4+ GASSCF(10e,20o), Fe 3d/4d, nonet
#
# LUMORB
# FILEORB = alfe2o4_gasscf_guessorb.INPORB
#
# Charge   = 1
# Spin     = 9
# Symmetry = 1
#
# Nactel   = 10 0 0
# Inactive = 43
#
# GASSCF
# 2
# 10
# 8 10
# 10
# 10 10
#
# CIRoot = 1 1 1
# CIMX = 200
# ITERations = 300 300
#
# Notes:
# The exact molecule and initial MOs were imported with mrh.
#
# OpenMolcas output:
#
# :: RASSCF root number 1 Total energy: -3066.32699432

openmolcas_energy = -3066.326994318111
target_s2 = 20.0  # S=4, multiplicity 9.

here = Path(__file__).resolve().parent
data_file = here / "data/alfe2o4_plus_mo.npz"

with numpy.load(data_file, allow_pickle=False) as data:
    mol = gto.loads(data["mol"].item())
    mo_coeff = data["mo_coeff"]

mol.max_memory = 16000  # MB
mol.verbose = 4

# One-cycle ROHF for mf generation.
mf = scf.ROHF(mol)
mf.max_cycle = 1
mf.kernel()
mf.mo_coeff = mo_coeff

# GAS1: 10 Fe 3d orbitals; GAS2: 10 Fe 4d orbitals.
# Allow up to two electrons to move from GAS1 to GAS2.
mc = gasscf.GASSCF(
    mf,
    ncas=20,
    nelecas=(9,1),
    ncore=43,
    gas_orbs=(10,10),
    gas_restr=((8,10),(10,10)),
    gas_restr_type="cumulative-occ",
)

mc.verbose = 4
mc.max_cycle_macro = 80
mc.conv_tol = 1e-9
mc.conv_tol_grad = 1e-5
mc.fcisolver.max_cycle = 300
mc.fcisolver.max_space = 40
mc.fcisolver.conv_tol = 1e-11
mc.fcisolver.conv_tol_residual = 1e-7

# Bias toward S=4 through the GAS spin penalty; check <S^2> after convergence.
# e_tot is the physical energy; spin_energy_report also exposes the objective.
mc.fix_spin_(shift=0.2, ss=target_s2)
mc.kernel(mo_coeff)

report = mc.spin_energy_report()
energy = mc.e_tot
ss, _ = mc.spin_square()

print()
print("OpenMolcas/PySCF nonet AlFe2O4+ GASSCF comparison")
print(f"E(OpenMolcas) / Eh     : {openmolcas_energy:.12f}")
print(f"E(PySCF, physical) / Eh: {energy:.12f}")
print(f"Difference / Eh       : {energy - openmolcas_energy:+.3e}")
print(f"Spin penalty / Eh     : {report['penalty']:.3e}")
print(f"<S^2>                 : {ss:.10f}")
print(f"GASSCF converged      : {mc.converged}")
print(f"GAS determinant count : {numpy.asarray(mc.ci).size}")

# Example output (last digits may depend on platform):
#
# E(OpenMolcas) / Eh     : -3066.326994318111
# E(PySCF, physical) / Eh: -3066.326994395571
# Difference / Eh       : -7.746e-08
# <S^2>                 : 20.0000000000
# GASSCF converged      : True

