#!/usr/bin/env python
#
# Author: Yi Deng <yideng@uchicago.edu>
#

"""Density-fitted GASSCF for the septet Kremer's dimer, with a CD reference."""

from pathlib import Path

import numpy

from pyscf import gto, scf
from pyscf.mcscf import gasscf


# OpenMolcas reference input:
#
# &GATEWAY
# ... (Geometry and basis are stored in the .npz file)
#
# &SEWARD
# CHOLESKY
# THRCholesky = 1.0d-6
#
# &RASSCF
#
# LUMORB
# FILEORB = kd_guessorb.INPORB
#
# Charge = 3
# Spin = 7
# Symmetry = 1
#
# Nactel = 30 2 0
# Inactive = 117
#
# Ras1 = 12
# Ras2 = 10
#
# CIRoot = 1 1 1
# CIMX = 200
# ITERations = 100 100
#
# OpenMolcas output:
#
# :: RASSCF root number 1 Total energy: -3345.73608985
#

openmolcas_energy = -3345.73608985
auxbasis = "def2-universal-jkfit"
target_s2 = 12.0

here = Path(__file__).resolve().parent
with numpy.load(here / "data/kremer_dimer_df_mo.npz", allow_pickle=False) as data:
    mol = gto.loads(data["mol"].item())
    mo_coeff = data["mo_coeff"]

mol.max_memory = 64000  # MB
mol.verbose = 4

# One-cycle DF-ROHF for mf generation.
mf = scf.ROHF(mol).density_fit(auxbasis=auxbasis)
mf.max_cycle = 1
mf.kernel()
mf.mo_coeff = mo_coeff

# RASSCF(30e,22o)
# The DF-SCF reference automatically selects DF-GASSCF. The explicit
# density_fit call below retains the matching SCF auxiliary-basis helper.
mc = gasscf.GASSCF(
    mf,
    ncas=22,
    nelecas=(18,12),
    ncore=117,
    gas_orbs=(12,10,0),
    gas_restr={"max_holes": 2, "max_particles": 0},
    gas_restr_type="ras",
).density_fit(auxbasis=auxbasis)

mc.verbose = 4
mc.max_memory = mol.max_memory
mc.max_cycle_macro = 80
mc.conv_tol = 1e-8
mc.conv_tol_grad = 1e-5
mc.fcisolver.max_cycle = 300
mc.fcisolver.max_space = 30
mc.fcisolver.conv_tol = 1e-10
mc.fcisolver.conv_tol_residual = 1e-6

# Bias toward S=3 through the GAS spin penalty; check <S^2> after convergence.
# e_tot is the physical energy; spin_energy_report also exposes the objective.
mc.fix_spin_(shift=0.2, ss=target_s2)
mc.kernel(mo_coeff)

report = mc.spin_energy_report()
pyscf_energy = mc.e_tot
ss, _ = mc.spin_square()

print()
print("OpenMolcas CD / PySCF DF GASSCF comparison")
print(f"Auxiliary basis         : {auxbasis}")
print(f"GASSCF converged        : {mc.converged}")
print(f"GAS determinant count   : {numpy.asarray(mc.ci).size}")
print(f"E(OpenMolcas CD) / Eh   : {openmolcas_energy:.12f}")
print(f"E(PySCF DF, physical)   : {pyscf_energy:.12f}")
print(f"DF - CD / Eh           : {pyscf_energy - openmolcas_energy:+.6e}")
print(f"Spin penalty / Eh      : {report['penalty']:.3e}")
print(f"<S^2>                  : {ss:.10f}")

# DF and CD use different integral approximations.  Their energies need not
# agree to the orbital-optimization tolerance; report the measured difference.
#
# Example output (last digits may depend on platform):
#
# GASSCF converged        : True
# E(OpenMolcas CD) / Eh   : -3345.736089850000
# E(PySCF DF, physical)   : -3345.735665001371
# DF - CD / Eh           : +4.248486e-04
# <S^2>                  : 12.0000000000

