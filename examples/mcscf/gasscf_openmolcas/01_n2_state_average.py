#!/usr/bin/env python
#
# Author: Yi Deng <yideng@uchicago.edu>
#

"""Three-state-averaged triplet N2 GASSCF, compared with OpenMolcas."""

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
#
# LUMORB
# FILEORB = n2_guessorb.INPORB
#
# Spin = 3
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
# CIRoot = 3 3 1
# ITERations = 200 100
# CIMX = 200
#
# OpenMolcas output:
#
# :: RASSCF root number 1 Total energy: -108.82664198
# :: RASSCF root number 2 Total energy: -108.80517976
# :: RASSCF root number 3 Total energy: -108.80517974
#

openmolcas_energies = numpy.array([
    -108.82664198, -108.80517976, -108.80517974,
])
weights = (1/3, 1/3, 1/3)

here = Path(__file__).resolve().parent
with numpy.load(here / "data/n2_sa_mo.npz", allow_pickle=False) as data:
    mol = gto.loads(data["mol"].item())
    mo_coeff = data["mo_coeff"]

mol.max_memory = 8000  # MB
mol.verbose = 4

# One-cycle ROHF for mf generation.
mf = scf.ROHF(mol)
mf.max_cycle = 1
mf.kernel()
mf.mo_coeff = mo_coeff

# GASSCF(10e,8o), with three GAS spaces containing 2, 4, and 2 orbitals.
# (6,4) fixes M_S=1; determinant roots are not constrained to total S=1.
# The reported <S^2> verifies the triplet character of these reference roots.
mc = gasscf.GASSCF(
    mf,
    ncas=8,
    nelecas=(6,4),
    ncore=2,
    gas_orbs=(2,4,2),
    gas_restr=((2,4),(7,9),(10,10)),
    gas_restr_type="cumulative-occ",
).state_average_(weights)

mc.verbose = 4
mc.max_cycle_macro = 80
mc.conv_tol = 1e-9
mc.conv_tol_grad = 1e-5
mc.fcisolver.max_cycle = 300
mc.fcisolver.max_space = 40
mc.fcisolver.conv_tol = 1e-11
mc.fcisolver.conv_tol_residual = 1e-7
mc.kernel(mo_coeff)

energies = numpy.asarray(mc.e_states)
ss, _ = mc.fcisolver.states_spin_square(mc.ci, mc.ncas, mc.nelecas)
reference_average = numpy.dot(weights, openmolcas_energies)

print()
print("OpenMolcas/PySCF triplet N2 SA3 GASSCF comparison")
print("Root  Weight    E(OpenMolcas) / Eh      E(PySCF) / Eh       diff / Eh       <S^2>")
for i, (weight, ref, energy, spin2) in enumerate(
        zip(weights, openmolcas_energies, energies, ss), start=1):
    print(f"{i:4d}  {weight:.6f}  {ref:20.12f}  {energy:20.12f}  "
          f"{energy - ref:+.3e}  {spin2:10.7f}")
print(f"SA energy (OpenMolcas) / Eh: {reference_average:.12f}")
print(f"SA energy (PySCF) / Eh    : {mc.e_tot:.12f}")
print(f"SA difference / Eh        : {mc.e_tot - reference_average:+.3e}")
print(f"GASSCF converged          : {mc.converged}")

# Example output (last digits may depend on platform):
#
# OpenMolcas/PySCF triplet N2 SA3 GASSCF comparison
# Root  Weight    E(OpenMolcas) / Eh      E(PySCF) / Eh       diff / Eh       <S^2>
#    1  0.333333     -108.826641980000     -108.826642023974  -4.397e-08   2.0000000
#    2  0.333333     -108.805179760000     -108.805179743497  +1.650e-08   2.0000000
#    3  0.333333     -108.805179740000     -108.805179731525  +8.475e-09   2.0000000
# SA energy (OpenMolcas) / Eh: -108.812333826667
# SA energy (PySCF) / Eh    : -108.812333832999
# SA difference / Eh        : -6.332e-09
# GASSCF converged          : True

