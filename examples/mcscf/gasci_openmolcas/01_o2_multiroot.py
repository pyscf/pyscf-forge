#!/usr/bin/env python
#
# Author: Yi Deng <yideng@uchicago.edu>
#

"""Fixed-orbital multiroot GASCI example using OpenMolcas orbitals."""

from pathlib import Path

import numpy

from pyscf import gto
from pyscf import scf
from pyscf.mcscf import gasci


# OpenMolcas reference input:
#
# &GATEWAY
# Title = O2 GASCI 2S+1=3
# Coord = o2.xyz
# Basis = def2-TZVP
# Group = C1
# noCD
#
# &SEWARD
#
# &RASSCF
# Title = O2 GASCI(12e,8o), GAS=2+4+2, triplet
#
# CIONLY
# FILEORB = o2_guessorb.INPORB
#
# Spin     = 3
# Symmetry = 1
#
# Nactel   = 12 0 0
# Inactive = 2
#
# GASSCF
# 3
# 2
# 2 4
# 4
# 8 10
# 2
# 12 12
#
# CIRoot = 20 20 1
# CIMX = 200
#
# Notes:
# - OpenMolcas Spin and PySCF spin use multiplicity 2S+1 and 2S,
#   respectively.
# - CIONLY keeps the molecular orbitals fixed.
# - noCD avoids the Cholesky approximation to the ERIs.
#
# The exact PySCF molecule and fixed MO coefficients were imported from
# the converged OpenMolcas HDF5 file as follows:
#
# import numpy
# from mrh.my_pyscf.tools.molcas2pyscf import get_mo_from_h5
# from mrh.my_pyscf.tools.molcas2pyscf import get_mol_from_h5
#
# h5_file = "o2_m3.rasscf.h5"
# mol = get_mol_from_h5(
#     h5_file,
#     charge=0,
#     spin=2,
# )
# mo_coeff = get_mo_from_h5(mol, h5_file)
# numpy.savez_compressed(
#     "data/o2_mo.npz",
#     mol=mol.dumps(),
#     mo_coeff=mo_coeff[:, :10],
# )
#
# OpenMolcas output:
#
# :: RASSCF root number  1 Total energy: -149.76505363
# :: RASSCF root number  2 Total energy: -149.53703734
# :: RASSCF root number  3 Total energy: -149.53703734
# :: RASSCF root number  4 Total energy: -149.53215375
# :: RASSCF root number  5 Total energy: -149.38576564
# :: RASSCF root number  6 Total energy: -149.28581775
# :: RASSCF root number  7 Total energy: -149.28581722
# :: RASSCF root number  8 Total energy: -149.15394925
# :: RASSCF root number  9 Total energy: -149.11022925
# :: RASSCF root number 10 Total energy: -149.11022908
# :: RASSCF root number 11 Total energy: -149.07315914
# :: RASSCF root number 12 Total energy: -149.07315679
# :: RASSCF root number 13 Total energy: -149.02767903
# :: RASSCF root number 14 Total energy: -149.02767903
# :: RASSCF root number 15 Total energy: -149.01720780
# :: RASSCF root number 16 Total energy: -149.01720383
# :: RASSCF root number 17 Total energy: -149.00345234
# :: RASSCF root number 18 Total energy: -149.00345234
# :: RASSCF root number 19 Total energy: -148.99747740
# :: RASSCF root number 20 Total energy: -148.99747411


openmolcas_energies = numpy.asarray([
    -149.76505363,
    -149.53703734,
    -149.53703734,
    -149.53215375,
    -149.38576564,
    -149.28581775,
    -149.28581722,
    -149.15394925,
    -149.11022925,
    -149.11022908,
    -149.07315914,
    -149.07315679,
    -149.02767903,
    -149.02767903,
    -149.01720780,
    -149.01720383,
    -149.00345234,
    -149.00345234,
    -148.99747740,
    -148.99747411,
])

here = Path(__file__).resolve().parent
data_file = here / "data" / "o2_mo.npz"

# Load the exact PySCF molecule and fixed OpenMolcas MO coefficients.
with numpy.load(data_file, allow_pickle=False) as data:
    mol = gto.loads(data["mol"].item())
    mo_coeff = data["mo_coeff"]

mol.max_memory = 8000  # MB

# One-cycle ROHF for mf generation.
mf = scf.ROHF(mol)
mf.max_cycle = 1
mf.kernel()
mf.mo_coeff = mo_coeff

# Active space
ncore = 2
ncas = 8
nelecas = (7, 5)

gas_orbs = (2, 4, 2)

# OpenMolcas cumulative GAS constraints.
gas_restr = (
    (2, 4),
    (8, 10),
    (12, 12),
)
gas_restr_type = "cumulative-occ"

# GASCI
mc = gasci.GASCI(
    mf,
    ncas,
    nelecas,
    ncore=ncore,
    gas_orbs=gas_orbs,
    gas_restr=gas_restr,
    gas_restr_type=gas_restr_type,
)

nroots = 30

mc.verbose = 4
mc.canonicalization = False
mc.fcisolver.spin = 2
mc.fcisolver.nroots = nroots
mc.fcisolver.max_cycle = 300
mc.fcisolver.conv_tol = 1e-10

mc.kernel()

# The PySCF M_S = 1 space contains states with different total spin.
# Select the triplet roots using <S^2> = 2.
pyscf_triplet_energies = []

print()
print(f"PySCF triplet roots among the lowest {nroots} states")
print("state          E(PySCF) / Eh          <S^2>")

for state, energy in enumerate(numpy.asarray(mc.e_tot).reshape(-1)):
    ss, _ = mc.spin_square(state=state)
    if abs(ss - 2.0) < 1e-6:
        pyscf_triplet_energies.append(energy)
        print(f"{state + 1:5d}  {energy:22.12f}  {ss:12.8f}")

ncompare = min(
    len(pyscf_triplet_energies),
    len(openmolcas_energies),
)
if ncompare == 0:
    raise RuntimeError(
        f"No triplet root was found among the lowest {nroots} states"
    )

pyscf_triplet_energies = numpy.asarray(
    pyscf_triplet_energies[:ncompare]
)
reference_energies = openmolcas_energies[:ncompare]
differences = pyscf_triplet_energies - reference_energies

print()
print("OpenMolcas/PySCF GASCI comparison")
print("root       E(OpenMolcas) / Eh       E(PySCF) / Eh         diff / Eh")

for root, (e_ref, e_pyscf, diff) in enumerate(
    zip(reference_energies, pyscf_triplet_energies, differences),
    start=1,
):
    print(f"{root:4d}  {e_ref:22.12f}  {e_pyscf:22.12f}  {diff: .3e}")

max_diff = numpy.max(numpy.abs(differences))
print()
print(f"Compared roots   : {ncompare}")
print(f"max |difference| : {max_diff:.3e} Eh")

# Example output (final comparison section only):
#
# Numerical values in the last few digits may depend on the platform.
#
# GASCI converged
#
# OpenMolcas/PySCF GASCI comparison
# root       E(OpenMolcas) / Eh       E(PySCF) / Eh         diff / Eh
#    1       -149.765053630000       -149.765053626447   3.553e-09
#    2       -149.537037340000       -149.537037344145  -4.145e-09
#    3       -149.537037340000       -149.537037343830  -3.830e-09
#    4       -149.532153750000       -149.532153753145  -3.145e-09
#    5       -149.385765640000       -149.385765640052  -5.247e-11
#    6       -149.285817750000       -149.285817749263   7.374e-10
#    7       -149.285817220000       -149.285817220731  -7.307e-10
#    8       -149.153949250000       -149.153949254242  -4.242e-09
#    9       -149.110229250000       -149.110229248524   1.476e-09
#   10       -149.110229080000       -149.110229082511  -2.511e-09
#   11       -149.073159140000       -149.073159141160  -1.160e-09
#   12       -149.073156790000       -149.073156787594   2.406e-09
#   13       -149.027679030000       -149.027679032858  -2.858e-09
#   14       -149.027679030000       -149.027679032856  -2.856e-09
#   15       -149.017207800000       -149.017207802629  -2.629e-09
#   16       -149.017203830000       -149.017203833468  -3.468e-09
#   17       -149.003452340000       -149.003452343080  -3.080e-09
#   18       -149.003452340000       -149.003452342985  -2.985e-09
#   19       -148.997477400000       -148.997477403623  -3.623e-09
#   20       -148.997474110000       -148.997474109395   6.051e-10
#
# Compared roots   : 20
# max |difference| : 4.242e-09 Eh
