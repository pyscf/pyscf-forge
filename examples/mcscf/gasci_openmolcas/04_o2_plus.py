#!/usr/bin/env python
#
# Author: Yi Deng <yideng@uchicago.edu>
#

"""Fixed-orbital GASCI validation against OpenMolcas for doublet O2+."""

from pathlib import Path

import numpy

from pyscf import gto
from pyscf import scf
from pyscf.mcscf import gasci


# OpenMolcas reference input:
#
# &GATEWAY
# Title = O2+ GASCI 2S+1=2
# Coord = o2.xyz
# Basis = def2-TZVP
# Group = C1
# noCD
#
# &SEWARD
#
# &RASSCF
# Title = O2+ GASCI(11e,8o), GAS=2+4+2, doublet
#
# CIONLY
# FILEORB = o2+_guessorb.INPORB
#
# Charge   = 1
# Spin     = 2
# Symmetry = 1
#
# Nactel   = 11 0 0
# Inactive = 2
#
# GASSCF
# 3
# 2
# 2 4
# 4
# 7 10
# 2
# 11 11
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
# h5_file = "o2+_m2.rasscf.h5"
# mol = get_mol_from_h5(
#     h5_file,
#     charge=1,
#     spin=1,
# )
# mo_coeff = get_mo_from_h5(mol, h5_file)
# numpy.savez_compressed(
#     "data/o2_plus_mo.npz",
#     mol=mol.dumps(),
#     mo_coeff=mo_coeff[:, :10],
# )
#
# OpenMolcas output:
#
# :: RASSCF root number  1 Total energy: -149.38201777
# :: RASSCF root number  2 Total energy: -149.35718636
# :: RASSCF root number  3 Total energy: -149.15593567
# :: RASSCF root number  4 Total energy: -149.13375064
# :: RASSCF root number  5 Total energy: -149.09335181
# :: RASSCF root number  6 Total energy: -149.08969726
# :: RASSCF root number  7 Total energy: -149.06506038
# :: RASSCF root number  8 Total energy: -149.04684794
# :: RASSCF root number  9 Total energy: -149.04589118
# :: RASSCF root number 10 Total energy: -149.03042212
# :: RASSCF root number 11 Total energy: -149.01721858
# :: RASSCF root number 12 Total energy: -148.99965663
# :: RASSCF root number 13 Total energy: -148.92394942
# :: RASSCF root number 14 Total energy: -148.88444954
# :: RASSCF root number 15 Total energy: -148.87703597
# :: RASSCF root number 16 Total energy: -148.86710034
# :: RASSCF root number 17 Total energy: -148.83846586
# :: RASSCF root number 18 Total energy: -148.83524349
# :: RASSCF root number 19 Total energy: -148.82450484
# :: RASSCF root number 20 Total energy: -148.82369240


openmolcas_energies = numpy.asarray([
    -149.38201777,
    -149.35718636,
    -149.15593567,
    -149.13375064,
    -149.09335181,
    -149.08969726,
    -149.06506038,
    -149.04684794,
    -149.04589118,
    -149.03042212,
    -149.01721858,
    -148.99965663,
    -148.92394942,
    -148.88444954,
    -148.87703597,
    -148.86710034,
    -148.83846586,
    -148.83524349,
    -148.82450484,
    -148.82369240,
])

here = Path(__file__).resolve().parent
data_file = here / "data" / "o2_plus_mo.npz"

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
nelecas = (6, 5)

gas_orbs = (2, 4, 2)

# OpenMolcas cumulative GAS constraints.
gas_restr = (
    (2, 4),
    (7, 10),
    (11, 11),
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
mc.fcisolver.spin = 1
mc.fcisolver.nroots = nroots
mc.fcisolver.max_cycle = 300
mc.fcisolver.conv_tol = 1e-10

mc.kernel()

# The PySCF M_S = 1/2 space contains states with different total spin.
# Select the doublet roots using <S^2> = 0.75.
pyscf_doublet_energies = []

print()
print(f"PySCF doublet roots among the lowest {nroots} states")
print("state          E(PySCF) / Eh          <S^2>")

for state, energy in enumerate(numpy.asarray(mc.e_tot).reshape(-1)):
    ss, _ = mc.spin_square(state=state)
    if abs(ss - 0.75) < 1e-6:
        pyscf_doublet_energies.append(energy)
        print(f"{state + 1:5d}  {energy:22.12f}  {ss:12.8f}")

ncompare = min(
    len(pyscf_doublet_energies),
    len(openmolcas_energies),
)
if ncompare == 0:
    raise RuntimeError(
        f"No doublet root was found among the lowest {nroots} states"
    )

pyscf_doublet_energies = numpy.asarray(
    pyscf_doublet_energies[:ncompare]
)
reference_energies = openmolcas_energies[:ncompare]
differences = pyscf_doublet_energies - reference_energies

print()
print("OpenMolcas/PySCF GASCI comparison")
print("root       E(OpenMolcas) / Eh       E(PySCF) / Eh         diff / Eh")

for root, (e_ref, e_pyscf, diff) in enumerate(
    zip(reference_energies, pyscf_doublet_energies, differences),
    start=1,
):
    print(f"{root:4d}  {e_ref:22.12f}  {e_pyscf:22.12f}  {diff: .3e}")

max_diff = numpy.max(numpy.abs(differences))
print()
print(f"Compared roots   : {ncompare}")
print(f"max |difference| : {max_diff:.3e} Eh")

numpy.testing.assert_allclose(
    pyscf_doublet_energies,
    reference_energies,
    atol=1e-8,
    rtol=0,
)

print("Validation passed")
