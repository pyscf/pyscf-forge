#!/usr/bin/env python
#
# Author: Yi Deng <yideng@uchicago.edu>
#

"""Fixed-orbital single-root RASCI example for sextet Al2FeO4+."""

from pathlib import Path

import numpy

from pyscf import gto
from pyscf import scf
from pyscf.mcscf import gasci


# OpenMolcas reference input:
#
# &GATEWAY
# Title = Al2FeO4+ RASCI 2S+1=6
# Coord = al2feo4.xyz
# Basis = cc-pVTZ
# Group = C1
# noCD
#
# &SEWARD
#
# &RASSCF
# Title = Al2FeO4+ RASCI(29e,22o), O2p/Fe3d/Fe4d, sextet
#
# CIONLY
# FILEORB = al2feo4_guessorb.INPORB
#
# Charge   = 1
# Spin     = 6
# Symmetry = 1
#
# Nactel   = 29 2 2
# Inactive = 27
#
# ! RAS1: MO28-39, O 2p,  12 orbitals, maximum 2 holes
# ! RAS2: MO40-44, Fe 3d,  5 orbitals
# ! RAS3: MO45-49, Fe 4d,  5 orbitals, maximum 2 electrons
#
# RAS1
# 12
#
# RAS2
# 5
#
# RAS3
# 5
#
# CIRoot = 1 1 1
# CIMX = 200
# ITERations = 300 300
#
# Notes:
# - OpenMolcas Spin = 6 means multiplicity 2S+1 = 6.
# - PySCF spin = 2S = 5 for the same sextet calculation.
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
# h5_file = "al2feo4_rasci.rasscf.h5"
# mol = get_mol_from_h5(
#     h5_file,
#     charge=1,
#     spin=5,
# )
# mo_coeff = get_mo_from_h5(mol, h5_file)
# numpy.savez_compressed(
#     "data/al2feo4_plus_rasci_mo.npz",
#     mol=mol.dumps(),
#     mo_coeff=mo_coeff[:, :49],
# )
#
# OpenMolcas output:
#
# :: RASSCF root number  1 Total energy: -2045.01359885


openmolcas_energy = -2045.01359885

here = Path(__file__).resolve().parent
data_file = here / "data" / "al2feo4_plus_rasci_mo.npz"

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

# Active space: RASCI(29e,22o)
ncore = 27
ncas = 22
nelecas = (17, 12)

gas_orbs = (12, 5, 5)

# OpenMolcas RAS constraints.
gas_restr = {
    "max_holes": 2,
    "max_particles": 2,
}
gas_restr_type = "ras"

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

mc.verbose = 4
mc.canonicalization = False
mc.fcisolver.spin = 5
mc.fcisolver.nroots = 1
mc.fcisolver.max_cycle = 300
mc.fcisolver.max_space = 30
mc.fcisolver.conv_tol = 1e-10
mc.fcisolver.conv_tol_residual = 1e-6

mc.kernel()

pyscf_energy = float(numpy.asarray(mc.e_tot).reshape(-1)[0])
ss, _ = mc.spin_square(state=0)
difference = pyscf_energy - openmolcas_energy

print()
print("OpenMolcas/PySCF RASCI comparison")
print("E(OpenMolcas) / Eh       E(PySCF) / Eh         diff / Eh          <S^2>")
print(
    f"{openmolcas_energy:20.12f}  "
    f"{pyscf_energy:20.12f}  "
    f"{difference: .3e}  "
    f"{ss:12.8f}"
)

# Example output (final comparison section only):
#
# Numerical values in the last few digits may depend on the platform.
#
# GASCI converged
# GASCI E = -2045.01359884909
# E(CI) = -363.115477794565
# S^2 = 8.7500000
#
# OpenMolcas/PySCF RASCI comparison
# E(OpenMolcas) / Eh       E(PySCF) / Eh         diff / Eh          <S^2>
#   -2045.013598850000    -2045.013598849092   9.081e-10    8.75000000
