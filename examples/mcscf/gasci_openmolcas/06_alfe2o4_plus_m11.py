#!/usr/bin/env python
#
# Author: Yi Deng <yideng@uchicago.edu>
#

"""Fixed-orbital GASCI validation against OpenMolcas for undecet AlFe2O4+."""

from pathlib import Path

import numpy

from pyscf import gto
from pyscf import scf
from pyscf.mcscf import gasci


# OpenMolcas reference input:
#
# &GATEWAY
# Title = AlFe2O4+ GASCI 2S+1=11
# Coord = alfe2o4.xyz
# Basis = cc-pVDZ
# Group = C1
# noCD
#
# &SEWARD
#
# &RASSCF
# Title = AlFe2O4+ GASCI(34e,34o), O2p/Fe3d/O3p, undecet
#
# CIONLY
# FILEORB = alfe2o4_guessorb.INPORB
#
# Charge   = 1
# Spin     = 11
# Symmetry = 1
#
# Nactel   = 34 0 0
# Inactive = 31
#
# ! GAS1: MO32-43, O 2p, 12 orbitals
# ! GAS2: MO44-53, Fe 3d, 10 orbitals
# ! GAS3: MO54-65, O 3p, 12 orbitals
# !
# ! Constraints:
# ! 23 <= N(O 2p)                    <= 24
# ! 33 <= N(O 2p + Fe 3d)           <= 34
# ! 34  = N(O 2p + Fe 3d + O 3p)    = 34
#
# GASSCF
# 3
# 12
# 23 24
# 10
# 33 34
# 12
# 34 34
#
# CIRoot = 1 1 1
# CIMX = 200
# ITERations = 300 300
#
# Notes:
# - OpenMolcas Spin = 11 means multiplicity 2S+1 = 11.
# - PySCF spin = 2S = 10 for the same undecet calculation.
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
# h5_file = "fe2_d2d_m11.rasscf.h5"
# mol = get_mol_from_h5(
#     h5_file,
#     charge=1,
#     spin=10,
# )
# mo_coeff = get_mo_from_h5(mol, h5_file)
# numpy.savez_compressed(
#     "data/alfe2o4_plus_mo.npz",
#     mol=mol.dumps(),
#     mo_coeff=mo_coeff[:, :65],
# )
#
# OpenMolcas output:
#
# :: RASSCF root number  1 Total energy: -3064.32294249


openmolcas_energy = -3064.32294249

here = Path(__file__).resolve().parent
data_file = here / "data" / "alfe2o4_plus_mo.npz"

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

# Active space: GASCI(34e,34o)
ncore = 31
ncas = 34
nelecas = (22, 12)

gas_orbs = (12, 10, 12)

# OpenMolcas cumulative GAS constraints.
gas_restr = (
    (23, 24),
    (33, 34),
    (34, 34),
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

mc.verbose = 4
mc.canonicalization = False
mc.fcisolver.spin = 10
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
print("OpenMolcas/PySCF GASCI comparison")
print("E(OpenMolcas) / Eh       E(PySCF) / Eh         diff / Eh          <S^2>")
print(
    f"{openmolcas_energy:20.12f}  "
    f"{pyscf_energy:20.12f}  "
    f"{difference: .3e}  "
    f"{ss:12.8f}"
)

numpy.testing.assert_allclose(
    pyscf_energy,
    openmolcas_energy,
    atol=1e-8,
    rtol=0,
)
numpy.testing.assert_allclose(
    ss,
    30.0,
    atol=1e-3,
    rtol=0,
)

print("Validation passed")
