#!/usr/bin/env python

import torch
from pyscf import gto
from pyscf.msdft import ldma


mol = gto.M(
    atom="H 0 0 -0.35; H 0 0 0.35",
    basis="sto-3g",
    spin=0,
    verbose=0,
)

matrix_density = ldma.MultistateMatrixDensityCAS.from_guess(
    mol,
    norb=2,
    nelec=2,
    spin_symmetry=True,
    spin_type=ldma.SpinType.UNPOLARIZED,
    guess="hcore",
)
hamiltonian = ldma.HamiltonianSemilocal(
    mol,
    spin_type=ldma.SpinType.UNPOLARIZED,
    grid_level=0,
)

energies = torch.linalg.eigvalsh(hamiltonian(matrix_density))
print("LDMA H2 energies (Hartree):", energies.tolist())
