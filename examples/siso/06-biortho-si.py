#!/usr/bin/env python
"""Overlap between CASCI wave functions at two molecular geometries.

The two CI vectors are expressed in different molecular-orbital bases, so
their direct dot product is not the wave-function overlap.  Biorthogonalizing
the orbitals and counter-transforming the CI vectors accounts for the
cross-geometry orbital overlap.
"""

import numpy as np

from pyscf import gto, mcscf, scf
from pyscf.siso.biortho import biorthogonalize


def run_casci(bond_length):
    """Run a full-valence CASCI calculation for H2."""
    mol = gto.M(
        atom=f"H 0 0 0; H 0 0 {bond_length}",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    mf = scf.RHF(mol).run()
    mc = mcscf.CASCI(mf, 2, 2).run()
    return mol, mc


mol_left, mc_left = run_casci(0.70)
mol_right, mc_right = run_casci(0.80)

# AO overlap between basis functions centered at the two geometries.
ao_overlap = gto.intor_cross("int1e_ovlp", mol_left, mol_right)

_, _, _, _, ci_left_bi, ci_right_bi = biorthogonalize(
    mc_left.mo_coeff,
    mc_right.mo_coeff,
    mc_left.ci,
    mc_right.ci,
    ao_overlap,
    mc_left.ncore,
    mc_left.ncas,
    mc_left.nelecas,
    mc_right.nelecas,
)

ci_overlap = np.vdot(ci_left_bi, ci_right_bi)
print(f"CI wave-function overlap: {ci_overlap:.12f}")
