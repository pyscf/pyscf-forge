#!/usr/bin/env python
"""Scalar state interaction between CAS states in different orbital bases.

Two singlet CASCI roots of LiH are first obtained in a common active-orbital
basis.  Each root is then represented in a different rotated active basis, with
its CI vector counter-rotated so that the physical wavefunction is unchanged.
The biorthogonal SI calculation recovers the original CASCI energies and also
provides pair-specific overlaps and transition density matrices.
"""

import numpy as np

from pyscf import fci, gto, mcscf, scf
from pyscf.siso.siso_biortho import SI


mol = gto.M(
    atom="Li 0 0 0; H 0 0 1.6",
    basis="sto-3g",
    symmetry=False,
    verbose=4,
)
mf = scf.RHF(mol).run()

# Compute two singlet roots in a common (2e,2o) active space.
mc = mcscf.CASCI(mf, 2, 2)
mc.fcisolver = fci.solver(mol, singlet=True)
mc.fcisolver.nroots = 2
mc.kernel()

# Give the two states distinct but physically equivalent active-orbital
# representations.  For genuinely state-specific calculations, pass their
# optimized MO coefficients and CI vectors directly instead of doing this step.
active = slice(mc.ncore, mc.ncore + mc.ncas)
mo_states = []
ci_states = []
for ci, angle in zip(mc.ci, (0.20, -0.30)):
    rotation = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    mo = mc.mo_coeff.copy()
    mo[:, active] = mo[:, active] @ rotation
    ci = fci.addons.transform_ci_for_orbital_rotation(
        ci, mc.ncas, mc.nelecas, rotation)
    mo_states.append(mo)
    ci_states.append(ci)

si = SI(
    mc,
    modelspace=[(2, 1)],
    ci=ci_states,
    mo_coeff=mo_states,
    energies=mc.e_tot,
)
si_energies, si_vectors = si.kernel()

print("\nCASCI energies")
print(mc.e_tot)
print("Biorthogonal SI energies")
print(si_energies)
print("Model-state overlap")
print(si.overlap)

# The AO transition density includes both inactive and active contributions.
dm1_ao = si.transition_rdm1(0, 1, basis="ao")
print("Trace of the (0,1) AO transition density")
print(np.einsum("uv,uv->", mf.get_ovlp(), dm1_ao))
