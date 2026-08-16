#!/usr/bin/env python
"""State interaction with state-specific orbital representations.

This example starts from three state-averaged CAS states and deliberately
represents every state in a different active-orbital basis.  The corresponding
CI vectors are counter-rotated so that the physical wave functions are
unchanged.  ``SI`` and ``SISO`` then recover their pairwise biorthogonal
representations before constructing either the scalar state-interaction
Hamiltonian or the state-interaction spin-orbit Hamiltonian.

Replace ``ci_states`` and ``mo_states`` by CI vectors and orbitals from
independently optimized CASSCF calculations for a genuine state-specific use.
"""

import numpy as np

from pyscf import fci, gto, mcscf, scf
from pyscf.csf_fci import csf_solver
from pyscf.siso.siso_biortho import SI, SISO


mol = gto.M(
    atom="B 0 0 0",
    basis="sto-3g",
    spin=1,
    symmetry=False,
    verbose=4,
)
mf = scf.ROHF(mol).run()

# Three spin-adapted doublet roots in a (3e,4o) active space.
solver = csf_solver(mol, smult=2)
solver.nroots = 3
mc = mcscf.CASSCF(mf, 4, 3)
mc = mcscf.state_average_mix_(mc, [solver], [1.0 / 3.0] * 3)
mc.kernel()

# Give each state a distinct (but equivalent) active-orbital representation.
# For independently optimized states, supply their orbitals and CI vectors
# directly and omit this rotation/counter-rotation step.
rng = np.random.default_rng(9)
ci_states = []
mo_states = []
for ci, angle_scale in zip(mc.ci, (0.10, 0.20, 0.30)):
    generator = rng.normal(size=(mc.ncas, mc.ncas))
    generator = angle_scale * (generator - generator.T)
    active_rotation = np.linalg.qr(np.eye(mc.ncas) + generator)[0]

    mo = mc.mo_coeff.copy()
    active = slice(mc.ncore, mc.ncore + mc.ncas)
    mo[:, active] = mo[:, active] @ active_rotation
    # For an orthogonal active rotation, PySCF's determinant-minor transform
    # gives the CI representation in the rotated orbital basis.
    ci_rotated = fci.addons.transform_ci_for_orbital_rotation(
        ci, mc.ncas, mc.nelecas, active_rotation
    )
    mo_states.append(mo)
    ci_states.append(ci_rotated)

# Scalar nonorthogonal state interaction without SOC.
si = SI(
    mc,
    modelspace=[(3, 2)],
    ci=ci_states,
    mo_coeff=mo_states,
    energies=mc.e_states,
)
si_energies, si_vectors = si.kernel()

print("\nBiorthogonal state-interaction relative energies (au)")
print(si_energies - si_energies[0])
print("Smallest scalar model-space overlap eigenvalue")
print(np.linalg.eigvalsh(si.overlap)[0])

# State-interaction spin-orbit calculation.
siso = SISO(
    mc,
    modelspace=[(3, 2)],
    ci=ci_states,
    mo_coeff=mo_states,
    energies=mc.e_states,
    ham="BP",
    amf=True,
)
siso_energies, siso_vectors = siso.kernel()

print("\nBiorthogonal SI-SO relative energies (cm^-1)")
print((siso_energies - siso_energies[0]) * 219474.6313705)
print("Smallest SI-SO model-space overlap eigenvalue")
print(np.linalg.eigvalsh(siso.overlap)[0])

# Pair-specific transformed data are retained for analysis.
pair = siso.pair_data[0, 1]
print("Maximum biorthonormality error for state pair (0,1)")
nocc = mc.ncore + mc.ncas
cross = (
    pair.mo_left[:, :nocc].T
    @ mf.get_ovlp()
    @ pair.mo_right[:, :nocc]
)
print(np.max(np.abs(cross - np.eye(nocc))))
