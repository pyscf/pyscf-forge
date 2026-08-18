#!/usr/bin/env python
"""Biorthogonal SISO using separately optimized states of atomic nitrogen.

The 4S term and the combined 2D/2P manifold are optimized in independent
CASSCF calculations using a (5e,13o) active space.  Their state-specific
orbital sets and CI vectors are then combined in a single spin-orbit
state-interaction calculation using pairwise biorthonormalization.
"""

import numpy as np

from pyscf import gto, mcscf, scf
from pyscf.csf_fci import csf_solver
from pyscf.siso.siso_biortho import SISO


mol = gto.M(
    atom="N 0 0 0",
    basis="CC-PVDZ",
    spin=3,
    symmetry=False,
    verbose=4,
)
mf = scf.ROHF(mol).run()


def run_manifold(smult, nroots):
    """Optimize one spin manifold in a (5e,13o) active space."""
    solver = csf_solver(mol, smult=smult)
    solver.spin = smult - 1
    solver.nroots = nroots

    mc = mcscf.CASSCF(mf, 13, 5)
    weights = np.ones(nroots) / nroots
    mc = mcscf.state_average_mix_(mc, [solver], weights)
    mc.conv_tol = 1e-9
    mc.kernel()
    return mc


# Optimize the 4S ground state.
mc_4s = run_manifold(smult=4, nroots=1)

# Optimize all eight doublet roots together.  The first five roots are the
# spatial components of 2D, and the following three are the components of 2P.
mc_doublets = run_manifold(smult=2, nroots=8)

# Supply the 4S state followed by the combined 2D/2P doublet manifold.
# SISO biorthonormalizes every state pair before evaluating its matrix elements.
modelspace = [(1, 4), (8, 2)]
ci_states = list(mc_4s.ci) + list(mc_doublets.ci)
mo_states = [mc_4s.mo_coeff] + [mc_doublets.mo_coeff] * 8
energies = np.hstack((
    mc_4s.e_states,
    mc_doublets.e_states,
))

mysiso = SISO(
    mc_4s,
    modelspace=modelspace,
    ci=ci_states,
    mo_coeff=mo_states,
    energies=energies,
    ham="BP",
    amf=True,
)
mysiso.kernel()
