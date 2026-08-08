"""Local-density matrix approximation for multistate DFT.

This package provides the analytic Dirac exchange and Chachiyo correlation
functionals with collinear matrix-density and Hamiltonian infrastructure.
"""

from .dft import (AOGridBatch, CASGuessComponents, ExactExchangeFunctionalAO, HamiltonianSemilocal,
                  HamiltonianTargetStateLDMA, LDA, PureXCFunctional,
                  MultistateMatrixDensityCAS,
                  MultistateMatrixDensityKohnSham, SpinType,
                  TargetStateMultistateMatrixDensityCAS, lda_c_chachiyo,
                  lda_x_dirac, lda_xc_dirac_chachiyo,
                  cas_guess_components, index_within_multiplicity,
                  lda_xc_dirac_chachiyo_unpolarized,
                  merge_multiplet_energies,
                  minimize_subspace_energy)

__all__ = [
    "AOGridBatch", "CASGuessComponents", "ExactExchangeFunctionalAO", "HamiltonianSemilocal",
    "HamiltonianTargetStateLDMA", "LDA", "PureXCFunctional",
    "MultistateMatrixDensityCAS", "MultistateMatrixDensityKohnSham", "SpinType",
    "TargetStateMultistateMatrixDensityCAS", "lda_c_chachiyo", "lda_x_dirac",
    "lda_xc_dirac_chachiyo", "lda_xc_dirac_chachiyo_unpolarized",
    "cas_guess_components", "index_within_multiplicity", "merge_multiplet_energies",
    "minimize_subspace_energy",
]
