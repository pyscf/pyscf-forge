# Local-Density Matrix Approximation

`pyscf.msdft.ldma` implements the local-density matrix approximation (LDMA)
for variational ground- and excited-state calculations in multistate density
functional theory. The local exchange-correlation energy is evaluated as an
analytic matrix function of the state and transition densities.

The implementation provides Dirac exchange and Chachiyo correlation for
unpolarized and collinearly polarized matrix densities. The polarized mode
spin-scales exchange while retaining the original paramagnetic Chachiyo
correlation approximation. Mixed-spin and noncollinear functionals are not
included.

## Installation

Install PySCF-Forge with the LDMA optional dependency:

```bash
pip install "pyscf-forge[msdft-ldma]"
```

## Example

```python
import torch
from pyscf import gto
from pyscf.msdft import ldma

mol = gto.M(
    atom="H 0 0 -0.35; H 0 0 0.35",
    basis="sto-3g",
    spin=0,
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
)

energies = torch.linalg.eigvalsh(hamiltonian(matrix_density))
print(energies)
```

The standalone example in `examples/msdft/02-ldma-h2.py` uses a low-cost grid
for a quick installation check.

## References

1. Y. Lu and J. Gao, *J. Phys. Chem. Lett.* **2022**, 13, 7762-7769.
   https://doi.org/10.1021/acs.jpclett.2c02088
2. A. Humeniuk, *J. Chem. Theory Comput.* **2024**, 20, 5497-5509.
   https://doi.org/10.1021/acs.jctc.4c00330

This component is distributed under the MIT License. See `LICENSE` in this
directory.
