# Nonorthogonal Wick theorem and NOCI

`pyscf.nonorthogonal` evaluates Hamiltonian and overlap matrix elements
between arbitrary excitations of nonorthogonal reference determinants with the
extended nonorthogonal Wick theorem, and solves nonorthogonal configuration
interaction (NOCI) over them.

Once the intermediates of every ordered reference pair are built, the cost of a
matrix element depends only on the excitation ranks and the numbers of
zero-overlap orbital pairs, not on the basis size.

## Quick start

```python
from pyscf import gto, scf, nonorthogonal

mol = gto.M(atom=[('H', (0, 0, 1.8 * i)) for i in range(6)], basis='sto-3g')
# uhf_a and uhf_b are broken-symmetry UHF solutions, constructed in
# examples/nonorthogonal/02-noci.py
refs = [scf.RHF(mol).run(), uhf_a, uhf_b]   # any SCF objects or (mo_coeff, mo_occ)

nonorthogonal.NOCI(mol, refs).run()            # reference NOCI
nonorthogonal.NOCISD(mol, refs).run()          # singles and doubles of every reference
mynoci = nonorthogonal.NOCI(mol, refs, level=3).run()   # any excitation level
mynoci.e_tot, mynoci.ci, mynoci.dets

# any Hamiltonian in the AO basis
h1e, eri, ovlp, ecore = nonorthogonal.get_hamiltonian(mol, refs)
nonorthogonal.NOCISD(mol, refs).kernel(h1e=h1e, eri=eri, ovlp=ovlp, ecore=ecore)
```

Individual matrix elements are available from `WickEngine`:

```python
wicks = nonorthogonal.WickEngine(mol, refs).build()
d0 = nonorthogonal.Determinant(0)
d1 = nonorthogonal.Determinant(1, alpha_holes=[2], alpha_particles=[3])
h, s = wicks.matrix_element(d0, d1)
H, S = wicks.build_matrices([d0, d1])
```

See `examples/nonorthogonal/` for complete examples.

## Modules

- `pyscf.nonorthogonal.noci`: `NOCI`, `NOCISD`, `NOCISDT` and `excitations`.
- `pyscf.nonorthogonal.wicks`: `WickEngine`, `Determinant` and `get_hamiltonian`.
- `pyscf.nonorthogonal._wicks`: reference-pair intermediates and the ctypes
  binding to `libnonorthwick` (`pyscf/lib/nonorthogonal`).

## Conventions

- References may be real or complex RHF, ROHF or UHF determinants, and must have
  the same number of MOs (at most 128) and the same numbers of alpha and beta
  electrons.
  Occupied orbitals may appear anywhere in the MO order.
- A `Determinant` is an excitation of one reference, with holes and particles
  given as MO indices of that reference.  It is the Slater determinant of its
  occupied MOs in ascending MO index order, alpha before beta, as in
  `pyscf.fci`.
- Complex matrix elements are Hermitian, with the bra conjugated.
- The Hamiltonian is given in the AO basis as `h1e`, `eri`, `ovlp` and
  `ecore` to `NOCI.kernel` (or `WickEngine.build`), and Hamiltonian elements
  include `ecore * S`.  Arguments that are not given are taken from the first
  reference that is an SCF object, through its `get_hcore`, `get_ovlp`,
  `energy_nuc` and incore integrals, and are otherwise computed from `mol`
  (`get_hamiltonian`).  The two-electron integrals are the full 4-index AO
  integrals, as the intermediates need their 4-index transformations, so a
  density-fitted reference does not reproduce its `e_tot` exactly.

## Implementation notes

- Matrix elements between excitations of the same reference use the
  Slater-Condon rules on occupation bit strings, adapted from
  `FCIpspace_h0tril_uhf` of `pyscf/lib/mcscf/fci_contract.c` to complex
  orbitals and up to 128 MOs; all other pairs use the extended nonorthogonal
  Wick theorem.  Beyond 64 MOs, NOCISD of one RHF reference matches
  `ci.CISD` to 4e-16 for He/cc-pV5Z (Cartesian, 70 MOs); this takes about a
  minute, so it is not a unit test.
- Singular values of the occupied orbital overlap below `tol` (default 1e-10)
  are treated as zero-overlap orbital pairs.
- Small nonzero singular values (below 0.1) would put large `1/S_k` terms
  into the intermediates, which cancel in the matrix elements only after
  rounding.  Each element is affine in `1/S_k`, so it is evaluated exactly from
  two well-conditioned corners per such pair.  n near-zero pairs cost about
  2^n ordinary evaluations per spin.
- Pairs without zero-overlap orbitals and with up to quadruple excitations on
  each side use fixed-size kernels, batched with AVX2 or AVX-512 when the CPU
  supports them (selected at run time).  Pairs with zero-overlap orbitals use a
  general evaluator, a few times to an order of magnitude slower per element.
- `NOCI` builds dense H and S over the whole space and solves the generalised
  eigenvalue problem with `lib.linalg_helper.safe_eigh`, removing overlap
  eigenvectors below `lindep`.  It is intended for spaces of up to a few tens
  of thousands of determinants.
- The intermediates of each reference pair hold O(nmo^4) two-electron tensors,
  whose number grows with the numbers of zero-overlap and near-zero pairs.

## References

- H. G. A. Burton, J. Chem. Phys. 154, 144109 (2021)
- H. G. A. Burton, J. Chem. Phys. 157, 204109 (2022)
