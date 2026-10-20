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
refs = [scf.RHF(mol).run(), uhf_a, uhf_b]   # any SCF objects or (mo_coeff, mo_occ)

nonorthogonal.NOCI(mol, refs).run()            # reference NOCI
nonorthogonal.NOCISD(mol, refs).run()          # singles and doubles of every reference
mynoci = nonorthogonal.NOCI(mol, refs, level=3).run()   # any excitation level
mynoci.e_tot, mynoci.ci, mynoci.dets
```

Individual matrix elements are available from `WickEngine`:

```python
wicks = nonorthogonal.WickEngine(mol, refs)
d0 = nonorthogonal.Determinant(0)
d1 = nonorthogonal.Determinant(1, alpha_holes=[2], alpha_particles=[3])
h, s = wicks.matrix_element(d0, d1)
H, S = wicks.build_matrices([d0, d1])
```

See `examples/nonorthogonal/` for complete examples.

## Modules

- `pyscf.nonorthogonal.noci`: `NOCI`, `NOCISD`, `NOCISDT` and `excitations`.
- `pyscf.nonorthogonal.wicks`: `WickEngine` and `Determinant`.
- `pyscf.nonorthogonal._wicks`: reference-pair intermediates and the ctypes
  binding to `libnonorthwick` (`pyscf/lib/nonorthogonal`).

## Conventions

- References may be real or complex, restricted or unrestricted, and must have
  the same number of MOs and the same numbers of alpha and beta electrons.
  Occupied orbitals may appear anywhere in the MO order.
- A `Determinant` is an excitation of one reference, with holes and particles
  given as MO indices of that reference.  It is the Slater determinant of its
  occupied MOs in ascending MO index order, alpha before beta, as in
  `pyscf.fci`.
- Complex matrix elements are Hermitian, with the bra conjugated.
- Hamiltonian elements include `E_nuc * S` unless `WickEngine.with_nuc` is
  False.

## Implementation notes

- Matrix elements between excitations of the same reference use the
  Slater-Condon rules; all other pairs use the extended nonorthogonal Wick
  theorem.
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
