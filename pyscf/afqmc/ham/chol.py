from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
from jax import tree_util

HamBasis = Literal["restricted", "generalized"]


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class HamChol:
    """
    cholesky hamiltonian.

    basis="restricted":
      h1:   (norb, norb)
      chol: (n_fields, norb, norb)

    basis="generalized":
      h1:   (nso, nso)   where nso = 2*norb
      chol: (n_fields, nso, nso)
    """

    h0: jax.Array
    h1: jax.Array
    chol: jax.Array
    basis: HamBasis = "restricted"
    nchol: int | None = None

    def __post_init__(self):
        if self.basis not in ("restricted", "generalized"):
            raise ValueError(f"unknown basis: {self.basis}")
        chol_shape = getattr(self.chol, "shape", None)
        if chol_shape is None:
            return

        n_chol_shape = int(chol_shape[0])
        nchol = self.nchol
        if nchol is None:
            object.__setattr__(self, "nchol", n_chol_shape)
        elif n_chol_shape not in (0, int(nchol)):
            raise ValueError(f"nchol={nchol} is inconsistent with chol.shape[0]={n_chol_shape}")

    def tree_flatten(self):
        children = (self.h0, self.h1, self.chol)
        nchol = self.nchol
        assert nchol is not None
        aux = (self.basis, int(nchol))
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        h0, h1, chol = children
        basis, nchol = aux
        return cls(h0=h0, h1=h1, chol=chol, basis=basis, nchol=nchol)


def n_fields(ham: HamChol) -> int:
    nchol = ham.nchol
    assert nchol is not None
    return int(nchol)


def slice_ham_level(ham: HamChol, *, norb_keep: int | None, nchol_keep: int | None) -> HamChol:
    """
    Build a HamChol view for measurement in MLMC:
      - slice orbitals as a prefix [:norb_keep]
      - slice chol as a prefix [:nchol_keep]
    """
    h0 = ham.h0
    h1 = ham.h1
    chol = ham.chol

    new_nchol = ham.nchol

    if norb_keep is not None:
        h1 = h1[:norb_keep, :norb_keep]
        chol = chol[:, :norb_keep, :norb_keep]

    if nchol_keep is not None:
        chol = chol[:nchol_keep]
        ham_nchol = ham.nchol
        assert ham_nchol is not None
        new_nchol = min(int(ham_nchol), nchol_keep)

    return HamChol(h0=h0, h1=h1, chol=chol, basis=ham.basis, nchol=new_nchol)
