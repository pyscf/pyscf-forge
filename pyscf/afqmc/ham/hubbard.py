from __future__ import annotations

from dataclasses import dataclass

import jax
from jax import tree_util


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class HamHubbard:
    """
    Hubbard Hamiltonian data.

    h1: one body term  ((norb, norb))
    u: on site interaction
    """

    h1: jax.Array
    u: float

    def tree_flatten(self):
        return (self.h1, self.u), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        h1, u = children
        return cls(h1=h1, u=u)
