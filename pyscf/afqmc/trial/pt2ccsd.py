from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import tree_util

from ..core.system import System


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Pt2ccsdTrial:
    """
    Restricted pt2CCSD trial in an MO basis where the reference
    determinant occupies the first nocc orbitals.

    Arrays:
      mo_t: (nocc, norb)                   mo_coeff |psi'> = e^t1 |psi_0> by Thouless Theorem in mo basis
      t2: (nocc, nvir, nocc, nvir)         doubles amplitudess t2_{i a j b} in mo basis
    """

    mo_t: jax.Array  # (norb, nocc)
    t2: jax.Array  # (norb, nvir, norb, nvir)

    @property
    def nocc(self) -> int:
        return int(self.t2.shape[0])

    @property
    def nvir(self) -> int:
        return int(self.t2.shape[1])

    @property
    def norb(self) -> int:
        return int(self.nocc + self.nvir)

    def tree_flatten(self):
        children = (self.mo_t, self.t2)
        aux = None
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        mo_t, t2 = children
        return cls(
            mo_t=mo_t,
            t2=t2,
        )


def overlap_r(walker: jax.Array, trial_data: Pt2ccsdTrial) -> jax.Array:
    # <exp(T1)HF|walker>
    return jnp.linalg.det(trial_data.mo_t.T.conj() @ walker) ** 2


def make_pt2ccsd_trial_data(data: dict, sys: System) -> Pt2ccsdTrial:
    mo_t = jnp.asarray(data["mo_t"])[:, : sys.nup]
    t2 = jnp.asarray(data["t2"])
    return Pt2ccsdTrial(mo_t=mo_t, t2=t2)
