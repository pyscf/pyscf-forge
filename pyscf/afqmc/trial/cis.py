from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import tree_util

from ..core.ops import TrialOps
from ..core.system import System


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CisTrial:
    """
    Restricted CIS trial in an MO basis where the reference
    determinant occupies the first nocc orbitals.

    Arrays:
      ci1: (nocc, nvir)                     singles coefficients c_{i a}
    """

    ci1: jax.Array

    @property
    def nocc(self) -> int:
        return int(self.ci1.shape[0])

    @property
    def nvir(self) -> int:
        return int(self.ci1.shape[1])

    @property
    def norb(self) -> int:
        return int(self.nocc + self.nvir)

    @property
    def nocc_full(self) -> int:
        return self.nocc

    @property
    def nvir_full(self) -> int:
        return self.nvir

    @property
    def occ_act_slice(self) -> slice:
        return slice(0, self.nocc)

    @property
    def vir_act_slice(self) -> slice:
        return slice(self.nocc, self.norb)

    def tree_flatten(self):
        children = (self.ci1,)
        aux = None
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (ci1,) = children
        return cls(ci1=ci1)


def get_rdm1(trial_data: CisTrial) -> jax.Array:
    # RHF
    norb, nocc = trial_data.norb, trial_data.nocc
    occ = jnp.arange(norb) < nocc
    dm = jnp.diag(occ)
    return jnp.stack([dm, dm], axis=0).astype(float)


def overlap_r(walker: jax.Array, trial_data: CisTrial) -> jax.Array:
    ci1 = trial_data.ci1
    nocc = trial_data.nocc

    g = (walker @ jnp.linalg.inv(walker[:nocc, :])).T
    o0 = jnp.linalg.det(walker[:nocc, :]) ** 2
    o1 = jnp.einsum("ia,ia", ci1, g[:, nocc:])
    return 2 * o1 * o0


def make_cis_trial_ops(sys: System) -> TrialOps:
    if sys.nup != sys.ndn:
        raise ValueError("Restricted CIS trial requires nup == ndn.")
    if sys.walker_kind.lower() != "restricted":
        raise ValueError(
            f"CIS trial currently supports only restricted walkers, got: {sys.walker_kind}"
        )
    return TrialOps(overlap=overlap_r, get_rdm1=get_rdm1)
