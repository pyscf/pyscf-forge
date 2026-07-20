from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import tree_util

from ..core.ops import TrialOps
from ..core.system import System


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GcisdTrial:
    """
    Generalized CISD trial in an MO basis where the reference
    determinant occupies the first nocc orbitals.

    Arrays:
      mo_coeff: (2*norb, 2*norb)      trial coefficients
      c1: (nocc, nvir)                singles coefficients c_{i a}
      c2: (nocc, nvir, nocc, nvir)    doubles coefficients c_{i a j b}
    """

    mo_coeff: jax.Array
    c1: jax.Array
    c2: jax.Array

    @property
    def norb(self) -> int:
        return int(self.mo_coeff.shape[0] // 2)

    @property
    def nocc(self) -> int:
        return int(self.c1.shape[0])

    @property
    def nvir(self) -> int:
        return int(self.c1.shape[1])

    def tree_flatten(self):
        return (
            self.mo_coeff,
            self.c1,
            self.c2,
        ), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        (
            mo_coeff,
            c1,
            c2,
        ) = children
        return cls(
            mo_coeff=mo_coeff,
            c1=c1,
            c2=c2,
        )


def get_rdm1(trial_data: GcisdTrial) -> jax.Array:
    c = trial_data.mo_coeff
    nocc = trial_data.nocc
    dm = c[:, :nocc] @ c[:, :nocc].conj().T  # (2*norb, 2*norb)
    return dm


def overlap_g(walker: jax.Array, trial_data: GcisdTrial) -> jax.Array:
    nocc = trial_data.nocc
    c1 = trial_data.c1
    c2 = trial_data.c2
    g = (walker @ jnp.linalg.inv(walker[:nocc, :])).T
    o0 = jnp.linalg.det(walker[:nocc, :])
    o1 = jnp.einsum("ia,ia", c1.conj(), g[:, nocc:])
    o2 = 2.0 * jnp.einsum("iajb, ia, jb", c2.conj(), g[:, nocc:], g[:, nocc:])
    o = (1.0 + o1 + 0.25 * o2) * o0
    return o


def make_gcisd_trial_ops(sys: System) -> TrialOps:
    wk = sys.walker_kind.lower()

    print(wk)
    if wk == "restricted" or wk == "unrestricted":
        raise NotImplementedError("GCISD trial is only implemented for generalized walkers.")
    elif wk == "generalized":
        overlap_fn = overlap_g
        get_rdm1_fn = get_rdm1
    else:
        raise ValueError(f"unknown walker_kind: {sys.walker_kind}")

    return TrialOps(
        overlap=overlap_fn,
        get_rdm1=get_rdm1_fn,
    )


def make_gcisd_trial_data(data: dict, sys: System) -> GcisdTrial:
    ci1 = jnp.asarray(data["ci1"])
    ci2 = jnp.asarray(data["ci2"])
    mo = jnp.asarray(data["mo_coeff"])
    return GcisdTrial(
        mo_coeff=mo,
        c1=ci1,
        c2=ci2,
    )
