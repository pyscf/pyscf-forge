from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import tree_util

from ..core.ops import TrialOps
from ..core.system import System


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class UhfTrial:
    mo_coeff_a: jax.Array  # (norb, nocc[0])
    mo_coeff_b: jax.Array  # (norb, nocc[1])

    @property
    def norb(self) -> int:
        return int(self.mo_coeff_a.shape[0])

    @property
    def nocc(self) -> tuple[int, int]:
        return (int(self.mo_coeff_a.shape[1]), int(self.mo_coeff_b.shape[1]))

    def tree_flatten(self):
        return (self.mo_coeff_a, self.mo_coeff_b), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        mo_coeff_a, mo_coeff_b = children
        return cls(mo_coeff_a=mo_coeff_a, mo_coeff_b=mo_coeff_b)


def _det(m: jax.Array) -> jax.Array:
    return jnp.linalg.det(m)


def get_rdm1(trial_data: UhfTrial) -> jax.Array:
    c_a = trial_data.mo_coeff_a
    c_b = trial_data.mo_coeff_b
    dm_a = c_a @ c_a.conj().T  # (norb, norb)
    dm_b = c_b @ c_b.conj().T  # (norb, norb)
    return jnp.stack([dm_a, dm_b], axis=0)  # (2, norb, norb)


def overlap_r(walker: jax.Array, trial_data: UhfTrial) -> jax.Array:
    n_elec_0 = trial_data.nocc[0]
    n_elec_1 = trial_data.nocc[1]
    return overlap_u((walker[:, :n_elec_0], walker[:, :n_elec_1]), trial_data)


def overlap_u(walker: tuple[jax.Array, jax.Array], trial_data: UhfTrial) -> jax.Array:
    wu, wd = walker
    cu = trial_data.mo_coeff_a.conj().T @ wu  # (nocc[0], nocc[0])
    cd = trial_data.mo_coeff_b.conj().T @ wd  # (nocc[1], nocc[1])
    return _det(cu) * _det(cd)


def overlap_g(walker: jax.Array, trial_data: UhfTrial) -> jax.Array:
    norb = trial_data.norb
    caH = trial_data.mo_coeff_a.conj().T  # (nocc[0], norb)
    cbH = trial_data.mo_coeff_b.conj().T  # (nocc[1], norb)
    top = caH @ walker[:norb, :]  # (nocc[0], sum(nocc))
    bot = cbH @ walker[norb:, :]  # (nocc[1], sum(nocc))
    m = jnp.vstack([top, bot])  # (sum(nocc), sum(nocc))
    return _det(m)


def make_uhf_trial_ops(sys: System) -> TrialOps:
    wk = sys.walker_kind.lower()

    if wk == "restricted":
        overlap_fn = overlap_r
        get_rdm1_fn = get_rdm1
    elif wk == "unrestricted":
        overlap_fn = overlap_u
        get_rdm1_fn = get_rdm1
    elif wk == "generalized":
        overlap_fn = overlap_g
        get_rdm1_fn = get_rdm1
    else:
        raise ValueError(f"unknown walker_kind: {sys.walker_kind}")

    return TrialOps(
        overlap=overlap_fn,
        get_rdm1=get_rdm1_fn,
    )


def make_uhf_trial_data(data: dict, sys: System) -> UhfTrial:
    if "mo_a" in data and "mo_b" in data:
        mo_a = jnp.asarray(data["mo_a"])
        mo_b = jnp.asarray(data["mo_b"])
    elif "mo" in data:
        mo_a = jnp.asarray(data["mo"])
        mo_b = jnp.asarray(data["mo"])
    else:
        raise KeyError("Failed to find the trial coeff.")

    mo_a = mo_a[:, : sys.nup]
    mo_b = mo_b[:, : sys.ndn]

    return UhfTrial(mo_a, mo_b)
