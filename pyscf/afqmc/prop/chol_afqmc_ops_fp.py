from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import tree_util

from ..ham.chol import HamChol
from .chol_afqmc_ops import _build_exp_h1_half_from_h1, _get_h1_eff, _mf_shifts


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FpCholAfqmcCtx:
    dt: jax.Array
    sqrt_dt: jax.Array
    exp_h1_half: jax.Array  # (n,n) or (ns,ns)
    mf_shifts: jax.Array  # (n_fields,)
    h0_prop: jax.Array  # scalar
    chol_flat: jax.Array  # (n_fields, n*n)
    norb: int

    def tree_flatten(self):
        return (
            self.dt,
            self.sqrt_dt,
            self.exp_h1_half,
            self.mf_shifts,
            self.h0_prop,
            self.chol_flat,
        ), (self.norb,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        dt, sqrt_dt, exp_h1_half, mf_shifts, h0_prop, chol_flat = children
        (norb,) = aux

        return cls(
            dt=dt,
            sqrt_dt=sqrt_dt,
            exp_h1_half=exp_h1_half,
            mf_shifts=mf_shifts,
            h0_prop=h0_prop,
            chol_flat=chol_flat,
            norb=norb,
        )


def _build_prop_ctx_fp(
    ham_data: HamChol,
    rdm1: jax.Array,
    dt: float,
    ene0: float = 0.0,
    chol_flat_precision: jnp.dtype = jnp.float64,
) -> FpCholAfqmcCtx:
    dt_a = jnp.array(dt)
    sqrt_dt = jnp.sqrt(dt_a)
    mf = _mf_shifts(ham_data, rdm1)
    h0_prop = -ham_data.h0 - 0.5 * jnp.sum(mf**2) + ene0

    h1_eff = _get_h1_eff(ham_data, mf)
    exp_h1_half = _build_exp_h1_half_from_h1(h1_eff, dt_a)
    chol_flat = ham_data.chol.reshape(ham_data.chol.shape[0], -1).astype(chol_flat_precision)
    norb = ham_data.chol.shape[1]
    return FpCholAfqmcCtx(
        dt=dt_a,
        sqrt_dt=sqrt_dt,
        exp_h1_half=exp_h1_half,
        mf_shifts=mf,
        h0_prop=h0_prop,
        chol_flat=chol_flat,
        norb=norb,
    )
