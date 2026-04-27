from __future__ import annotations

import jax
import jax.numpy as jnp


def taylor_expm_matrix(a: jax.Array, mat: jax.Array, n_terms: int) -> jax.Array:
    """
    Build exp(a * mat) using truncated Taylor series
    """
    n = mat.shape[0]
    out = jnp.eye(n, dtype=mat.dtype)
    term = jnp.eye(n, dtype=mat.dtype)
    for k in range(1, n_terms + 1):
        term = (a / k) * (mat @ term)
        out = out + term
    return out


def taylor_expm_action(a: jax.Array, mat: jax.Array, vecs: jax.Array, n_terms: int) -> jax.Array:
    """
    Apply exp(a * mat) to vecs without forming expm explicitly:
    """
    out = vecs
    term = vecs
    for k in range(1, n_terms + 1):
        term = (a / k) * (mat @ term)
        out = out + term
    return out


def block_diag(a: jax.Array, b: jax.Array) -> jax.Array:
    z1 = jnp.zeros((a.shape[0], b.shape[1]), dtype=jnp.result_type(a, b))
    z2 = jnp.zeros((b.shape[0], a.shape[1]), dtype=jnp.result_type(a, b))
    return jnp.block([[a, z1], [z2, b]])
