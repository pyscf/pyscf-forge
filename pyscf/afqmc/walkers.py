from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import NamedSharding

from .core.system import System
from .core.typing import walkers


def _natorbs(dm: jax.Array, n_occ: int) -> jax.Array:
    dm = 0.5 * (dm + jnp.conj(dm.T))
    vecs = jnp.linalg.eigh(dm)[1][:, ::-1]
    return vecs[:, :n_occ]


def init_walkers(sys: System, rdm1: jax.Array, n_walkers: int) -> walkers:
    """
    Initialize walkers from natural orbitals of a trial rdm1.
    """
    wk = (sys.walker_kind).lower()
    norb = sys.norb
    nup, ndn = sys.nup, sys.ndn

    if wk == "generalized":
        ne = nup + ndn

        if rdm1.ndim == 2:
            if rdm1.shape[0] != 2 * norb or rdm1.shape[1] != 2 * norb:
                raise ValueError(
                    "For generalized walkers, a 2D rdm1 must have shape (2*norb, 2*norb)."
                )
            w0 = _natorbs(rdm1, ne) + 0.0j  # (2*norb, ne)
            return jnp.broadcast_to(w0, (n_walkers, *w0.shape))

        if rdm1.ndim != 3 or rdm1.shape[0] != 2:
            raise ValueError(
                "Expected rdm1 with shape (2, norb, norb) for generalized init from spin blocks."
            )

        natorbs_up = _natorbs(rdm1[0], nup)  # (norb, nup)
        natorbs_dn = _natorbs(rdm1[1], ndn)  # (norb, ndn)

        z_up = jnp.zeros((norb, ndn))
        z_dn = jnp.zeros((norb, nup))

        top = jnp.concatenate([natorbs_up, z_up], axis=1) + 0.0j  # (norb, ne)
        bot = jnp.concatenate([z_dn, natorbs_dn], axis=1) + 0.0j  # (norb, ne)
        w0 = jnp.concatenate([top, bot], axis=0)  # (2*norb, ne)

        return jnp.broadcast_to(w0, (n_walkers, *w0.shape))

    if rdm1.ndim == 2:
        raise ValueError(
            "For walker_kind in {'restricted','unrestricted'}, rdm1 must be spin-block (2, norb, norb)."
        )
    if rdm1.ndim != 3 or rdm1.shape[0] != 2:
        raise ValueError("Expected rdm1 with shape (2, norb, norb).")

    dm_up, dm_dn = rdm1[0], rdm1[1]

    if wk == "restricted":
        dm_tot = dm_up + dm_dn
        natorbs = _natorbs(dm_tot, nup)  # (norb, nup)
        w0 = natorbs + 0.0j
        return jnp.broadcast_to(w0, (n_walkers, *w0.shape))

    if wk == "unrestricted":
        natorbs_up = _natorbs(dm_up, nup) + 0.0j
        natorbs_dn = _natorbs(dm_dn, ndn) + 0.0j
        wu = jnp.broadcast_to(natorbs_up, (n_walkers, *natorbs_up.shape))
        wd = jnp.broadcast_to(natorbs_dn, (n_walkers, *natorbs_dn.shape))
        return (wu, wd)

    raise ValueError(f"unknown walker_kind: {wk}")


def is_unrestricted(w: walkers) -> bool:
    return isinstance(w, tuple) and len(w) == 2


def _batch_size0(x: Any) -> int:
    leaf = jax.tree_util.tree_leaves(x)[0]
    return int(leaf.shape[0])


def n_walkers(w: walkers) -> int:
    return _batch_size0(w)


def vmap_chunked(
    fn: Callable[..., Any],
    n_chunks: int,
    *,
    in_axes: int | tuple[int | None, ...] = 0,
):
    """
    Memory friendly vmap: map over axis-0 in micro-batches using lax.map.

    Usage like vmap:
        out = vmap_chunked(fn, n_chunks, in_axes=...)(*args, **kwargs)
    """

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        g = lambda *a: fn(*a, **kwargs)

        if n_chunks == 1:
            return jax.vmap(g, in_axes=in_axes)(*args)
        if not isinstance(in_axes, tuple):
            in_axes_ = (in_axes,) * len(args)
        else:
            in_axes_ = in_axes

        mapped_pos = [i for i, ax in enumerate(in_axes_) if ax == 0]
        if not mapped_pos:
            return g(*args)

        nw = _batch_size0(args[mapped_pos[0]])
        batch_size = (nw + n_chunks - 1) // n_chunks

        mapped_args = tuple(args[i] for i in mapped_pos)

        def f(xi):
            full = list(args)
            for j, i in enumerate(mapped_pos):
                full[i] = xi[j]
            return g(*full)

        return lax.map(f, mapped_args, batch_size=batch_size)

    return wrapped


def _qr(mat: jax.Array) -> tuple[jax.Array, jax.Array]:
    """
    QR with a phase convention that makes diag(R) real nonnegative.
    """
    q, r = jnp.linalg.qr(mat, mode="reduced")
    d = jnp.diag(r)
    abs_d = jnp.abs(d)
    phase = d / jnp.where(abs_d == 0, 1.0, abs_d)
    q = q * jnp.conj(phase)[None, :]
    r = phase[:, None] * r  # check this is correct for free projection
    det_r = jnp.prod(jnp.diag(r))
    return q, det_r


def orthogonalize(
    w: walkers,
    walker_kind: str,
) -> tuple[walkers, jax.Array]:
    """
    Keeps track of normalization constants.
    """
    wk = walker_kind.lower()

    if wk == "unrestricted":
        wu, wd = w
        q_u, det_u = jax.vmap(_qr, in_axes=0)(wu)
        q_d, det_d = jax.vmap(_qr, in_axes=0)(wd)
        norm = det_u * det_d
        return (q_u, q_d), norm
    elif wk in ("restricted", "generalized"):
        q, det_r = jax.vmap(_qr, in_axes=0)(w)
        norm = det_r * det_r if wk == "restricted" else det_r
        return q, norm

    raise ValueError(f"unknown walker_kind: {walker_kind}")


def orthonormalize(w: walkers, walker_kind: str) -> walkers:
    """
    Throws away normalization constants.
    """
    w_new, _ = orthogonalize(w, walker_kind)
    return w_new


def multiply_constants(w: walkers, constants: jax.Array, walker_kind: str) -> walkers:
    """
    Distribute a per walker constant across walker columns.
    """
    wk = walker_kind.lower()
    constants = jnp.asarray(constants)
    if wk == "unrestricted":
        wu, wd = w
        n_total = wu.shape[-1] + wd.shape[-1]
        c = (constants ** (1.0 / n_total)).reshape(-1, 1, 1)
        return (wu * c, wd * c)
    elif wk == "restricted":
        n_total = 2 * w.shape[-1]
        c = (constants ** (1.0 / n_total)).reshape(-1, 1, 1)
        return w * c
    raise ValueError(f"multiply_constants does not handle '{walker_kind}' walkers")


def SrFn(Protocol):
    def __call__(
        self,
        w: walkers,
        weights: jax.Array,
        zeta: jax.Array | float,
        walker_kind: str,
    ) -> tuple[walkers, jax.Array]: ...


def no_sr(
    w: walkers,
    weights: jax.Array,
    zeta: jax.Array | float,
    walker_kind: str,
) -> tuple[walkers, jax.Array]:
    return w, weights


def _sr_indices(weights: jax.Array, zeta: jax.Array | float, n_walkers: int) -> jax.Array:
    cw = jnp.cumsum(jnp.abs(weights))
    tot = cw[-1]
    z = tot * (jnp.arange(n_walkers) + zeta) / n_walkers
    idx = jnp.searchsorted(cw, z, side="left")
    return idx


def stochastic_reconfiguration(
    w: walkers,
    weights: jax.Array,
    zeta: jax.Array | float,
    walker_kind: str,
    *,
    data_sharding: NamedSharding | None = None,
) -> tuple[walkers, jax.Array]:
    wk = walker_kind.lower()
    n = w[0].shape[0] if wk == "unrestricted" else w.shape[0]

    cw = jnp.cumsum(jnp.abs(weights))
    avg = cw[-1] / n
    weights_new = jnp.full((n,), avg, dtype=weights.dtype)

    idx = _sr_indices(weights, zeta, n)

    if wk == "unrestricted":
        wu, wd = w
        w_new = (wu[idx], wd[idx])
    elif wk in ("restricted", "generalized"):
        w_new = w[idx]
    else:
        raise ValueError(f"unknown walker_kind: {walker_kind}")

    if data_sharding is not None:
        if wk == "unrestricted":
            wu_new, wd_new = w_new
            wu_new = lax.with_sharding_constraint(wu_new, data_sharding)
            wd_new = lax.with_sharding_constraint(wd_new, data_sharding)
            w_new = (wu_new, wd_new)
        else:
            w_new = lax.with_sharding_constraint(w_new, data_sharding)

        weights_new = lax.with_sharding_constraint(weights_new, data_sharding)

    return w_new, weights_new


def slice_walkers(walkers: Any, walker_kind: str, norb_keep: int | None) -> Any:
    """
    Slice walkers to a prefix of orbital rows for measurement only truncations.
    Used in MLMC.

    Parameters
    ----------
    walkers:
      Walker batch.

    walker_kind:
      sys.walker_kind

    norb_keep:
      If None, return walkers unchanged. Otherwise keep orbital indices [0:norb_keep).
    """
    if norb_keep is None:
        return walkers

    wk = walker_kind.lower()

    if wk == "restricted" or wk == "generalized":
        # walkers: (nw, norb, nocc)
        return walkers[:, :norb_keep, :]

    if wk == "unrestricted":
        wu, wd = walkers
        return (wu[:, :norb_keep, :], wd[:, :norb_keep, :])

    raise ValueError(f"unknown walker_kind: {walker_kind}")


def take_walkers(walkers: Any, idx: jnp.ndarray) -> Any:
    return jax.tree_util.tree_map(lambda x: x[idx, ...], walkers)
