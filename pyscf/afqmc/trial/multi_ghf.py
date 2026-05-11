from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util

from ..core.ops import TrialOps
from ..core.system import System
from .ghf import _eff_idx, _ratio_full_rank2, _update_full_rank2


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MultiGhfTrial:
    """
    Multi-GHF trial (mostly used for CPMC)

    ci_coeffs: (ndets,)
    mo_coeffs: (ndets, 2*norb, nelec_tot)
    """

    ci_coeffs: jax.Array
    mo_coeffs: jax.Array
    green_real_dtype: Any = jnp.float32
    green_complex_dtype: Any = jnp.complex64

    @property
    def ndets(self) -> int:
        return int(self.ci_coeffs.shape[0])

    @property
    def norb(self) -> int:
        return int(self.mo_coeffs.shape[1] // 2)

    @property
    def nelec_total(self) -> int:
        return int(self.mo_coeffs.shape[2])

    def tree_flatten(self):
        # keep only arrays as children, dtypes go in aux to stay static
        children = (self.ci_coeffs, self.mo_coeffs)
        aux = (self.green_real_dtype, self.green_complex_dtype)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        green_real_dtype, green_complex_dtype = aux
        ci_coeffs, mo_coeffs = children
        return cls(
            ci_coeffs=ci_coeffs,
            mo_coeffs=mo_coeffs,
            green_real_dtype=green_real_dtype,
            green_complex_dtype=green_complex_dtype,
        )


# def _det_stable(a: jax.Array) -> jax.Array:
#     sign, logabs = jnp.linalg.slogdet(a)
#     return sign * jnp.exp(logabs)


def _det_stable(a: jax.Array) -> jax.Array:
    return jnp.linalg.det(a)


def get_rdm1(trial_data: MultiGhfTrial) -> jax.Array:
    """
    Return a spin-block RDM1 with shape (2, norb, norb).
    Uses only the leading determinant.
    """
    ci = trial_data.ci_coeffs
    mo = trial_data.mo_coeffs
    k0 = jnp.argmax(jnp.abs(ci))
    C = mo[k0]  # (2n, ne)

    dm = C @ C.conj().T  # (2n,2n)
    norb = trial_data.norb
    dm_up = dm[:norb, :norb]
    dm_dn = dm[norb:, norb:]
    return jnp.stack([dm_up, dm_dn], axis=0)


def overlap_u(walker: tuple[jax.Array, jax.Array], trial_data: MultiGhfTrial) -> jax.Array:
    """
    Overlap for unrestricted walker: W = (W_up, W_dn)
    """
    wu, wd = walker
    ci = trial_data.ci_coeffs
    mo = trial_data.mo_coeffs  # (nd,2n,ne)

    norb = wu.shape[0]

    def per_det(Ck: jax.Array, ck: jax.Array) -> jax.Array:
        Cu = Ck[:norb, :]  # (n,ne)
        Cd = Ck[norb:, :]  # (n,ne)

        O_left = Cu.conj().T @ wu  # (ne,nup)
        O_right = Cd.conj().T @ wd  # (ne,ndn)
        O = jnp.concatenate([O_left, O_right], axis=1)  # (ne,ne)

        Ok = _det_stable(O)
        return ck * Ok

    contrib = jax.vmap(per_det, in_axes=(0, 0))(mo, ci)
    return jnp.sum(contrib)


def overlap_g(walker: jax.Array, trial_data: MultiGhfTrial) -> jax.Array:
    """
    Overlap for generalized walker
    """
    ci = trial_data.ci_coeffs
    mo = trial_data.mo_coeffs  # (nd,2n,ne)

    def per_det(Ck: jax.Array, ck: jax.Array) -> jax.Array:
        O = Ck.conj().T @ walker  # (ne,ne)
        Ok = _det_stable(O)
        return ck * Ok

    contrib = jax.vmap(per_det, in_axes=(0, 0))(mo, ci)
    return jnp.sum(contrib)


# Greens for CPMC updates: walker -> greens
#   - returns {"G": (nd,2n,2n), "w": (nd,)}
def calc_green_u(
    walker: tuple[jax.Array, jax.Array], trial_data: MultiGhfTrial
) -> dict[str, jax.Array]:
    wu, wd = walker
    ci = trial_data.ci_coeffs.astype(trial_data.green_complex_dtype)
    mo = trial_data.mo_coeffs.astype(trial_data.green_complex_dtype)

    norb = wu.shape[0]
    nup = wu.shape[1]

    wu_r = wu.astype(trial_data.green_real_dtype)
    wd_r = wd.astype(trial_data.green_real_dtype)

    def per_det(Ck: jax.Array, ck: jax.Array):
        Cu = Ck[:norb, :]
        Cd = Ck[norb:, :]

        O_left = Cu.conj().T @ wu_r
        O_right = Cd.conj().T @ wd_r
        O = jnp.concatenate([O_left, O_right], axis=1)  # (ne,ne)

        X = jnp.linalg.solve(O, Ck.conj().T)  # (ne,2n)

        top = wu_r @ X[:nup, :]  # (n,2n)
        bot = wd_r @ X[nup:, :]  # (n,2n)
        Gk = jnp.concatenate([top, bot], axis=0).T  # (2n,2n)

        Ok = _det_stable(O)
        wk = ck * Ok
        return Gk, wk

    Gk, wk = jax.vmap(per_det, in_axes=(0, 0))(mo, ci)
    return {"G": Gk, "w": wk}


def calc_green_g(walker: jax.Array, trial_data: MultiGhfTrial) -> dict[str, jax.Array]:
    ci = trial_data.ci_coeffs.astype(trial_data.green_complex_dtype)
    mo = trial_data.mo_coeffs.astype(trial_data.green_complex_dtype)

    W = walker.astype(trial_data.green_real_dtype)

    def per_det(Ck: jax.Array, ck: jax.Array):
        O = Ck.conj().T @ W  # (ne,ne)
        X = jnp.linalg.solve(O, Ck.conj().T)  # (ne,2n)
        Gk = (W @ X).T  # (2n,2n)

        Ok = _det_stable(O)
        wk = ck * Ok
        return Gk, wk

    Gk, wk = jax.vmap(per_det, in_axes=(0, 0))(mo, ci)
    return {"G": Gk, "w": wk}


def calc_overlap_ratio(
    greens: dict[str, jax.Array],
    update_indices: jax.Array,
    update_constants: jax.Array,
) -> jax.Array:
    """
    Ratio for the multi-det overlap:
      R = (sum_k w_k r_k) / (sum_k w_k)
    where w_k = c_k det(C_k^H W) and r_k is the determinant-lemma ratio for det k.
    """
    G_states = greens["G"]  # (nd,2n,2n)
    w_states = greens["w"]  # (nd,)

    norb = G_states.shape[-1] // 2
    i_eff, j_eff = _eff_idx(update_indices, norb)
    u0, u1 = update_constants[0], update_constants[1]

    r_k = jax.vmap(lambda G: _ratio_full_rank2(G, i_eff, j_eff, u0, u1), in_axes=0)(G_states)

    W_old = jnp.sum(w_states)
    W_new = jnp.sum(w_states * r_k)

    ratio = jnp.where(jnp.abs(W_old) < 1.0e-16, 0.0 + 0.0j, W_new / W_old)
    return jnp.real(ratio)


def update_green(
    greens: dict[str, jax.Array],
    update_indices: jax.Array,
    update_constants: jax.Array,
) -> dict[str, jax.Array]:
    """
    Update per-determinant greens and weights:
      G_k <- SMW_update(G_k)
      w_k <- w_k * r_k
    """
    G_states = greens["G"]
    w_states = greens["w"]

    norb = G_states.shape[-1] // 2
    i_eff, j_eff = _eff_idx(update_indices, norb)
    u0, u1 = update_constants[0].astype(G_states.dtype), update_constants[1].astype(G_states.dtype)

    r_k = jax.vmap(lambda G: _ratio_full_rank2(G, i_eff, j_eff, u0, u1), in_axes=0)(G_states)

    G_new = jax.vmap(
        lambda G, r: _update_full_rank2(G, i_eff, j_eff, u0, u1, eps=1.0e-8, sanitize=True),
        in_axes=(0, 0),
    )(G_states, r_k)

    w_new = w_states * r_k
    return {"G": G_new, "w": w_new}


def make_multi_ghf_trial_ops(sys: System) -> TrialOps:
    wk = sys.walker_kind.lower()

    if wk == "unrestricted":
        return TrialOps(
            overlap=overlap_u,
            get_rdm1=get_rdm1,
            calc_green=calc_green_u,
            calc_overlap_ratio=calc_overlap_ratio,
            update_green=update_green,
        )

    if wk == "generalized":
        return TrialOps(
            overlap=overlap_g,
            get_rdm1=get_rdm1,
            calc_green=calc_green_g,
            calc_overlap_ratio=calc_overlap_ratio,
            update_green=update_green,
        )

    raise ValueError(
        f"multi-GHF trial only implemented for unrestricted/generalized walkers; got walker_kind={sys.walker_kind}"
    )


# ------------------------------------------------------------
# building multi ghf trial from symmetry projection
# probably belongs in a separate symmetry projection file
# ------------------------------------------------------------


def _apply_pg_to_ket(mo: jax.Array, U: jax.Array, *, norb: int) -> jax.Array:
    """Apply an orbital space operator U to both spin blocks of a GHF ket."""
    up = U @ mo[:norb, :]
    dn = U @ mo[norb:, :]
    return jnp.concatenate([up, dn], axis=0)


def _compress_unique_pg_dets(
    mo_rotated: jax.Array,  # (n_ops, 2*norb, nelec)
    chars: jax.Array,  # (n_ops,)
    *,
    tol_same: float,
    tol_keep: float,
) -> tuple[jax.Array, jax.Array]:
    """
    Given PG-rotated determinants and characters, combine duplicates:
      - duplicates detected via det(Cref^H Ck)
      - coefficients accumulated with phase detS.
    """
    mo_np = np.asarray(mo_rotated)
    ch_np = np.asarray(chars)

    unique_mos: list[np.ndarray] = []
    unique_coeffs: list[np.complex128] = []

    for Ck, chi_k in zip(mo_np, np.conj(ch_np)):
        if not unique_mos:
            unique_mos.append(Ck)
            unique_coeffs.append(chi_k)
            continue

        matched = False
        for m, Cref in enumerate(unique_mos):
            S = Cref.conj().T @ Ck
            detS = np.linalg.det(S)
            if abs(abs(detS) - 1.0) < tol_same:
                unique_coeffs[m] += chi_k * detS
                matched = True
                break

        if not matched:
            unique_mos.append(Ck)
            unique_coeffs.append(chi_k)

    mo_u = jnp.asarray(np.stack(unique_mos, axis=0))
    ci_u = jnp.asarray(np.asarray(unique_coeffs))

    # drop tiny coeff dets
    mask = jnp.abs(ci_u) > tol_keep
    mo_u = mo_u[mask]
    ci_u = ci_u[mask]
    return mo_u, ci_u


def _make_alpha_grid(n_alpha: int) -> tuple[jax.Array, float]:
    # midpoint rule on [0, pi]
    alpha_vals = jnp.pi * (jnp.arange(n_alpha) + 0.5) / float(n_alpha)
    w_alpha = 1.0 / float(n_alpha)
    return alpha_vals, w_alpha


def _make_beta_grid(n_beta: int) -> tuple[jax.Array, jax.Array]:
    # Gauss–Legendre on cos(beta), beta in [0, pi]
    from numpy.polynomial.legendre import leggauss

    x, w = leggauss(int(n_beta))
    beta = np.arccos(x)
    order = np.argsort(beta)
    beta_vals = jnp.asarray(beta[order])
    w_beta = jnp.asarray(w[order])
    return beta_vals, w_beta


def _rotate_spin_trial(mo: jax.Array, *, norb: int, beta: jax.Array, alpha: jax.Array) -> jax.Array:
    """
    Apply the (beta, alpha) spin rotation to a GHF ket
    """
    c, s = jnp.cos(beta / 2.0), jnp.sin(beta / 2.0)

    phase_up = jnp.exp(+0.5j * alpha)
    phase_dn = jnp.exp(-0.5j * alpha)

    C_up = phase_up * mo[:norb, :]
    C_dn = phase_dn * mo[norb:, :]

    a = c * C_up + s * C_dn
    b = -s * C_up + c * C_dn
    return jnp.concatenate([a, b], axis=0)


def _build_pg_s2_expansion(
    mo_pg: jax.Array,  # (n_pg, 2*norb, ne)
    ci_pg: jax.Array,  # (n_pg,)
    alpha: tuple[jax.Array, float],
    beta: tuple[jax.Array, jax.Array],
    *,
    norb: int,
) -> tuple[jax.Array, jax.Array]:
    """
    Returns:
      mo_coeffs: (n_beta*n_alpha*n_pg, 2*norb, ne)
      ci_coeffs: (n_beta*n_alpha*n_pg,)
    """
    alpha_vals, w_alpha = alpha
    beta_vals, w_beta = beta

    # mo grid: (n_beta, n_alpha, n_pg, 2n, ne)
    mo_grid = jax.vmap(
        lambda b: jax.vmap(
            lambda a: jax.vmap(lambda C: _rotate_spin_trial(C, norb=norb, beta=b, alpha=a))(mo_pg)
        )(alpha_vals)
    )(beta_vals)

    mo_coeffs = mo_grid.reshape((-1, mo_grid.shape[-2], mo_grid.shape[-1]))
    n_alpha = alpha_vals.shape[0]
    w_alpha_vec = jnp.full((n_alpha,), w_alpha, dtype=ci_pg.dtype)  # (n_alpha,)

    ci_grid = (w_beta[:, None, None] * w_alpha_vec[None, :, None]) * ci_pg[None, None, :]
    ci_coeffs = ci_grid.reshape((-1,))
    return mo_coeffs, ci_coeffs


def _apply_k_projection(
    mo_coeffs: jax.Array, ci_coeffs: jax.Array, parity: int = 1
) -> tuple[jax.Array, jax.Array]:
    mo_all = jnp.concatenate([mo_coeffs, jnp.conj(mo_coeffs)], axis=0)
    ci_all = jnp.concatenate([ci_coeffs, parity * jnp.conj(ci_coeffs)], axis=0)
    return mo_all, ci_all


def _rdm1_to_test_walker(
    rdm1_up: jax.Array, rdm1_dn: jax.Array, *, nelec: tuple[int, int]
) -> tuple[jax.Array, jax.Array]:
    _, evecs_u = jnp.linalg.eigh(rdm1_up)
    _, evecs_d = jnp.linalg.eigh(rdm1_dn)
    evecs_u = evecs_u[:, ::-1]
    evecs_d = evecs_d[:, ::-1]
    n_up, n_dn = nelec
    return evecs_u[:, :n_up], evecs_d[:, :n_dn]


EnergyFn = Callable[
    [
        tuple[jax.Array, jax.Array],
        Any,
        Any,
        Any,
    ],  # (walker, ham, mo_coeffs, ci_coeffs)
    jax.Array,
]


def build_multi_ghf_expansion(
    *,
    mo0: jax.Array,  # (2*norb, nelec_tot)
    nelec: tuple[int, int],
    pg_ops: jax.Array | None = None,  # (n_ops, norb, norb)
    pg_chars: jax.Array | None = None,  # (n_ops,)
    alpha: tuple[jax.Array, float] | None = None,
    beta: tuple[jax.Array, jax.Array] | None = None,
    n_alpha: int | None = None,
    n_beta: int | None = None,
    auto_grid: bool = False,
    k_projection: bool = False,
    k_parity: int = 1,
    # optional convergence selection
    ham_data: Any | None = None,
    rdm1: jax.Array | None = None,
    test_walker: tuple[jax.Array, jax.Array] | None = None,
    energy_fn: EnergyFn | None = None,
    energy_tol: float = 1.0e-3,
    candidate_grids: Iterable[tuple[int, int]] = (
        (3, 4),
        (4, 4),
        (5, 4),
        (5, 6),
        (6, 6),
        (6, 8),
        (8, 8),
        (10, 10),
    ),
    tol_same: float = 1.0e-5,
    tol_keep: float = 1.0e-5,
) -> MultiGhfTrial:
    """
    Build a multi-GHF trial wavefunction via PG, S^2, and K projection.
    """
    mo0 = jnp.asarray(mo0)
    norb = int(mo0.shape[0] // 2)

    # point group projection
    if pg_ops is not None:
        if pg_chars is None:
            raise ValueError("pg_chars must be provided when pg_ops is provided.")
        pg_ops = jnp.asarray(pg_ops)
        pg_chars = jnp.asarray(pg_chars)

        mo_rot = jax.vmap(lambda U: _apply_pg_to_ket(mo0, U, norb=norb))(pg_ops)
        mo_pg, ci_pg = _compress_unique_pg_dets(
            mo_rot, pg_chars, tol_same=tol_same, tol_keep=tol_keep
        )
        # mo_pg, ci_pg = mo_rot, pg_chars

        # normalize by number of group ops
        n_pg = int(pg_ops.shape[0])
        ci_pg = ci_pg / float(n_pg)
    else:
        mo_pg = mo0[None, ...]
        ci_pg = jnp.asarray([1.0 + 0.0j])

    # choose / auto select (alpha, beta) grid
    if test_walker is None and (rdm1 is not None):
        test_walker = _rdm1_to_test_walker(rdm1[0], rdm1[1], nelec=nelec)

    if auto_grid:
        if ham_data is None or test_walker is None or energy_fn is None:
            raise ValueError(
                "auto_grid=True requires ham_data, test_walker (or rdm1), and energy_fn."
            )
        print("Auto-selecting (alpha, beta) grid via energy convergence...")
        wu, wd = test_walker

        Es: list[float] = []
        grids: list[tuple[tuple[jax.Array, float], tuple[jax.Array, jax.Array]]] = []

        for n_alpha, n_beta in candidate_grids:
            a = _make_alpha_grid(n_alpha)
            b = _make_beta_grid(n_beta)
            mo_coeffs_tmp, ci_tmp = _build_pg_s2_expansion(mo_pg, ci_pg, a, b, norb=norb)
            if k_projection:
                mo_coeffs_tmp, ci_tmp = _apply_k_projection(mo_coeffs_tmp, ci_tmp, parity=k_parity)

            E = energy_fn((wu, wd), ham_data, mo_coeffs_tmp, ci_tmp)
            Es.append(float(E))
            grids.append((a, b))
            print(f"(n_alpha={n_alpha}, n_beta={n_beta}) -> E = {float(E):.8f}")

        E_ref = Es[-1]
        best = len(Es) - 1
        for k, Ek in enumerate(Es):
            if abs(Ek - E_ref) < energy_tol:
                best = k
                break

        alpha = grids[best][0]
        beta = grids[best][1]
        print(
            f"Selected (n_alpha={alpha[0].shape[0]}, n_beta={beta[0].shape[0]}) with E = {Es[best]:.8f}"
        )

    if alpha is None:
        if n_alpha is not None:
            alpha = _make_alpha_grid(n_alpha)
        else:
            alpha = _make_alpha_grid(5)
    if beta is None:
        if n_beta is not None:
            beta = _make_beta_grid(n_beta)
        else:
            beta = _make_beta_grid(8)
    print(f"Using (n_alpha={alpha[0].shape[0]}, n_beta={beta[0].shape[0]}) grid.")
    # build PG × S^2 expansion
    mo_coeffs, ci_coeffs = _build_pg_s2_expansion(mo_pg, ci_pg, alpha, beta, norb=norb)

    # k projection
    if k_projection:
        mo_coeffs, ci_coeffs = _apply_k_projection(mo_coeffs, ci_coeffs, parity=k_parity)

    print(f"Final multi-GHF expansion: {ci_coeffs.shape[0]} determinants.")

    return MultiGhfTrial(
        ci_coeffs=ci_coeffs,
        mo_coeffs=mo_coeffs,
    )
