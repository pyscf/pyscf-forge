from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import tree_util, vmap

from ..core.ops import MeasOps, k_energy, k_force_bias
from ..core.system import System
from ..ham.chol import HamChol
from ..trial.gcisd import GcisdTrial, overlap_g


def _half_green_from_overlap_matrix(w: jax.Array, ovlp_mat: jax.Array) -> jax.Array:
    """
    green_half = (w @ inv(ovlp_mat)).T
    """
    return jnp.linalg.solve(ovlp_mat.T, w.T)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GcisdMeasCtx:
    # half-rotated:
    rot_h1: jax.Array  # (nocc, norb)
    rot_chol: jax.Array  # (n_chol, nocc, norb)
    rot_chol_flat: jax.Array  # (n_chol, nocc*norb)

    def tree_flatten(self):
        return (
            self.rot_h1,
            self.rot_chol,
            self.rot_chol_flat,
        ), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        (
            rot_h1,
            rot_chol,
            rot_chol_flat,
        ) = children
        return cls(
            rot_h1=rot_h1,
            rot_chol=rot_chol,
            rot_chol_flat=rot_chol_flat,
        )


def force_bias_kernel_gw_gh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: GcisdMeasCtx,
    trial_data: GcisdTrial,
) -> jax.Array:
    """Calculates force bias < psi_T | chol_gamma | walker > / < psi_T | walker >"""
    c1 = trial_data.c1
    c2 = trial_data.c2

    norb = trial_data.norb
    nocc = trial_data.nocc

    w = walker
    wo = w[:nocc, :]

    green = _half_green_from_overlap_matrix(w, wo)
    green_occ = green[:, nocc:].copy()
    greenp = jnp.vstack((green_occ, -jnp.eye(2 * norb - nocc)))

    chol = ham_data.chol
    rot_chol = meas_ctx.rot_chol

    # Ref
    nu0 = jnp.einsum("gpq,pq->g", rot_chol, green)

    # Single excitations
    nu1 = jnp.einsum("gpq,ia,pq,ia->g", rot_chol, c1.conj(), green, green_occ)
    nu1 -= jnp.einsum("gpq,ia,iq,pa->g", chol, c1.conj(), green, greenp)

    # Double excitations
    nu2 = 2.0 * jnp.einsum(
        "gpq,iajb,pq,ia,jb->g",
        rot_chol,
        c2.conj(),
        green,
        green_occ,
        green_occ,
    )
    nu2 -= 4.0 * jnp.einsum("gpq,iajb,pa,iq,jb->g", chol, c2.conj(), greenp, green, green_occ)
    nu2 *= 0.25

    nu = nu0 + nu1 + nu2
    o1 = jnp.einsum("ia,ia->", c1.conj(), green_occ)
    o2 = 0.25 * 2.0 * jnp.einsum("iajb, ia, jb->", c2.conj(), green_occ, green_occ)
    overlap = 1.0 + o1 + o2
    nu = nu / overlap
    return nu


def energy_kernel_gw_gh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: GcisdMeasCtx,
    trial_data: GcisdTrial,
) -> jax.Array:
    c1 = trial_data.c1
    c2 = trial_data.c2

    norb = trial_data.norb
    nocc = trial_data.nocc

    w = walker
    wo = w[:nocc, :]

    green = _half_green_from_overlap_matrix(w, wo)
    green_occ = green[:, nocc:].copy()
    greenp = jnp.vstack((green_occ, -jnp.eye(2 * norb - nocc)))

    chol = ham_data.chol
    rot_chol = meas_ctx.rot_chol
    h1 = ham_data.h1
    rot_h1 = meas_ctx.rot_h1

    # 0 body energy
    e0 = ham_data.h0

    # 1 body energy
    # ref
    e1_0 = jnp.einsum("pq,pq->", rot_h1, green)

    # single excitations
    e1_1 = jnp.einsum("pq,ia,pq,ia->", rot_h1, c1.conj(), green, green_occ, optimize="optimal")
    e1_1 -= jnp.einsum("pq,ia,iq,pa->", h1, c1.conj(), green, greenp, optimize="optimal")

    # double excitations
    # e1_2  = jnp.einsum("pq,iajb,pq,ia,jb->", h1[nocc:,:], c2.conj(), green, green_occ, green_occ)
    # e1_2 -= jnp.einsum("pq,iajb,pq,ib,ja->", h1[nocc:,:], c2.conj(), green, green_occ, green_occ)
    # e1_2 -= jnp.einsum("pq,iajb,pa,iq,jb->", h1, c2.conj(), greenp, green, green_occ)
    # e1_2 += jnp.einsum("pq,iajb,pa,ib,jq->", h1, c2.conj(), greenp, green_occ, green)
    # e1_2 += jnp.einsum("pq,iajb,pb,iq,ja->", h1, c2.conj(), greenp, green, green_occ)
    # e1_2 -= jnp.einsum("pq,iajb,pb,ia,jq->", h1, c2.conj(), greenp, green_occ, green)

    e1_2 = 2.0 * jnp.einsum("rq,rq,iajb,ia,jb", rot_h1, green, c2.conj(), green_occ, green_occ)
    e1_2 -= 4.0 * jnp.einsum("pq,iajb,pa,iq,jb", h1, c2.conj(), greenp, green, green_occ)
    e1_2 *= 0.25

    # 2 body energy
    # ref
    f = jnp.einsum("gij,jk->gik", rot_chol, green.T, optimize="optimal")
    c = vmap(jnp.trace)(f)
    exc = jnp.sum(vmap(lambda x: x * x.T)(f))
    e2_0 = (jnp.sum(c * c) - exc) / 2.0

    # single excitations
    e2_1 = jnp.einsum(
        "gpr,gqs,ia,ir,ps,qa->",
        rot_chol,
        chol,
        c1.conj(),
        green,
        green,
        greenp,
    )
    e2_1 -= jnp.einsum(
        "gpr,gqs,ia,ir,pa,qs->",
        chol,
        rot_chol,
        c1.conj(),
        green,
        greenp,
        green,
    )
    e2_1 -= jnp.einsum(
        "gpr,gqs,ia,pr,is,qa->",
        rot_chol,
        chol,
        c1.conj(),
        green,
        green,
        greenp,
    )
    e2_1 += jnp.einsum(
        "gpr,gqs,ia,pr,ia,qs->",
        rot_chol,
        rot_chol,
        c1.conj(),
        green,
        green_occ,
        green,
    )
    e2_1 += jnp.einsum(
        "gpr,gqs,ia,qr,is,pa->",
        chol,
        rot_chol,
        c1.conj(),
        green,
        green,
        greenp,
    )
    e2_1 -= jnp.einsum(
        "gpr,gqs,ia,qr,ia,ps->",
        rot_chol,
        rot_chol,
        c1.conj(),
        green,
        green_occ,
        green,
    )
    e2_1 *= 0.5

    # double excitations
    e2_2 = 2.0 * jnp.einsum(
        "gpr,gqs,iajb,ir,js,pa,qb->",
        chol,
        chol,
        c2.conj(),
        green,
        green,
        greenp,
        greenp,
    )
    # e2_2 -= jnp.einsum("gpr,gqs,iajb,ir,js,pb,qa->", chol           , chol           , c2.conj(), green, green, greenp   , greenp)
    e2_2 -= 2.0 * jnp.einsum(
        "gpr,gqs,iajb,ir,ps,ja,qb->",
        rot_chol,
        chol,
        c2.conj(),
        green,
        green,
        green_occ,
        greenp,
    )
    # e2_2 += jnp.einsum("gpr,gqs,iajb,ir,ps,jb,qa->", chol[:,:nocc,:], chol           , c2.conj(), green, green, green_occ, greenp)
    e2_2 += 2.0 * jnp.einsum(
        "gpr,gqs,iajb,ir,qs,ja,pb->",
        chol,
        rot_chol,
        c2.conj(),
        green,
        green,
        green_occ,
        greenp,
    )
    # e2_2 -= jnp.einsum("gpr,gqs,iajb,ir,qs,jb,pa->", chol           , chol[:,:nocc,:], c2.conj(), green, green, green_occ, greenp)
    # P_ij
    e2_2 *= 2.0

    e2_2 += 4.0 * jnp.einsum(
        "gpr,gqs,iajb,pr,is,ja,qb->",
        rot_chol,
        chol,
        c2.conj(),
        green,
        green,
        green_occ,
        greenp,
    )
    # e2_2 -= 2.0 * jnp.einsum("gpr,gqs,iajb,pr,is,jb,qa->", chol[:,:nocc,:], chol           , c2.conj(), green, green, green_occ, greenp   )
    e2_2 += 2.0 * jnp.einsum(
        "gpr,gqs,iajb,pr,qs,ia,jb->",
        rot_chol,
        rot_chol,
        c2.conj(),
        green,
        green,
        green_occ,
        green_occ,
    )

    # e2_2 -=       jnp.einsum("gpr,gqs,iajb,pr,qs,ib,ja->", chol[:,:nocc,:], chol[:,:nocc,:], c2.conj(), green, green, green_occ, green_occ)
    # P_pq
    e2_2 -= 4.0 * jnp.einsum(
        "gpr,gqs,iajb,qr,is,ja,pb->",
        chol,
        rot_chol,
        c2.conj(),
        green,
        green,
        green_occ,
        greenp,
    )
    # e2_2 += 2.0 * jnp.einsum("gpr,gqs,iajb,qr,is,jb,pa->", chol           , chol[:,:nocc,:], c2.conj(), green, green, green_occ, greenp   )
    e2_2 -= 2.0 * jnp.einsum(
        "gpr,gqs,iajb,qr,ps,ia,jb->",
        rot_chol,
        rot_chol,
        c2.conj(),
        green,
        green,
        green_occ,
        green_occ,
    )
    # e2_2 +=       jnp.einsum("gpr,gqs,iajb,qr,ps,ib,ja->", chol[:,:nocc,:], chol[:,:nocc,:], c2.conj(), green, green, green_occ, green_occ)
    e2_2 *= 0.5 * 0.25

    e = e1_0 + e1_1 + e1_2 + e2_0 + e2_1 + e2_2
    o1 = jnp.einsum("ia,ia->", c1.conj(), green_occ)
    o2 = 0.25 * 2.0 * jnp.einsum("iajb, ia, jb->", c2.conj(), green_occ, green_occ)
    overlap = 1.0 + o1 + o2
    e = e / overlap

    return e + e0


def build_meas_ctx(ham_data: HamChol, trial_data: GcisdTrial) -> GcisdMeasCtx:
    if ham_data.basis != "generalized":
        raise ValueError("GCISD MeasOps currently assumes HamChol.basis == 'generalized'.")
    nocc = trial_data.nocc
    rot_h1 = ham_data.h1[:nocc, :]  # (nocc, norb)
    rot_chol = ham_data.chol[:, :nocc, :]
    rot_chol_flat = rot_chol.reshape(rot_chol.shape[0], -1)

    return GcisdMeasCtx(
        rot_h1=rot_h1,
        rot_chol=rot_chol,
        rot_chol_flat=rot_chol_flat,
    )


def make_gcisd_meas_ops(sys: System, mixed_precision: bool = False) -> MeasOps:
    wk = sys.walker_kind.lower()

    # Note: mixed_precision is not used for now

    if wk == "restricted" or wk == "unrestricted":
        raise NotImplementedError("GCISD MeasOps only implemented for generalized walkers.")
    elif wk == "generalized":
        overlap_fn = overlap_g
        kernels = {
            k_force_bias: force_bias_kernel_gw_gh,
            k_energy: energy_kernel_gw_gh,
        }
    else:
        raise ValueError(f"unknown walker_kind: {sys.walker_kind}")

    return MeasOps(
        overlap=overlap_fn,
        build_meas_ctx=build_meas_ctx,
        kernels=kernels,
    )
