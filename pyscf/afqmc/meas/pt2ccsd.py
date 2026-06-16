from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import lax, tree_util

from ..core.ops import MeasOps, k_energy
from ..core.system import System
from ..ham.chol import HamChol
from ..trial.pt2ccsd import Pt2ccsdTrial
from ..trial.pt2ccsd import overlap_r
from .. import walkers as wk
from ..prop.types import PropState, QmcParams


@dataclass(frozen=True)
class Pt2ccsdMeasCfg:
    memory_mode: str = "low"  # or Literal["low","high"]
    mixed_real_dtype: jnp.dtype = jnp.float64
    mixed_complex_dtype: jnp.dtype = jnp.complex128
    mixed_real_dtype_testing: jnp.dtype = jnp.float32
    mixed_complex_dtype_testing: jnp.dtype = jnp.complex64


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Pt2ccsdMeasCtx:
    cfg: Pt2ccsdMeasCfg  # static

    def tree_flatten(self):
        children = ()
        aux = (self.cfg,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (cfg,) = aux
        return cls(cfg=cfg)


def build_meas_ctx(
    ham_data: HamChol, trial_data: Pt2ccsdTrial, cfg: Pt2ccsdMeasCfg = Pt2ccsdMeasCfg()
) -> Pt2ccsdMeasCtx:
    if ham_data.basis != "restricted":
        raise ValueError("pt2CCSD MeasOps currently assumes HamChol.basis == 'restricted'.")

    return Pt2ccsdMeasCtx(cfg=cfg)


def _greens_restricted(walker: jax.Array, mo_t: jax.Array) -> jax.Array:
    return (walker @ (jnp.linalg.inv(mo_t.T @ walker)) @ mo_t.T).T


def _greenp_from_green(green: jax.Array, nocc: int) -> jax.Array:
    norb = green.shape[0]
    return (green - jnp.eye(norb))[:, nocc:]


def energy_kernel_rw_rh(
    walker: jax.Array, ham_data: HamChol, meas_ctx: Pt2ccsdMeasCtx, trial_data: Pt2ccsdTrial
) -> jax.Array:
    mo_t, t2 = trial_data.mo_t, trial_data.t2
    nocc = trial_data.nocc

    green = _greens_restricted(walker, mo_t)  # (norb, norb)
    greenp = _greenp_from_green(green, nocc)  # (norb, nvir)

    h1 = ham_data.h1
    chol = ham_data.chol

    hg = jnp.einsum("pq,pq->", h1, green, optimize="optimal")
    e1_0 = 2 * hg

    # one-body double excitations
    t2g_c = jnp.einsum("iajb,ia->jb", t2, green[:nocc, nocc:], optimize="optimal")
    t2g_e = jnp.einsum("iajb,ib->ja", t2, green[:nocc, nocc:], optimize="optimal")
    t2_green_c = (greenp @ t2g_c.T) @ green[:nocc, :]
    t2_green_e = (greenp @ t2g_e.T) @ green[:nocc, :]
    t2_green = 2 * t2_green_c - t2_green_e
    t2g = 2 * t2g_c - t2g_e
    gt2g = jnp.einsum("ia,ia->", t2g, green[:nocc, nocc:], optimize="optimal")
    e1_2_1 = 2 * hg * gt2g
    e1_2_2 = -2 * jnp.einsum("pq,pq->", h1, t2_green, optimize="optimal")
    e1_2 = e1_2_1 + e1_2_2  # <exp(T1)HF|T2 h1|walker>/<exp(T1)HF|walker>

    # two body energy
    lg = jnp.einsum("gpq,pq->g", chol, green, optimize="optimal")

    # two body double excitations
    lt2g = jnp.einsum("gpq,pq->g", chol, t2_green, optimize="optimal")
    e2_2_2_1 = -lt2g @ lg

    def scanned_fun(carry, x):
        chol_i = x
        # e2_0
        gl_i = jnp.einsum("pr,qr->pq", green, chol_i, optimize="optimal")
        e2_0_1_i = (2 * jnp.trace(gl_i)) ** 2 / 2.0
        e2_0_2_i = -jnp.einsum("pq,qp->", gl_i, gl_i, optimize="optimal")
        carry[0] += e2_0_1_i + e2_0_2_i
        # e2_2_2_2
        lt2_green_i = jnp.einsum("pr,qr->pq", chol_i, t2_green, optimize="optimal")
        carry[1] += 0.5 * jnp.einsum("pq,pq->", gl_i, lt2_green_i, optimize="optimal")
        # e2_2_3
        glgp_i = jnp.einsum("iq,qa->ia", gl_i[:nocc, :], greenp, optimize="optimal")
        l2t2_1 = jnp.einsum("ia,jb,iajb->", glgp_i, glgp_i, t2, optimize="optimal")
        l2t2_2 = jnp.einsum("ib,ja,iajb->", glgp_i, glgp_i, t2, optimize="optimal")
        carry[2] += 2 * l2t2_1 - l2t2_2
        return carry, 0.0

    [e2_0, e2_2_2_2, e2_2_3], _ = lax.scan(scanned_fun, [0.0, 0.0, 0.0], chol)
    e2_2_1 = e2_0 * gt2g
    e2_2_2 = 4 * (e2_2_2_1 + e2_2_2_2)
    e2_2 = e2_2_1 + e2_2_2 + e2_2_3

    t2 = gt2g  # <exp(T1)HF|T2|walker>/<exp(T1)HF|walker>
    e0 = e1_0 + e2_0  # * t1 # <exp(T1)HF|h1+h2|walker>/<exp(T1)HF|walker>
    e1 = e1_2 + e2_2  # * t1 # <exp(T1)HF|T2 (h1+h2)|walker>/<exp(T1)HF|walker>

    return jnp.stack([t2, e0, e1])


def make_pt2ccsd_meas_ops(
    sys: System,
    memory_mode: str = "low",
    mixed_precision: bool = False,
    testing: bool = False,
) -> MeasOps:
    if sys.walker_kind.lower() != "restricted":
        raise ValueError(
            f"pt2CCSD MeasOps currently supports only restricted walkers, got: {sys.walker_kind}"
        )

    cfg = Pt2ccsdMeasCfg(
        memory_mode=memory_mode,
        mixed_real_dtype=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed_precision else jnp.complex128,
        mixed_real_dtype_testing=jnp.float64 if testing else jnp.float32,
        mixed_complex_dtype_testing=jnp.complex128 if testing else jnp.complex64,
    )

    return MeasOps(
        overlap=overlap_r,
        build_meas_ctx=lambda ham_data, trial_data: build_meas_ctx(ham_data, trial_data, cfg),
        kernels={k_energy: energy_kernel_rw_rh},
    )


def get_init_pt2trial_energy(
    init_state: PropState,
    ham_data: HamChol,
    trial_data: Pt2ccsdTrial,
    trial_meas_ops: MeasOps,
    trial_meas_ctx: Pt2ccsdMeasCtx,
    params: QmcParams,
):

    walker_0 = wk.take_walkers(init_state.walkers, jnp.array([0]))
    trial_e_kernel = trial_meas_ops.require_kernel(k_energy)
    pt2results = wk.vmap_chunked(trial_e_kernel, n_chunks=1, in_axes=(0, None, None, None))(
        walker_0, ham_data, trial_meas_ctx, trial_data
    )
    t2, e0, e1 = pt2results[:, 0], pt2results[:, 1], pt2results[:, 2]
    trial_overlap = wk.vmap_chunked(
        trial_meas_ops.overlap, n_chunks=params.n_chunks, in_axes=(0, None)
    )(walker_0, trial_data)
    guide_overlap = init_state.overlaps[0]
    trial_weights = init_state.weights * trial_overlap / guide_overlap
    trial_energy = (ham_data.h0 + e0 + e1 - t2 * e1).mean()

    return trial_energy + 0j, jnp.sum(trial_weights)
