from __future__ import annotations

import jax
import jax.numpy as jnp

from ..core.ops import TrialOps
from ..core.system import System
from . import eom_cisd
from .eom_cisd import EomCisdTrial


def overlap_r(walker: jax.Array, trial_data: EomCisdTrial) -> jax.Array:
    ci1 = trial_data.ci1
    ci2 = trial_data.ci2
    r1 = trial_data.r1
    r2 = trial_data.r2
    nocc = trial_data.nocc

    wocc = walker[:nocc, :]  # (nocc, nocc)
    green = jnp.linalg.solve(wocc.T, walker.T)  # (nocc, norb)
    green_occ = green[:, nocc:]

    det0 = jnp.linalg.det(wocc)
    o0 = det0 * det0

    # r1 terms
    # r1 1
    r1g = 2 * jnp.einsum("pt,pt", r1, green_occ)
    r1_1 = r1g
    # r1 c1
    c1g = 2 * jnp.einsum("pt,pt", ci1, green_occ)
    r1_c1_1 = r1g * c1g
    r1_g = r1 @ green_occ.T
    c1_g = ci1 @ green_occ.T
    r1_c1_2 = -2 * jnp.einsum("pq,qp", r1_g, c1_g)
    r1_c1 = r1_c1_1 + r1_c1_2
    # r1 c2
    c2g2 = 2 * jnp.einsum("ptqu,pt,qu", ci2, green_occ, green_occ) - jnp.einsum(
        "ptqu,pu,qt", ci2, green_occ, green_occ
    )
    r1_c2_1 = r1g * c2g2
    c2g_1 = jnp.einsum("ptqu,qu->pt", ci2, green_occ)
    c2g_2 = 0.5 * jnp.einsum("ptqu,pu->qt", ci2, green_occ)
    gc2_g = (c2g_1 - c2g_2) @ green_occ.T
    r1_c2_2 = -4 * jnp.einsum("pq,qp", r1_g, gc2_g)
    r1_c2 = r1_c2_1 + r1_c2_2

    # r2 terms
    # r2 1
    r2g2 = 2 * jnp.einsum("ptqu,pt,qu", r2, green_occ, green_occ) - jnp.einsum(
        "ptqu,pu,qt", r2, green_occ, green_occ
    )
    r2_1 = r2g2
    # r2 c1
    r2_c1_1 = r2g2 * c1g
    r2g_1 = jnp.einsum("ptqu,qu->pt", r2, green_occ)
    r2g_2 = 0.5 * jnp.einsum("ptqu,pu->qt", r2, green_occ)
    gr2_g = (r2g_1 - r2g_2) @ green_occ.T
    r2_c1_2 = -4 * jnp.einsum("pq,qp", gr2_g, c1_g)
    r2_c1 = r2_c1_1 + r2_c1_2

    return (r1_1 + r1_c1 + r1_c2 + r2_1 + r2_c1) * o0


def make_eom_t_cisd_trial_ops(sys: System) -> TrialOps:
    if sys.nup != sys.ndn:
        raise ValueError("Restricted EOM T CISD trial requires nup == ndn.")
    if sys.walker_kind.lower() != "restricted":
        raise ValueError(
            f"EOM CISD trial currently supports only restricted walkers, got: {sys.walker_kind}"
        )
    return TrialOps(overlap=overlap_r, get_rdm1=eom_cisd.get_rdm1)
