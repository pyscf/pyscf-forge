#!/usr/bin/env python

import os
from functools import partial

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

import pyscf.afqmc.testing as testing
import pyscf.afqmc.walkers as wk
from pyscf.afqmc.core.system import System
from pyscf.afqmc.meas.auto import make_auto_meas_ops
from pyscf.afqmc.prop.afqmc import init_prop_state, make_prop_ops
from pyscf.afqmc.prop.blocks import block
from pyscf.afqmc.prop.chol_afqmc_ops import _build_prop_ctx
from pyscf.afqmc.prop.types import QmcParams
from pyscf.afqmc.sharding import make_data_mesh, shard_prop_state


mesh = make_data_mesh()
assert mesh.size == 4, mesh.size

n_per_dev = 2
norb = 4
nocc = 2
n_chol = 3
n_walkers = mesh.size * n_per_dev

rng = jax.random.PRNGKey(42)
k_ham, k_walk1, k_walk2, k_wt = jax.random.split(rng, 4)

ham = testing.make_random_ham_chol(k_ham, norb, n_chol)
sys = System(norb=norb, nelec=(nocc, nocc), walker_kind="restricted")
trial_ops = testing.make_dummy_trial_ops()
meas_ops = make_auto_meas_ops(sys, trial_ops)
trial_data = {"rdm1": jnp.eye(norb, dtype=jnp.float64)}
meas_ctx = meas_ops.build_meas_ctx(ham, trial_data)

params = QmcParams(
    dt=0.1,
    n_chunks=1,
    n_exp_terms=4,
    n_prop_steps=1,
    n_blocks=1,
    n_walkers=n_walkers,
    seed=0,
    pop_control_damping=0.1,
)

prop_ops = make_prop_ops(ham.basis, sys.walker_kind)
prop_ctx = _build_prop_ctx(ham, trial_data["rdm1"], params.dt)

initial_walkers = jax.random.normal(
    k_walk1, (n_walkers, norb, nocc), dtype=jnp.float64
) + 1.0j * jax.random.normal(k_walk2, (n_walkers, norb, nocc), dtype=jnp.float64)

state_u = init_prop_state(
    sys=sys,
    ham_data=ham,
    trial_ops=trial_ops,
    trial_data=trial_data,
    meas_ops=meas_ops,
    params=params,
    initial_walkers=initial_walkers,
    mesh=None,
)
rand_weights = jax.random.uniform(
    k_wt, (n_walkers,), dtype=jnp.float64, minval=0.5, maxval=2.0
)
state_u = state_u._replace(weights=rand_weights)
state_s = shard_prop_state(state_u, mesh)

data_sharding = NamedSharding(mesh, P("data"))
sr_sharded = partial(wk.stochastic_reconfiguration, data_sharding=data_sharding)
sr_unsharded = partial(wk.stochastic_reconfiguration, data_sharding=None)


def call_block(state, *, sr_fn):
    return block(
        state,
        sys=sys,
        params=params,
        ham_data=ham,
        trial_data=trial_data,
        trial_ops=trial_ops,
        meas_ops=meas_ops,
        meas_ctx=meas_ctx,
        prop_ops=prop_ops,
        prop_ctx=prop_ctx,
        sr_fn=sr_fn,
    )


run_u = jax.jit(partial(call_block, sr_fn=sr_unsharded))
run_s = jax.jit(partial(call_block, sr_fn=sr_sharded))

state1_u, obs_u = run_u(state_u)
state1_s, obs_s = run_s(state_s)

np.testing.assert_allclose(
    np.asarray(jax.device_get(obs_s.scalars["energy"])),
    np.asarray(jax.device_get(obs_u.scalars["energy"])),
    rtol=1e-12,
    atol=1e-12,
)
np.testing.assert_allclose(
    np.asarray(jax.device_get(state1_s.weights)),
    np.asarray(jax.device_get(state1_u.weights)),
    rtol=1e-12,
    atol=1e-12,
)
for leaf_s, leaf_u in zip(
    tree_util.tree_leaves(state1_s.walkers),
    tree_util.tree_leaves(state1_u.walkers),
):
    np.testing.assert_allclose(
        np.asarray(jax.device_get(leaf_s)),
        np.asarray(jax.device_get(leaf_u)),
        rtol=1e-12,
        atol=1e-12,
    )

print("ok")
