import jax
import jax.numpy as jnp

from . import driver
from .core.ops import TrialOps
from .core.system import System
from .ham.chol import HamBasis, HamChol
from .meas.auto import make_auto_meas_ops
from .prop.afqmc import make_prop_ops
from .staging import StagedMfOrCc, _stage_ham_input


def rand_orthonormal_cols(key, nrow, ncol, dtype=jnp.complex128):
    """
    Random (nrow, ncol) matrix with orthonormal columns via QR.
    """
    k1, k2 = jax.random.split(key)

    if dtype in (jnp.complex128, jnp.complex64):
        a = jax.random.normal(k1, (nrow, ncol), dtype=jnp.float64) + 1.0j * jax.random.normal(
            k2, (nrow, ncol), dtype=jnp.float64
        )
    elif dtype in (jnp.float64, jnp.float32):
        a = jax.random.normal(k1, (nrow, ncol), dtype=jnp.float64)
    else:
        raise TypeError(f"Received unsupported type {dtype}.")

    q, _ = jnp.linalg.qr(a, mode="reduced")
    return q.astype(dtype)


def make_random_ham_chol(
    key, norb, n_chol, basis: HamBasis = "restricted", dtype=jnp.float64
) -> HamChol:
    """
    Build a small HamChol with:
      - symmetric real h1
      - symmetric real chol[g]
    """
    assert basis in ["restricted", "generalized"]

    if basis == "generalized":
        norb = 2 * norb

    k1, k2, k3 = jax.random.split(key, 3)

    a = jax.random.normal(k1, (norb, norb), dtype=dtype)
    h1 = 0.5 * (a + a.T)

    b = jax.random.normal(k2, (n_chol, norb, norb), dtype=dtype)
    chol = 0.5 * (b + jnp.swapaxes(b, 1, 2))

    h0 = jax.random.normal(k3, (), dtype=dtype)

    return HamChol(basis=basis, h0=h0, h1=h1, chol=chol)


def make_walkers(key, sys: System, dtype=jnp.complex128):
    """
    Build a random walker that can be either
    - restricted (norb, nocc)
    - unrestricted ((norb, na), (norb, nb))
    - generalized (2*norb, na+nb)
    """
    norb, na, nb = sys.norb, sys.nup, sys.ndn
    wk = sys.walker_kind.lower()

    if wk == "restricted":
        w = rand_orthonormal_cols(key, norb, na, dtype=dtype)
        return w

    if wk == "unrestricted":
        k1, k2 = jax.random.split(key)
        wu = rand_orthonormal_cols(k1, norb, na, dtype=dtype)
        wd = rand_orthonormal_cols(k2, norb, nb, dtype=dtype)
        return (wu, wd)

    if wk == "generalized":
        w = rand_orthonormal_cols(key, 2 * norb, na + nb, dtype=dtype)
        return w

    raise ValueError(f"unknown walker_kind: {sys.walker_kind}")


def make_restricted_walker_near_ref(
    key, norb: int, nocc: int, *, mix: float = 0.2, dtype=jnp.complex128
) -> jax.Array:
    """
    Make a restricted walker (norb, nocc) whose occupied block isn't near-singular.

    Start from the reference [I;0] and add a small random perturbation, then QR.
    This avoids tiny det(w[:nocc,:]) which can make overlap-based finite differences noisy.
    """
    k1, k2 = jax.random.split(key)
    w0 = jnp.zeros((norb, nocc), dtype=jnp.complex128)
    w0 = w0.at[:nocc, :].set(jnp.eye(nocc, dtype=jnp.complex128))
    noise = jax.random.normal(k1, (norb, nocc), dtype=jnp.float64) + 1.0j * jax.random.normal(
        k2, (norb, nocc), dtype=jnp.float64
    )
    w = w0 + mix * noise
    q, _ = jnp.linalg.qr(w, mode="reduced")
    return q.astype(dtype)


def make_dummy_trial_ops():
    def get_rdm1(trial_data):
        return trial_data["rdm1"]

    def overlap(walker, trial_data):
        return jnp.sum(walker) * 0.0 + (1.0 + 0.0j)

    return TrialOps(overlap=overlap, get_rdm1=get_rdm1)


def make_common_auto(
    key,
    walker_kind,
    norb: int,
    nelec: tuple[int, int],
    n_chol: int,
    *,
    make_trial_fn,
    make_trial_fn_kwargs=(),
    make_trial_ops_fn,
    make_meas_ops_fn,
    ham_basis: HamBasis = "restricted",
):
    sys = System(norb=norb, nelec=nelec, walker_kind=walker_kind)

    k_ham, k_trial = jax.random.split(key, 2)

    ham = make_random_ham_chol(k_ham, norb=norb, n_chol=n_chol, basis=ham_basis)
    trial = make_trial_fn(k_trial, **make_trial_fn_kwargs)

    t_ops = make_trial_ops_fn(sys)
    meas_manual = make_meas_ops_fn(sys)
    meas_auto = make_auto_meas_ops(sys, t_ops, eps=1.0e-4)

    ctx_manual = meas_manual.build_meas_ctx(ham, trial)
    ctx_auto = meas_auto.build_meas_ctx(ham, trial)

    return sys, ham, trial, meas_manual, ctx_manual, meas_auto, ctx_auto


def make_common_manual_only(
    key,
    walker_kind,
    norb: int,
    nelec: tuple[int, int],
    n_chol: int,
    *,
    make_trial_fn,
    make_trial_fn_kwargs=(),
    make_trial_ops_fn,
    build_meas_ctx_fn,
):
    sys = System(norb=norb, nelec=nelec, walker_kind=walker_kind)

    k_ham, k_trial = jax.random.split(key, 2)

    ham = make_random_ham_chol(k_ham, norb=norb, n_chol=n_chol)
    trial = make_trial_fn(k_trial, **make_trial_fn_kwargs)
    ctx = build_meas_ctx_fn(ham, trial)

    return sys, ham, trial, ctx


def run_calc(sys, meas_ops, ham_data, trial_ops, trial_data, params, block_fn, prop_ops):
    mean, err, block_e_all, block_w_all = driver.run_qmc_energy(
        sys=sys,
        params=params,
        ham_data=ham_data,
        trial_ops=trial_ops,
        trial_data=trial_data,
        meas_ops=meas_ops,
        prop_ops=prop_ops,
        block_fn=block_fn,
    )
    return mean, err, block_e_all, block_w_all


def make_common_pyscf(
    mf,
    make_meas_ops_fn,
    make_trial_ops_fn,
    walker_kind,
    ham_basis: HamBasis = "restricted",
):
    obj = StagedMfOrCc(mf, frozen=0)
    ham_input = _stage_ham_input(obj, chol_cut=1e-6, verbose=False)
    h0 = jnp.asarray(ham_input.h0)
    h1 = jnp.asarray(ham_input.h1)
    chol = jnp.asarray(ham_input.chol)
    sys = System(
        norb=mf.mol.nao,
        nelec=mf.mol.nelec,
        walker_kind=walker_kind,
    )
    meas_ops = make_meas_ops_fn(sys)
    ham_data = HamChol(h0, h1, chol, basis=ham_basis)
    prop_ops = make_prop_ops(ham_data.basis, sys.walker_kind)
    trial_ops = make_trial_ops_fn(sys=sys)

    return sys, ham_data, trial_ops, prop_ops, meas_ops
