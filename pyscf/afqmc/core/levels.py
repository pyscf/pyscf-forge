from dataclasses import dataclass
from typing import Any

from jax import tree_util

# These are used in MLMC


@dataclass(frozen=True)
class LevelSpec:
    # orbital truncation in MO basis: keep first nvir virtuals (occupied always kept)
    nvir_keep: int | None = None
    # cholesky truncation: keep first nchol vectors
    nchol_keep: int | None = None


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LevelPack:
    """
    Bundle of level-specific inputs for measurement kernels.
    """

    level: LevelSpec
    ham_data: Any
    trial_data: Any
    meas_ctx: Any
    norb_keep: int | None = None

    def tree_flatten(self):
        children = (self.ham_data, self.trial_data, self.meas_ctx)
        aux = (self.level, self.norb_keep)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        ham_data, trial_data, meas_ctx = children
        level, norb_keep = aux
        return cls(
            level=level,
            ham_data=ham_data,
            trial_data=trial_data,
            meas_ctx=meas_ctx,
            norb_keep=norb_keep,
        )
