from __future__ import annotations

from ..core.ops import OverlapFn, Rdm1Fn, TrialOps
from ..core.system import System


def make_auto_trial_ops(
    sys: System,
    *,
    overlap_r: OverlapFn,
    overlap_u: OverlapFn,
    overlap_g: OverlapFn,
    get_rdm1: Rdm1Fn,
) -> TrialOps:
    """
    For convenience.
    """
    wk = sys.walker_kind.lower()

    if wk == "restricted":
        return TrialOps(overlap=overlap_r, get_rdm1=get_rdm1)

    if wk == "unrestricted":
        return TrialOps(overlap=overlap_u, get_rdm1=get_rdm1)

    if wk == "generalized":
        return TrialOps(overlap=overlap_g, get_rdm1=get_rdm1)

    raise ValueError(f"unknown walker_kind: {sys.walker_kind}")
