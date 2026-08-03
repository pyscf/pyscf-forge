"""Deprecated compatibility namespace for :mod:`pyscf.gbci`."""

import warnings

from pyscf import lib
from pyscf.gbci.gbci import GBCI

warnings.warn(
    "pyscf.sfnoci is deprecated; use pyscf.gbci instead.",
    lib.exceptions.DeprecationWarning,
    stacklevel=2,
)

SFNOCI = GBCI


def sfnoci(mf, ncas, nelecas, ncore=None, group_a=None):
    """Deprecated alias for :class:`pyscf.gbci.gbci.GBCI`."""
    warnings.warn(
        "pyscf.sfnoci.sfnoci is deprecated; use pyscf.gbci.gbci "
        "or pyscf.gbci.GBCI instead.",
        lib.exceptions.DeprecationWarning,
        stacklevel=2,
    )
    return GBCI(mf, ncas, nelecas, ncore=ncore, group_a=group_a)


__all__ = ["GBCI", "SFNOCI", "sfnoci"]
