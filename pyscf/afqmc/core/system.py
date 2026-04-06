from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

WalkerKind = Literal["restricted", "unrestricted", "generalized"]


@dataclass(frozen=True)
class System:
    """
    Static system configuration
      - norb: number of spatial orbitals
      - nelec: (n_up, n_dn)
      - walker_kind: how walkers are represented
    """

    norb: int
    nelec: Tuple[int, int]
    walker_kind: WalkerKind

    def __post_init__(self):
        object.__setattr__(self, "walker_kind", self.walker_kind.lower())

    @property
    def nup(self) -> int:
        return self.nelec[0]

    @property
    def ndn(self) -> int:
        return self.nelec[1]

    @property
    def ne(self) -> int:
        return self.nelec[0] + self.nelec[1]
