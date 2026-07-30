"""Deprecated compatibility wrapper for :mod:`pyscf.gbci.direct_gbci`."""

import warnings

from pyscf import lib
from pyscf.gbci import direct_gbci

warnings.warn(
    "pyscf.sfnoci.direct_sfnoci is deprecated; use "
    "pyscf.gbci.direct_gbci instead.",
    lib.exceptions.DeprecationWarning,
    stacklevel=2,
)

str2occ = direct_gbci.str2occ
make_hdiag = direct_gbci.make_hdiag
absorb_h1e = direct_gbci.absorb_h1e
gen_excitations = direct_gbci.gen_excitations
gen_nonzero_excitations = direct_gbci.gen_nonzero_excitations
contract_h = direct_gbci.contract_h
contract_h_slow = direct_gbci.contract_h_slow
kernel = direct_gbci.kernel
fix_spin = direct_gbci.fix_spin
fix_spin_ = direct_gbci.fix_spin_
GBCISolver = direct_gbci.GBCISolver
SpinPenaltyGBCISolver = direct_gbci.SpinPenaltyGBCISolver

SFNOCISolver = GBCISolver
SpinPenaltySFNOCISolver = SpinPenaltyGBCISolver
contract_H = contract_h
contract_H_slow = contract_h_slow

__all__ = [
    "str2occ", "make_hdiag", "absorb_h1e",
    "gen_excitations", "gen_nonzero_excitations",
    "contract_h", "contract_h_slow", "contract_H", "contract_H_slow",
    "kernel", "fix_spin", "fix_spin_",
    "GBCISolver", "SpinPenaltyGBCISolver",
    "SFNOCISolver", "SpinPenaltySFNOCISolver",
]
