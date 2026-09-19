#!/usr/bin/env python
#
# Author: Yi Deng <yideng@uchicago.edu>
#

"""GASSCF orbital optimization for an H4 chain with a restricted GAS space."""

from pyscf import gto, scf
from pyscf.mcscf import gasscf


mol = gto.M(
    atom="; ".join("H 0 0 %.1f" % z for z in (0.0, 0.8, 1.8, 2.6)),
    basis="sto-3g",
    spin=0,
    verbose=0,
)
mf = scf.RHF(mol).run()

# The two active orbitals are split into two one-orbital GAS spaces.  The
# cumulative bounds enforce one electron after GAS1 and two electrons after
# GAS2, equivalent to one electron in each GAS space.
mc = gasscf.GASSCF(
    mf,
    ncas=2,
    nelecas=(1, 1),
    ncore=1,
    gas_orbs=(1, 1),
    gas_restr=((1, 1), (2, 2)),
    gas_restr_type="cumulative-occ",
)
mc.verbose = 4
e_tot, e_gas, ci, mo, mo_energy = mc.kernel()

print("GASSCF total energy        = %.12f" % e_tot)
print("GASSCF active-space energy = %.12f" % e_gas)
print("GASSCF converged           = %s" % mc.converged)
print("GAS determinant count     = %d" % ci.size)

