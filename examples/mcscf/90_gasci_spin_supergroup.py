#!/usr/bin/env python
#
# Author: Yi Deng <yideng@uchicago.edu>
#

"""Fixed-orbital GASCI with an explicit spin-supergroup set D."""

import numpy

from pyscf import gto
from pyscf import scf
from pyscf.mcscf import gasci


mol = gto.M(
    atom="N 0 0 0; N 0 0 1.10",
    basis="sto-3g",
    spin=0,
    verbose=0,
)
mf = scf.RHF(mol).run()

# Each row is (alpha occupations, beta occupations).  Rows are normalized to
# lexicographic order and duplicates are removed before entering the C kernel.
gas_restr = numpy.asarray([
    [0, 2, 2, 0],
    [1, 1, 1, 1],
    [2, 0, 0, 2],
], dtype=numpy.int32)

mc = gasci.GASCI(
    mf, 4, (2, 2), ncore=5,
    gas_orbs=(2, 2),
    gas_restr=gas_restr,
    gas_restr_type="spin-supergroup",
)
mc.verbose = 4
e_tot, e_gas, ci, _, _ = mc.kernel()

print("GASCI total energy       = %.12f" % e_tot)
print("GASCI active-space energy = %.12f" % e_gas)
print("GAS determinant count     = %d" % ci.size)
print("GAS space information     = %s" % mc.gas_space_info())
