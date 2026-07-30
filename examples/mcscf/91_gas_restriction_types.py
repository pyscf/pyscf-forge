#!/usr/bin/env python
#
# Author: Yi Deng <yideng@uchicago.edu>
#

"""Normalize the four supported GAS restriction input formats."""

import numpy

from pyscf.mcscf import addons_gas


gas_orbs = (2, 2)
nelec = (2, 2)

spin_supergroups = numpy.asarray([
    [2, 0, 0, 2],
    [1, 1, 1, 1],
    [0, 2, 2, 0],
    [1, 1, 1, 1],  # Duplicate rows are removed.
])
supergroups = numpy.asarray([[2, 2]])
cumulative_occ = numpy.asarray([[2, 2], [4, 4]])

for restriction_type, restriction in (
        ("spin-supergroup", spin_supergroups),
        ("supergroup", supergroups),
        ("cumulative-occ", cumulative_occ)):
    blocks = addons_gas.normalize_gas_restr(
        gas_orbs, nelec, restriction, restriction_type)
    print("%-17s ->\n%s" % (restriction_type, blocks))

ras_blocks = addons_gas.normalize_gas_restr(
    (1, 2, 1), nelec,
    {"max_holes": 1, "max_particles": 1},
    "ras",
)
print("ras               ->\n%s" % ras_blocks)
