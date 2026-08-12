#!/usr/bin/env python
#
# Author: Yi Deng <yideng@uchicago.edu>
#

"""Multiroot GASCI with cumulative restrictions and state-resolved RDMs."""

import numpy

from pyscf import gto
from pyscf import scf
from pyscf.mcscf import gasci


mol = gto.M(
    atom="N 0 0 0; N 0 0 1.10", basis="sto-3g", spin=0, verbose=0)
mf = scf.RHF(mol).run()

# Each row gives the lower and upper cumulative electron occupation after
# adding the corresponding GAS subspace.
mc = gasci.GASCI(
    mf, 4, (2, 2), ncore=5,
    gas_orbs=(2, 2),
    gas_restr=[[1, 3], [4, 4]],
    gas_restr_type="cumulative-occ",
)
mc.fcisolver.nroots = 3
mc.verbose = 4
energies, _, _, _, _ = mc.kernel()

for state, energy in enumerate(numpy.asarray(energies).reshape(-1)):
    dm1, dm2 = mc.make_gasdm12(state=state)
    ss, multiplicity = mc.spin_square(state=state)
    pair_trace = numpy.einsum("pprr", dm2).real
    print("state %d  E = %.12f  Tr(D1) = %.8f  S^2 = %.8f  mult = %.4f" %
          (state, energy, numpy.trace(dm1), ss, multiplicity))
    print("state %d  Tr_pair(D2) = %.8f" % (state, pair_trace))

tdm1, tdm2 = mc.trans_gasdm12(bra_state=0, ket_state=2)
print("transition 0 -> 2: ||D1|| = %.8f  ||D2|| = %.8f" %
      (numpy.linalg.norm(tdm1), numpy.linalg.norm(tdm2)))

_, natural_occ = mc.get_gas_natorb(state=0)
_, pseudo_occ = mc.get_gas_pseudo_natorb(state=0)
print("natural occupations        = %s" % natural_occ)
for igas, occupations in enumerate(pseudo_occ):
    print("GAS%d pseudo-natural occupations = %s" %
          (igas + 1, occupations))
