#!/usr/bin/env python

"""
Response-theory calculation of the L-PDFT permanent dipole moment
"""

from pyscf import gto, scf, mcpdft
from pyscf.mcpdft.lpdft import _LPDFT
from pyscf.prop.dip_moment.lpdft import _LPDFTDipole

mol = gto.M(
    atom="""O 0 0 0
          H 0 -0.757 0.587
          H 0  0.757 0.587""",
    basis="6-31G",
)
mf = scf.RHF(mol).run()
mc = mcpdft.CASSCF(mf, "tPBE", 4, 4)

mc_lin = mc.multi_state(
    [
        1.0 / 4,
    ]
    * 4,
    method="lin",
)
mc_lin.run()

for state in range(4):
    d_state = mc_lin.dip_moment(unit="Debye", state=state)
    print("Electric dipole moment of {} excited state".format(state))
    print(" {:8.5f} {:8.5f} {:8.5f}\n".format(*d_state))
