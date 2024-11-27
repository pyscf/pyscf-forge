#!/usr/bin/env python

"""
Response-theory calculation of the transition dipole moment.

Can only be used with CMS-PDFT currently.
"""

from pyscf import gto, scf, mcpdft

mol = gto.M(
    atom="""O 0 0 0
          H 0 -0.757 0.587
          H 0  0.757 0.587""",
    basis="6-31G",
)
mf = scf.RHF(mol).run()
mc = mcpdft.CASSCF(mf, "tPBE", 4, 4)

mc_cms = mc.multi_state(
    [
        1.0 / 4,
    ]
    * 4,
    method="cms",
)
mc_cms.run()

td_state = mc_cms.trans_moment(unit="Debye", state=(0, 2))
print("Transition dipole moment of <{}|mu|{}> excited state".format(0, 2))
print(" {:8.5f} {:8.5f} {:8.5f}\n".format(*td_state))

# Note that the transition dipole moment is symmetric
h_td_state = mc_cms.trans_moment(unit="Debye", state=(2, 0))
print(max(abs(td_state - h_td_state)))
