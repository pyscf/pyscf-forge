#!/usr/bin/env python

"""
Response-theory calculation of the permanent dipole moment for excited states.
For MC-PDFT, excited state dipole moments can be computed with either
SA-MC-PDFT or CMS-PDFT currently.
"""

from pyscf import gto, scf, mcpdft

mol = gto.M(
    atom="""O 0 0 0
          H 0 -0.757 0.587
          H 0  0.757 0.587""",
    basis='6-31G',
)
mf = scf.RHF(mol).run()
mc = mcpdft.CASSCF(mf, 'tPBE', 4, 4)

mc_sa = mc.state_average([1.0/4,]*4)
mc_sa.run()

for state in range(4):
    d_state = mc_sa.dip_moment(unit="Debye", state=state)
    print("Electric dipole moment of {} excited state".format(state))
    print(" {:8.5f} {:8.5f} {:8.5f}\n".format(*d_state))


mc_cms = mc.multi_state([1.0/4,]*4, method="cms")
mc_cms.run()

for state in range(4):
    d_state = mc_cms.dip_moment(unit="Debye", state=state)
    print("Electric dipole moment of {} excited state".format(state))
    print(" {:8.5f} {:8.5f} {:8.5f}\n".format(*d_state))
