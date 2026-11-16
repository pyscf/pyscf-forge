from pyscf import gto, scf, mcpdft
from pyscf.prop.trans_dip_moment import lpdft

mol = gto.M(
    atom="""
    O 0.000  0.000  0.000
    H 0.000 -0.757  0.587
    H 0.000  0.757  0.587
    """,
    basis="6-31G",
)

mf = scf.RHF(mol).run()

mc = mcpdft.CASSCF(mf, "tPBE", 4, 4)

mc_lpdft = mc.multi_state(
    [1.0 / 4.0] * 4,
    method="lin",
)

mc_lpdft.kernel()

td_state = mc_lpdft.trans_moment(
    unit="Debye",
    state=(0, 2),
)

print("L-PDFT transition dipole <0|mu|2> (Debye):")
print(td_state)
