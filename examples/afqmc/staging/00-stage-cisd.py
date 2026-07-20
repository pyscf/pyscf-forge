# Requires a separate JAX installation.

from pyscf import cc, gto, scf

from pyscf import afqmc


STAGED_PATH = "h2o_afqmc.h5"

mol = gto.M(
    atom="""
    O        0.0000000000      0.0000000000      0.0000000000
    H        0.9562300000      0.0000000000      0.0000000000
    H       -0.2353791634      0.9268076728      0.0000000000
    """,
    basis="6-31g",
    verbose=3,
)

mf = scf.RHF(mol)
mf.kernel()

mycc = cc.CCSD(mf)
mycc.frozen = 1
mycc.kernel()

af = afqmc.AFQMC(mycc)
af.save_staged(STAGED_PATH)

print(f"Wrote staged AFQMC inputs with a CISD trial to {STAGED_PATH}.")
