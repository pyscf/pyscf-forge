# Requires a separate JAX installation.
# Phaseless AFQMC with a CISD trial state.

from pyscf import afqmc, cc, gto, scf

mol = gto.M(
    atom="""
    O        0.0000000000      0.0000000000      0.0000000000
    H        0.9562300000      0.0000000000      0.0000000000
    H       -0.2353791634      0.9268076728      0.0000000000
    """,
    basis="6-31g",
    verbose=3,
)

# RHF
mf = scf.RHF(mol)
mf.kernel()

mycc = cc.CCSD(mf)
mycc.frozen = 1  # freeze O 1s core
mycc.kernel()
et = mycc.ccsd_t()  # for comparison
print(f"CCSD(T) total energy: {mycc.e_tot + et}")

af = afqmc.AFQMC(mycc)
af.n_walkers = 100
af.n_eql_blocks = 10
af.n_blocks = 100
af.seed = 548280
mean, err = af.kernel()
