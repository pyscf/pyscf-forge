# Requires a separate JAX installation.
# Phaseless AFQMC with a UCISD trial state.

from pyscf import afqmc, cc, gto, scf

mol = gto.M(
    atom="""
    N  -1.67119571   -1.44021737    0.00000000
    H  -2.12619571   -0.65213425    0.00000000
    H  -0.76119571   -1.44021737    0.00000000
    """,
    spin=1,
    basis="6-31g",
    verbose=3,
)

mf = scf.UHF(mol)
mf.kernel()

mo1 = mf.stability()[0]
dm1 = mf.make_rdm1(mo1, mf.mo_occ)
mf = mf.run(dm1)
mf.stability()

mycc = cc.UCCSD(mf)
mycc.kernel()

af = afqmc.AFQMC(mycc)
af.n_walkers = 100
af.n_blocks = 100
af.verbose = 4
mean, err = af.kernel()
