# Requires a separate JAX installation.

from pyscf import afqmc


STAGED_PATH = "h2o_afqmc.h5"

af = afqmc.AFQMC.from_staged(STAGED_PATH)
# The staged file can come from an expensive CPU-side CCSD calculation.
af.n_walkers = 20
af.n_eql_blocks = 10
af.n_blocks = 200
af.seed = 42

mean, err = af.kernel()
print(f"AFQMC energy: {mean:.10f} +/- {err:.10f}")
