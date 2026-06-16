# Requires a separate JAX installation.

from pyscf import afqmc
from pyscf.afqmc.sharding import make_data_mesh


STAGED_PATH = "h2o_afqmc.h5"

af = afqmc.AFQMC.from_staged(STAGED_PATH)
af.n_walkers = 20
af.n_eql_blocks = 10
af.n_blocks = 200
af.seed = 42

# shards the walker data across all devices, but replicates other data
mesh = make_data_mesh()

# Choleskies can be sharded by using
# from pyscf.afqmc.sharding import make_data_model_mesh
# for 4 devices, with 2 data shards and 2 model shards
# mesh = make_data_model_mesh(n_data=2, n_model=2)

mean, err = af.kernel(mesh=mesh)
print(f"AFQMC energy: {mean:.10f} +/- {err:.10f}")
