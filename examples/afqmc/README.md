# AFQMC Examples

AFQMC currently requires Python 3.10 or newer and [JAX](https://github.com/jax-ml/jax).
For CPU runs, install JAX with `pip install -U jax`.
For NVIDIA GPU runs, install `pip install -U "jax[cuda12]"` or
`pip install -U "jax[cuda13]"`.
If preferred, we also provide convenience extras:
`pyscf-forge[afqmc-cpu]`, `pyscf-forge[afqmc-cuda12]`, and
`pyscf-forge[afqmc-cuda13]`.
