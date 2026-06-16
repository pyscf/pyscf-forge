"""JAX-backed AFQMC methods for PySCF.

To use this module the users must install JAX separately
before importing ``pyscf.afqmc``. AFQMC currently requires Python 3.10 or
newer. For CPU runs, use ``pip install -U jax``. For NVIDIA GPU runs, use
``pip install -U "jax[cuda12]"`` or ``pip install -U "jax[cuda13]"``.
We also provide convenience extras: ``pyscf-forge[afqmc-cpu]``,
``pyscf-forge[afqmc-cuda12]``, and ``pyscf-forge[afqmc-cuda13]``.
"""

import sys

_PYTHON_IMPORT_ERROR = (
    "pyscf.afqmc currently requires Python 3.10 or newer. "
    "The ported AFQMC code uses Python 3.10+ syntax and has not been "
    "backported to older Python versions."
)

_JAX_IMPORT_ERROR = (
    "pyscf.afqmc requires a separate JAX installation. "
    "For CPU runs, install JAX with `pip install -U jax`. "
    'For NVIDIA GPU runs, install `pip install -U "jax[cuda12]"` or '
    '`pip install -U "jax[cuda13]"`. If preferred, we also provide '
    "convenience extras: `pyscf-forge[afqmc-cpu]`, "
    "`pyscf-forge[afqmc-cuda12]`, and `pyscf-forge[afqmc-cuda13]`."
)


def _python_version_key(version_info: object = sys.version_info) -> tuple[int, int]:
    if hasattr(version_info, "major") and hasattr(version_info, "minor"):
        return int(getattr(version_info, "major")), int(getattr(version_info, "minor"))

    version_tuple = tuple(version_info)  # type: ignore[arg-type]
    if len(version_tuple) < 2:
        raise ValueError(
            "version_info must provide at least major and minor components"
        )
    return int(version_tuple[0]), int(version_tuple[1])


def _require_supported_python(version_info: object = sys.version_info) -> None:
    if _python_version_key(version_info) < (3, 10):
        raise ImportError(_PYTHON_IMPORT_ERROR)


def _is_missing_jax_import(err: ModuleNotFoundError) -> bool:
    missing_name = (err.name or "").split(".", 1)[0]
    if missing_name in {"jax", "jaxlib"}:
        return True

    message = str(err)
    return (
        "No module named 'jax'" in message
        or 'No module named "jax"' in message
        or "No module named 'jaxlib'" in message
        or 'No module named "jaxlib"' in message
    )


_require_supported_python()

try:
    from .afqmc import AFQMC, AFQMCFP, Afqmc, AFQMCFp, AfqmcFp, banner_afqmc
except ModuleNotFoundError as err:
    if _is_missing_jax_import(err):
        raise ImportError(_JAX_IMPORT_ERROR) from err
    raise


__all__ = [
    "AFQMC",
    "AFQMCFP",
    "AFQMCFp",
    "Afqmc",
    "AfqmcFp",
    "banner_afqmc",
]
