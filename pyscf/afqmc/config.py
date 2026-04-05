from __future__ import annotations

import os
import platform
import socket
import sys
import warnings
from dataclasses import dataclass


def is_jupyter_notebook() -> bool:
    try:
        from IPython.core.getipython import get_ipython

        ip = get_ipython()
        return ip is not None and "IPKernelApp" in ip.config
    except Exception:
        return False


def _parse_visible_devices(value: str | None) -> int | None:
    if value is None:
        return None

    text = value.strip()
    if text == "":
        return 0

    lowered = text.lower()
    if lowered in {"-1", "none", "novisibledevices"}:
        return 0

    return len([item for item in text.split(",") if item.strip()])


def visible_gpu_count() -> int:
    """
    Number of GPUs visible to the current process before JAX initializes.

    Preference order mirrors common CUDA launcher conventions.
    If no visibility env var is set, fall back to counting NVIDIA GPUs via
    `nvidia-smi -L`; otherwise return 0 if detection is unavailable.
    """
    for name in ("CUDA_VISIBLE_DEVICES",):
        count = _parse_visible_devices(os.getenv(name))
        if count is not None:
            return count

    import shutil
    import subprocess

    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if r.returncode == 0:
                return sum(1 for line in r.stdout.splitlines() if "GPU" in line)
        except Exception:
            pass

    return 0


@dataclass
class AfqmcConfig:
    """
    Global configuration.

    use_gpu:
      - None  : auto (prefer GPU if available, else CPU)
      - True  : force GPU (error if unavailable)
      - False : force CPU
    """

    use_gpu: bool | None = None
    single_precision: bool = False
    disable_tf32: bool = False  # Disable TF32 on gpu if true
    quiet: bool = True  # suppress prints


afqmc_config = AfqmcConfig()

_configured_once = False


def configure_once(
    *,
    use_gpu: bool | None = None,
    single_precision: bool | None = None,
    disable_tf32: bool | None = None,
    quiet: bool | None = None,
) -> None:
    """
    Configure JAX once, subsequent calls do nothing.
    Use GPU if available by default.
    """
    global _configured_once
    if _configured_once:
        return

    assert (
        isinstance(use_gpu, bool) or use_gpu is None
    ), f"Expect a bool | None for 'use_gpu', but got '{type(use_gpu)}'."
    assert (
        isinstance(single_precision, bool) or single_precision is None
    ), f"Expect a bool | None for 'single_precision', but got '{type(single_precision)}'."
    assert (
        isinstance(quiet, bool) or quiet is None
    ), f"Expect a bool | None for 'quiet', but got '{type(quiet)}'."

    if use_gpu is not None:
        afqmc_config.use_gpu = use_gpu
    if single_precision is not None:
        afqmc_config.single_precision = single_precision
    if disable_tf32 is not None:
        afqmc_config.disable_tf32 = disable_tf32
    if quiet is not None:
        afqmc_config.quiet = quiet

    setup_jax(
        use_gpu=afqmc_config.use_gpu,
        single_precision=afqmc_config.single_precision,
        disable_tf32=afqmc_config.disable_tf32,
        quiet=afqmc_config.quiet,
    )
    _configured_once = True


def _detect_gpu() -> bool:
    """Detect NVIDIA GPU hardware without importing JAX."""
    return visible_gpu_count() > 0


def setup_jax(
    *, use_gpu: bool | None, single_precision: bool, disable_tf32: bool, quiet: bool
) -> None:
    """
    Configure JAX runtime.
    """
    jax_already_imported = "jax" in sys.modules

    # resolve auto-detection before touching JAX
    if use_gpu is None:
        use_gpu = _detect_gpu()
    afqmc_config.use_gpu = use_gpu

    # env vars only take effect if JAX hasn't been imported yet
    if jax_already_imported and use_gpu:
        warnings.warn(
            "JAX was imported before AFQMC configuration; "
            "GPU memory settings (XLA_PYTHON_CLIENT_PREALLOCATE, "
            "XLA_PYTHON_CLIENT_ALLOCATOR) may not take effect. "
            "For full control, call config.configure_once(...) before importing jax.",
            stacklevel=3,
        )
    if not jax_already_imported:
        if not single_precision:
            os.environ.setdefault("JAX_ENABLE_X64", "1")
        if disable_tf32:
            os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
        if use_gpu:
            os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")
            os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
        else:
            os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

    from jax import config as jax_config

    # these two work even after import
    jax_config.update("jax_threefry_partitionable", False)
    if not single_precision:
        jax_config.update("jax_enable_x64", True)

    # platform_name only works before backend init
    if not jax_already_imported:
        if use_gpu:
            jax_config.update("jax_platform_name", "gpu")
        else:
            jax_config.update("jax_platform_name", "cpu")

    # verify GPU actually initialized
    if use_gpu:
        import jax

        platforms = {d.platform for d in jax.devices()}
        if "gpu" not in platforms:
            raise RuntimeError(
                "GPU was detected/requested, but JAX did not initialize a GPU backend. "
                "Ensure JAX with NVIDIA CUDA support is installed."
            )

    if not quiet and use_gpu:
        _print_host_info()


def _print_host_info() -> None:
    hostname = socket.gethostname()
    uname_info = platform.uname()
    print(f"# Hostname: {hostname}")
    print("# Using GPU (Policy A).")
    print(f"# System: {uname_info.system}")
    print(f"# Node Name: {uname_info.node}")
    print(f"# Release: {uname_info.release}")
    print(f"# Version: {uname_info.version}")
    print(f"# Machine: {uname_info.machine}")
    print(f"# Processor: {uname_info.processor}")
