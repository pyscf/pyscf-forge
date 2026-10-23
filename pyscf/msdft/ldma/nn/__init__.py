"""Differentiable matrix-function utilities used by LDMA."""

from .functional import MatrixFunction, ScalarFunction, robust_symmetric_eigh, trace_average

__all__ = ["MatrixFunction", "ScalarFunction", "robust_symmetric_eigh", "trace_average"]
