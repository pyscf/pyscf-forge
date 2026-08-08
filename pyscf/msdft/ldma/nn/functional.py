"""Differentiable analytic functions of symmetric matrices."""

from abc import ABC, abstractmethod

import torch
from torch import Size, Tensor
from torch.autograd import Function
from torch.autograd.function import once_differentiable


def robust_symmetric_eigh(tensor, epsilon=1.0e-12):
    """Diagonalize a finite symmetric tensor, retrying with small diagonal jitter."""
    _check_dimensions(tensor, "tensor")
    if not torch.isfinite(tensor).all():
        raise ValueError("symmetric eigendecomposition requires finite input")
    if not torch.allclose(tensor, tensor.transpose(-1, -2), rtol=0.0, atol=epsilon):
        raise ValueError("symmetric eigendecomposition requires symmetric input")
    try:
        return torch.linalg.eigh(tensor)
    except torch.linalg.LinAlgError:
        identity = torch.eye(tensor.size(-1), dtype=tensor.dtype, device=tensor.device)
        for scale in (1.0, 1.0e2, 1.0e4, 1.0e6, 1.0e8):
            try:
                return torch.linalg.eigh(tensor + epsilon * scale * identity)
            except torch.linalg.LinAlgError:
                continue
        raise


class ScalarFunction(ABC):
    """Scalar function used to define an analytic matrix function."""

    @staticmethod
    @abstractmethod
    def value(x: Tensor, *args) -> Tensor:
        pass

    @staticmethod
    @abstractmethod
    def derivative1(x: Tensor, *args) -> Tensor:
        pass


def _check_dimensions(tensor: Tensor, name: str) -> None:
    if tensor.ndim < 2 or tensor.size(-1) != tensor.size(-2):
        raise ValueError(
            f"The tensor {name} must have shape (..., n, n); got {tensor.size()}."
        )


class MatrixFunction(Function):
    """Apply a scalar function to the eigenvalues of symmetric matrices."""

    @staticmethod
    def forward(ctx, function: ScalarFunction, tensor: Tensor, *args) -> Tensor:
        _check_dimensions(tensor, "tensor")
        with torch.no_grad():
            eigenvalues, eigenvectors = robust_symmetric_eigh(tensor)
            function_values = function.value(eigenvalues, *args)
            result = torch.einsum(
                "...ia,...a,...ja->...ij",
                eigenvectors,
                function_values,
                eigenvectors,
            )
        ctx.save_for_backward(eigenvalues, function_values, eigenvectors)
        ctx.function = function
        ctx.args = args
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor):
        _check_dimensions(grad_output, "grad_output")
        eigenvalues, function_values, eigenvectors = ctx.saved_tensors
        la = eigenvalues.unsqueeze(-1)
        lb = eigenvalues.unsqueeze(-2)
        fla = function_values.unsqueeze(-1)
        flb = function_values.unsqueeze(-2)
        difference = la - lb
        same = torch.abs(difference) < 1.0e-12
        safe_difference = torch.where(same, torch.ones_like(difference), difference)
        divided_difference = (fla - flb) / safe_difference
        derivative = ctx.function.derivative1(0.5 * (la + lb), *ctx.args)
        spectral_derivative = torch.where(same, derivative, divided_difference)
        transformed_gradient = torch.einsum(
            "...ia,...ij,...jb->...ab", eigenvectors, grad_output, eigenvectors
        )
        grad_input = torch.einsum(
            "...ka,...ab,...lb->...kl",
            eigenvectors,
            spectral_derivative * transformed_gradient,
            eigenvectors,
        )
        return (None, grad_input, *(None for _ in ctx.args))


def trace_average(tensor: Tensor, weights=None, subspace_dim=None) -> Tensor:
    """Average the trace, optionally over the lowest weighted eigenvalues."""
    _check_dimensions(tensor, "tensor")
    nstate = tensor.size(-1)
    naverage = nstate if subspace_dim is None else subspace_dim
    if not 0 < naverage <= nstate:
        raise ValueError("subspace_dim must be between 1 and the matrix dimension")
    if weights is None:
        weights = torch.ones(nstate, dtype=tensor.dtype, device=tensor.device)
    elif weights.size() != Size([nstate]):
        raise ValueError(f"weights must have shape ({nstate},), got {weights.size()}")
    weights = weights.to(dtype=tensor.dtype, device=tensor.device)
    if not torch.isfinite(weights).all() or torch.any(weights < 0.0):
        raise ValueError("weights must be finite and non-negative")
    if torch.sum(weights) <= 0.0:
        raise ValueError("at least one weight must be positive")
    weights = nstate * weights / torch.sum(weights)
    sqrt_weights = torch.sqrt(weights)
    weighted_tensor = torch.einsum(
        "i,...ij,j->...ij", sqrt_weights, tensor, sqrt_weights)
    eigenvalues = torch.linalg.eigvalsh(weighted_tensor)
    result = torch.sum(eigenvalues[..., :naverage], dim=-1) / naverage
    return result.unsqueeze(-1).unsqueeze(-1)


__all__ = ["MatrixFunction", "ScalarFunction", "robust_symmetric_eigh", "trace_average"]
