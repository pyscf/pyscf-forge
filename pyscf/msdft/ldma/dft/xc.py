"""Analytic Dirac exchange and Chachiyo correlation matrix functionals."""

import math

import torch
from torch import Tensor

from ..nn.functional import MatrixFunction, ScalarFunction


class _LDAExchangeDirac(ScalarFunction):
    coefficient = 2.0 ** (1.0 / 3.0) * 0.75 * (3.0 / math.pi) ** (1.0 / 3.0)

    @staticmethod
    def value(density: Tensor) -> Tensor:
        return -_LDAExchangeDirac.coefficient * torch.abs(density) ** (4.0 / 3.0)

    @staticmethod
    def derivative1(density: Tensor) -> Tensor:
        return -(4.0 / 3.0) * _LDAExchangeDirac.coefficient * torch.abs(density) ** (1.0 / 3.0)


class _LDACorrelationChachiyo(ScalarFunction):
    @staticmethod
    def parameters(spin: int):
        if spin not in (0, 1):
            raise ValueError("spin must be 0 or 1")
        a = (math.log(2.0) - 1.0) / (2.0 * math.pi**2)
        b = 20.4562557
        if spin == 1:
            a *= 0.5
            b = 27.4203609
        return a, (4.0 * math.pi / 3.0) ** (1.0 / 3.0) * b, (4.0 * math.pi / 3.0) ** (2.0 / 3.0) * b

    @staticmethod
    def value(density: Tensor, spin: int) -> Tensor:
        a, b1, b2 = _LDACorrelationChachiyo.parameters(spin)
        rho = torch.abs(density) + 1.0e-15
        argument = 1.0 + b1 * rho ** (1.0 / 3.0) + b2 * rho ** (2.0 / 3.0)
        return a * torch.log(argument) * rho

    @staticmethod
    def derivative1(density: Tensor, spin: int) -> Tensor:
        a, b1, b2 = _LDACorrelationChachiyo.parameters(spin)
        rho = torch.abs(density) + 1.0e-15
        argument = 1.0 + b1 * rho ** (1.0 / 3.0) + b2 * rho ** (2.0 / 3.0)
        return a * (
            ((b1 / 3.0) * rho ** (1.0 / 3.0) + (2.0 * b2 / 3.0) * rho ** (2.0 / 3.0)) / argument
            + torch.log(argument)
        )


class _LDAXCDiracChachiyo(ScalarFunction):
    @staticmethod
    def value(density: Tensor, spin: int) -> Tensor:
        return _LDAExchangeDirac.value(density) + _LDACorrelationChachiyo.value(density, spin)

    @staticmethod
    def derivative1(density: Tensor, spin: int) -> Tensor:
        return _LDAExchangeDirac.derivative1(density) + _LDACorrelationChachiyo.derivative1(density, spin)


class _LDAXCDiracChachiyoUnpolarized(ScalarFunction):
    @staticmethod
    def value(density: Tensor, spin: int) -> Tensor:
        return 2.0 * _LDAExchangeDirac.value(density / 2.0) + _LDACorrelationChachiyo.value(density, spin)

    @staticmethod
    def derivative1(density: Tensor, spin: int) -> Tensor:
        return _LDAExchangeDirac.derivative1(density / 2.0) + _LDACorrelationChachiyo.derivative1(density, spin)


def lda_x_dirac(matrix_density: Tensor, grad_dummy=None, lapl_dummy=None) -> Tensor:
    return MatrixFunction.apply(_LDAExchangeDirac, matrix_density)


def lda_c_chachiyo(matrix_density: Tensor, grad_dummy=None, lapl_dummy=None, spin=0) -> Tensor:
    return MatrixFunction.apply(_LDACorrelationChachiyo, matrix_density, spin)


def lda_xc_dirac_chachiyo(matrix_density: Tensor, grad_dummy=None, lapl_dummy=None, spin=0) -> Tensor:
    return MatrixFunction.apply(_LDAXCDiracChachiyo, matrix_density, spin)


def lda_xc_dirac_chachiyo_unpolarized(matrix_density: Tensor, grad_dummy=None, lapl_dummy=None, spin=0) -> Tensor:
    return MatrixFunction.apply(_LDAXCDiracChachiyoUnpolarized, matrix_density, spin)


__all__ = [
    "lda_x_dirac",
    "lda_c_chachiyo",
    "lda_xc_dirac_chachiyo",
    "lda_xc_dirac_chachiyo_unpolarized",
]
