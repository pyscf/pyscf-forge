"""Composite analytic LDMA functionals."""

from abc import ABC

from .xc import lda_c_chachiyo, lda_x_dirac, lda_xc_dirac_chachiyo_unpolarized


class PureXCFunctional(ABC):
    exact_exchange = None

    def __init__(self, mol=None):
        pass

    def exchange(self, matrix_density, grad_density=None, lapl_density=None):
        return 0.0 * matrix_density

    def correlation(self, matrix_density, grad_density=None, lapl_density=None):
        return 0.0 * matrix_density

    def exchange_correlation(self, matrix_density, grad_density=None, lapl_density=None):
        return self.exchange(matrix_density, grad_density, lapl_density) + self.correlation(
            matrix_density, grad_density, lapl_density)


class LDA(PureXCFunctional):
    def exchange(self, matrix_density, grad_density=None, lapl_density=None):
        return lda_x_dirac(matrix_density)

    def correlation(self, matrix_density, grad_density=None, lapl_density=None):
        return lda_c_chachiyo(matrix_density)

    def exchange_correlation(self, matrix_density, grad_density=None, lapl_density=None):
        return lda_xc_dirac_chachiyo_unpolarized(matrix_density)


__all__ = ["PureXCFunctional", "LDA"]
