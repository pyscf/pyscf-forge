import torch
import numpy
from pyscf.dft import libxc

from pyscf.msdft.ldma import (lda_c_chachiyo, lda_x_dirac,
                              lda_xc_dirac_chachiyo,
                              lda_xc_dirac_chachiyo_unpolarized)


def positive_density(nbatch=3, nstate=3):
    torch.manual_seed(12)
    factor = torch.randn(nbatch, nstate, nstate, dtype=torch.double)
    return factor @ factor.transpose(-1, -2) + 0.2 * torch.eye(nstate, dtype=torch.double)


def test_fused_functionals_match_analytic_terms():
    density = positive_density()
    torch.testing.assert_close(
        lda_xc_dirac_chachiyo(density),
        lda_x_dirac(density) + lda_c_chachiyo(density),
    )
    torch.testing.assert_close(
        lda_xc_dirac_chachiyo_unpolarized(density),
        2.0 * lda_x_dirac(density / 2.0) + lda_c_chachiyo(density),
    )


def test_matrix_function_backward_handles_repeated_eigenvalues():
    density = torch.eye(3, dtype=torch.double).repeat(2, 1, 1).requires_grad_(True)
    lda_xc_dirac_chachiyo(density).sum().backward()
    assert torch.isfinite(density.grad).all()


def test_matrix_function_gradcheck():
    density = positive_density(1, 2).requires_grad_(True)
    def symmetric_xc(value):
        return lda_xc_dirac_chachiyo(0.5 * (value + value.transpose(-1, -2)))

    assert torch.autograd.gradcheck(symmetric_xc, (density,),
                                    eps=1.0e-6, atol=1.0e-5, rtol=1.0e-4)


def test_scalar_functionals_match_libxc():
    density = torch.linspace(0.05, 2.0, 12, dtype=torch.double)[:, None, None]
    rho = density[:, 0, 0].numpy()[None, :]
    exchange_reference = libxc.eval_xc("LDA_X,", rho)[0] * rho[0]
    correlation_reference = libxc.eval_xc(",LDA_C_CHACHIYO", rho)[0] * rho[0]
    torch.testing.assert_close(
        2.0 * lda_x_dirac(density / 2.0)[:, 0, 0],
        torch.from_numpy(exchange_reference), rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(
        lda_c_chachiyo(density)[:, 0, 0],
        torch.from_numpy(correlation_reference), rtol=1.0e-5, atol=1.0e-5)


def test_matrix_function_is_orthogonally_equivariant():
    density = positive_density(2, 3)
    generator = torch.randn(3, 3, dtype=torch.double)
    rotation = torch.matrix_exp(generator - generator.T)
    transformed = rotation @ density @ rotation.T
    expected = rotation @ lda_xc_dirac_chachiyo(density) @ rotation.T
    torch.testing.assert_close(lda_xc_dirac_chachiyo(transformed), expected)
