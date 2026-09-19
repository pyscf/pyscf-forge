import numpy
import pytest
import torch

from pyscf.msdft.ldma.optim.minimize import minimize
from pyscf.msdft.ldma.optim.torch_optimizer import (TorchLBFGSOptimizer,
                                                    WrappedOptimizer)


def rosenbrock(x, requires_grad=True):
    value = (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2
    if not requires_grad:
        return value
    gradient = numpy.array([
        -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] ** 2),
        200.0 * (x[1] - x[0] ** 2),
    ])
    return value, gradient


def circle_constraint(x):
    return numpy.array([1.0 - x[0] ** 2 - x[1] ** 2]), numpy.array([[-2*x[0], -2*x[1]]])


@pytest.mark.parametrize("method", ["Newton", "BFGS", "Steepest Descent"])
@pytest.mark.parametrize("line_search", ["Armijo", "Wolfe"])
def test_original_minimizer_algorithms(method, line_search):
    result = minimize(rosenbrock, numpy.zeros(2), method=method,
                      line_search_method=line_search,
                      gtol=1.0e-7 if method == "Steepest Descent" else 1.0e-6)
    numpy.testing.assert_allclose(result.x, [1.0, 1.0], atol=1.0e-4, rtol=1.0e-4)


@pytest.mark.parametrize("method", ["Newton", "BFGS", "Steepest Descent"])
def test_original_minimizer_constraints(method):
    result = minimize(rosenbrock, numpy.zeros(2), method=method,
                      constraints=circle_constraint, gtol=1.0e-7)
    numpy.testing.assert_allclose(result.x, [0.7864, 0.6177], atol=1.0e-4)
    numpy.testing.assert_allclose(result.fun, 0.0457, atol=1.0e-4)


def test_unsupported_optimizer_modes_fail_explicitly():
    with pytest.raises(ValueError, match="method"):
        minimize(rosenbrock, numpy.zeros(2), method="CG")
    with pytest.raises(ValueError, match="line_search"):
        minimize(rosenbrock, numpy.zeros(2), line_search_method="fake")


@pytest.mark.parametrize("optimizer_class", [WrappedOptimizer, TorchLBFGSOptimizer])
def test_torch_optimizer_wrappers_and_telemetry(optimizer_class):
    parameter = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.double))
    optimizer = optimizer_class([parameter], maxiter=20, gtol=1.0e-8)

    def closure():
        optimizer.zero_grad()
        return torch.sum((parameter - 1.0) ** 2)

    optimizer.step(closure)
    torch.testing.assert_close(parameter, torch.ones_like(parameter), atol=1.0e-5, rtol=1.0e-5)
    assert optimizer.telemetry.closure_calls > 0
