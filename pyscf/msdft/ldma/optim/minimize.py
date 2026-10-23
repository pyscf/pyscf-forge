"""Unconstrained and log-barrier Newton, steepest-descent, and BFGS methods."""

import numpy as np
import numpy.linalg as la

from . import finite_differences


def line_search_backtracking(xk, fk, grad_fk, pk, func, constraints=None,
                             a0=1.0, rho=0.3, c=0.0001, lmax=100):
    a = a0
    directional_derivative = np.dot(grad_fk, pk)
    if directional_derivative > 0.0:
        raise ValueError("search direction is not a descent direction")
    for _ in range(lmax):
        candidate = xk + a * pk
        if constraints is not None and np.any(constraints(candidate)[0] <= 0.0):
            a *= rho
            continue
        if func(candidate) <= fk + c * a * directional_derivative:
            return candidate
        a *= rho
    raise RuntimeError("Linesearch failed to satisfy the sufficient decrease condition")


def line_search_wolfe(xk, fk, grad_fk, pk, func, constraints=None,
                      max_steplen=None, a0=1.0, amax=50.0,
                      c1=0.0001, c2=0.9, lmax=100, debug=0):
    if not 0 < c1 < c2 < 1:
        raise ValueError("Wolfe constants must satisfy 0 < c1 < c2 < 1")
    s0 = fk
    ds0 = np.dot(grad_fk, pk)
    if ds0 > 0.0:
        raise ValueError("search direction is not a descent direction")

    def scalar_function(a):
        value, gradient = func(xk + a * pk)
        return value, np.dot(gradient, pk)

    def zoom(alo, ahi, slo):
        for _ in range(lmax):
            aj = 0.5 * (alo + ahi)
            sj, dsj = scalar_function(aj)
            if sj > s0 + c1 * aj * ds0 or sj >= slo:
                ahi = aj
            else:
                if abs(dsj) <= c2 * abs(ds0):
                    return aj
                if dsj * (ahi - alo) >= 0.0:
                    ahi = alo
                alo, slo = aj, sj
        if debug:
            print("WARNING: zoom did not satisfy the Wolfe conditions")
        return 0.5 * (alo + ahi)

    def feasible(a):
        return constraints is None or np.all(constraints(xk + a * pk)[0] > 0.0)

    if max_steplen is not None:
        maximum = max_steplen(xk, pk)
        if maximum != np.inf:
            amax = maximum
    else:
        while not feasible(amax):
            amax *= 0.99
    if amax <= 0.0:
        raise RuntimeError("No positive feasible step length")
    if a0 >= amax:
        a0 = 0.5 * amax

    previous_a = 0.0
    previous_s = s0
    a = a0
    for iteration in range(1, lmax):
        value, derivative = scalar_function(a)
        if value > s0 + c1 * a * ds0 or (iteration > 1 and value >= previous_s):
            return xk + zoom(previous_a, a, previous_s) * pk
        if abs(derivative) <= c2 * abs(ds0):
            return xk + a * pk
        if derivative >= 0.0:
            return xk + zoom(a, previous_a, value) * pk
        previous_a, previous_s = a, value
        a = 2.0 * previous_a
        if a >= amax:
            a = 0.5 * (previous_a + amax)
    raise RuntimeError("Linesearch failed to satisfy the Wolfe conditions")


def bfgs_update(inv_hessian, step, gradient_difference, iteration):
    identity = np.eye(len(step))
    if iteration < 1:
        raise ValueError("BFGS update iteration must be at least 1")
    if iteration == 1:
        return np.dot(gradient_difference, step) / np.dot(
            gradient_difference, gradient_difference) * identity
    rho = 1.0 / np.dot(gradient_difference, step)
    left = identity - rho * np.outer(step, gradient_difference)
    right = identity - rho * np.outer(gradient_difference, step)
    return left.dot(inv_hessian).dot(right) + rho * np.outer(step, step)


def log_barrier(values, gradients):
    return -np.sum(np.log(values)), -np.dot(1.0 / values, gradients)


class OptimizationResult:
    def __init__(self, x, fun, grad, nit):
        self.x = x
        self.fun = fun
        self.grad = grad
        self.nit = nit


def minimize(objfunc, x0, method="BFGS", line_search_method="Wolfe",
             constraints=None, max_steplen=None, callback=None,
             maxiter=100000, gtol=1.0e-6, ftol=1.0e-8, debug=0):
    if ftol <= 0.0 or gtol <= 0.0:
        raise ValueError("ftol and gtol must be positive")
    if method not in ("Newton", "Steepest Descent", "BFGS"):
        raise ValueError("method must be 'Newton', 'Steepest Descent', or 'BFGS'")
    if line_search_method not in ("Armijo", "Wolfe"):
        raise ValueError("line_search_method must be 'Armijo' or 'Wolfe'")
    nvariable = len(x0)

    def barrier(x):
        if constraints is None:
            return 0.0, np.zeros(nvariable)
        values, gradients = constraints(x)
        barrier_value, barrier_gradient = log_barrier(values, gradients)
        return 1.0e-5 * barrier_value, 1.0e-5 * barrier_gradient

    def function(x):
        value = objfunc(x, requires_grad=False)
        return value + barrier(x)[0]

    def gradient(x):
        _, objective_gradient = objfunc(x, requires_grad=True)
        return objective_gradient + barrier(x)[1]

    def function_gradient(x):
        value, objective_gradient = objfunc(x, requires_grad=True)
        barrier_value, barrier_gradient = barrier(x)
        return value + barrier_value, objective_gradient + barrier_gradient

    xk = np.asarray(x0)
    fk, grad_fk = function_gradient(xk)
    inverse_hessian = None
    step = None
    gradient_difference = None
    epsilon = np.finfo(float).eps

    for iteration in range(maxiter):
        if method == "Newton":
            hessian = finite_differences.numerical_hessian_G(gradient, xk)
            positive_hessian, _ = modified_cholesky(hessian)
            direction = la.solve(positive_hessian, -grad_fk)
        elif method == "Steepest Descent":
            direction = -grad_fk
        else:
            if iteration == 0:
                inverse_hessian = np.eye(nvariable)
            else:
                if np.dot(gradient_difference, step) <= 0.0 and debug:
                    print("WARNING: BFGS curvature condition y.s > 0 was not satisfied")
                inverse_hessian = bfgs_update(
                    inverse_hessian, step, gradient_difference, iteration)
            direction = np.dot(inverse_hessian, -grad_fk)

        if line_search_method == "Armijo":
            next_x = line_search_backtracking(
                xk, fk, grad_fk, direction, function, constraints=constraints)
        else:
            next_x = line_search_wolfe(
                xk, fk, grad_fk, direction, function_gradient,
                constraints=constraints, max_steplen=max_steplen, debug=debug)
        next_f, next_gradient = function_gradient(next_x)
        function_change = abs(next_f - fk)
        gradient_norm = la.norm(next_gradient)
        converged = function_change < ftol and gradient_norm < gtol
        if function_change < epsilon:
            converged = True
        step = next_x - xk
        gradient_difference = next_gradient - grad_fk
        xk, fk, grad_fk = next_x, next_f, next_gradient
        if callback is not None:
            callback(xk)
        if debug:
            print("k=%d f(x)=%.10f |step|=%e |df|=%e |grad|=%e" % (
                iteration, fk, la.norm(step), function_change, gradient_norm))
        if converged:
            return OptimizationResult(xk, fk, grad_fk, iteration)
    raise RuntimeError("No convergence in %s method after %d iterations" % (method, maxiter))


def modified_cholesky(matrix, beta=0.01):
    nrow, ncol = matrix.shape
    if nrow != ncol:
        raise ValueError("matrix must be square")
    minimum_diagonal = np.diag(matrix).min()
    tau = 0.0 if minimum_diagonal > 0.0 else -minimum_diagonal + beta
    identity = np.eye(nrow)
    while True:
        try:
            la.cholesky(matrix + tau * identity)
            return matrix + tau * identity, tau
        except la.LinAlgError:
            tau = max(2.0 * tau, beta)


__all__ = ["OptimizationResult", "bfgs_update", "line_search_backtracking",
           "line_search_wolfe", "log_barrier", "minimize", "modified_cholesky"]
