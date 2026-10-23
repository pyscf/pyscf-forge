"""Torch optimizers and host-synchronization telemetry for LDMA."""

import numpy
import torch

from .minimize import minimize


class OptimizerTelemetry:
    def __init__(self):
        self.numpy_parameter_transfers = 0
        self.numpy_gradient_transfers = 0
        self.scalar_item_calls = 0
        self.closure_calls = 0

    def as_dict(self):
        return vars(self).copy()


class ClosureProfiler:
    def __init__(self):
        self.telemetry = OptimizerTelemetry()

    def wrap(self, closure):
        def profiled_closure(*args, **kwargs):
            self.telemetry.closure_calls += 1
            return closure(*args, **kwargs)
        return profiled_closure

    def as_dict(self):
        return self.telemetry.as_dict()


def parameters_to_vector(parameters, telemetry=None):
    values = []
    for parameter in parameters:
        values.append(parameter.detach().reshape(-1).numpy(force=True))
        if telemetry is not None:
            telemetry.numpy_parameter_transfers += 1
    return numpy.hstack(values)


def parameter_gradients(parameters, telemetry=None):
    values = []
    for parameter in parameters:
        values.append(parameter.grad.detach().reshape(-1).numpy(force=True))
        if telemetry is not None:
            telemetry.numpy_gradient_transfers += 1
    return numpy.hstack(values)


def vector_to_parameters(vector, parameters):
    offset = 0
    with torch.no_grad():
        for parameter in parameters:
            count = parameter.numel()
            value = torch.as_tensor(vector[offset:offset + count], dtype=parameter.dtype, device=parameter.device)
            parameter.copy_(value.reshape(parameter.size()))
            offset += count


class WrappedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, method="BFGS", line_search_method="Wolfe",
                 constraints=None, max_steplen=None, callback=None,
                 maxiter=100000, gtol=1.0e-6, ftol=1.0e-8, debug=0):
        defaults = {
            "method": method,
            "line_search_method": line_search_method,
            "constraints": constraints,
            "max_steplen": max_steplen,
            "callback": callback,
            "maxiter": maxiter,
            "gtol": gtol,
            "ftol": ftol,
            "debug": debug,
        }
        super().__init__(params, defaults)
        if len(self.param_groups) != 1:
            raise ValueError("per-parameter options are unsupported")
        self._params = self.param_groups[0]["params"]
        self.telemetry = OptimizerTelemetry()

    def step(self, closure):
        def objective(vector, requires_grad=True):
            vector_to_parameters(vector, self._params)
            self.zero_grad()
            value = closure()
            self.telemetry.closure_calls += 1
            if requires_grad:
                value.backward()
                gradient = parameter_gradients(self._params, self.telemetry)
                self.telemetry.scalar_item_calls += 1
                return value.item(), gradient
            self.telemetry.scalar_item_calls += 1
            return value.item()

        initial = parameters_to_vector(self._params, self.telemetry)
        result = minimize(objective, initial, **self.defaults)
        vector_to_parameters(result.x, self._params)
        return result.fun


class TorchLBFGSOptimizer(torch.optim.LBFGS):
    def __init__(self, params, maxiter=100, gtol=1.0e-6, line_search_fn="strong_wolfe", **kwargs):
        super().__init__(params, max_iter=maxiter, tolerance_grad=gtol,
                         line_search_fn=line_search_fn, **kwargs)
        self.telemetry = OptimizerTelemetry()

    def step(self, closure):
        def profiled():
            with torch.enable_grad():
                self.zero_grad()
                value = closure()
                self.telemetry.closure_calls += 1
                value.backward()
                return value
        return super().step(profiled)


__all__ = ["ClosureProfiler", "OptimizerTelemetry", "WrappedOptimizer", "TorchLBFGSOptimizer"]
