import numpy


def numerical_hessian_G(gradient, x0, h=1.0e-8):
    hessian = numpy.zeros((len(x0), len(x0)))
    for index in range(len(x0)):
        direction = numpy.zeros(len(x0))
        direction[index] = 1.0
        hessian[index] = (gradient(x0 + h * direction) - gradient(x0 - h * direction)) / (2.0 * h)
    return 0.5 * (hessian + hessian.T)


def condition_number(matrix):
    return numpy.linalg.norm(numpy.linalg.inv(matrix)) * numpy.linalg.norm(matrix)
