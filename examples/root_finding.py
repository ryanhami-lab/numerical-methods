"""Solve a scalar equation and inspect why the algorithm stopped."""

import numpy as np

from numerical_methods import bisection, fixed_point, newton, scan_brackets, secant


def f(x):
    return x * x - 2


for result in (
    bisection(f, 1, 2),
    newton(f, lambda x: 2 * x, 1.5),
    secant(f, 1, 2),
    fixed_point(lambda x: 0.5 * (x + 2 / x), 1.5, f=f),
):
    print(result.method, result.root, result.reason, result.residual)
    assert result.converged and abs(result.root - np.sqrt(2)) < 1e-10

print("sin brackets:", scan_brackets(np.sin, 0.5, 10, num=101))

