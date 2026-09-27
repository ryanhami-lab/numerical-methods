"""Solve several right-hand sides while reusing one LU factorization."""

import numpy as np

from numerical_methods import factorization_error, forward_error, lu_factor, residual_diagnostics

a = np.array([[1e-20, 1.0], [1.0, 1.0]])
expected = np.array([[1.0, 2.0], [1.0, -1.0]])
b = a @ expected
factor = lu_factor(a)
x = factor.solve(b)
print("Solutions (one per column):")
print(x)
print("Permutation satisfying a[perm] = L @ U:", factor.perm)
print("Relative forward error:", forward_error(x, expected))
print("Relative factorization error:", factorization_error(a, factor))
print("First-column residual diagnostics:", residual_diagnostics(a, x[:, 0], b[:, 0]))
