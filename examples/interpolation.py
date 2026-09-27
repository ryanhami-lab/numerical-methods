"""Approximate the Runge function on [-1, 1] using Chebyshev nodes."""

import numpy as np

from numerical_methods import BarycentricInterpolator, chebyshev_nodes

nodes = chebyshev_nodes(21)
values = 1.0 / (1.0 + 25.0 * nodes**2)
interpolant = BarycentricInterpolator(nodes, values)
grid = np.linspace(-1.0, 1.0, 1001)
truth = 1.0 / (1.0 + 25.0 * grid**2)
print(f"Degree bound: {interpolant.degree_bound}")
print(f"Maximum error on a 1,001-point grid: {np.max(np.abs(interpolant(grid) - truth)):.6g}")
print(f"Exact supplied values at nodes: {np.array_equal(interpolant(nodes), values)}")
