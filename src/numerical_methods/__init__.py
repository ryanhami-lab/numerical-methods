"""numerical_methods: small, tested numerical-analysis and Monte Carlo library.

Supported release scope (real float64 only):

* :mod:`~numerical_methods.root_finding` - bisection, Newton, secant,
  fixed-point iteration for scalar equations; array-based bracket scan.
* :mod:`~numerical_methods.linear_systems` - Gaussian elimination and
  reusable LU factorization with partial pivoting; triangular solves;
  residual / backward-error diagnostics.
* :mod:`~numerical_methods.interpolation` - barycentric polynomial
  interpolation and Chebyshev nodes.
* :mod:`~numerical_methods.monte_carlo` - fixed-budget batched Monte Carlo
  estimator with stable variance accumulation.
* :mod:`~numerical_methods.stochastic` - Brownian motion and geometric
  Brownian motion on a user time grid.
"""

from ._validation import InvalidInputError, InvalidSampleError, SingularMatrixError
from .interpolation import BarycentricInterpolator, barycentric_weights, chebyshev_nodes
from .linear_systems import (
    LUFactorization,
    ResidualDiagnostics,
    back_substitution,
    factorization_error,
    forward_error,
    forward_substitution,
    gaussian_elimination,
    lu_factor,
    residual_diagnostics,
    solve,
)
from .monte_carlo import MonteCarloResult, RunningMoments, monte_carlo
from .root_finding import RootResult, bisection, fixed_point, newton, scan_brackets, secant
from .stochastic import (
    bm_moments,
    brownian_motion,
    gbm_terminal_moments,
    geometric_brownian_motion,
    path_sampler,
)

__version__ = "1.0.0"

__all__ = [
    "InvalidInputError", "InvalidSampleError", "SingularMatrixError",
    "RootResult", "bisection", "newton", "secant", "fixed_point", "scan_brackets",
    "LUFactorization", "lu_factor", "gaussian_elimination", "solve",
    "forward_substitution", "back_substitution", "ResidualDiagnostics",
    "residual_diagnostics", "factorization_error", "forward_error",
    "BarycentricInterpolator", "barycentric_weights", "chebyshev_nodes",
    "MonteCarloResult", "RunningMoments", "monte_carlo",
    "brownian_motion", "geometric_brownian_motion", "path_sampler",
    "bm_moments", "gbm_terminal_moments",
    "__version__",
]
