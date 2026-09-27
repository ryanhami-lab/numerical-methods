"""Barycentric polynomial interpolation (second, "true" barycentric form).

For distinct nodes ``x_0..x_{n-1}`` and values ``y_j`` the interpolating
polynomial of degree <= n-1 is evaluated as

    p(x) = sum_j (w_j / (x - x_j)) y_j  /  sum_j w_j / (x - x_j),

with weights ``w_j = 1 / prod_{k != j} (x_j - x_k)``. Weights are computed
from sums of logarithms of differences and normalized to ``max|w_j| = 1``.
The common factor cancels in the formula. Node sets whose relative weights
underflow in float64 are rejected.

Practical limitations
---------------------
* Numerical stability of the evaluation does not make a bad interpolant
  good: with equally spaced nodes the Lebesgue constant grows exponentially
  with degree, so the Runge phenomenon and amplification of data errors
  remain.  Chebyshev nodes (:func:`chebyshev_nodes`) avoid this.
* The second form is forward stable for evaluation inside ``[min x, max x]``
  when the node set has a small Lebesgue constant (Higham 2004);
  extrapolation outside the node interval is allowed but not recommended.
* Weight computation costs O(n^2) time and memory; evaluation costs
  O(n) per point and is processed in blocks.
"""

from __future__ import annotations

import numpy as np

from ._validation import InvalidInputError, as_real_array, as_real_scalar, check_positive_int

__all__ = ["BarycentricInterpolator", "barycentric_weights", "chebyshev_nodes"]

_BLOCK_ELEMENTS = 1 << 20  # max evaluation-points x nodes per block


def barycentric_weights(nodes) -> np.ndarray:
    """Barycentric weights for distinct 1-D ``nodes``, normalized to ``max|w| = 1``."""
    x = as_real_array(nodes, "nodes", ndim=1)
    n = x.size
    if n == 1:
        return np.ones(1)
    if np.unique(x).size != n:
        raise InvalidInputError("nodes must be distinct.")
    with np.errstate(over="ignore"):
        diff = x[:, None] - x[None, :]
    np.fill_diagonal(diff, 1.0)
    sign = np.prod(np.sign(diff), axis=1)
    logdiff = np.log(np.abs(diff))
    overflow = np.isinf(diff)
    if overflow.any():
        halfdiff = x[:, None] * 0.5 - x[None, :] * 0.5
        logdiff[overflow] = np.log(np.abs(halfdiff[overflow])) + np.log(2.0)
    logw = -np.sum(logdiff, axis=1)
    weights = sign * np.exp(logw - logw.max())
    if np.any(weights == 0.0):
        raise InvalidInputError("nodes produce relative barycentric weights outside the float64 range.")
    return weights


class BarycentricInterpolator:
    """Polynomial interpolant through ``(nodes[j], values[j])``.

    Parameters
    ----------
    nodes : array_like, shape (n,)
        Distinct, finite, real nodes (any order).  ``n >= 1``.
    values : array_like, shape (n,)
        Finite real data values.

    Calling ``p(x)`` accepts a scalar (returns ``float``) or an array of any
    shape (returns an array of the same shape).  Points that coincide
    exactly with a node return that node's value exactly.  Non-finite
    evaluation points raise :class:`InvalidInputError`.

    Raises
    ------
    InvalidInputError
        On duplicate nodes, shape mismatch, empty or non-finite input, or
        relative weights that underflow in float64.
    """

    def __init__(self, nodes, values):
        x = as_real_array(nodes, "nodes", ndim=1)
        y = as_real_array(values, "values", ndim=1)
        if x.shape != y.shape:
            raise InvalidInputError(
                f"nodes and values must have equal shape; got {x.shape} and {y.shape}."
            )
        self.nodes = x.copy()
        self.values = y.copy()
        self.weights = barycentric_weights(x)
        self.nodes.flags.writeable = False
        self.values.flags.writeable = False
        self.weights.flags.writeable = False

    @property
    def degree_bound(self) -> int:
        """Maximum polynomial degree ``n - 1``."""
        return self.nodes.size - 1

    def __call__(self, x):
        is_scalar = np.ndim(x) == 0
        xa = as_real_array(x, "x", allow_empty=True)
        flat = xa.ravel()
        out = np.empty_like(flat)
        block = max(1, _BLOCK_ELEMENTS // self.nodes.size)
        for start in range(0, flat.size, block):
            out[start:start + block] = self._eval_block(flat[start:start + block])
        out = out.reshape(xa.shape)
        return float(out) if is_scalar else out

    def _eval_block(self, x: np.ndarray) -> np.ndarray:
        exact = x[:, None] == self.nodes[None, :]
        hit = exact.any(axis=1)
        res = np.empty(x.size)
        if hit.any():
            res[hit] = self.values[np.argmax(exact[hit], axis=1)]
        if (~hit).any():
            points = x[~hit, None]
            with np.errstate(over="ignore"):
                diff = points - self.nodes[None, :]
            overflow_rows = np.isinf(diff).any(axis=1)
            if overflow_rows.any():
                diff[overflow_rows] = points[overflow_rows] * 0.5 - self.nodes[None, :] * 0.5
            # A common row scale cancels; this avoids reciprocal overflow
            # even one subnormal ULP from a node.
            nearest = np.min(np.abs(diff), axis=1, keepdims=True)
            c = self.weights * (nearest / diff)
            value_scale = float(np.max(np.abs(self.values))) or 1.0
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                res[~hit] = ((c @ (self.values / value_scale)) / c.sum(axis=1)) * value_scale
        return res


def chebyshev_nodes(n: int, a: float = -1.0, b: float = 1.0, *, kind: int = 2) -> np.ndarray:
    """Chebyshev nodes on ``[a, b]`` in increasing order.

    ``kind=2``: Chebyshev extreme points ``cos(pi j/(n-1))``, j=0..n-1
    (include the endpoints; ``n >= 2``).  ``kind=1``: Chebyshev roots
    ``cos(pi (2j+1)/(2n))`` (interior points; ``n >= 1``).
    ``kind`` must be an integer. Raises :class:`InvalidInputError` if
    the interval is too narrow to represent distinct float64 nodes.
    """
    n = check_positive_int(n, "n")
    kind = check_positive_int(kind, "kind")
    a = as_real_scalar(a, "a")
    b = as_real_scalar(b, "b")
    if not a < b:
        raise InvalidInputError("chebyshev_nodes requires a < b.")
    if kind == 2:
        if n < 2:
            raise InvalidInputError("kind=2 requires n >= 2.")
        t = np.cos(np.pi * np.arange(n - 1, -1, -1) / (n - 1))
    elif kind == 1:
        t = np.cos(np.pi * (2 * np.arange(n - 1, -1, -1) + 1) / (2 * n))
    else:
        raise InvalidInputError("kind must be 1 or 2.")
    nodes = ((1.0 - t) * 0.5) * a + ((1.0 + t) * 0.5) * b
    if kind == 2:
        nodes[0], nodes[-1] = a, b
    if np.unique(nodes).size != n:
        raise InvalidInputError("interval is too narrow to represent distinct Chebyshev nodes in float64.")
    return nodes
