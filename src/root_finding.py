"""Deprecated compatibility shim for the original ``src/root_finding.py`` API.

The notebook and the original tests import this module after adding
``src/`` to ``sys.path``.  It forwards to :mod:`numerical_methods.root_finding`
and returns the original result shape (``history``/``residuals`` lists and a
single ``tol``).  New code should use ``numerical_methods`` directly.

Behavioural differences from the pre-1.0 implementation (bug fixes):
an exact root at a bisection endpoint is accepted instead of rejected;
bisection's sign test no longer overflows; secant and Newton no longer
re-evaluate ``f`` redundantly; non-finite values stop the iteration.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from numerical_methods import root_finding as _rf

__all__ = ["RootResult", "bisection", "newton", "secant", "fixed_point"]


@dataclass
class RootResult:
    root: float
    iterations: int
    converged: bool
    residual: float
    history: list[float]
    residuals: list[float]
    method: str
    message: str


def _warn():
    warnings.warn(
        "The top-level 'root_finding' module is deprecated; use 'numerical_methods' instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def _legacy(res: _rf.RootResult) -> RootResult:
    h = res.history or {"x": [], "residual": []}
    return RootResult(
        root=res.root, iterations=res.iterations, converged=res.converged,
        residual=res.residual, history=[float(v) for v in h["x"]],
        residuals=[float(v) for v in h["residual"]], method=res.method,
        message=res.message,
    )


def bisection(f, a, b, tol=1e-12, max_iter=100):
    _warn()
    return _legacy(_rf.bisection(f, a, b, xtol=2 * tol, rtol=0.0, ftol=tol,
                                 max_iter=max_iter, record_history=True))


def newton(f, df, x0, tol=1e-12, max_iter=50):
    _warn()
    return _legacy(_rf.newton(f, df, x0, xtol=0.0, rtol=0.0, ftol=tol,
                              max_iter=max_iter, record_history=True))


def secant(f, x0, x1, tol=1e-12, max_iter=50):
    _warn()
    return _legacy(_rf.secant(f, x0, x1, xtol=0.0, rtol=0.0, ftol=tol,
                              max_iter=max_iter, record_history=True))


def fixed_point(g, x0, tol=1e-12, max_iter=100):
    _warn()
    return _legacy(_rf.fixed_point(g, x0, xtol=tol, rtol=0.0, max_iter=max_iter,
                                   record_history=True))
