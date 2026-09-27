"""Brownian motion and geometric Brownian motion sampled on a time grid.

Both processes are simulated **exactly at the grid points** from their
Gaussian transition laws (no SDE discretization error at the grid points):

* Brownian motion with drift ``X_t = x0 + mu (t - t0) + sigma W_{t - t0}``,
  using independent increments ``N(mu dt, sigma^2 dt)``.
* Geometric Brownian motion ``S_t = s0 exp((mu - sigma^2/2)(t - t0) + sigma W_{t - t0})``,
  the explicit solution of ``dS = mu S dt + sigma S dW``.

``t0 = times[0]`` is the start time and column 0 of the output equals the
initial value.  Output arrays have shape ``(n_paths, len(times))`` (sample
axis 0), which matches the sampler contract of
:func:`numerical_methods.monte_carlo`.

Grid-point sampling is exact only for functionals of the values at the grid
points.  Quantities of the continuous path *between* grid points
(running maximum, barrier crossing, time integrals) computed from these
samples carry a time-discretization bias that shrinks with the grid
spacing; Monte Carlo confidence intervals do not include it.
"""

from __future__ import annotations

import numpy as np

from ._validation import (
    InvalidInputError,
    as_real_array,
    as_real_scalar,
    check_callable,
    check_nonneg,
    check_positive_int,
)

__all__ = ["brownian_motion", "geometric_brownian_motion", "path_sampler",
           "gbm_terminal_moments", "bm_moments"]


def _grid(times) -> tuple[np.ndarray, np.ndarray]:
    t = as_real_array(times, "times", ndim=1)
    if t.size < 2:
        raise InvalidInputError("times must contain at least two points.")
    if t[0] < 0:
        raise InvalidInputError("times must be non-negative.")
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise InvalidInputError("times must be strictly increasing.")
    return t, dt


def _check_rng(rng):
    if not isinstance(rng, np.random.Generator):
        raise InvalidInputError("rng must be a numpy.random.Generator.")


def _sigma(sigma) -> float:
    s = as_real_scalar(sigma, "sigma")
    if s < 0:
        raise InvalidInputError("sigma must be >= 0.")
    return s


def _brownian_increments(dt: np.ndarray, n_paths: int, rng) -> np.ndarray:
    """Cumulative standard Brownian motion at grid points, shape (n_paths, len(dt))."""
    z = rng.standard_normal((n_paths, dt.size))
    return np.cumsum(z * np.sqrt(dt), axis=1)


def brownian_motion(times, n_paths: int, *, rng: np.random.Generator, x0: float = 0.0,
                    mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    """Sample Brownian motion with drift at ``times``.

    Parameters
    ----------
    times : array_like, shape (m,), m >= 2
        Strictly increasing, non-negative, finite; ``times[0]`` is the start.
    n_paths : int >= 1
    rng : numpy.random.Generator
    x0, mu : finite floats; sigma : finite float >= 0

    Returns
    -------
    ndarray, shape (n_paths, m)
        ``X[:, 0] == x0``; ``E[X_t] = x0 + mu (t - t0)``,
        ``Var[X_t] = sigma^2 (t - t0)``.  With ``sigma == 0`` the paths are
        exactly ``x0 + mu (t - t0)`` (random numbers are still drawn).
    """
    t, dt = _grid(times)
    n_paths = check_positive_int(n_paths, "n_paths")
    _check_rng(rng)
    x0 = as_real_scalar(x0, "x0")
    mu = as_real_scalar(mu, "mu")
    sigma = _sigma(sigma)
    out = np.empty((n_paths, t.size))
    out[:, 0] = x0
    w = _brownian_increments(dt, n_paths, rng)
    out[:, 1:] = x0 + mu * (t[1:] - t[0]) + sigma * w
    return out


def geometric_brownian_motion(times, n_paths: int, *, rng: np.random.Generator,
                              s0: float = 1.0, mu: float = 0.0,
                              sigma: float = 1.0) -> np.ndarray:
    """Sample geometric Brownian motion at ``times`` using the exact solution.

    ``s0 > 0``, ``sigma >= 0``.  Returns shape ``(n_paths, m)`` with
    ``S[:, 0] == s0``, ``E[S_t] = s0 exp(mu tau)`` and
    ``Var[S_t] = s0^2 exp(2 mu tau) (exp(sigma^2 tau) - 1)``, ``tau = t - t0``.
    With ``sigma == 0`` the paths equal ``s0 exp(mu tau)``.
    """
    t, dt = _grid(times)
    n_paths = check_positive_int(n_paths, "n_paths")
    _check_rng(rng)
    s0 = as_real_scalar(s0, "s0")
    if s0 <= 0:
        raise InvalidInputError("s0 must be > 0.")
    mu = as_real_scalar(mu, "mu")
    sigma = _sigma(sigma)
    out = np.empty((n_paths, t.size))
    out[:, 0] = s0
    w = _brownian_increments(dt, n_paths, rng)
    out[:, 1:] = s0 * np.exp((mu - 0.5 * sigma**2) * (t[1:] - t[0]) + sigma * w)
    return out


def path_sampler(process, times, **params):
    """Adapt a process function to the Monte Carlo sampler contract.

    Returns ``sampler(rng, n) = process(times, n, rng=rng, **params)``, so
    paths are generated batch by batch inside :func:`monte_carlo` and never
    all retained at once.  ``process`` is :func:`brownian_motion` or
    :func:`geometric_brownian_motion` (or any function with that signature).
    """
    check_callable(process, "process")
    t, _ = _grid(times)
    t = t.copy()  # private copy, made only after validating the original dtype

    def sampler(rng, n):
        return process(t, n, rng=rng, **params)

    return sampler


def bm_moments(tau: float, *, mu: float = 0.0, sigma: float = 1.0, x0: float = 0.0):
    """Analytic ``(mean, variance)`` after finite ``tau >= 0`` with ``sigma >= 0``."""
    tau = check_nonneg(tau, "tau")
    mu = as_real_scalar(mu, "mu")
    sigma = _sigma(sigma)
    x0 = as_real_scalar(x0, "x0")
    return x0 + mu * tau, sigma**2 * tau


def gbm_terminal_moments(tau: float, *, s0: float = 1.0, mu: float = 0.0, sigma: float = 1.0):
    """Analytic ``(mean, variance)`` after finite ``tau >= 0``; ``s0 > 0``, ``sigma >= 0``."""
    tau = check_nonneg(tau, "tau")
    s0 = as_real_scalar(s0, "s0")
    if s0 <= 0:
        raise InvalidInputError("s0 must be > 0.")
    mu = as_real_scalar(mu, "mu")
    sigma = _sigma(sigma)
    m = s0 * np.exp(mu * tau)
    return float(m), float(m**2 * np.expm1(sigma**2 * tau))
