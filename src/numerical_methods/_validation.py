"""Shared input validation helpers and exception types.

All public routines raise :class:`InvalidInputError` (a ``ValueError``
subclass) for API misuse: wrong shapes, non-real or non-finite inputs,
invalid tolerances, invalid brackets, and so on.  Ordinary numerical
non-convergence is *not* an exception; it is reported through result
objects (for example ``RootResult.converged`` / ``RootResult.reason``).
"""

from __future__ import annotations

import numbers

import numpy as np

__all__ = ["InvalidInputError", "SingularMatrixError", "InvalidSampleError"]


class InvalidInputError(ValueError):
    """Raised when arguments violate a documented API contract."""


class SingularMatrixError(np.linalg.LinAlgError):
    """Raised when a factorization or triangular solve meets a negligible pivot."""


class InvalidSampleError(ValueError):
    """Raised when a Monte Carlo sampler/quantity returns invalid output."""


def as_real_array(x, name: str, *, ndim: int | tuple[int, ...] | None = None,
                  allow_empty: bool = False, require_finite: bool = True) -> np.ndarray:
    """Convert ``x`` to a float64 array, rejecting non-real or non-numeric data.

    Integers and floats are accepted; booleans, complex numbers, strings and
    object arrays are rejected rather than silently converted.
    """
    arr = np.asarray(x)
    if arr.dtype == bool or arr.dtype.kind not in "iuf":
        raise InvalidInputError(
            f"{name} must contain real integer or floating-point numbers; got dtype {arr.dtype}."
        )
    arr = arr.astype(np.float64, copy=False)
    if ndim is not None:
        allowed = (ndim,) if isinstance(ndim, int) else ndim
        if arr.ndim not in allowed:
            raise InvalidInputError(f"{name} must have ndim in {allowed}; got shape {arr.shape}.")
    if not allow_empty and arr.size == 0:
        raise InvalidInputError(f"{name} must not be empty.")
    if require_finite and not np.all(np.isfinite(arr)):
        raise InvalidInputError(f"{name} must contain only finite values.")
    return arr


def as_real_scalar(x, name: str) -> float:
    """Validate a finite real scalar and return it as a Python float."""
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, (numbers.Real, np.floating, np.integer)):
        arr = np.asarray(x)
        if arr.ndim != 0:
            raise InvalidInputError(f"{name} must be a real scalar; got shape {arr.shape}.")
        if arr.dtype == bool or arr.dtype.kind not in "iuf":
            raise InvalidInputError(f"{name} must be a real number; got {type(x).__name__}.")
        x = arr[()]
    value = float(x)
    if not np.isfinite(value):
        raise InvalidInputError(f"{name} must be finite; got {value}.")
    return value


def check_nonneg(x, name: str) -> float:
    value = as_real_scalar(x, name)
    if value < 0:
        raise InvalidInputError(f"{name} must be >= 0; got {value}.")
    return value


def check_positive_int(x, name: str, *, minimum: int = 1) -> int:
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, (numbers.Integral, np.integer)):
        raise InvalidInputError(f"{name} must be an integer; got {type(x).__name__}.")
    value = int(x)
    if value < minimum:
        raise InvalidInputError(f"{name} must be >= {minimum}; got {value}.")
    return value


def check_callable(f, name: str) -> None:
    if not callable(f):
        raise InvalidInputError(f"{name} must be callable.")
