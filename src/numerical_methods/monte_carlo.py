"""Fixed-budget Monte Carlo estimation of ``E[Q(X)]`` with batched sampling.

Contracts
---------
``sampler(rng, n) -> samples``
    Draws ``n`` independent realizations of ``X`` using **only** the
    supplied :class:`numpy.random.Generator`.  The first axis of the
    returned array is the sample axis and must have length ``n``; any
    trailing shape is allowed (e.g. ``(n, n_times)`` paths).
``quantity(samples) -> y``
    Maps a batch to the scalar quantity of interest per sample, returning a
    real array of shape exactly ``(n,)``.  It should use array operations.
    If ``quantity`` is omitted the sampler output itself must have shape
    ``(n,)``.

Statistical assumptions
-----------------------
Observations ``Y_i = Q(X_i)`` are independent and identically distributed
with finite variance.  The confidence interval is the normal-approximation
(central limit theorem) interval ``mean +/- z * s / sqrt(N)``.  It is an
*approximation to sampling uncertainty only*: its coverage is not
guaranteed for finite ``N`` (particularly for heavy-tailed ``Y``), and it
does not account for model error or for bias from discretizing a
continuous-time quantity on a finite time grid.

Accumulation is numerically stable: each batch uses a two-pass mean and
centered sum of squares, and batches are merged with the pairwise update of
Chan, Golub & LeVeque (1983).  ``E[Y^2] - E[Y]^2`` is never formed.

Reproducibility: for a fixed seed, ``batch_size``, sampler and the recorded
NumPy version, results are repeatable.  Different ``batch_size`` values
may consume the random stream differently and agree only statistically
(and, when the stream happens to be identical, up to floating-point
summation order).  Identical streams across NumPy versions are not
promised.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from ._validation import (
    InvalidInputError,
    InvalidSampleError,
    as_real_array,
    as_real_scalar,
    check_callable,
    check_positive_int,
)

__all__ = ["MonteCarloResult", "monte_carlo", "RunningMoments"]


@dataclass(frozen=True)
class MonteCarloResult:
    """Result of :func:`monte_carlo`.

    estimate : sample mean of the quantity
    variance : unbiased sample variance (ddof=1) of the quantity
    std_error : ``sqrt(variance / n_samples)``
    n_samples : number of observations used
    ci_low, ci_high : approximate ``confidence`` normal interval for the mean
    confidence : nominal level of the interval
    n_batches, batch_size : work metadata (``batch_size`` = requested maximum)
    elapsed_seconds : wall-clock time of sampling and accumulation
    """

    estimate: float
    variance: float
    std_error: float
    n_samples: int
    ci_low: float
    ci_high: float
    confidence: float
    n_batches: int
    batch_size: int
    elapsed_seconds: float

    @property
    def ci(self) -> tuple[float, float]:
        return (self.ci_low, self.ci_high)


class RunningMoments:
    """Stable streaming mean and centered sum of squares (Chan et al. merge).

    ``update(y)`` merges a 1-D batch.  If every observation seen is
    identical, ``mean`` equals that value exactly and ``m2`` is exactly 0.
    """

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, y: np.ndarray) -> None:
        y = as_real_array(y, "y", ndim=1, allow_empty=True)
        nb = y.size
        if nb == 0:
            return
        if np.all(y == y[0]):
            mb, m2b = float(y[0]), 0.0
        else:
            mb = float(np.mean(y))
            m2b = float(np.sum((y - mb) ** 2))
        if self.n == 0:
            self.n, self.mean, self.m2 = nb, mb, m2b
            return
        n = self.n + nb
        delta = mb - self.mean
        if delta != 0.0:
            self.mean += delta * (nb / n)
            self.m2 += m2b + delta * delta * (self.n * nb / n)
        else:
            self.m2 += m2b
        self.n = n

    @property
    def variance(self) -> float:
        """Unbiased sample variance (requires ``n >= 2``)."""
        return self.m2 / (self.n - 1) if self.n >= 2 else float("nan")


def monte_carlo(sampler, n_samples: int, *, rng: np.random.Generator, quantity=None,
                batch_size: int = 100_000, confidence: float = 0.95) -> MonteCarloResult:
    """Estimate ``E[quantity(X)]`` from ``n_samples`` i.i.d. draws produced in batches.

    Parameters
    ----------
    sampler : callable ``(rng, n) -> array`` with sample axis 0.
    n_samples : int, >= 2 (needed for the sample variance).
    rng : numpy.random.Generator
        Required explicitly; global NumPy RNG state is never used.
    quantity : callable ``(samples) -> array of shape (n,)``, optional.
    batch_size : int, >= 1
        Maximum samples requested from ``sampler`` per call, bounding memory
        (for paths: ``batch_size * n_times`` floats at a time).
    confidence : float in (0, 1)

    Constant output: if every observation is identical, ``variance`` and
    ``std_error`` are exactly 0 and the interval collapses to the estimate.

    Raises
    ------
    InvalidInputError
        Invalid arguments (including a non-Generator ``rng``).
    InvalidSampleError
        If the sampler/quantity returns the wrong shape, non-real or
        non-finite values.  Observations are never silently dropped.
    """
    check_callable(sampler, "sampler")
    if quantity is not None:
        check_callable(quantity, "quantity")
    if not isinstance(rng, np.random.Generator):
        raise InvalidInputError("rng must be a numpy.random.Generator (e.g. np.random.default_rng(seed)).")
    n_samples = check_positive_int(n_samples, "n_samples", minimum=2)
    batch_size = check_positive_int(batch_size, "batch_size")
    confidence = as_real_scalar(confidence, "confidence")
    if not 0.0 < confidence < 1.0:
        raise InvalidInputError("confidence must lie strictly between 0 and 1.")

    moments = RunningMoments()
    n_batches = 0
    t0 = time.perf_counter()
    remaining = n_samples
    while remaining > 0:
        nb = min(batch_size, remaining)
        try:
            samples = as_real_array(sampler(rng, nb), "sampler output", allow_empty=True)
        except InvalidInputError as exc:
            raise InvalidSampleError(f"sampler returned non-real or non-finite values: {exc}") from exc
        if samples.ndim == 0 or samples.shape[0] != nb:
            raise InvalidSampleError(
                f"sampler must return an array with leading dimension {nb}; got shape {samples.shape}."
            )
        y = np.asarray(quantity(samples)) if quantity is not None else samples
        if y.shape != (nb,):
            raise InvalidSampleError(f"quantity values must have shape ({nb},); got {y.shape}.")
        if y.dtype == bool or y.dtype.kind not in "iuf":
            raise InvalidSampleError(f"quantity values must be real numbers; got dtype {y.dtype}.")
        y = y.astype(np.float64, copy=False)
        if not np.all(np.isfinite(y)):
            bad = int(np.count_nonzero(~np.isfinite(y)))
            raise InvalidSampleError(f"{bad} non-finite observation(s) in batch {n_batches}.")
        moments.update(y)
        n_batches += 1
        remaining -= nb
    elapsed = time.perf_counter() - t0

    var = moments.variance
    se = float(np.sqrt(var / moments.n))
    # The lower-tail probability remains representable even when confidence
    # is the greatest float below 1; 0.5 + confidence / 2 can round to 1.
    z = -NormalDist().inv_cdf((1.0 - confidence) / 2.0)
    mean = moments.mean
    return MonteCarloResult(
        estimate=mean, variance=var, std_error=se, n_samples=moments.n,
        ci_low=mean - z * se, ci_high=mean + z * se, confidence=confidence,
        n_batches=n_batches, batch_size=batch_size, elapsed_seconds=elapsed,
    )
