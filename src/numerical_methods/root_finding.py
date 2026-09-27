"""Scalar root-finding: bisection, Newton, secant and fixed-point iteration.

Conventions shared by every solver
----------------------------------
* Problems are real scalar equations ``f(x) = 0`` (or ``x = g(x)`` for
  fixed-point iteration) in float64.  User callables receive a Python
  ``float`` and must return a real scalar; returning an array, a complex
  number or a non-numeric value raises :class:`InvalidInputError`.
* Tolerances

  ``ftol``  residual tolerance, test ``|f(x)| <= ftol``.
  ``xtol``, ``rtol``  absolute/relative *x* tolerance, test
  ``width <= xtol + rtol * |x|``.  ``width`` is the bracket width for
  bisection, the last step length for Newton/secant, and ``|g(x) - x|``
  for fixed-point iteration.

* ``max_iter`` bounds the number of iterations (updates of the iterate).
* API misuse (non-finite start values, invalid brackets, non-callables,
  bad tolerances) raises :class:`InvalidInputError`, a ``ValueError``.
  Numerical failure is *not* an exception: it is reported in the returned
  :class:`RootResult` with ``converged=False`` and a ``reason``.

Termination reasons (``RootResult.reason``)
-------------------------------------------
``"residual"``          ``|f(x)| <= ftol`` (converged).
``"bracket"``           bisection bracket with a sign change is narrower than
                        ``xtol + rtol*|x|`` or cannot be split further in
                        floating point (converged; assumes continuity).
``"stagnation"``        Newton/secant step fell below the *x* tolerance but
                        the residual test failed.  Not reported as converged:
                        a tiny step is not proof of a root.
``"max_iter"``          iteration budget exhausted.
``"zero_derivative"``   Newton met ``f'(x) == 0``.
``"zero_denominator"``  secant met ``f(x_k) == f(x_{k-1})``.
``"nonfinite"``         a function value, derivative or iterate was inf/NaN.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._validation import (
    InvalidInputError,
    as_real_scalar,
    check_callable,
    check_nonneg,
    check_positive_int,
)

__all__ = [
    "RootResult",
    "bisection",
    "newton",
    "secant",
    "fixed_point",
    "scan_brackets",
    "CONVERGED_REASONS",
]

CONVERGED_REASONS = frozenset({"residual", "bracket"})

_EPS = np.finfo(np.float64).eps
_DEFAULT_XTOL = 1e-12
_DEFAULT_RTOL = 4 * _EPS
_DEFAULT_FTOL = 1e-12


@dataclass(frozen=True)
class RootResult:
    """Outcome of a scalar root-finding run.

    Attributes
    ----------
    root : float
        Final estimate (the point at which ``residual`` was evaluated).
    residual : float
        ``|f(root)|`` for bisection/Newton/secant; ``|g(root) - root|`` for
        fixed-point iteration (see ``residual_kind``).
    converged : bool
        True only when ``reason`` is ``"residual"`` or ``"bracket"``.
    reason : str
        Termination reason (see module docstring).
    message : str
        Human-readable explanation.
    iterations : int
        Number of iterate updates performed.
    nfev : int
        Evaluations of ``f`` (or ``g`` for fixed-point).
    njev : int
        Evaluations of the derivative (Newton only; 0 otherwise).
    method : str
    residual_kind : str
        ``"|f(x)|"`` or ``"|g(x)-x|"``.
    bracket : tuple[float, float] | None
        Final sign-change bracket (bisection only).
    equation_residual : float | None
        ``|f(root)|`` for fixed-point iteration when the original equation
        ``f`` was supplied; otherwise None.
    history : dict[str, numpy.ndarray] | None
        Present only when ``record_history=True``.  Keys ``"x"`` and
        ``"residual"`` (one entry per evaluated iterate, including the
        start); bisection also records ``"a"`` and ``"b"``.
    """

    root: float
    residual: float
    converged: bool
    reason: str
    message: str
    iterations: int
    nfev: int
    njev: int
    method: str
    residual_kind: str = "|f(x)|"
    bracket: tuple[float, float] | None = None
    equation_residual: float | None = None
    history: dict[str, np.ndarray] | None = field(default=None, repr=False)


class _Recorder:
    """Optional iteration history; stores nothing when disabled."""

    def __init__(self, enabled: bool, keys: tuple[str, ...] = ("x", "residual")):
        self.enabled = bool(enabled)
        self.data: dict[str, list[float]] = {k: [] for k in keys} if self.enabled else {}

    def add(self, **values: float) -> None:
        if self.enabled:
            for k, v in values.items():
                self.data[k].append(float(v))

    def result(self) -> dict[str, np.ndarray] | None:
        if not self.enabled:
            return None
        return {k: np.asarray(v, dtype=np.float64) for k, v in self.data.items()}


def _call_scalar(func, x: float, name: str) -> float:
    """Evaluate a user callable and validate that it returned a real scalar."""
    try:
        value = func(x)
    except OverflowError:
        # math.exp and Python powers raise instead of returning IEEE infinity.
        # Treat arithmetic overflow like a non-finite callback result.
        return float("inf")
    arr = np.asarray(value)
    if arr.shape != ():
        raise InvalidInputError(f"{name} must return a scalar; got shape {arr.shape}.")
    if arr.dtype == bool or arr.dtype.kind not in "iuf":
        raise InvalidInputError(f"{name} must return a real number; got dtype {arr.dtype}.")
    return float(arr)


def _check_common(xtol, rtol, ftol, max_iter):
    return (
        check_nonneg(xtol, "xtol"),
        check_nonneg(rtol, "rtol"),
        check_nonneg(ftol, "ftol"),
        check_positive_int(max_iter, "max_iter"),
    )


def bisection(f, a, b, *, xtol=_DEFAULT_XTOL, rtol=_DEFAULT_RTOL, ftol=_DEFAULT_FTOL,
              max_iter: int = 200, record_history: bool = False) -> RootResult:
    """Find a root of a continuous ``f`` in the sign-change bracket ``[a, b]``.

    Assumptions: ``f`` is continuous on ``[a, b]`` and ``f(a)``, ``f(b)``
    are finite with opposite signs (or one of them is exactly zero).
    Under these assumptions the returned ``root`` lies in the final
    ``bracket``, which contains a root.

    Stopping rules, checked after every midpoint evaluation:

    1. ``|f(c)| <= ftol``  -> ``reason="residual"``.
    2. updated bracket width ``b - a <= xtol + rtol*|c|`` -> ``"bracket"``;
       since ``c`` is an endpoint of the bracket, ``|c - root| <= b - a``.
    3. the bracket consists of adjacent floats -> ``"bracket"``.
    4. a non-finite ``f(c)`` -> ``"nonfinite"`` (not converged).
    5. ``max_iter`` midpoints -> ``"max_iter"``.

    Pass ``ftol=0`` to rely on the bracket criterion only.  If ``f`` is
    discontinuous (e.g. a pole with a sign change) the bracket criterion can
    still be met; the large ``residual`` then reveals the violated assumption.

    Raises
    ------
    InvalidInputError
        If ``a >= b``, an endpoint value is non-finite, or ``f(a)`` and
        ``f(b)`` are nonzero with the same sign.
    """
    check_callable(f, "f")
    a = as_real_scalar(a, "a")
    b = as_real_scalar(b, "b")
    xtol, rtol, ftol, max_iter = _check_common(xtol, rtol, ftol, max_iter)
    if not a < b:
        raise InvalidInputError(f"bisection requires a < b; got a={a}, b={b}.")

    fa = _call_scalar(f, a, "f")
    fb = _call_scalar(f, b, "f")
    nfev = 2
    if not (np.isfinite(fa) and np.isfinite(fb)):
        raise InvalidInputError(f"f must be finite at both endpoints; got f(a)={fa}, f(b)={fb}.")
    rec = _Recorder(record_history, ("x", "residual", "a", "b"))

    def done(root, froot, reason, message, it):
        return RootResult(root=root, residual=abs(froot), converged=reason in CONVERGED_REASONS,
                          reason=reason, message=message, iterations=it, nfev=nfev, njev=0,
                          method="bisection", bracket=(a, b), history=rec.result())

    if fa == 0.0 or fb == 0.0:
        root, froot = (a, fa) if fa == 0.0 else (b, fb)
        rec.add(x=root, residual=abs(froot), a=a, b=b)
        return done(root, froot, "residual", "Endpoint is an exact root.", 0)
    if np.sign(fa) == np.sign(fb):
        raise InvalidInputError(
            f"f(a) and f(b) must have opposite signs; got f(a)={fa}, f(b)={fb}."
        )

    c, fc = (a, fa) if abs(fa) <= abs(fb) else (b, fb)
    for it in range(1, max_iter + 1):
        mid = a + 0.5 * (b - a)  # safe for endpoints of the same sign
        if not np.isfinite(mid):
            mid = 0.5 * a + 0.5 * b  # opposite signs can overflow b - a
        if not a < mid < b:
            c, fc = (a, fa) if abs(fa) <= abs(fb) else (b, fb)
            return done(c, fc, "bracket",
                        "Bracket reached floating-point resolution.", it - 1)
        c = mid
        fc = _call_scalar(f, c, "f")
        nfev += 1
        if not np.isfinite(fc):
            return done(c, fc, "nonfinite", f"f returned non-finite value {fc} at x={c}.", it)
        if np.sign(fc) == np.sign(fa):
            a, fa = c, fc
        else:
            b, fb = c, fc
        rec.add(x=c, residual=abs(fc), a=a, b=b)
        if abs(fc) <= ftol:
            return done(c, fc, "residual", "Residual tolerance satisfied.", it)
        if b - a <= xtol + rtol * abs(c):
            return done(c, fc, "bracket", "Bracket width tolerance satisfied.", it)
    return done(c, fc, "max_iter", "Maximum iterations reached.", max_iter)


def newton(f, df, x0, *, xtol=_DEFAULT_XTOL, rtol=_DEFAULT_RTOL, ftol=_DEFAULT_FTOL,
           max_iter: int = 100, record_history: bool = False) -> RootResult:
    """Newton's method ``x_{k+1} = x_k - f(x_k)/f'(x_k)`` with a user derivative.

    Assumptions: ``f`` is differentiable near the root and ``df`` is its
    derivative.  Convergence is local: quadratic near a simple root,
    linear near a multiple root, and not guaranteed from a poor ``x0``.
    No safeguarding (damping, bracketing) is performed.

    Each iteration costs one ``f`` and one ``df`` evaluation (``nfev``/
    ``njev``), plus one initial ``f`` evaluation.

    Stopping rules: ``|f(x)| <= ftol`` gives ``"residual"`` (checked at
    ``x0`` and after each step).  A step ``|x_{k+1}-x_k| <= xtol + rtol*|x_{k+1}|``
    while the residual test still fails gives ``"stagnation"``
    (not converged).  ``f'(x) == 0`` gives ``"zero_derivative"``; non-finite
    derivative, step or value gives ``"nonfinite"`` and returns the last
    finite iterate.  Choose ``ftol`` relative to the scale of ``f``: a
    residual below ``ftol`` may be unattainable for badly scaled ``f``.
    """
    check_callable(f, "f")
    check_callable(df, "df")
    x = as_real_scalar(x0, "x0")
    xtol, rtol, ftol, max_iter = _check_common(xtol, rtol, ftol, max_iter)
    rec = _Recorder(record_history)
    nfev = njev = 0

    def done(x, fx, reason, message, it):
        return RootResult(root=x, residual=abs(fx), converged=reason in CONVERGED_REASONS,
                          reason=reason, message=message, iterations=it, nfev=nfev,
                          njev=njev, method="newton", history=rec.result())

    fx = _call_scalar(f, x, "f")
    nfev += 1
    if not np.isfinite(fx):
        return done(x, fx, "nonfinite", f"f(x0) is non-finite ({fx}).", 0)
    rec.add(x=x, residual=abs(fx))
    if abs(fx) <= ftol:
        return done(x, fx, "residual", "Residual tolerance satisfied at x0.", 0)

    for it in range(1, max_iter + 1):
        dfx = _call_scalar(df, x, "df")
        njev += 1
        if not np.isfinite(dfx):
            return done(x, fx, "nonfinite", f"df returned non-finite value at x={x}.", it - 1)
        if dfx == 0.0:
            return done(x, fx, "zero_derivative", f"Derivative is zero at x={x}.", it - 1)
        x_new = x - fx / dfx
        if not np.isfinite(x_new):
            return done(x, fx, "nonfinite", "Newton step overflowed.", it - 1)
        f_new = _call_scalar(f, x_new, "f")
        nfev += 1
        if not np.isfinite(f_new):
            return done(x, fx, "nonfinite",
                        f"f returned non-finite value at x={x_new}; returning last finite iterate.",
                        it - 1)
        step = abs(x_new - x)
        x, fx = x_new, f_new
        rec.add(x=x, residual=abs(fx))
        if abs(fx) <= ftol:
            return done(x, fx, "residual", "Residual tolerance satisfied.", it)
        if step <= xtol + rtol * abs(x):
            return done(x, fx, "stagnation",
                        f"Step {step:.3e} below x tolerance but residual {abs(fx):.3e} > ftol.", it)
    return done(x, fx, "max_iter", "Maximum iterations reached.", max_iter)


def secant(f, x0, x1, *, xtol=_DEFAULT_XTOL, rtol=_DEFAULT_RTOL, ftol=_DEFAULT_FTOL,
           max_iter: int = 100, record_history: bool = False) -> RootResult:
    """Secant method from two distinct starting points ``x0`` and ``x1``.

    ``x_{k+1} = x_k - f(x_k) (x_k - x_{k-1}) / (f(x_k) - f(x_{k-1}))``.
    One new ``f`` evaluation per iteration (two initial ones).  Local
    convergence of order about 1.618 near a simple root; no bracketing is
    maintained, so iterates can leave any interval.

    Stopping rules match :func:`newton`; ``f(x_k) == f(x_{k-1})`` gives
    ``"zero_denominator"``.  The start values are checked against ``ftol``
    first (``x1`` then ``x0``).
    """
    check_callable(f, "f")
    xp = as_real_scalar(x0, "x0")
    x = as_real_scalar(x1, "x1")
    xtol, rtol, ftol, max_iter = _check_common(xtol, rtol, ftol, max_iter)
    if xp == x:
        raise InvalidInputError("secant requires distinct starting points x0 != x1.")
    rec = _Recorder(record_history)

    fp = _call_scalar(f, xp, "f")
    fx = _call_scalar(f, x, "f")
    nfev = 2

    def done(x, fx, reason, message, it):
        return RootResult(root=x, residual=abs(fx), converged=reason in CONVERGED_REASONS,
                          reason=reason, message=message, iterations=it, nfev=nfev, njev=0,
                          method="secant", history=rec.result())

    if not np.isfinite(fp):
        return done(xp, fp, "nonfinite", "f(x0) is non-finite.", 0)
    if not np.isfinite(fx):
        return done(x, fx, "nonfinite", "f(x1) is non-finite.", 0)
    rec.add(x=xp, residual=abs(fp))
    rec.add(x=x, residual=abs(fx))
    if abs(fx) <= ftol:
        return done(x, fx, "residual", "Residual tolerance satisfied at x1.", 0)
    if abs(fp) <= ftol:
        return done(xp, fp, "residual", "Residual tolerance satisfied at x0.", 0)

    for it in range(1, max_iter + 1):
        denom = fx - fp
        if not np.isfinite(denom):
            return done(x, fx, "nonfinite", "Secant function difference overflowed.", it - 1)
        if denom == 0.0:
            return done(x, fx, "zero_denominator",
                        "f(x_k) == f(x_{k-1}); secant slope is zero.", it - 1)
        x_new = x - fx * (x - xp) / denom
        if not np.isfinite(x_new):
            return done(x, fx, "nonfinite", "Secant step overflowed.", it - 1)
        f_new = _call_scalar(f, x_new, "f")
        nfev += 1
        if not np.isfinite(f_new):
            return done(x, fx, "nonfinite",
                        f"f returned non-finite value at x={x_new}; returning last finite iterate.",
                        it - 1)
        step = abs(x_new - x)
        xp, fp, x, fx = x, fx, x_new, f_new
        rec.add(x=x, residual=abs(fx))
        if abs(fx) <= ftol:
            return done(x, fx, "residual", "Residual tolerance satisfied.", it)
        if step <= xtol + rtol * abs(x):
            return done(x, fx, "stagnation",
                        f"Step {step:.3e} below x tolerance but residual {abs(fx):.3e} > ftol.", it)
    return done(x, fx, "max_iter", "Maximum iterations reached.", max_iter)


def fixed_point(g, x0, *, xtol=_DEFAULT_XTOL, rtol=_DEFAULT_RTOL, max_iter: int = 500,
                f=None, record_history: bool = False) -> RootResult:
    """Fixed-point iteration ``x_{k+1} = g(x_k)``.

    The residual is the fixed-point residual ``|g(x) - x|`` evaluated at the
    returned ``root`` (``residual_kind="|g(x)-x|"``).  Convergence
    (``reason="residual"``) means ``|g(x) - x| <= xtol + rtol*|x|``.  If
    ``g`` is a contraction with Lipschitz constant ``L < 1`` near the
    fixed point ``x*``, then ``|x - x*| <= |g(x) - x| / (1 - L)``; without
    that assumption the residual does not bound the error.

    If the original equation ``f`` is supplied, ``equation_residual =
    |f(root)|`` is reported (one extra call to ``f``, not counted in
    ``nfev``).  It does not affect the stopping rule.

    ``iterations`` counts updates; the returned ``root`` is the last iterate
    at which ``g`` was evaluated, so ``nfev = iterations + 1`` on
    convergence.  Non-finite ``g`` values give ``"nonfinite"``; there is
    no divergence detection other than that and ``"max_iter"``.
    """
    check_callable(g, "g")
    if f is not None:
        check_callable(f, "f")
    x = as_real_scalar(x0, "x0")
    xtol = check_nonneg(xtol, "xtol")
    rtol = check_nonneg(rtol, "rtol")
    max_iter = check_positive_int(max_iter, "max_iter")
    rec = _Recorder(record_history)
    nfev = 0

    def done(x, r, reason, message, it):
        eq = None
        if f is not None and np.isfinite(x):
            eq = abs(_call_scalar(f, x, "f"))
        return RootResult(root=x, residual=r, converged=reason in CONVERGED_REASONS,
                          reason=reason, message=message, iterations=it, nfev=nfev, njev=0,
                          method="fixed_point", residual_kind="|g(x)-x|",
                          equation_residual=eq, history=rec.result())

    r = float("inf")
    for it in range(0, max_iter + 1):
        gx = _call_scalar(g, x, "g")
        nfev += 1
        if not np.isfinite(gx):
            return done(x, float("inf"), "nonfinite", f"g returned non-finite value at x={x}.", it)
        r = abs(gx - x)
        rec.add(x=x, residual=r)
        if r <= xtol + rtol * abs(x):
            return done(x, r, "residual", "Fixed-point residual tolerance satisfied.", it)
        if it == max_iter:
            break
        x = gx
    return done(x, r, "max_iter", "Maximum iterations reached.", max_iter)


def scan_brackets(f, a, b, num: int = 1001) -> np.ndarray:
    """Locate sign-change brackets of ``f`` on a uniform grid of ``[a, b]``.

    ``f`` is evaluated **once on the whole grid as an array**
    (``f(np.linspace(a, b, num))``), so it must accept and return float64
    arrays of the same shape.  Returns an array of shape ``(k, 2)`` of
    adjacent grid intervals ``[x_i, x_{i+1}]`` where ``f`` changes sign or
    has an exact zero at ``x_i`` (or at the last node), each suitable for
    :func:`bisection`.  Roots of even multiplicity and pairs of roots inside
    one cell are not detected.  Non-finite values raise
    :class:`InvalidInputError`.
    """
    check_callable(f, "f")
    a = as_real_scalar(a, "a")
    b = as_real_scalar(b, "b")
    num = check_positive_int(num, "num", minimum=2)
    if not a < b:
        raise InvalidInputError("scan_brackets requires a < b.")
    x = np.linspace(a, b, num)
    y = np.asarray(f(x))
    if y.shape != x.shape or y.dtype == bool or y.dtype.kind not in "iuf":
        raise InvalidInputError("f must map a float array to a real array of the same shape.")
    if not np.all(np.isfinite(y)):
        raise InvalidInputError("f returned non-finite values on the scan grid.")
    s = np.sign(y)
    flag = (s[:-1] * s[1:] < 0) | (s[:-1] == 0)
    if s[-1] == 0 and s[-2] != 0:
        flag[-1] = True
    idx = np.nonzero(flag)[0]
    return np.column_stack([x[idx], x[idx + 1]])
