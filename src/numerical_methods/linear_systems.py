"""Dense linear systems: Gaussian elimination / LU with partial pivoting.

Supported problem class: square, real, finite float64 matrices ``A`` of
shape ``(n, n)`` with right-hand sides of shape ``(n,)`` or ``(n, k)``.

Permutation convention: :func:`lu_factor` returns ``perm`` such that
``A[perm] == L @ U`` in exact arithmetic, i.e. ``P @ A = L @ U`` with
``P = np.eye(n)[perm]``.  ``L`` is unit lower triangular with
``|L[i, j]| <= 1`` (partial pivoting), ``U`` is upper triangular.

The elimination, pivot search and triangular solves are implemented here;
NumPy provides only array arithmetic, rank-1 updates and norms.  No matrix
is ever inverted explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._validation import InvalidInputError, SingularMatrixError, as_real_array, check_nonneg

__all__ = [
    "LUFactorization",
    "lu_factor",
    "gaussian_elimination",
    "solve",
    "forward_substitution",
    "back_substitution",
    "ResidualDiagnostics",
    "residual_diagnostics",
    "factorization_error",
    "forward_error",
]

_EPS = np.finfo(np.float64).eps


def _square(A, name="A") -> np.ndarray:
    A = as_real_array(A, name, ndim=2)
    if A.shape[0] != A.shape[1]:
        raise InvalidInputError(f"{name} must be square; got shape {A.shape}.")
    return A


def _rhs(b, n: int, name="b") -> np.ndarray:
    b = as_real_array(b, name, ndim=(1, 2))
    if b.shape[0] != n:
        raise InvalidInputError(f"{name} must have {n} rows; got shape {b.shape}.")
    return b


def _default_pivot_tol(A: np.ndarray) -> float:
    return A.shape[0] * _EPS * float(np.max(np.abs(A))) if A.size else 0.0


@dataclass(frozen=True)
class LUFactorization:
    """Reusable LU factorization ``P @ A = L @ U`` (partial pivoting).

    Attributes
    ----------
    lu : ndarray, shape (n, n)
        Packed factors: strict lower part holds ``L`` (unit diagonal
        implied), upper part holds ``U``.
    perm : ndarray of int, shape (n,)
        Row permutation with ``A[perm] = L @ U``.
    pivot_tol : float
        Threshold used for the singularity test during factorization.
    n_swaps : int
        Number of row interchanges (sign of det(P) is ``(-1)**n_swaps``).
    """

    lu: np.ndarray
    perm: np.ndarray
    pivot_tol: float
    n_swaps: int

    @property
    def n(self) -> int:
        return self.lu.shape[0]

    @property
    def L(self) -> np.ndarray:
        return np.tril(self.lu, -1) + np.eye(self.n)

    @property
    def U(self) -> np.ndarray:
        return np.triu(self.lu)

    @property
    def P(self) -> np.ndarray:
        return np.eye(self.n)[self.perm]

    @property
    def pivots(self) -> np.ndarray:
        """Diagonal of ``U``."""
        return np.diag(self.lu).copy()

    def solve(self, b) -> np.ndarray:
        """Solve ``A x = b`` for ``b`` of shape ``(n,)`` or ``(n, k)`` without refactorizing.

        Cost is O(n^2) per right-hand side (forward then back substitution).
        """
        b = _rhs(b, self.n)
        y = forward_substitution(self.lu, b[self.perm], unit_diagonal=True, check_triangular=False)
        return back_substitution(self.lu, y, check_triangular=False)

    def det(self) -> float:
        """Determinant from the factors (may over/underflow for large n)."""
        return float((-1) ** self.n_swaps * np.prod(np.diag(self.lu)))


def lu_factor(A, *, pivot_tol: float | None = None) -> LUFactorization:
    """LU factorization with partial (row) pivoting, ``P @ A = L @ U``.

    At step ``k`` the entry of largest magnitude in column ``k`` on or below
    the diagonal is chosen as pivot, rows are swapped, multipliers are stored
    and the trailing submatrix receives a rank-1 update.

    Singularity test (scale-aware): the factorization fails with
    :class:`SingularMatrixError` if a chosen pivot satisfies
    ``|u_kk| <= pivot_tol``.  The default is ``n * eps * max|A_ij|``: a pivot
    that small is at the level of rounding errors committed on entries of
    size ``max|A_ij|``, so the matrix is singular to working precision
    relative to its own scale.  ``pivot_tol=0`` only rejects exactly zero
    pivots.  This test is a heuristic, not a condition-number estimate:
    passing it does **not** mean ``A`` is well conditioned (ill-conditioned
    matrices can have unremarkable pivots), and badly row-scaled but
    nonsingular matrices can trip it.  Inspect residual and forward-error
    diagnostics for accuracy.

    Parameters
    ----------
    A : array_like, shape (n, n)
        Real, finite, square.  Not modified.
    pivot_tol : float, optional
        Non-negative absolute pivot threshold.
    """
    A = _square(A)
    n = A.shape[0]
    if pivot_tol is None:
        pivot_tol = _default_pivot_tol(A)
    else:
        pivot_tol = check_nonneg(pivot_tol, "pivot_tol")
    lu = A.copy()
    perm = np.arange(n)
    n_swaps = 0
    for k in range(n):
        p = k + int(np.argmax(np.abs(lu[k:, k])))
        pivot = lu[p, k]
        if abs(pivot) <= pivot_tol or pivot == 0.0:
            raise SingularMatrixError(
                f"Matrix is singular to working precision: pivot {pivot:.3e} at column {k} "
                f"is <= pivot_tol {pivot_tol:.3e}."
            )
        if p != k:
            lu[[k, p]] = lu[[p, k]]
            perm[[k, p]] = perm[[p, k]]
            n_swaps += 1
        if k + 1 < n:
            lu[k + 1:, k] /= pivot
            lu[k + 1:, k + 1:] -= np.outer(lu[k + 1:, k], lu[k, k + 1:])
    return LUFactorization(lu=lu, perm=perm, pivot_tol=pivot_tol, n_swaps=n_swaps)


def gaussian_elimination(A, b, *, pivot_tol: float | None = None) -> np.ndarray:
    """Solve ``A x = b`` by Gaussian elimination with partial pivoting.

    Equivalent to ``lu_factor(A).solve(b)``; the elimination is the same
    and the multipliers are simply retained.  Use :func:`lu_factor` directly
    to reuse the factorization for further right-hand sides.
    Raises :class:`SingularMatrixError` under the rule described in
    :func:`lu_factor`.
    """
    fac = lu_factor(A, pivot_tol=pivot_tol)
    return fac.solve(b)


solve = gaussian_elimination


def _check_tri(T: np.ndarray, lower: bool, name: str) -> None:
    off = np.triu(T, 1) if lower else np.tril(T, -1)
    if np.any(off != 0):
        kind = "lower" if lower else "upper"
        raise InvalidInputError(f"{name} must be {kind} triangular.")


def forward_substitution(L, b, *, unit_diagonal: bool = False,
                         check_triangular: bool = True) -> np.ndarray:
    """Solve ``L y = b`` for lower-triangular ``L`` (row-oriented).

    ``b`` has shape ``(n,)`` or ``(n, k)``.  With ``unit_diagonal=True`` the
    diagonal is taken to be 1 and entries on/above it are ignored (packed
    LU storage).  Otherwise a zero diagonal entry raises
    :class:`SingularMatrixError`, and a nonzero strictly-upper entry raises
    :class:`InvalidInputError` when ``check_triangular`` is true.
    """
    L = _square(L, "L")
    b = _rhs(b, L.shape[0])
    if check_triangular and not unit_diagonal:
        _check_tri(L, True, "L")
    n = L.shape[0]
    y = np.empty_like(b)
    for i in range(n):
        s = b[i] - L[i, :i] @ y[:i]
        if unit_diagonal:
            y[i] = s
        else:
            if L[i, i] == 0.0:
                raise SingularMatrixError(f"Zero diagonal entry L[{i},{i}].")
            y[i] = s / L[i, i]
    return y


def back_substitution(U, b, *, check_triangular: bool = True) -> np.ndarray:
    """Solve ``U x = b`` for upper-triangular ``U`` (row-oriented).

    Entries strictly below the diagonal are ignored when
    ``check_triangular=False`` (packed LU storage).  A zero diagonal entry
    raises :class:`SingularMatrixError`.
    """
    U = _square(U, "U")
    b = _rhs(b, U.shape[0])
    if check_triangular:
        _check_tri(U, False, "U")
    n = U.shape[0]
    x = np.empty_like(b)
    for i in range(n - 1, -1, -1):
        if U[i, i] == 0.0:
            raise SingularMatrixError(f"Zero diagonal entry U[{i},{i}].")
        x[i] = (b[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


@dataclass(frozen=True)
class ResidualDiagnostics:
    """Residual-based quality measures of a computed solution (infinity norms).

    residual_norm : ``||b - A x||``
    relative_residual : ``||b - A x|| / ||b||`` (inf if ``b == 0`` and r != 0)
    backward_error : normwise backward error
        ``||b - A x|| / (||A|| ||x|| + ||b||)`` (Rigal-Gaches; Higham,
        *Accuracy and Stability of Numerical Algorithms*, Thm 7.1): the
        smallest ``eps`` such that ``(A + dA) x = b + db`` with
        ``||dA|| <= eps ||A||`` and ``||db|| <= eps ||b||``.

    These measure how well ``x`` solves a nearby problem.  They do not
    measure ``||x - x_true||``; roughly, forward error <= cond(A) * backward
    error, so an ill-conditioned system can have a tiny backward error and a
    large forward error.
    """

    residual_norm: float
    relative_residual: float
    backward_error: float


def residual_diagnostics(A, x, b) -> ResidualDiagnostics:
    """Compute :class:`ResidualDiagnostics` for 1-D ``x`` and ``b``."""
    A = _square(A)
    n = A.shape[0]
    x = as_real_array(x, "x", ndim=1)
    b = as_real_array(b, "b", ndim=1)
    if x.shape != (n,) or b.shape != (n,):
        raise InvalidInputError("x and b must have shape (n,).")
    r = b - A @ x
    rn = float(np.linalg.norm(r, np.inf))
    bn = float(np.linalg.norm(b, np.inf))
    denom = float(np.linalg.norm(A, np.inf)) * float(np.linalg.norm(x, np.inf)) + bn
    rel = rn / bn if bn > 0 else (0.0 if rn == 0 else float("inf"))
    be = rn / denom if denom > 0 else 0.0
    return ResidualDiagnostics(residual_norm=rn, relative_residual=rel, backward_error=be)


def factorization_error(A, fac: LUFactorization) -> float:
    """Relative reconstruction error ``||P A - L U||_inf / ||A||_inf``."""
    A = _square(A)
    if A.shape != fac.lu.shape:
        raise InvalidInputError("A and factorization have different shapes.")
    na = float(np.linalg.norm(A, np.inf))
    err = float(np.linalg.norm(A[fac.perm] - fac.L @ fac.U, np.inf))
    return err / na if na > 0 else err


def forward_error(x, x_true) -> float:
    """Relative forward error ``max|x - x_true| / max|x_true|`` (absolute if ``x_true == 0``).

    For vectors this is the infinity-norm relative error; for ``(n, k)``
    arrays it uses the largest entry over all columns.
    """
    x = as_real_array(x, "x", ndim=(1, 2))
    x_true = as_real_array(x_true, "x_true", ndim=(1, 2))
    if x.shape != x_true.shape:
        raise InvalidInputError("x and x_true must have the same shape.")
    d = float(np.max(np.abs(x - x_true)))
    t = float(np.max(np.abs(x_true)))
    return d / t if t > 0 else d
