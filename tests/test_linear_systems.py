import numpy as np
import pytest

from numerical_methods import (
    InvalidInputError,
    LUFactorization,
    SingularMatrixError,
    back_substitution,
    factorization_error,
    forward_error,
    forward_substitution,
    gaussian_elimination,
    lu_factor,
    residual_diagnostics,
    solve,
)

EPS = np.finfo(float).eps


def random_system(n, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    x = rng.standard_normal(n)
    return A, x, A @ x


class TestLU:
    @pytest.mark.parametrize("n", [1, 2, 5, 50, 120])
    def test_reconstruction_and_structure(self, n):
        A, _, _ = random_system(n, n)
        fac = lu_factor(A)
        L, U, P = fac.L, fac.U, fac.P
        assert np.allclose(np.diag(L), 1.0)
        assert np.all(np.triu(L, 1) == 0) and np.all(np.tril(U, -1) == 0)
        assert np.max(np.abs(L)) <= 1.0  # partial pivoting bound
        # P @ A = L @ U, and the perm convention A[perm] = L @ U
        assert np.array_equal(P @ A, A[fac.perm])
        assert factorization_error(A, fac) <= 10 * n * EPS
        assert sorted(fac.perm) == list(range(n))

    def test_known_pivot_permutation(self):
        # a11 = 0 forces a swap; largest entry in column 0 is row 2
        A = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [4.0, -3.0, 8.0]])
        fac = lu_factor(A)
        assert fac.perm[0] == 2
        assert fac.n_swaps >= 1
        assert np.allclose(fac.P @ A, fac.L @ fac.U, atol=1e-15)

    def test_pivoting_needed_for_accuracy(self):
        # tiny (1,1) entry: elimination without pivoting loses all accuracy
        d = 1e-20
        A = np.array([[d, 1.0], [1.0, 1.0]])
        b = np.array([1.0, 2.0])
        x_true = np.array([1 / (1 - d), (1 - 2 * d) / (1 - d)])
        x = solve(A, b)
        assert np.allclose(x, x_true, rtol=1e-15, atol=0)
        # the naive (unpivoted) formula for comparison gives x0 = 0
        l21 = 1 / d
        u22 = 1 - l21
        x1 = (2 - l21 * 1) / u22
        x0 = (1 - x1) / d
        assert abs(x0 - 1) > 0.5

    def test_matches_numpy_det(self):
        A, _, _ = random_system(7, 3)
        fac = lu_factor(A)
        assert fac.det() == pytest.approx(np.linalg.det(A), rel=1e-12)
        assert np.array_equal(fac.pivots, np.diag(fac.U))
        assert np.all(np.abs(fac.pivots) > fac.pivot_tol)

    def test_does_not_modify_input(self):
        A, _, b = random_system(6, 1)
        A0, b0 = A.copy(), b.copy()
        lu_factor(A).solve(b)
        assert np.array_equal(A, A0) and np.array_equal(b, b0)


class TestSolve:
    @pytest.mark.parametrize("n", [1, 3, 40, 150])
    def test_against_known_solution_and_scipy(self, n):
        A, x_true, b = random_system(n, 100 + n)
        x = gaussian_elimination(A, b)
        cond = np.linalg.cond(A, np.inf)
        assert forward_error(x, x_true) <= 50 * n * cond * EPS
        diag = residual_diagnostics(A, x, b)
        assert diag.backward_error <= 10 * n * EPS
        sla = pytest.importorskip("scipy.linalg")
        assert np.allclose(x, sla.solve(A, b), rtol=0, atol=50 * n * cond * EPS * np.max(abs(x)))

    def test_reuse_factorization_multiple_rhs(self):
        A, _, _ = random_system(30, 7)
        fac = lu_factor(A)
        rng = np.random.default_rng(8)
        for _ in range(3):
            xt = rng.standard_normal(30)
            assert forward_error(fac.solve(A @ xt), xt) < 1e-11
        X = rng.standard_normal((30, 4))
        Xs = fac.solve(A @ X)
        assert Xs.shape == (30, 4) and forward_error(Xs, X) < 1e-11
        # matrix RHS agrees with column-by-column solves up to rounding (BLAS summation
        # order differs between matrix and vector products; bitwise equality is not promised)
        cols = np.column_stack([fac.solve((A @ X)[:, j]) for j in range(4)])
        cond = np.linalg.cond(A, np.inf)
        assert np.max(np.abs(Xs - cols)) <= 10 * 30 * cond * EPS * np.max(np.abs(X))

    def test_identity_and_permutation_exact(self):
        b = np.array([1.0, 2.0, 3.0])
        assert np.array_equal(solve(np.eye(3), b), b)
        P = np.eye(3)[[2, 0, 1]]
        assert np.array_equal(solve(P, P @ b), b)

    def test_integer_input_accepted_as_float(self):
        x = solve([[2, 0], [0, 4]], [2, 8])
        assert x.dtype == np.float64 and np.array_equal(x, [1.0, 2.0])

    def test_exactly_singular(self):
        A = np.array([[1.0, 2.0], [2.0, 4.0]])
        with pytest.raises(SingularMatrixError):
            solve(A, [1.0, 2.0])
        with pytest.raises(SingularMatrixError):
            lu_factor(np.zeros((3, 3)))

    def test_singular_to_working_precision_rank_deficient(self):
        rng = np.random.default_rng(0)
        u = rng.standard_normal((10, 9))
        A = u @ rng.standard_normal((9, 10))  # rank 9 in exact arithmetic
        with pytest.raises(SingularMatrixError):
            lu_factor(A)

    def test_pivot_tol_is_scale_aware(self):
        # scaling a nonsingular matrix by 1e-200 must not make it "singular"
        A, xt, b = random_system(8, 2)
        s = 1e-200
        x = solve(A * s, b * s)
        assert forward_error(x, xt) < 1e-10
        # pivot_tol=0 accepts a nearly-singular matrix the default rejects
        # second pivot is eps, below the default n*eps*max|A| = 2 eps threshold
        B = np.array([[1.0, 1.0], [1.0, 1.0 + EPS]])
        lu_factor(B, pivot_tol=0.0)
        with pytest.raises(SingularMatrixError):
            lu_factor(B)

    def test_ill_conditioned_not_flagged_singular(self):
        # Hilbert(10): cond ~ 1.6e13, nonsingular; small backward error, large forward error
        n = 10
        i = np.arange(n)
        H = 1.0 / (i[:, None] + i[None, :] + 1)
        xt = np.ones(n)
        b = H @ xt
        x = solve(H, b)
        assert residual_diagnostics(H, x, b).backward_error < 10 * n * EPS
        assert forward_error(x, xt) > 1e-8  # accuracy genuinely lost to conditioning

    @pytest.mark.parametrize("pivot_tol", [True, False, "0", 1j, [0.0], np.nan, np.inf])
    def test_pivot_tolerance_rejects_unsupported_scalars(self, pivot_tol):
        with pytest.raises(InvalidInputError):
            lu_factor(np.eye(2), pivot_tol=pivot_tol)

    def test_invalid_inputs(self):
        with pytest.raises(InvalidInputError):
            solve(np.ones((2, 3)), np.ones(2))
        with pytest.raises(InvalidInputError):
            solve(np.eye(2), np.ones(3))
        with pytest.raises(InvalidInputError):
            solve(np.array([[1.0, np.nan], [0.0, 1.0]]), np.ones(2))
        with pytest.raises(InvalidInputError):
            solve(np.eye(2), [np.inf, 1.0])
        with pytest.raises(InvalidInputError):
            solve(np.eye(2) * 1j, np.ones(2))
        with pytest.raises(InvalidInputError):
            solve(np.ones(3), np.ones(3))
        with pytest.raises(InvalidInputError):
            lu_factor(np.eye(2), pivot_tol=-1.0)
        with pytest.raises(InvalidInputError):
            solve(np.eye(2), np.ones((2, 2, 2)))


class TestTriangular:
    def test_forward_and_back(self):
        rng = np.random.default_rng(4)
        L = np.tril(rng.standard_normal((6, 6))) + 6 * np.eye(6)
        U = np.triu(rng.standard_normal((6, 6))) + 6 * np.eye(6)
        x = rng.standard_normal(6)
        assert np.allclose(forward_substitution(L, L @ x), x, rtol=1e-13)
        assert np.allclose(back_substitution(U, U @ x), x, rtol=1e-13)

    def test_unit_diagonal_ignores_upper(self):
        L = np.array([[5.0, 9.0], [2.0, 7.0]])  # treated as [[1,0],[2,1]]
        assert np.array_equal(forward_substitution(L, [1.0, 3.0], unit_diagonal=True), [1.0, 1.0])

    def test_zero_diagonal_and_non_triangular(self):
        with pytest.raises(SingularMatrixError):
            back_substitution(np.array([[1.0, 1.0], [0.0, 0.0]]), [1.0, 1.0])
        with pytest.raises(SingularMatrixError):
            forward_substitution(np.array([[0.0, 0.0], [1.0, 1.0]]), [1.0, 1.0])
        with pytest.raises(InvalidInputError):
            back_substitution(np.ones((2, 2)), [1.0, 1.0])
        with pytest.raises(InvalidInputError):
            forward_substitution(np.ones((2, 2)), [1.0, 1.0])


class TestDiagnostics:
    def test_backward_error_definition(self):
        A = np.array([[2.0, 0.0], [0.0, 1.0]])
        b = np.array([2.0, 1.0])
        x = np.array([1.0, 1.1])
        d = residual_diagnostics(A, x, b)
        assert d.residual_norm == pytest.approx(0.1)
        assert d.relative_residual == pytest.approx(0.1 / 2)
        assert d.backward_error == pytest.approx(0.1 / (2 * 1.1 + 2))

    def test_exact_solution_zero(self):
        d = residual_diagnostics(np.eye(2), [0.0, 0.0], [0.0, 0.0])
        assert d.backward_error == 0 and d.relative_residual == 0

    def test_forward_error_zero_reference(self):
        assert forward_error([1e-3, 0.0], [0.0, 0.0]) == pytest.approx(1e-3)

    def test_shape_checks(self):
        with pytest.raises(InvalidInputError):
            residual_diagnostics(np.eye(2), np.ones(3), np.ones(2))
        with pytest.raises(InvalidInputError):
            forward_error(np.ones(2), np.ones(3))
        with pytest.raises(InvalidInputError):
            factorization_error(np.eye(3), lu_factor(np.eye(2)))
        assert isinstance(lu_factor(np.eye(2)), LUFactorization)
