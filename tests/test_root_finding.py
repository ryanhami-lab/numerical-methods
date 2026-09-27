import math

import numpy as np
import pytest

from numerical_methods import (
    InvalidInputError,
    bisection,
    fixed_point,
    newton,
    scan_brackets,
    secant,
)

CUBIC_ROOT = 1.5213797068045676  # real root of x^3 - x - 2 (Cardano, verified with scipy.brentq)


def cubic(x):
    return x**3 - x - 2


def dcubic(x):
    return 3 * x**2 - 1


def test_cubic_root_reference_value():
    scipy_opt = pytest.importorskip("scipy.optimize")
    assert scipy_opt.brentq(cubic, 1, 2, xtol=1e-15) == pytest.approx(CUBIC_ROOT, abs=4e-16)
    # Cardano's formula as an independent reference
    c = (1 + math.sqrt(26 / 27)) ** (1 / 3) + (1 - math.sqrt(26 / 27)) ** (1 / 3)
    assert c == pytest.approx(CUBIC_ROOT, abs=1e-15)


# ---------------------------------------------------------------- bisection
class TestBisection:
    def test_converges_and_brackets_root(self):
        r = bisection(cubic, 1, 2, ftol=0.0, xtol=1e-12)
        assert r.converged and r.reason == "bracket"
        a, b = r.bracket
        assert a <= CUBIC_ROOT <= b and b - a <= 1e-12 + 4 * np.finfo(float).eps * abs(r.root)
        assert abs(r.root - CUBIC_ROOT) <= b - a
        assert r.nfev == r.iterations + 2 and r.njev == 0

    def test_iteration_count_matches_theory(self):
        # width after k halvings is 2^-k; need 2^-k <= 1e-6
        r = bisection(cubic, 1, 2, ftol=0.0, xtol=1e-6, rtol=0.0)
        assert r.iterations == math.ceil(math.log2(1 / 1e-6))

    def test_residual_stop(self):
        r = bisection(cubic, 1, 2, ftol=1e-3, xtol=0.0)
        assert r.reason == "residual" and r.residual <= 1e-3

    @pytest.mark.parametrize("a,b,root", [(0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)])
    def test_endpoint_root(self, a, b, root):
        r = bisection(lambda x: x, a, b)
        assert r.converged and r.root == root and r.iterations == 0 and r.residual == 0

    def test_same_sign_rejected(self):
        with pytest.raises(InvalidInputError, match="opposite signs"):
            bisection(lambda x: x**2 + 1, -1, 1)

    def test_reversed_or_degenerate_bracket_rejected(self):
        with pytest.raises(InvalidInputError):
            bisection(cubic, 2, 1)
        with pytest.raises(InvalidInputError):
            bisection(cubic, 1, 1)

    def test_nonfinite_endpoint_rejected(self):
        with pytest.raises(InvalidInputError):
            bisection(lambda x: math.inf if x == 0 else x, 0, 1)
        with pytest.raises(InvalidInputError):
            bisection(cubic, -np.inf, 1)

    def test_nonfinite_midpoint_reported(self):
        r = bisection(lambda x: math.nan if 0.4 < x < 0.6 else x - 0.7, 0.0, 1.0)
        assert not r.converged and r.reason == "nonfinite"

    def test_max_iter(self):
        r = bisection(cubic, 1, 2, ftol=0, xtol=0, rtol=0, max_iter=5)
        assert not r.converged and r.reason == "max_iter" and r.iterations == 5

    def test_floating_point_resolution(self):
        # tolerances of zero: stops when bracket holds adjacent floats
        # (x^2 - 2 has no exactly representable root, unlike the cubic whose midpoint hits f == 0)
        r = bisection(lambda x: x * x - 2, 1, 2, ftol=0, xtol=0, rtol=0, max_iter=200)
        assert r.converged and r.reason == "bracket"
        a, b = r.bracket
        assert np.nextafter(a, np.inf) == b
        assert r.iterations < 60

    def test_large_values_no_overflow(self):
        # old implementation formed f(a)*f(b), which overflows here
        r = bisection(lambda x: 1e200 * (x - 0.3), 0, 1)
        assert r.converged and abs(r.root - 0.3) < 1e-11

    def test_opposite_extreme_endpoints_do_not_claim_false_convergence(self):
        r = bisection(lambda x: x, -1e308, 1e308)
        assert r.reason == "residual" and r.root == 0.0 and r.iterations == 1

    def test_pole_sign_change_flagged_by_residual(self):
        r = bisection(lambda x: 1 / (x - 0.5) if x != 0.5 else math.inf, 0.1, 0.8, ftol=0)
        assert r.reason in {"bracket", "nonfinite"}
        assert r.residual > 1e6  # residual exposes the violated continuity assumption

    def test_history_off_by_default_and_on_request(self):
        assert bisection(cubic, 1, 2).history is None
        r = bisection(cubic, 1, 2, record_history=True)
        h = r.history
        assert set(h) == {"x", "residual", "a", "b"}
        assert len(h["x"]) == r.iterations
        assert np.all(np.diff(h["b"] - h["a"]) < 0)

    def test_invalid_arguments(self):
        with pytest.raises(InvalidInputError):
            bisection(cubic, 1, 2, xtol=-1)
        with pytest.raises(InvalidInputError):
            bisection(cubic, 1, 2, max_iter=0)
        with pytest.raises(InvalidInputError):
            bisection(cubic, 1, 2, max_iter=2.5)
        with pytest.raises(InvalidInputError):
            bisection("f", 1, 2)
        with pytest.raises(InvalidInputError):
            bisection(cubic, 1 + 1j, 2)
        with pytest.raises(InvalidInputError):
            bisection(cubic, True, 2)

    def test_function_must_return_scalar(self):
        with pytest.raises(InvalidInputError):
            bisection(lambda x: np.array([x, x]), 1, 2)
        with pytest.raises(InvalidInputError):
            bisection(lambda x: complex(x), -1, 2)


# ---------------------------------------------------------------- newton
class TestNewton:
    def test_quadratic_convergence_simple_root(self):
        r = newton(cubic, dcubic, 1.5, ftol=0, xtol=0, rtol=0, max_iter=20, record_history=True)
        e = np.abs(r.history["x"] - CUBIC_ROOT)
        # ratio e_{k+1}/e_k^2 approaches |f''/(2f')| = 6r/(2(3r^2-1)) ~ 0.765
        c = e[1:4] / e[0:3] ** 2
        assert np.all(np.abs(c - 0.765) < 0.2)

    def test_default_converges(self):
        r = newton(cubic, dcubic, 1.5)
        assert r.converged and r.reason == "residual"
        assert abs(r.root - CUBIC_ROOT) < 1e-13
        assert r.nfev == r.iterations + 1 and r.njev == r.iterations

    def test_root_at_start(self):
        r = newton(lambda x: x**3, lambda x: 3 * x**2, 0.0)
        assert r.converged and r.iterations == 0 and r.njev == 0

    def test_zero_derivative(self):
        r = newton(lambda x: x**2 - 1, lambda x: 2 * x, 0.0)
        assert not r.converged and r.reason == "zero_derivative" and r.root == 0.0

    def test_repeated_root_linear_convergence(self):
        # f = (x-1)^2: Newton error halves each step
        r = newton(lambda x: (x - 1) ** 2, lambda x: 2 * (x - 1), 2.0,
                   ftol=1e-20, max_iter=200, record_history=True)
        e = np.abs(r.history["x"][:15] - 1)
        assert np.allclose(e[1:] / e[:-1], 0.5, atol=1e-12)
        assert r.converged

    def test_tiny_step_with_large_residual_is_stagnation(self):
        # wrong (huge) derivative: steps are tiny but f is ~1, far from a root
        r = newton(lambda x: 1.0 + 1e-12 * x, lambda x: 1e20, 0.0, xtol=1e-10)
        assert r.reason == "stagnation" and not r.converged
        assert r.residual > 0.9

    def test_unattainable_ftol_gives_stagnation_not_success(self):
        # near sqrt(2), |x^2 - 2| cannot reach 1e-30 in float64; steps shrink to rounding level
        f, df = (lambda x: x * x - 2), (lambda x: 2 * x)
        r = newton(f, df, 1.5, ftol=1e-30)
        assert r.reason == "stagnation" and not r.converged
        assert abs(r.root - math.sqrt(2)) <= 2 * np.finfo(float).eps  # iterate nonetheless accurate
        s = secant(f, 1.0, 2.0, ftol=1e-30)
        assert s.reason == "stagnation" and not s.converged

    def test_divergent_start_reports_failure(self):
        # atan: Newton diverges for |x0| > ~1.39
        # iterates grow until 1/(1+x^2) underflows to 0 -> zero_derivative (a failure, not success)
        r = newton(math.atan, lambda x: 1 / (1 + x * x), 1.5, max_iter=100)
        assert not r.converged and r.reason in {"nonfinite", "max_iter", "zero_derivative"}
        assert abs(r.root) > 1e100
        assert newton(math.atan, lambda x: 1 / (1 + x * x), 1.0).converged

    def test_cycle_hits_max_iter(self):
        # x^3 - 2x + 2 from 0 cycles 0 -> 1 -> 0
        r = newton(lambda x: x**3 - 2 * x + 2, lambda x: 3 * x * x - 2, 0.0, max_iter=30)
        assert r.reason == "max_iter" and not r.converged and r.iterations == 30

    def test_nonfinite_derivative(self):
        r = newton(lambda x: x - 1, lambda x: math.nan, 0.0)
        assert r.reason == "nonfinite" and not r.converged

    def test_overflowing_step_returns_last_finite_iterate(self):
        r = newton(lambda x: 1e308, lambda x: 1e-10, 3.0)
        assert r.reason == "nonfinite" and r.root == 3.0 and r.iterations == 0

    def test_nonfinite_value_after_step_returns_last_finite_iterate(self):
        r = newton(lambda x: x - 1 if x < 5 else math.nan, lambda x: 0.1, 0.0)
        assert r.reason == "nonfinite" and r.root == 0.0 and r.nfev == 2

    def test_nonfinite_function_at_start(self):
        r = newton(lambda x: math.inf, lambda x: 1.0, 0.0)
        assert r.reason == "nonfinite"

    def test_invalid_inputs(self):
        with pytest.raises(InvalidInputError):
            newton(cubic, None, 1.0)
        with pytest.raises(InvalidInputError):
            newton(cubic, dcubic, math.nan)
        with pytest.raises(InvalidInputError):
            newton(cubic, dcubic, [1.0])


# ---------------------------------------------------------------- secant
class TestSecant:
    def test_converges(self):
        r = secant(cubic, 1.0, 2.0)
        assert r.converged and abs(r.root - CUBIC_ROOT) < 1e-13
        assert r.nfev == r.iterations + 2 and r.njev == 0

    def test_superlinear_order(self):
        r = secant(cubic, 1.0, 2.0, ftol=0, xtol=0, rtol=0, max_iter=12, record_history=True)
        e = np.abs(r.history["x"] - CUBIC_ROOT)
        e = e[e > 1e-14]
        p = np.log(e[2:] / e[1:-1]) / np.log(e[1:-1] / e[:-2])
        assert 1.4 < p[-1] < 1.9

    def test_zero_denominator(self):
        r = secant(lambda x: x * x - 1, -2.0, 2.0)
        assert r.reason == "zero_denominator" and not r.converged

    def test_equal_start_rejected(self):
        with pytest.raises(InvalidInputError):
            secant(cubic, 1.0, 1.0)

    def test_start_is_root(self):
        assert secant(lambda x: x - 2, 2.0, 3.0).root == 2.0
        assert secant(lambda x: x - 2, 1.0, 2.0).iterations == 0

    def test_nonfinite(self):
        r = secant(lambda x: math.inf if x > 5 else x - 10, 0.0, 1.0)
        assert r.reason == "nonfinite" and not r.converged
        assert secant(lambda x: math.nan, 0.0, 1.0).reason == "nonfinite"
        r1 = secant(lambda x: math.nan if x == 1.0 else x, 0.5, 1.0)
        assert r1.reason == "nonfinite" and r1.iterations == 0

    def test_overflowing_step(self):
        # nearly equal f values over a huge interval -> step overflows to inf
        r = secant(lambda x: 1.0 if x == 0 else 1.0 + 2.0**-52, 0.0, 1e300)
        assert r.reason == "nonfinite" and r.root == 1e300

    def test_overflowing_function_difference_is_not_stagnation(self):
        r = secant(lambda x: 1e308 * x, -1.0, 1.0)
        assert not r.converged and r.reason == "nonfinite"

    def test_max_iter(self):
        r = secant(cubic, 1.0, 2.0, ftol=0, xtol=0, rtol=0, max_iter=3)
        assert r.reason == "max_iter" and r.iterations == 3


# ---------------------------------------------------------------- fixed point
class TestFixedPoint:
    def test_converges_with_equation_residual(self):
        r = fixed_point(lambda x: (x + 2) ** (1 / 3), 1.5, f=cubic)
        assert r.converged and r.residual_kind == "|g(x)-x|"
        assert abs(r.root - CUBIC_ROOT) < 1e-12
        assert r.equation_residual is not None and r.equation_residual < 1e-10
        assert r.nfev == r.iterations + 1

    def test_contraction_error_bound(self):
        # g = cos, near x* L = |sin x*| ~ 0.674
        x_star = 0.7390851332151607
        r = fixed_point(math.cos, 1.0, xtol=1e-10)
        L = 0.68
        assert abs(r.root - x_star) <= r.residual / (1 - L)

    def test_residual_is_g_minus_x_at_returned_point(self):
        r = fixed_point(math.cos, 1.0, xtol=1e-8)
        assert r.residual == abs(math.cos(r.root) - r.root)

    def test_divergent_map(self):
        r = fixed_point(lambda x: 2 * x + 1, 1.0, max_iter=2000)
        assert not r.converged and r.reason == "nonfinite"

    def test_callback_arithmetic_overflow_is_numerical_failure(self):
        r = fixed_point(math.exp, 1.0)
        assert not r.converged and r.reason == "nonfinite" and math.isfinite(r.root)

    def test_nonconvergent_oscillation(self):
        r = fixed_point(lambda x: -x, 1.0, max_iter=50)
        assert r.reason == "max_iter" and not r.converged and r.iterations == 50

    def test_exact_fixed_point_start(self):
        r = fixed_point(lambda x: x, 3.0)
        assert r.converged and r.iterations == 0 and r.residual == 0

    def test_history(self):
        r = fixed_point(math.cos, 1.0, record_history=True)
        assert len(r.history["x"]) == r.nfev
        assert r.history["x"][0] == 1.0

    def test_invalid(self):
        with pytest.raises(InvalidInputError):
            fixed_point(math.cos, 1.0, f=3)
        with pytest.raises(InvalidInputError):
            fixed_point(math.cos, np.inf)


# ---------------------------------------------------------------- array evaluation
class TestScanBrackets:
    def test_finds_all_sign_changes_of_sin(self):
        br = scan_brackets(np.sin, 0.5, 10.0, num=1000)
        assert br.shape == (3, 2)
        roots = [bisection(np.sin, a, b).root for a, b in br]
        assert np.allclose(roots, [np.pi, 2 * np.pi, 3 * np.pi], atol=1e-11)

    def test_exact_zero_on_grid(self):
        br = scan_brackets(lambda x: x, -1.0, 1.0, num=3)  # node 0 is exact root
        assert br.shape[0] >= 1 and np.any(br[:, 0] == 0.0)
        # exact root at the right end of the grid is reported in the last cell
        br = scan_brackets(lambda x: x - 1.0, 0.0, 1.0, num=5)
        assert br.shape == (1, 2) and br[0, 1] == 1.0
        assert bisection(lambda x: x - 1.0, *br[0]).root == 1.0

    def test_zero_dim_array_start_value_accepted(self):
        assert bisection(cubic, np.array(1.0), np.float32(2.0)).converged
        with pytest.raises(InvalidInputError):
            bisection(cubic, np.array([1.0]), 2.0)

    def test_evaluates_once_on_array(self):
        calls = []

        def f(x):
            calls.append(np.shape(x))
            return np.cos(x)

        scan_brackets(f, 0, 10, num=101)
        assert calls == [(101,)]

    def test_rejects_non_array_function(self):
        with pytest.raises(InvalidInputError):
            scan_brackets(lambda x: 1.0, 0, 1)
        with np.errstate(divide="ignore"), pytest.raises(InvalidInputError):
            scan_brackets(lambda x: 1 / x, -1, 1, num=3)
        with pytest.raises(InvalidInputError):
            scan_brackets(lambda x: x[:-1], 0, 1)
        with pytest.raises(InvalidInputError):
            scan_brackets(lambda x: x > 0.5, 0, 1)
        with pytest.raises(InvalidInputError):
            scan_brackets(np.sin, 1, 0)
