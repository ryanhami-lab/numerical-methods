import numpy as np
import pytest

from numerical_methods import (
    BarycentricInterpolator,
    InvalidInputError,
    barycentric_weights,
    chebyshev_nodes,
)


def runge(x):
    return 1.0 / (1.0 + 25.0 * x**2)


class TestBarycentric:
    @pytest.mark.parametrize("deg", [0, 1, 2, 5, 8])
    def test_reproduces_polynomials(self, deg):
        rng = np.random.default_rng(deg)
        coef = rng.standard_normal(deg + 1)
        nodes = chebyshev_nodes(deg + 1, -2, 3, kind=1)
        p = BarycentricInterpolator(nodes, np.polyval(coef, nodes))
        xs = np.linspace(-2, 3, 201)
        assert np.allclose(p(xs), np.polyval(coef, xs), rtol=0, atol=1e-12 * np.abs(coef).sum() * 3**deg)

    def test_exact_at_nodes_scalar_and_array(self):
        x = np.array([0.3, -1.0, 2.5, 0.0, 1.0])  # unsorted, includes 0
        y = np.array([1.0, -2.0, 7.0, 3.5, -0.25])
        p = BarycentricInterpolator(x, y)
        for xi, yi in zip(x, y, strict=True):
            v = p(xi)
            assert isinstance(v, float) and v == yi
        assert np.array_equal(p(x), y)
        assert np.all(np.isfinite(p(x + 1e-300)))

    def test_near_node_is_continuous(self):
        x = np.linspace(0, 1, 6)
        p = BarycentricInterpolator(x, np.sin(x))
        assert p(0.4 + 1e-14) == pytest.approx(np.sin(0.4), abs=1e-12)

    def test_one_subnormal_ulp_from_node(self):
        p = BarycentricInterpolator([0.0, 1.0], [2.0, 3.0])
        assert p(np.nextafter(0.0, 1.0)) == 2.0

    @pytest.mark.parametrize("scale", [1e-310, 1e308])
    def test_extreme_node_scales(self, scale):
        nodes = np.array([-1.0, 0.0, 1.0]) * scale
        p = BarycentricInterpolator(nodes, [1.0, 0.0, 1.0])
        assert np.all(np.isfinite(p.weights))
        assert p(scale * 0.5) == pytest.approx(0.25, rel=1e-12)

    def test_large_distance_extrapolation_and_values(self):
        p = BarycentricInterpolator([-1e308, -9e307], [-1.0, -0.9])
        assert p(1e308) == pytest.approx(1.0, abs=1e-13)
        q = BarycentricInterpolator([-1.0, 0.0, 1.0], [1e308, 1e308, 1e308])
        assert q(0.5) == pytest.approx(1e308, rel=1e-15)

    def test_subnormal_constant_values(self):
        value = np.nextafter(0.0, 1.0)
        p = BarycentricInterpolator([-1.0, 0.0, 1.0], [value, value, value])
        assert p(0.5) == value

    def test_unrepresentable_relative_weights_rejected(self):
        with pytest.raises(InvalidInputError, match="weights outside the float64 range"):
            barycentric_weights([0.0, np.nextafter(0.0, 1.0), 1e308])

    def test_shapes(self):
        p = BarycentricInterpolator([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
        grid = np.linspace(0, 2, 12).reshape(3, 4)
        out = p(grid)
        assert out.shape == (3, 4) and np.allclose(out, grid**2)
        assert p(np.array([])).shape == (0,)
        assert isinstance(p(np.float64(0.5)), float)
        assert p.degree_bound == 2

    def test_block_evaluation_consistent(self):
        import numerical_methods.interpolation as mod
        x = chebyshev_nodes(30)
        p = BarycentricInterpolator(x, runge(x))
        xs = np.linspace(-1, 1, 5000)
        full = p(xs)
        old = mod._BLOCK_ELEMENTS
        try:
            mod._BLOCK_ELEMENTS = 64  # force many small blocks
            assert np.allclose(p(xs), full, rtol=1e-15, atol=1e-15)
        finally:
            mod._BLOCK_ELEMENTS = old

    def test_single_node_constant(self):
        p = BarycentricInterpolator([2.0], [5.0])
        assert np.array_equal(p(np.array([0.0, 2.0, 9.0])), [5.0, 5.0, 5.0])

    def test_matches_scipy(self):
        interp = pytest.importorskip("scipy.interpolate")
        x = chebyshev_nodes(21, 0, 3)
        y = np.exp(np.sin(3 * x))
        xs = np.linspace(0, 3, 333)
        ref = interp.BarycentricInterpolator(x, y)(xs)
        assert np.allclose(BarycentricInterpolator(x, y)(xs), ref, rtol=1e-12, atol=1e-12)

    def test_weights_match_formula_and_no_overflow(self):
        x = np.array([0.0, 1.0, 3.0])
        w = 1 / np.array([(0 - 1) * (0 - 3), (1 - 0) * (1 - 3), (3 - 0) * (3 - 1)])
        wb = barycentric_weights(x)
        assert np.allclose(wb / wb[0], w / w[0])
        assert np.max(np.abs(wb)) == 1.0
        w_many = barycentric_weights(chebyshev_nodes(2000, 0, 1e-6))
        assert np.all(np.isfinite(w_many)) and np.all(w_many != 0)

    def test_chebyshev_converges_on_runge_equispaced_does_not(self):
        xs = np.linspace(-1, 1, 2001)
        errs_c, errs_e = [], []
        for n in (11, 21, 41):
            c = chebyshev_nodes(n)
            e = np.linspace(-1, 1, n)
            errs_c.append(np.max(np.abs(BarycentricInterpolator(c, runge(c))(xs) - runge(xs))))
            errs_e.append(np.max(np.abs(BarycentricInterpolator(e, runge(e))(xs) - runge(xs))))
        assert errs_c[0] > errs_c[1] > errs_c[2] and errs_c[2] < 1e-3
        assert errs_e[0] < errs_e[1] < errs_e[2] and errs_e[2] > 100

    def test_duplicate_nodes_rejected(self):
        with pytest.raises(InvalidInputError, match="distinct"):
            BarycentricInterpolator([0.0, 1.0, 1.0], [1.0, 2.0, 3.0])
        with pytest.raises(InvalidInputError):
            barycentric_weights([0.0, 0.0])

    def test_invalid_shapes_and_values(self):
        with pytest.raises(InvalidInputError):
            BarycentricInterpolator([0.0, 1.0], [1.0])
        with pytest.raises(InvalidInputError):
            BarycentricInterpolator([[0.0, 1.0]], [[1.0, 2.0]])
        with pytest.raises(InvalidInputError):
            BarycentricInterpolator([], [])
        with pytest.raises(InvalidInputError):
            BarycentricInterpolator([0.0, np.nan], [1.0, 2.0])
        with pytest.raises(InvalidInputError):
            BarycentricInterpolator([0.0, 1.0], [1.0, np.inf])
        with pytest.raises(InvalidInputError):
            BarycentricInterpolator([0.0, 1.0], [1.0 + 1j, 2.0])
        p = BarycentricInterpolator([0.0, 1.0], [1.0, 2.0])
        with pytest.raises(InvalidInputError):
            p(np.nan)
        with pytest.raises(InvalidInputError):
            p("a")

    def test_internal_state_read_only(self):
        p = BarycentricInterpolator([0.0, 1.0], [1.0, 2.0])
        with pytest.raises(ValueError):
            p.values[0] = 3.0


class TestChebyshevNodes:
    def test_kind2_endpoints_and_order(self):
        x = chebyshev_nodes(5, 2.0, 4.0)
        assert x[0] == pytest.approx(2.0) and x[-1] == pytest.approx(4.0)
        assert np.all(np.diff(x) > 0)
        assert np.allclose(x, 3 + np.cos(np.pi * np.arange(4, -1, -1) / 4))

    def test_kind1_roots_of_T_n(self):
        n = 7
        x = chebyshev_nodes(n, kind=1)
        assert np.allclose(np.cos(n * np.arccos(x)), 0, atol=1e-14)
        assert np.all(np.abs(x) < 1)

    @pytest.mark.parametrize("bounds", [(-1e308, 1e308), (1e308, 1.7e308)])
    def test_extreme_finite_endpoints(self, bounds):
        nodes = chebyshev_nodes(5, *bounds)
        assert np.all(np.isfinite(nodes))
        assert nodes[0] == bounds[0] and nodes[-1] == bounds[1]
        assert np.all(nodes[1:] > nodes[:-1])

    def test_unrepresentable_nodes_rejected(self):
        with pytest.raises(InvalidInputError, match="too narrow"):
            chebyshev_nodes(3, 1.0, np.nextafter(1.0, np.inf))

    @pytest.mark.parametrize("kind", [True, 1.0, "1"])
    def test_kind_must_be_an_integer(self, kind):
        with pytest.raises(InvalidInputError):
            chebyshev_nodes(3, kind=kind)

    def test_invalid(self):
        with pytest.raises(InvalidInputError):
            chebyshev_nodes(1)
        with pytest.raises(InvalidInputError):
            chebyshev_nodes(3, 1, 0)
        with pytest.raises(InvalidInputError):
            chebyshev_nodes(3, kind=3)
        with pytest.raises(InvalidInputError):
            chebyshev_nodes(0, kind=1)
