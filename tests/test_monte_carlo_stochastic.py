import numpy as np
import pytest

from numerical_methods import (
    InvalidInputError,
    InvalidSampleError,
    RunningMoments,
    bm_moments,
    brownian_motion,
    gbm_terminal_moments,
    geometric_brownian_motion,
    monte_carlo,
    path_sampler,
)


def normal_sampler(rng, n):
    return rng.standard_normal(n)


class TestMonteCarlo:
    def test_matches_numpy_moments_single_batch(self):
        res = monte_carlo(normal_sampler, 1000, rng=np.random.default_rng(1), batch_size=1000)
        y = np.random.default_rng(1).standard_normal(1000)
        assert res.estimate == pytest.approx(y.mean(), rel=1e-13, abs=1e-15)
        assert res.variance == pytest.approx(y.var(ddof=1), rel=1e-13)
        assert res.std_error == pytest.approx(y.std(ddof=1) / np.sqrt(1000), rel=1e-13)
        assert res.n_samples == 1000 and res.n_batches == 1

    def test_batching_same_stream_agrees_to_rounding(self):
        # standard_normal(n) in pieces consumes the stream identically for this generator,
        # so batching changes only summation order (not guaranteed for all samplers)
        a = monte_carlo(normal_sampler, 10_007, rng=np.random.default_rng(5), batch_size=10_007)
        b = monte_carlo(normal_sampler, 10_007, rng=np.random.default_rng(5), batch_size=333)
        assert b.n_batches == 31
        assert b.estimate == pytest.approx(a.estimate, rel=1e-12, abs=1e-15)
        assert b.variance == pytest.approx(a.variance, rel=1e-12)

    def test_reproducible_with_seed(self):
        r1 = monte_carlo(normal_sampler, 5000, rng=np.random.default_rng(42), batch_size=700)
        r2 = monte_carlo(normal_sampler, 5000, rng=np.random.default_rng(42), batch_size=700)
        assert r1.estimate == r2.estimate and r1.variance == r2.variance

    def test_quantity_and_ci(self):
        # E[U^2] = 1/3 for U ~ U(0,1); Var = 1/5 - 1/9
        res = monte_carlo(lambda rng, n: rng.random(n), 200_000, rng=np.random.default_rng(3),
                          quantity=lambda u: u**2, batch_size=50_000)
        assert abs(res.estimate - 1 / 3) < 5 * res.std_error  # ~5-sigma, non-flaky with fixed seed
        assert res.variance == pytest.approx(4 / 45, rel=0.02)
        lo, hi = res.ci
        assert lo < res.estimate < hi
        assert hi - lo == pytest.approx(2 * 1.959963984540054 * res.std_error, rel=1e-12)

    def test_constant_output(self):
        res = monte_carlo(lambda rng, n: np.full(n, 7.25), 1001, rng=np.random.default_rng(0),
                          batch_size=100)
        assert res.estimate == 7.25 and res.variance == 0.0 and res.std_error == 0.0
        assert res.ci == (7.25, 7.25)

    def test_large_offset_small_variance_stable(self):
        # Y = 1e9 + U(0,1) * 1e-3: naive E[Y^2]-E[Y]^2 in float64 is dominated by cancellation
        offset, scale = 1e9, 1e-3
        res = monte_carlo(lambda rng, n: offset + scale * rng.random(n), 100_000,
                          rng=np.random.default_rng(11), batch_size=7_000)
        true_var = scale**2 / 12
        assert res.variance == pytest.approx(true_var, rel=0.02)
        y = offset + scale * np.random.default_rng(11).random(100_000)
        naive = (np.mean(y**2) - np.mean(y) ** 2) * 100_000 / 99_999
        assert abs(naive - true_var) > 10 * true_var  # the fragile formula fails here

    def test_confidence_level(self):
        r90 = monte_carlo(normal_sampler, 100, rng=np.random.default_rng(0), confidence=0.90)
        r99 = monte_carlo(normal_sampler, 100, rng=np.random.default_rng(0), confidence=0.99)
        assert (r99.ci_high - r99.ci_low) > (r90.ci_high - r90.ci_low)

    def test_confidence_nearest_float_to_one(self):
        res = monte_carlo(normal_sampler, 100, rng=np.random.default_rng(0),
                          confidence=np.nextafter(1.0, 0.0))
        assert np.isfinite(res.ci).all()
        assert res.ci_high - res.estimate > 8 * res.std_error

    @pytest.mark.parametrize("confidence", ["0.95", True, 0.95 + 0j, [0.95], np.nan])
    def test_confidence_rejects_unsupported_inputs(self, confidence):
        with pytest.raises(InvalidInputError):
            monte_carlo(normal_sampler, 10, rng=np.random.default_rng(0), confidence=confidence)

    @pytest.mark.parametrize("bad_value", [np.nan, np.inf, 1j, "1", True])
    def test_quantity_cannot_hide_invalid_samples(self, bad_value):
        with pytest.raises(InvalidSampleError):
            monte_carlo(lambda rng, n: np.full((n, 2), bad_value), 10,
                        quantity=lambda samples: np.ones(samples.shape[0]), rng=np.random.default_rng(0))

    def test_invalid_arguments(self):
        rng = np.random.default_rng(0)
        with pytest.raises(InvalidInputError):
            monte_carlo(normal_sampler, 1, rng=rng)
        with pytest.raises(InvalidInputError):
            monte_carlo(normal_sampler, 10, rng=None)
        with pytest.raises(InvalidInputError):
            monte_carlo(normal_sampler, 10, rng=np.random.RandomState(0))
        with pytest.raises(InvalidInputError):
            monte_carlo(normal_sampler, 10, rng=rng, batch_size=0)
        with pytest.raises(InvalidInputError):
            monte_carlo(normal_sampler, 10, rng=rng, confidence=1.0)
        with pytest.raises(InvalidInputError):
            monte_carlo(3, 10, rng=rng)
        with pytest.raises(InvalidInputError):
            monte_carlo(normal_sampler, 10.0, rng=rng)

    def test_invalid_samples(self):
        rng = np.random.default_rng(0)
        with pytest.raises(InvalidSampleError):
            monte_carlo(lambda r, n: r.standard_normal(n + 1), 10, rng=rng)
        with pytest.raises(InvalidSampleError):
            monte_carlo(lambda r, n: r.standard_normal((n, 2)), 10, rng=rng)  # no quantity
        with pytest.raises(InvalidSampleError):
            monte_carlo(lambda r, n: 1.0, 10, rng=rng)
        with pytest.raises(InvalidSampleError, match="non-finite"):
            monte_carlo(lambda r, n: np.where(np.arange(n) == 3, np.nan, 1.0), 10, rng=rng)
        with pytest.raises(InvalidSampleError):
            monte_carlo(lambda r, n: np.full(n, np.inf), 10, rng=rng)
        with pytest.raises(InvalidSampleError):
            monte_carlo(normal_sampler, 10, rng=rng, quantity=lambda y: y + 0j)
        with pytest.raises(InvalidSampleError):
            monte_carlo(normal_sampler, 10, rng=rng, quantity=lambda y: y[:, None])
        with pytest.raises(InvalidSampleError, match="non-finite"):
            monte_carlo(normal_sampler, 10, rng=rng, quantity=lambda y: np.full(y.shape, np.nan))


class TestRunningMoments:
    @pytest.mark.parametrize("bad", [[[1.0, 2.0]], [np.inf], [True], [1j], ["1"]])
    def test_rejects_invalid_batches(self, bad):
        moments = RunningMoments()
        with pytest.raises(InvalidInputError):
            moments.update(bad)
        assert moments.n == 0

    def test_merge_equals_two_pass(self):
        rng = np.random.default_rng(9)
        y = rng.standard_normal(10_000) * 3 + 50
        m = RunningMoments()
        for chunk in np.array_split(y, 37):
            m.update(chunk)
        assert m.n == y.size
        assert m.mean == pytest.approx(y.mean(), rel=1e-14)
        assert m.variance == pytest.approx(y.var(ddof=1), rel=1e-12)

    def test_single_element_batches(self):
        m = RunningMoments()
        for v in (1.0, 2.0, 4.0):
            m.update(np.array([v]))
        assert m.mean == pytest.approx(7 / 3) and m.variance == pytest.approx(np.var([1, 2, 4], ddof=1))
        m.update(np.array([]))
        assert m.n == 3
        assert np.isnan(RunningMoments().variance)


class TestProcesses:
    def test_shapes_and_initial_values(self):
        t = np.array([0.5, 0.75, 1.0, 2.0])
        rng = np.random.default_rng(0)
        X = brownian_motion(t, 7, rng=rng, x0=2.0)
        S = geometric_brownian_motion(t, 7, rng=rng, s0=3.0)
        assert X.shape == (7, 4) and S.shape == (7, 4)
        assert np.all(X[:, 0] == 2.0) and np.all(S[:, 0] == 3.0)
        assert np.all(S > 0)

    def test_zero_volatility_deterministic(self):
        t = np.linspace(0.0, 2.0, 5)
        rng = np.random.default_rng(1)
        X = brownian_motion(t, 3, rng=rng, x0=1.0, mu=0.5, sigma=0.0)
        assert np.allclose(X, 1.0 + 0.5 * t, rtol=0, atol=1e-15)
        S = geometric_brownian_motion(t, 3, rng=rng, s0=2.0, mu=0.1, sigma=0.0)
        assert np.allclose(S, 2.0 * np.exp(0.1 * t), rtol=1e-15)

    def test_reproducible(self):
        t = np.linspace(0, 1, 11)
        a = geometric_brownian_motion(t, 50, rng=np.random.default_rng(3), mu=0.1, sigma=0.3)
        b = geometric_brownian_motion(t, 50, rng=np.random.default_rng(3), mu=0.1, sigma=0.3)
        assert np.array_equal(a, b)

    def test_gbm_is_exp_of_bm_with_same_stream(self):
        t = np.array([0.0, 0.3, 1.0])
        mu, sig = 0.2, 0.4
        W = brownian_motion(t, 20, rng=np.random.default_rng(8), sigma=1.0)
        S = geometric_brownian_motion(t, 20, rng=np.random.default_rng(8), s0=1.5, mu=mu, sigma=sig)
        assert np.allclose(S, 1.5 * np.exp((mu - sig**2 / 2) * t + sig * W), rtol=1e-14)

    def test_bm_moments_statistically(self):
        # fixed seed; 6-sigma bands keep this deterministic and non-flaky
        t = np.array([0.0, 0.5, 2.0])
        n = 200_000
        X = brownian_motion(t, n, rng=np.random.default_rng(2), x0=1.0, mu=0.3, sigma=0.7)
        for j in (1, 2):
            m, v = bm_moments(t[j], mu=0.3, sigma=0.7, x0=1.0)
            assert abs(X[:, j].mean() - m) < 6 * np.sqrt(v / n)
            assert abs(X[:, j].var(ddof=1) - v) < 6 * v * np.sqrt(2 / n)
        # independent increments: correlation of disjoint increments ~ 0
        c = np.corrcoef(X[:, 1] - X[:, 0], X[:, 2] - X[:, 1])[0, 1]
        assert abs(c) < 6 / np.sqrt(n)

    def test_gbm_terminal_via_monte_carlo(self):
        t = np.linspace(0.0, 1.0, 5)
        mu, sig, s0 = 0.05, 0.2, 100.0
        res = monte_carlo(path_sampler(geometric_brownian_motion, t, s0=s0, mu=mu, sigma=sig),
                          100_000, rng=np.random.default_rng(4), batch_size=8_192,
                          quantity=lambda paths: paths[:, -1])
        m, v = gbm_terminal_moments(1.0, s0=s0, mu=mu, sigma=sig)
        assert abs(res.estimate - m) < 5 * res.std_error
        assert res.variance == pytest.approx(v, rel=0.03)
        assert res.n_batches == 13

    def test_path_sampler_copies_grid(self):
        t = np.linspace(0, 1, 3)
        s = path_sampler(brownian_motion, t)
        t[1] = 5.0
        assert s(np.random.default_rng(0), 2).shape == (2, 3)

    @pytest.mark.parametrize("times", [[False, True], ["0", "1"], [0j, 1j]])
    def test_path_sampler_validates_before_conversion(self, times):
        with pytest.raises(InvalidInputError):
            path_sampler(brownian_motion, times)

    def test_path_sampler_requires_callable(self):
        with pytest.raises(InvalidInputError):
            path_sampler(None, [0.0, 1.0])

    @pytest.mark.parametrize("moment_function", [bm_moments, gbm_terminal_moments])
    @pytest.mark.parametrize("kwargs", [{"tau": -1}, {"tau": "1"}, {"tau": 1, "sigma": -1},
                                       {"tau": 1, "mu": np.inf}, {"tau": True}])
    def test_moment_helpers_validate_inputs(self, moment_function, kwargs):
        with pytest.raises(InvalidInputError):
            moment_function(**kwargs)

    def test_gbm_moments_requires_positive_initial_value(self):
        with pytest.raises(InvalidInputError):
            gbm_terminal_moments(1, s0=0)

    def test_invalid(self):
        rng = np.random.default_rng(0)
        for bad in ([0.0], [0.0, 0.0], [1.0, 0.5], [-1.0, 0.0], [0.0, np.nan], [[0.0, 1.0]]):
            with pytest.raises(InvalidInputError):
                brownian_motion(bad, 2, rng=rng)
        with pytest.raises(InvalidInputError):
            brownian_motion([0, 1], 0, rng=rng)
        with pytest.raises(InvalidInputError):
            brownian_motion([0, 1], 2, rng=rng, sigma=-1)
        with pytest.raises(InvalidInputError):
            brownian_motion([0, 1], 2, rng=None)
        with pytest.raises(InvalidInputError):
            geometric_brownian_motion([0, 1], 2, rng=rng, s0=0.0)
        with pytest.raises(InvalidInputError):
            geometric_brownian_motion([0, 1], 2, rng=rng, mu=np.inf)
        with pytest.raises(InvalidInputError):
            path_sampler(brownian_motion, [1.0, 0.0])

    def test_analytic_moment_helpers(self):
        assert bm_moments(2.0, mu=1.0, sigma=3.0, x0=1.0) == (3.0, 18.0)
        m, v = gbm_terminal_moments(1.0, s0=2.0, mu=0.0, sigma=0.0)
        assert m == 2.0 and v == 0.0
