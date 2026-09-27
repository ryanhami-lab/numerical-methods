"""Batched E[U²] estimation and exact-grid GBM terminal mean, using explicit RNGs."""

import numpy as np

from numerical_methods import gbm_terminal_moments, geometric_brownian_motion, monte_carlo, path_sampler


def main():
    squared_uniform = monte_carlo(lambda rng, n: rng.random(n), 50_000,
                                 quantity=lambda samples: samples**2,
                                 rng=np.random.default_rng(42), batch_size=4096)
    print(f"E[U^2] = 1/3; estimate={squared_uniform.estimate:.6f}, 95% CI={squared_uniform.ci}")

    times = np.linspace(0, 1, 17)
    sampler = path_sampler(geometric_brownian_motion, times, s0=100, mu=0.05, sigma=0.2)
    terminal = monte_carlo(sampler, 50_000, rng=np.random.default_rng(43),
                           quantity=lambda paths: paths[:, -1], batch_size=4096)
    expected, _ = gbm_terminal_moments(1, s0=100, mu=0.05, sigma=0.2)
    print(f"GBM terminal mean: analytic={expected:.6f}, estimate={terminal.estimate:.6f}")
    print(f"Approximate sampling CI={terminal.ci}; generated {terminal.n_batches} batches")


if __name__ == "__main__":
    main()
