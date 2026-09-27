"""Fixed-budget Monte Carlo, exact-grid BM/GBM, and variance stability study.

Run: python studies/study_monte_carlo.py [--smoke] [--seed N] [--out DIR]
Full mode uses 400 independent replications at each of five sample counts.
Seeds and budgets are fixed before the results are observed; no rerun selects
favorable coverage. The same replications measure RMSE, spread, and coverage.
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import NormalDist

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

from numerical_methods import (  # noqa: E402
    RunningMoments,
    bm_moments,
    brownian_motion,
    gbm_terminal_moments,
    geometric_brownian_motion,
    monte_carlo,
)


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def wilson_interval(successes, total):
    """95% Wilson binomial interval for the observed replication coverage."""
    z = NormalDist().inv_cdf(0.975)
    p = successes / total
    denominator = 1 + z * z / total
    midpoint = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / denominator
    return midpoint - half, midpoint + half


def uniform_squared(rng, n):
    return rng.random(n) ** 2


def process_moments(config, seeds):
    times = np.linspace(0.5, 2.5, config["moment_grid_points"])
    cases = [
        ("BM", brownian_motion, bm_moments, {"x0": 1.0, "mu": 0.3, "sigma": 0.7}),
        ("GBM", geometric_brownian_motion, gbm_terminal_moments,
         {"s0": 100.0, "mu": 0.05, "sigma": 0.2}),
    ]
    rows = []
    for (name, process, analytic, params), seed in zip(cases, seeds, strict=True):
        moments = [RunningMoments() for _ in times]
        rng = np.random.default_rng(seed)
        for start in range(0, config["moment_paths"], config["batch_size"]):
            paths = process(times, min(config["batch_size"], config["moment_paths"] - start),
                            rng=rng, **params)
            for moment, column in zip(moments, paths.T, strict=True):
                moment.update(column)
        n = config["moment_paths"]
        for t, moment in zip(times, moments, strict=True):
            tau = t - times[0]
            mean, variance = analytic(tau, **params)
            mean_se = math.sqrt(variance / n)
            if name == "BM":
                central_fourth = 3 * variance**2
            else:
                # Lognormal fourth central moment, expressed through its
                # excess kurtosis to avoid subtracting nearly equal raw moments.
                a = params["sigma"] ** 2 * tau
                kurtosis = np.exp(4 * a) + 2 * np.exp(3 * a) + 3 * np.exp(2 * a) - 3
                central_fourth = kurtosis * variance**2
            variance_se = math.sqrt(max(0.0, (central_fourth - (n - 3) / (n - 1) * variance**2) / n))
            rows.append({
                "process": name, "time": float(t), "elapsed_time": float(tau), "n_paths": n,
                "empirical_mean": moment.mean, "analytic_mean": mean,
                "mean_error": moment.mean - mean, "analytic_mean_se": mean_se,
                "mean_error_in_se": (moment.mean - mean) / mean_se if mean_se else 0.0,
                "empirical_variance": moment.variance, "analytic_variance": variance,
                "variance_error": moment.variance - variance, "analytic_variance_se": variance_se,
                "variance_error_in_se": (moment.variance - variance) / variance_se if variance_se else 0.0,
            })
    return rows


def replication_study(config, seed):
    rows, observations = [], []
    for n, n_seed in zip(config["sample_counts"], seed.spawn(len(config["sample_counts"])), strict=True):
        estimates, standard_errors, covered = [], [], 0
        for replication, rep_seed in enumerate(n_seed.spawn(config["replications"])):
            result = monte_carlo(uniform_squared, n, rng=np.random.default_rng(rep_seed),
                                 batch_size=config["batch_size"])
            hit = result.ci_low <= 1 / 3 <= result.ci_high
            covered += int(hit)
            estimates.append(result.estimate)
            standard_errors.append(result.std_error)
            observations.append({
                "n_samples": n, "replication": replication, "estimate": result.estimate,
                "standard_error": result.std_error, "ci_low": result.ci_low,
                "ci_high": result.ci_high, "covered": int(hit),
            })
        estimates = np.asarray(estimates)
        rmse = float(np.sqrt(np.mean((estimates - 1 / 3) ** 2)))
        empirical_sd = float(np.std(estimates, ddof=1))
        mean_se = float(np.mean(standard_errors))
        coverage = covered / config["replications"]
        lo, hi = wilson_interval(covered, config["replications"])
        rows.append({
            "n_samples": n, "replications": config["replications"], "rmse": rmse,
            "bias": float(np.mean(estimates) - 1 / 3), "empirical_sd": empirical_sd,
            "mean_reported_se": mean_se, "analytic_se": math.sqrt((4 / 45) / n),
            "empirical_sd_over_mean_se": empirical_sd / mean_se,
            "covered_count": covered, "coverage": coverage,
            "coverage_binomial_se": math.sqrt(coverage * (1 - coverage) / config["replications"]),
            "coverage_wilson95_low": lo, "coverage_wilson95_high": hi,
        })
    slope = float(np.polyfit(np.log(config["sample_counts"]), np.log([row["rmse"] for row in rows]), 1)[0])
    return rows, observations, slope


def stability_study(config, seed):
    offset, scale = 1e9, 1e-3
    y = offset + scale * np.random.default_rng(seed).random(config["stability_samples"])
    moments = RunningMoments()
    for start in range(0, y.size, config["batch_size"]):
        moments.update(y[start:start + config["batch_size"]])
    reference = float(np.var(y - offset, ddof=1))
    naive = float((np.mean(y**2) - np.mean(y)**2) * y.size / (y.size - 1))
    return {
        "n_samples": y.size, "offset": offset, "scale": scale,
        "analytic_variance_before_float64_quantization": scale**2 / 12,
        "centered_reference_variance": reference, "stable_variance": moments.variance,
        "naive_variance": naive,
        "stable_relative_error_vs_centered": abs(moments.variance - reference) / reference,
        "naive_relative_error_vs_centered": abs(naive - reference) / reference,
        "reference_note": "Variance of y - offset uses the same rounded observations near zero.",
    }


def scalar_paths(times, n_paths, rng, *, geometric):
    """Same exact-grid law and path-major draws as the array implementation."""
    result = np.empty((n_paths, times.size))
    dt = np.diff(times)
    scales = np.sqrt(dt)
    initial, mu, sigma = (100.0, 0.05, 0.2) if geometric else (1.0, 0.3, 0.7)
    for i in range(n_paths):
        result[i, 0] = initial
        w = 0.0
        for j, scale in enumerate(scales, start=1):
            w += rng.standard_normal() * scale
            tau = times[j] - times[0]
            result[i, j] = (initial * math.exp((mu - 0.5 * sigma**2) * tau + sigma * w)
                            if geometric else initial + mu * tau + sigma * w)
    return result


def path_timings(config, seed):
    times = np.linspace(0, 1, config["timing_steps"] + 1)
    n = config["timing_paths"]
    rows = []
    for name, process, params, geometric in [
        ("BM", brownian_motion, {"x0": 1.0, "mu": 0.3, "sigma": 0.7}, False),
        ("GBM", geometric_brownian_motion, {"s0": 100.0, "mu": 0.05, "sigma": 0.2}, True),
    ]:
        def array_run(process=process, params=params):
            return process(times, n, rng=np.random.default_rng(seed), **params)

        def scalar_run(geometric=geometric):
            return scalar_paths(times, n, np.random.default_rng(seed), geometric=geometric)

        scalar, array = scalar_run(), array_run()
        discrepancy = float(np.max(np.abs(scalar - array)))
        if not np.allclose(scalar, array, rtol=3e-14, atol=3e-14):
            raise RuntimeError("Scalar and array paths do not agree for the recorded random stream.")
        for implementation, fn in [("scalar_loop", scalar_run), ("array", array_run)]:
            rows.append({"process": name, "implementation": implementation, "n_paths": n,
                         "n_steps": config["timing_steps"], "max_scalar_array_difference": discrepancy,
                         **C.timeit(fn, repeats=config["timing_repeats"]), "warmup": 1})
    return rows


def make_figures(out, moments, convergence, timings):
    plt = C.plt()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for col, name in enumerate(("BM", "GBM")):
        rows = [row for row in moments if row["process"] == name]
        times = [row["elapsed_time"] for row in rows]
        for ax, measure in zip(axes[:, col], ("mean", "variance"), strict=True):
            ax.plot(times, [row[f"analytic_{measure}"] for row in rows], label="Analytic", color="black")
            ax.errorbar(times, [row[f"empirical_{measure}"] for row in rows],
                        yerr=[1.96 * row[f"analytic_{measure}_se"] for row in rows],
                        fmt="o", ms=4, capsize=3, label="Simulation +/- 1.96 analytic SE")
            ax.set(title=f"{name}: {measure}", xlabel="Elapsed time", ylabel=measure.capitalize())
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Exact grid-point process moments ({moments[0]['n_paths']:,} independent paths)")
    fig.savefig(out / "process_moments.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    ns = [row["n_samples"] for row in convergence]
    axes[0].loglog(ns, [row["rmse"] for row in convergence], "o-", label="Empirical RMSE")
    axes[0].loglog(ns, [row["analytic_se"] for row in convergence], "--", label="sqrt((4/45)/N)")
    axes[0].set(xlabel="Samples N", ylabel="Error", title="E[U²] = 1/3")
    axes[0].legend(fontsize=8)
    axes[1].semilogx(ns, [row["empirical_sd_over_mean_se"] for row in convergence], "o-")
    axes[1].axhline(1, color="black", ls="--")
    axes[1].set(xlabel="Samples N", ylabel="Empirical SD / mean reported SE", title="SE calibration")
    coverage = np.array([row["coverage"] for row in convergence])
    bounds = np.array([[row["coverage_wilson95_low"], row["coverage_wilson95_high"]] for row in convergence])
    axes[2].errorbar(ns, coverage, yerr=[coverage - bounds[:, 0], bounds[:, 1] - coverage],
                     fmt="o", capsize=4, label="95% Wilson uncertainty")
    axes[2].axhline(0.95, color="black", ls="--", label="Nominal coverage")
    axes[2].set(xscale="log", xlabel="Samples N", ylabel="Observed coverage",
                title="Normal-interval coverage")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle(f"{convergence[0]['replications']} independent replications per N; fixed seed and budget")
    fig.savefig(out / "monte_carlo_convergence.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for i, row in enumerate(timings):
        ax.bar(i, row["median_s"] * 1000, yerr=row["iqr_s"] * 500, capsize=4,
               color="#287c8e" if row["implementation"] == "array" else "#df9c48")
    ax.set_xticks(range(len(timings)), [f"{row['process']}\n{row['implementation']}" for row in timings])
    ax.set(yscale="log", ylabel="Milliseconds per call (log scale)",
           title="Exact-grid paths: median timing with half-IQR bars")
    fig.savefig(out / "path_timings.png", dpi=150)
    plt.close(fig)


def main():
    args = C.parse_args("monte_carlo", __doc__)
    config = {
        "smoke": args.smoke,
        "sample_counts": [128, 512, 2048] if args.smoke else [128, 512, 2048, 8192, 32768],
        "replications": 40 if args.smoke else 400, "batch_size": 8192,
        "moment_paths": 10_000 if args.smoke else 120_000, "moment_grid_points": 9,
        "known_expectation_samples": 10_000 if args.smoke else 200_000,
        "stability_samples": 10_000 if args.smoke else 200_000,
        "timing_paths": 100 if args.smoke else 2000, "timing_steps": 16 if args.smoke else 128,
        "timing_repeats": 3 if args.smoke else 7,
        "seed_scheme": "SeedSequence(base).spawn(6): expectation, BM, GBM, N sweep, stability, timings; "
                       "N sweep child.spawn(len(N)); each N child.spawn(R). Replications are independent.",
        "confidence": 0.95,
    }
    seeds = np.random.SeedSequence(args.seed).spawn(6)
    known = monte_carlo(uniform_squared, config["known_expectation_samples"],
                        rng=np.random.default_rng(seeds[0]), batch_size=config["batch_size"])
    moments = process_moments(config, seeds[1:3])
    convergence, replications, slope = replication_study(config, seeds[3])
    stability = stability_study(config, seeds[4])
    timings = path_timings(config, seeds[5])
    speedups = {name: timings[i]["median_s"] / timings[i + 1]["median_s"]
                for i, name in [(0, "BM"), (2, "GBM")]}
    results = {
        "config": config, "known_expectation": {"truth": 1 / 3, **asdict(known)},
        "process_moments": moments, "convergence": convergence, "rmse_loglog_slope": slope,
        "variance_stability": stability, "timings": timings, "array_speedup": speedups,
        "limitations": [
            "Intervals approximate sampling uncertainty for iid finite-variance draws; "
            "finite-N coverage is not guaranteed.",
            "Coverage uncertainty is binomial; Wilson intervals are pointwise, not simultaneous.",
            "Process moment checks share paths across times, so errors across grid points are correlated.",
            "Exact grid-point BM/GBM sampling does not eliminate bias in continuous-path functionals.",
            "Timings include RNG construction and allocation, using equivalent path-major random streams.",
            "Reproducibility is within the recorded environment; wall times vary with system load.",
        ],
    }
    C.save_json(args.out / "results.json", results)
    C.save_json(args.out / "metadata.json", C.metadata(config, args.seed))
    save_csv(args.out / "process_moments.csv", moments)
    save_csv(args.out / "convergence.csv", convergence)
    save_csv(args.out / "replications.csv", replications)
    save_csv(args.out / "timings.csv", timings)
    make_figures(args.out, moments, convergence, timings)
    print(f"Monte Carlo: RMSE slope {slope:.6f}; coverage {[row['coverage'] for row in convergence]}")
    print(f"Array speedups: BM {speedups['BM']:.3f}x; GBM {speedups['GBM']:.3f}x")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
