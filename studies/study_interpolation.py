"""Polynomial recovery and the Runge phenomenon with recorded timing uncertainty.

Run from any working directory after installing ``numerical-methods[studies]``:
    python path/to/studies/study_interpolation.py --seed 20260927

The reported sup norms and Lebesgue constants are estimates on a dense uniform
grid, not rigorous bounds. Timing samples include one warm-up and report the
median and IQR; construction and evaluation are measured separately.
"""

from __future__ import annotations

import csv

import numpy as np
from _common import metadata, parse_args, plt, save_json, timeit

from numerical_methods import BarycentricInterpolator, chebyshev_nodes


def runge(x):
    return 1.0 / (1.0 + 25.0 * x * x)


def polynomial(coefficients, x):
    """Horner evaluation; coefficients are in descending power order."""
    value = np.zeros_like(x)
    for coefficient in coefficients:
        value = value * x + coefficient
    return value


def lebesgue_estimate(interpolant, grid):
    """Maximum sum of absolute cardinal functions on the recorded grid."""
    largest = 1.0
    for start in range(0, grid.size, 2048):
        diff = grid[start:start + 2048, None] - interpolant.nodes[None, :]
        away = np.all(diff != 0.0, axis=1)
        diff = diff[away]
        if diff.size:
            coefficients = interpolant.weights * (np.min(np.abs(diff), axis=1, keepdims=True) / diff)
            values = np.sum(np.abs(coefficients), axis=1) / np.abs(np.sum(coefficients, axis=1))
            largest = max(largest, float(np.max(values)))
    return largest


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args("interpolation", __doc__)
    config = {
        "smoke": args.smoke,
        "interval": [-1.0, 1.0],
        "degrees": [4, 12, 20] if args.smoke else list(range(4, 41, 2)),
        "polynomial_degrees": [0, 1, 2, 5, 8],
        "evaluation_points": 1001 if args.smoke else 10001,
        "timing_repeats": 3 if args.smoke else 9,
        "timing_warmup": 1,
        "timing_min_sample_seconds": 0.002,
        "chebyshev_kind": 2,
        "error_metric": "maximum absolute error on uniform evaluation grid",
        "lebesgue_metric": "maximum sum of absolute cardinal functions on same grid",
        "seed_use": "one default_rng stream for polynomial coefficients; Runge experiment is deterministic",
    }
    rng = np.random.default_rng(args.seed)
    grid = np.linspace(-1.0, 1.0, config["evaluation_points"])
    target = runge(grid)
    polynomial_rows = []
    for degree in config["polynomial_degrees"]:
        coefficients = rng.normal(size=degree + 1)
        nodes = chebyshev_nodes(degree + 1, kind=1)
        values = polynomial(coefficients, nodes)
        interpolant = BarycentricInterpolator(nodes, values)
        error = float(np.max(np.abs(interpolant(grid) - polynomial(coefficients, grid))))
        rounding_scale = np.finfo(float).eps * (degree + 1)**2 * np.sum(np.abs(coefficients))
        polynomial_rows.append({
            "degree": degree,
            "max_abs_error": error,
            "rounding_scale": float(rounding_scale),
            "error_over_rounding_scale": error / rounding_scale,
            "exact_values_at_nodes": bool(np.array_equal(interpolant(nodes), values)),
        })
    rows = []
    for degree in config["degrees"]:
        for node_type in ("equispaced", "chebyshev"):
            nodes = (np.linspace(-1.0, 1.0, degree + 1) if node_type == "equispaced"
                     else chebyshev_nodes(degree + 1))
            values = runge(nodes)
            interpolant = BarycentricInterpolator(nodes, values)
            construction = timeit(
                lambda nodes=nodes, values=values: BarycentricInterpolator(nodes, values),
                repeats=config["timing_repeats"],
            )
            evaluation = timeit(
                lambda interpolant=interpolant: interpolant(grid), repeats=config["timing_repeats"],
            )
            rows.append({
                "node_type": node_type,
                "degree": degree,
                "n_nodes": degree + 1,
                "max_abs_error": float(np.max(np.abs(interpolant(grid) - target))),
                "lebesgue_estimate": lebesgue_estimate(interpolant, grid),
                "exact_values_at_nodes": bool(np.array_equal(interpolant(nodes), values)),
                **{f"construction_{key}": value for key, value in construction.items()},
                **{f"evaluation_{key}": value for key, value in evaluation.items()},
            })
    if not all(row["exact_values_at_nodes"] for row in polynomial_rows + rows):
        raise AssertionError("Interpolation must return supplied values exactly at nodes.")
    if not all(row["error_over_rounding_scale"] < 10.0 for row in polynomial_rows):
        raise AssertionError("Polynomial recovery exceeded its roundoff-scaled check.")
    if not all(np.isfinite(row["max_abs_error"]) and np.isfinite(row["lebesgue_estimate"]) for row in rows):
        raise AssertionError("Runge experiment produced non-finite results.")
    save_json(args.out / "metadata.json", metadata(config, args.seed))
    save_json(args.out / "results.json", {"polynomial_recovery": polynomial_rows, "runge": rows})
    save_csv(args.out / "polynomial_recovery.csv", polynomial_rows)
    save_csv(args.out / "runge.csv", rows)
    plots = plt()
    fig, axes = plots.subplots(1, 2, figsize=(10.4, 4.0), constrained_layout=True)
    for node_type, color in (("equispaced", "#b34a35"), ("chebyshev", "#1e6c99")):
        selected = [row for row in rows if row["node_type"] == node_type]
        degrees = [row["degree"] for row in selected]
        for axis, field in zip(axes, ("max_abs_error", "lebesgue_estimate"), strict=True):
            axis.semilogy(degrees, [row[field] for row in selected], "o-", color=color, label=node_type)
            axis.set_xlabel("Polynomial degree")
            axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("Maximum absolute error (grid estimate)")
    axes[1].set_ylabel("Lebesgue constant (grid estimate)")
    axes[0].set_title("Runge function: 1 / (1 + 25x²)")
    axes[1].set_title("Sensitivity to perturbations in data")
    axes[0].legend()
    fig.savefig(args.out / "runge_error_lebesgue.png", dpi=160)
    plots.close(fig)

    illustrated_degree = 20
    fig, axes = plots.subplots(1, 2, figsize=(10.4, 4.0), constrained_layout=True)
    for axis, node_type, color in zip(axes, ("equispaced", "chebyshev"), ("#b34a35", "#1e6c99"),
                                     strict=True):
        nodes = (np.linspace(-1.0, 1.0, illustrated_degree + 1) if node_type == "equispaced"
                 else chebyshev_nodes(illustrated_degree + 1))
        interpolant = BarycentricInterpolator(nodes, runge(nodes))
        axis.plot(grid, interpolant(grid), color=color, label="Interpolant")
        axis.plot(grid, target, color="#222222", linestyle="--", label="Runge function")
        axis.plot(nodes, runge(nodes), ".", color=color, label="Nodes")
        axis.set(xlabel="x", ylabel="Value", title=f"Degree {illustrated_degree}, {node_type}")
        axis.grid(True, alpha=0.25)
        axis.legend()
    fig.suptitle("Each panel uses its own vertical scale")
    fig.savefig(args.out / "runge_interpolants.png", dpi=160)
    plots.close(fig)

    fig, axes = plots.subplots(1, 2, figsize=(10.4, 4.0), constrained_layout=True)
    for node_type, color in (("equispaced", "#b34a35"), ("chebyshev", "#1e6c99")):
        selected = [row for row in rows if row["node_type"] == node_type]
        degrees = np.array([row["degree"] for row in selected])
        for axis, operation in zip(axes, ("construction", "evaluation"), strict=True):
            medians = np.array([row[f"{operation}_median_s"] for row in selected]) * 1000.0
            iqrs = np.array([row[f"{operation}_iqr_s"] for row in selected]) * 1000.0
            axis.plot(degrees, medians, "o-", color=color, label=node_type)
            axis.errorbar(degrees, medians, yerr=iqrs / 2.0, color=color, linestyle="none", capsize=2)
            axis.set(xlabel="Polynomial degree", ylabel="Time (ms)", title=operation.capitalize())
            axis.grid(True, alpha=0.25)
    axes[0].legend()
    fig.suptitle(f"Median timings; bars span one IQR; evaluation at {grid.size:,} points")
    fig.savefig(args.out / "timings.png", dpi=160)
    plots.close(fig)
    last = {row["node_type"]: row for row in rows if row["degree"] == config["degrees"][-1]}
    print(f"Saved interpolation study to {args.out}")
    print(f"Degree {config['degrees'][-1]} max error: "
          f"equispaced={last['equispaced']['max_abs_error']:.6g}, "
          f"Chebyshev={last['chebyshev']['max_abs_error']:.6g}")


if __name__ == "__main__":
    main()
