"""Reproducible LU accuracy, conditioning, factor-reuse and timing study.

Run from any directory with an installed numerical_methods package:
    python /path/to/studies/study_linear_systems.py [--smoke] [--seed 20260927] [--out DIR]

SciPy is the reference implementation. Condition numbers are obtained from
SciPy singular values only for analysis; the package itself implements LU.
"""

from __future__ import annotations

import csv
import warnings
from dataclasses import asdict
from functools import partial

import numpy as np
import scipy.linalg as sla
from _common import metadata, parse_args, plt, save_json, timeit

from numerical_methods import (
    SingularMatrixError,
    factorization_error,
    forward_error,
    lu_factor,
    residual_diagnostics,
    solve,
)


def condition_2(a):
    singular_values = sla.svdvals(a)
    return float(singular_values[0] / singular_values[-1])


def errors(a, x, b, x_true):
    return {"forward_error": forward_error(x, x_true), **asdict(residual_diagnostics(a, x, b))}


def scipy_solution(a, b):
    # Ill-conditioning warnings are evidence and are recorded in the results.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", sla.LinAlgWarning)
        x = sla.solve(a, b, assume_a="gen")
    return x, [str(w.message) for w in caught]


def scipy_factorization_error(a, packed, piv):
    perm = np.arange(len(a))
    for i, j in enumerate(piv):
        perm[[i, j]] = perm[[j, i]]
    lower = np.tril(packed, -1) + np.eye(len(a))
    upper = np.triu(packed)
    return float(np.linalg.norm(a[perm] - lower @ upper, np.inf) / np.linalg.norm(a, np.inf))


def random_systems(rng, config):
    rows = []
    timing = config["timing"]
    for n in config["sizes"]:
        a = rng.standard_normal((n, n))
        x_true = rng.standard_normal(n)
        b = a @ x_true
        fac = lu_factor(a)
        packed, piv = sla.lu_factor(a)
        reference, reference_warnings = scipy_solution(a, b)
        cond = condition_2(a)
        methods = [
            ("package_solve", partial(solve, a, b), fac.solve(b), factorization_error(a, fac)),
            ("scipy_solve", partial(sla.solve, a, b, assume_a="gen"), reference,
             scipy_factorization_error(a, packed, piv)),
            ("package_lu_factor", partial(lu_factor, a), None, factorization_error(a, fac)),
            ("scipy_lu_factor", partial(sla.lu_factor, a), None,
             scipy_factorization_error(a, packed, piv)),
        ]
        for method, fn, x, reconstruction in methods:
            row = {
                "n": n, "method": method, "condition_2": cond,
                "factorization_error": reconstruction, **timeit(fn, **timing),
            }
            if x is not None:
                row.update(errors(a, x, b, x_true))
            if method == "scipy_solve":
                row["warnings"] = reference_warnings
            rows.append(row)
    return rows


def conditioning(config):
    rows = []
    for n in config["hilbert_sizes"]:
        a = sla.hilbert(n)
        x_true = np.ones(n)
        b = a @ x_true
        cond = condition_2(a)
        for policy, tolerance in [("default", None), ("exact_zero_only", 0.0)]:
            row = {"n": n, "method": "package", "pivot_policy": policy, "condition_2": cond}
            try:
                fac = lu_factor(a, pivot_tol=tolerance)
            except SingularMatrixError as exc:
                row.update(status="rejected", message=str(exc))
            else:
                row.update(
                    status="solved", pivot_tol=fac.pivot_tol,
                    minimum_abs_pivot=float(np.min(np.abs(fac.pivots))),
                    factorization_error=factorization_error(a, fac),
                    **errors(a, fac.solve(b), b, x_true),
                )
            rows.append(row)
        reference, caught = scipy_solution(a, b)
        rows.append({
            "n": n, "method": "scipy", "pivot_policy": "scipy_default",
            "status": "solved", "condition_2": cond, "warnings": caught,
            **errors(a, reference, b, x_true),
        })
    return rows


def special_cases():
    a = np.array([[1e-20, 1.0], [1.0, 1.0]])
    x_true = np.ones(2)
    b = a @ x_true
    fac = lu_factor(a)
    # Deliberately unpivoted two-by-two elimination for the comparison only.
    multiplier = a[1, 0] / a[0, 0]
    x1 = (b[1] - multiplier * b[0]) / (a[1, 1] - multiplier * a[0, 1])
    unpivoted = np.array([(b[0] - a[0, 1] * x1) / a[0, 0], x1])
    pivot = {
        "a": a, "b": b, "x_true": x_true, "perm": fac.perm,
        "pivoted_solution": fac.solve(b), "unpivoted_solution": unpivoted,
        "pivoted_errors": errors(a, fac.solve(b), b, x_true),
        "unpivoted_errors": errors(a, unpivoted, b, x_true),
        "factorization_error": factorization_error(a, fac),
    }
    singular = np.array([[1.0, 2.0], [2.0, 4.0]])
    try:
        lu_factor(singular)
    except SingularMatrixError as exc:
        singular_result = {"status": "rejected", "exception": type(exc).__name__, "message": str(exc)}
    else:
        raise AssertionError("Exactly singular test matrix unexpectedly accepted")
    # The threshold is a heuristic. A nonsingular badly scaled diagonal can
    # be rejected; callers may deliberately select pivot_tol=0 in this case.
    badly_scaled = np.diag([1e20, 1.0])
    try:
        lu_factor(badly_scaled)
    except SingularMatrixError as exc:
        scaled_default = {"status": "rejected", "message": str(exc)}
    else:
        raise AssertionError("Documented global pivot threshold was not applied")
    scaled_x = lu_factor(badly_scaled, pivot_tol=0.0).solve(badly_scaled @ x_true)
    return {
        "tiny_leading_pivot": pivot,
        "exactly_singular": {"a": singular, **singular_result},
        "badly_scaled_nonsingular": {
            "a": badly_scaled, "default_policy": scaled_default,
            "exact_zero_only_solution": scaled_x, "forward_error": forward_error(scaled_x, x_true),
        },
    }


def reuse(rng, config):
    n, k = config["reuse_n"], config["right_hand_sides"]
    a = rng.standard_normal((n, n))
    x_true = rng.standard_normal((n, k))
    b = a @ x_true
    fac = lu_factor(a)
    scipy_fac = sla.lu_factor(a)

    def factor_once_then_columns():
        one_fac = lu_factor(a)
        return np.column_stack([one_fac.solve(b[:, j]) for j in range(k)])

    operations = [
        ("package_reused_matrix_rhs", "solve only; factors prepared", lambda: fac.solve(b)),
        ("package_reused_columns", "solve only; factors prepared",
         lambda: np.column_stack([fac.solve(b[:, j]) for j in range(k)])),
        ("scipy_reused_matrix_rhs", "solve only; factors prepared", lambda: sla.lu_solve(scipy_fac, b)),
        ("package_factor_once_then_columns", "one factorization plus all solves", factor_once_then_columns),
        ("package_refactor_each_column", "k factorizations plus all solves",
         lambda: np.column_stack([solve(a, b[:, j]) for j in range(k)])),
    ]
    rows = []
    for method, scope, fn in operations:
        x = fn()
        rows.append({
            "n": n, "n_rhs": k, "method": method, "timing_scope": scope,
            "condition_2": condition_2(a), "forward_error": forward_error(x, x_true),
            "max_column_backward_error": max(
                residual_diagnostics(a, x[:, j], b[:, j]).backward_error for j in range(k)
            ),
            "factorization_error": factorization_error(a, fac), **timeit(fn, **config["timing"]),
        })
    return rows


def save_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def figures(out, random_rows, hilbert_rows, reuse_rows):
    pyplot = plt()
    fig, axes = pyplot.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    for method in ["package_solve", "scipy_solve", "package_lu_factor", "scipy_lu_factor"]:
        rows = [r for r in random_rows if r["method"] == method]
        axes[0].loglog([r["n"] for r in rows], [r["median_s"] for r in rows], "o-", label=method)
    axes[0].set(xlabel="Matrix size n", ylabel="Median seconds per call", title="Random dense systems")
    axes[0].legend(fontsize=8)
    for key, label in [("forward_error", "forward error"), ("backward_error", "backward error")]:
        rows = [r for r in random_rows if r["method"] == "package_solve"]
        axes[1].loglog([r["n"] for r in rows], [r[key] for r in rows], "o-", label=label)
    axes[1].set(xlabel="Matrix size n", ylabel="Relative error", title="Package solve accuracy")
    axes[1].legend()
    for ax in axes:
        ax.grid(True, which="both", alpha=0.25)
    fig.savefig(out / "random_accuracy_timing.png", dpi=160)
    pyplot.close(fig)

    fig, ax = pyplot.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    for method, policy in [("package", "exact_zero_only"), ("scipy", "scipy_default")]:
        rows = [r for r in hilbert_rows if r["method"] == method and r["pivot_policy"] == policy]
        ax.loglog([r["condition_2"] for r in rows], [max(r["forward_error"], 1e-18) for r in rows],
                  "o-", label=f"{method}: forward error")
        ax.loglog([r["condition_2"] for r in rows], [max(r["backward_error"], 1e-18) for r in rows],
                  "s--", label=f"{method}: backward error")
    ax.set(xlabel="Estimated 2-norm condition number", ylabel="Relative error (zeros shown at 1e-18)",
           title="Hilbert systems: small backward error can coexist with large forward error")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(out / "conditioning.png", dpi=160)
    pyplot.close(fig)

    fig, ax = pyplot.subplots(figsize=(9, 4.5), constrained_layout=True)
    ax.barh([r["method"].replace("_", " ") for r in reuse_rows], [r["median_s"] for r in reuse_rows])
    ax.set(xscale="log", xlabel="Median seconds per call (log scale)",
           title=f"Reuse: n={reuse_rows[0]['n']}, {reuse_rows[0]['n_rhs']} right-hand sides")
    ax.grid(True, axis="x", alpha=0.25)
    fig.savefig(out / "factor_reuse.png", dpi=160)
    pyplot.close(fig)


def main():
    args = parse_args("linear_systems", __doc__)
    config = {
        "smoke": args.smoke,
        "sizes": [10, 30] if args.smoke else [10, 25, 50, 100, 200, 400],
        "hilbert_sizes": [2, 6, 10, 12] if args.smoke else list(range(2, 13)),
        "reuse_n": 30 if args.smoke else 200,
        "right_hand_sides": 4 if args.smoke else 8,
        "timing": {"repeats": 3 if args.smoke else 9, "warmup": 2, "min_time": 0.003},
        "random_matrix_distribution": "independent standard normal entries",
        "known_solution_distribution": "independent standard normal entries; Hilbert solution all ones",
        "condition_norm": "2; from scipy.linalg.svdvals, an estimate near floating-point limits",
        "error_norm": "infinity; matrix RHS forward error uses largest entry across all columns",
        "timing_contract": "default finite checks enabled in package and SciPy; inputs preallocated",
    }
    rng = np.random.default_rng(args.seed)
    meta = metadata(config, args.seed)
    random_rows = random_systems(rng, config)
    hilbert_rows = conditioning(config)
    reuse_rows = reuse(rng, config)
    results = {
        "random_systems": random_rows, "conditioning": hilbert_rows,
        "special_cases": special_cases(), "factor_reuse": reuse_rows,
        "interpretation": [
            "Backward error measures how well the computed solution solves a nearby problem.",
            "Ill-conditioning amplifies small data and arithmetic perturbations into forward error.",
            "Hilbert b=A@ones is formed in float64: forward error includes RHS formation rounding.",
            "The pivot threshold is a global-scale heuristic, not proof of mathematical singularity.",
            "Timings describe this recorded machine and environment; SciPy uses compiled LAPACK.",
            "Factor-reuse timing scopes are recorded explicitly; solve-only timings exclude factorization.",
        ],
    }
    save_json(args.out / "metadata.json", meta)
    save_json(args.out / "results.json", results)
    for name, rows in [("random_systems", random_rows), ("conditioning", hilbert_rows),
                       ("factor_reuse", reuse_rows)]:
        save_csv(args.out / f"{name}.csv", rows)
    figures(args.out, random_rows, hilbert_rows, reuse_rows)
    worst = max(r["forward_error"] for r in hilbert_rows if r.get("status") == "solved")
    print(f"Saved linear-system study to {args.out}; largest Hilbert forward error={worst:.6g}")


if __name__ == "__main__":
    main()
