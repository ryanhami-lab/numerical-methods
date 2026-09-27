"""Study 1: scalar root-finding (bisection, Newton, secant, fixed point).

Run:  python studies/study_root_finding.py [--smoke] [--seed N] [--out DIR]

Outputs (results/root_finding/): results.json, runs.csv, metadata.json,
convergence_cubic.png, convergence_repeated.png.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

from numerical_methods import (  # noqa: E402
    InvalidInputError,
    bisection,
    fixed_point,
    newton,
    scan_brackets,
    secant,
)

EPS = np.finfo(float).eps


def reference_roots():
    """Independent reference values (closed forms, or scipy.optimize.brentq)."""
    from scipy.optimize import brentq
    from scipy.special import lambertw

    s = math.sqrt(26 / 27)
    cubic = (1 + s) ** (1 / 3) + (1 - s) ** (1 / 3)  # Cardano
    dottie = brentq(lambda x: math.cos(x) - x, 0, 1, xtol=1e-16, rtol=4 * EPS)
    exp_small = float((-lambertw(-1 / 3, 0)).real)  # e^x = 3x  <=>  x = -W(-1/3)
    return {"cubic": cubic, "cos_minus_x": dottie, "exp_minus_3x": exp_small,
            "repeated_quadratic": 1.0, "repeated_cubic": 1.0}


def problems(ref):
    return [
        dict(name="cubic", f=lambda x: x**3 - x - 2, df=lambda x: 3 * x * x - 1,
             g=lambda x: (x + 2) ** (1 / 3), bracket=(1.0, 2.0), x0=1.5, secant=(1.0, 2.0),
             root=ref["cubic"], note="simple root; g'(r)~0.14"),
        dict(name="cos_minus_x", f=lambda x: math.cos(x) - x, df=lambda x: -math.sin(x) - 1,
             g=math.cos, bracket=(0.0, 1.0), x0=0.5, secant=(0.0, 1.0),
             root=ref["cos_minus_x"], note="simple root; g'(r)~-0.67"),
        dict(name="exp_minus_3x", f=lambda x: math.exp(x) - 3 * x, df=lambda x: math.exp(x) - 3,
             g=lambda x: math.exp(x) / 3, bracket=(0.0, 1.0), x0=0.5, secant=(0.0, 1.0),
             root=ref["exp_minus_3x"], note="smaller of two roots; g'(r)~0.62"),
        dict(name="repeated_quadratic", f=lambda x: (x - 1) ** 2, df=lambda x: 2 * (x - 1),
             g=None, bracket=None, x0=2.0, secant=(2.0, 2.5),
             root=1.0, note="double root; Newton linear with ratio 1/2; no sign-change bracket"),
        dict(name="repeated_cubic", f=lambda x: (x - 1) ** 3, df=lambda x: 3 * (x - 1) ** 2,
             g=None, bracket=(0.0, 3.0), x0=2.0, secant=(2.0, 2.5),
             root=1.0, note="triple root; Newton linear with ratio 2/3"),
    ]


def run_method(p, method, **kw):
    if method == "bisection":
        return bisection(p["f"], *p["bracket"], **kw)
    if method == "newton":
        return newton(p["f"], p["df"], p["x0"], **kw)
    if method == "secant":
        return secant(p["f"], *p["secant"], **kw)
    kw.pop("ftol", None)
    return fixed_point(p["g"], p["x0"], f=p["f"], **kw)


def row(problem, method, res, root, extra=""):
    return {
        "problem": problem, "method": method, "reason": res.reason, "converged": res.converged,
        "root": res.root, "abs_error": abs(res.root - root) if root is not None else None,
        "residual": res.residual, "residual_kind": res.residual_kind,
        "equation_residual": res.equation_residual, "iterations": res.iterations,
        "nfev": res.nfev, "njev": res.njev, "note": extra,
    }


def order_estimates(x_hist, root, floor):
    """p_k = log(e_{k+1}/e_k) / log(e_k/e_{k-1}) using only errors above the saturation floor."""
    e = np.abs(np.asarray(x_hist) - root)
    keep = np.nonzero(e <= floor)[0]
    e = e[: keep[0]] if keep.size else e
    if e.size < 3:
        return [], e
    p = np.log(e[2:] / e[1:-1]) / np.log(e[1:-1] / e[:-2])
    return p.tolist(), e


def main():
    a = C.parse_args("root_finding", __doc__)
    repeats = 5 if a.smoke else 30
    grid_n = 10_000 if a.smoke else 1_000_000
    ref = reference_roots()
    tol = dict(xtol=1e-12, rtol=4 * EPS, ftol=1e-12)
    config = {"tolerances": tol, "timing_repeats": repeats, "grid_points": grid_n,
              "stopping_note": "Fixed point uses xtol+rtol*abs(x) on |g(x)-x|, not ftol on |f(x)|; "
                               "equation residual and actual error are reported separately.",
              "reference_roots": ref,
              "reference_sources": {"cubic": "Cardano", "cos_minus_x": "scipy brentq xtol=1e-16",
                                    "exp_minus_3x": "-lambertw(-1/3)",
                                    "repeated_quadratic": "exact", "repeated_cubic": "exact"}}

    runs, timings, orders = [], [], {}
    methods = ["bisection", "newton", "secant", "fixed_point"]

    # --- 1. ordinary convergence + repeated root, common tolerances -------------
    for p in problems(ref):
        for m in methods:
            if m == "fixed_point" and p["g"] is None:
                continue
            if m == "bisection" and p["bracket"] is None:
                continue
            res = run_method(p, m, **tol)
            runs.append(row(p["name"], m, res, p["root"], p["note"]))
            t = C.timeit(lambda p=p, m=m: run_method(p, m, **tol), repeats=repeats)
            timings.append({"problem": p["name"], "method": m, **t,
                            "abs_error": abs(res.root - p["root"])})

    # --- 2. convergence order before saturation (tolerances 0, history on) -----
    for p in problems(ref):
        floor = 100 * EPS * max(1.0, abs(p["root"]))
        orders[p["name"]] = {}
        for m in methods:
            if m == "fixed_point" and p["g"] is None:
                continue
            if m == "bisection" and p["bracket"] is None:
                continue
            kw = dict(xtol=0.0, rtol=0.0, ftol=0.0, max_iter=200, record_history=True)
            if m == "fixed_point":
                kw.pop("ftol")
            res = run_method(p, m, **kw)
            ps, e = order_estimates(res.history["x"], p["root"], floor)
            metric, metric_values = "point error", e
            if m == "bisection":
                # Point errors can oscillate within nested brackets. The
                # shrinking bracket width, not those point errors, is linear.
                widths = res.history["b"] - res.history["a"]
                ps, metric_values = order_estimates(widths, 0.0, floor)
                metric = "bracket width"
            ratios = (metric_values[1:] / metric_values[:-1]).tolist() if metric_values.size > 1 else []
            orders[p["name"]][m] = {"errors": e.tolist(), "order_estimates": ps,
                                    "order_metric": metric, "order_metric_values": metric_values.tolist(),
                                    "error_ratios": ratios,
                                    "last_order_estimate": ps[-1] if ps else None,
                                    "last_ratio": ratios[-1] if ratios else None}

    # --- 3. unsuitable starts and failure cases --------------------------------
    atan, datan = math.atan, (lambda x: 1 / (1 + x * x))
    fail = []
    for x0 in (1.3, 1.39, 1.4, 1.5):
        r = newton(atan, datan, x0, max_iter=100)
        fail.append(row(f"atan x0={x0}", "newton", r, 0.0,
                        "Newton on atan converges only for |x0| < ~1.3917"))
    r = newton(lambda x: x**3 - 2 * x + 2, lambda x: 3 * x * x - 2, 0.0, max_iter=50,
               record_history=True)
    fail.append(row("x^3-2x+2 x0=0", "newton", r, None,
                    f"2-cycle, iterates {r.history['x'][:5].tolist()}"))
    fail.append(row("x^3-2x+2 bracket [-2,-1]", "bisection",
                    bisection(lambda x: x**3 - 2 * x + 2, -2.0, -1.0), -1.7692923542386314,
                    "same problem, bracketing method succeeds (reference checked with scipy brentq)"))
    fail.append(row("x^2-1 x0=0", "newton", newton(lambda x: x * x - 1, lambda x: 2 * x, 0.0), 1.0,
                    "f'(x0)=0"))
    fail.append(row("x^2-1 x0=-2,x1=2", "secant", secant(lambda x: x * x - 1, -2.0, 2.0), 1.0,
                    "f(x0)=f(x1)"))
    fail.append(row("e^x=3x, upper root", "fixed_point",
                    fixed_point(lambda x: math.exp(x) / 3, 1.6, max_iter=1000), 1.5121345516578424,
                    "g'(r)=r~1.51>1: repelling fixed point, iterates diverge"))
    fail.append(row("e^x=3x, upper root", "fixed_point (log form)",
                    fixed_point(lambda x: math.log(3 * x), 1.6, f=lambda x: math.exp(x) - 3 * x),
                    1.5121345516578424, "g=log(3x), g'(r)=1/r~0.66: converges"))
    try:
        bisection(lambda x: x * x + 1, -1.0, 1.0)
        rejected = "not rejected (unexpected)"
    except InvalidInputError as exc:
        rejected = f"InvalidInputError: {exc}"
    fail.append({"problem": "x^2+1 on [-1,1]", "method": "bisection", "reason": "rejected",
                 "converged": False, "note": rejected})
    r = newton(lambda x: x * x - 2, lambda x: 2 * x, 1.5, ftol=1e-30)
    fail.append(row("x^2-2, ftol=1e-30", "newton", r, math.sqrt(2),
                    "unattainable residual -> stagnation, not claimed converged"))

    # repeated root under the residual test: small |f| does not imply small error
    rr = newton(lambda x: (x - 1) ** 3, lambda x: 3 * (x - 1) ** 2, 2.0, **tol)
    fail.append(row("(x-1)^3 residual test", "newton", rr, 1.0,
                    "|f| <= 1e-12 reached with error ~1e-4: residual is |e|^3"))

    # --- 4. array-based nonlinear function evaluation --------------------------
    def fv(x):
        return np.sin(x) * np.exp(-x / 10)

    x = np.linspace(0.5, 50.0, grid_n)
    t_arr = C.timeit(lambda: fv(x), repeats=repeats)
    xs_list = x.tolist()
    sub = xs_list[: min(len(xs_list), 100_000)]
    t_loop = C.timeit(lambda: [math.sin(v) * math.exp(-v / 10) for v in sub], repeats=max(3, repeats // 5))
    br = scan_brackets(fv, 0.5, 50.0, num=grid_n)
    roots = np.array([bisection(fv, lo, hi).root for lo, hi in br])
    expected = np.pi * np.arange(1, roots.size + 1)
    array_eval = {
        "function": "sin(x) exp(-x/10) on [0.5, 50]",
        "grid_points": grid_n,
        "array_eval_median_s": t_arr["median_s"],
        "array_ns_per_point": 1e9 * t_arr["median_s"] / grid_n,
        "python_loop_points": len(sub),
        "python_loop_ns_per_point": 1e9 * t_loop["median_s"] / len(sub),
        "array_timing": t_arr,
        "python_loop_timing": t_loop,
        "n_brackets": int(br.shape[0]),
        "max_abs_error_vs_k_pi": float(np.max(np.abs(roots - expected))),
    }

    # --- save -------------------------------------------------------------------
    C.save_json(a.out / "metadata.json", C.metadata(config, a.seed))
    C.save_json(a.out / "results.json", {"runs": runs, "failures": fail, "timings": timings,
                                         "orders": orders, "array_evaluation": array_eval})
    cols = ["problem", "method", "reason", "converged", "root", "abs_error", "residual",
            "residual_kind", "equation_residual", "iterations", "nfev", "njev", "note"]
    with open(a.out / "runs.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r_ in runs + fail:
            w.writerow(r_)

    plt = C.plt()
    for name, fname in (("cubic", "convergence_cubic.png"),
                        ("repeated_quadratic", "convergence_repeated.png")):
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for m, d in orders[name].items():
            if d["errors"]:
                ax.semilogy(d["errors"], marker="o", ms=3, label=m)
        ax.set_xlabel("recorded iterate index")
        ax.set_ylabel("|x_k - root|")
        ax.set_title(f"Error vs iteration: {name} (stopped at 100 eps floor)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(a.out / fname, dpi=110)
        plt.close(fig)

    # --- console summary --------------------------------------------------------
    print(f"{'problem':16s} {'method':12s} {'reason':10s} {'abs_err':>9s} {'resid':>9s} "
          f"{'it':>3s} {'nfev':>4s} {'njev':>4s} {'median_us':>9s}")
    for r_, t in zip(runs, timings, strict=True):
        print(f"{r_['problem']:16s} {r_['method']:12s} {r_['reason']:10s} {r_['abs_error']:9.1e} "
              f"{r_['residual']:9.1e} {r_['iterations']:3d} {r_['nfev']:4d} {r_['njev']:4d} "
              f"{1e6 * t['median_s']:9.2f}")
    print("\nlast order / ratio before saturation (bisection: bracket widths; others: point errors):")
    for name, d in orders.items():
        print(" ", name,
              {m: (None if v["last_order_estimate"] is None else round(v["last_order_estimate"], 3),
                              None if v["last_ratio"] is None else float(f"{v['last_ratio']:.3g}"))
                          for m, v in d.items()})
    print("\nfailure / unsuitable cases:")
    for r_ in fail:
        print(f"  {r_['problem']:26s} {r_['method']:22s} {r_['reason']:16s} conv={r_['converged']}  "
              f"{r_.get('note', '')[:70]}")
    print("\narray evaluation:", array_eval)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
