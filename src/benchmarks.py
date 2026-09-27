import time
from root_finding import bisection, newton, secant, fixed_point


def benchmark_root_finders():
    test_cases = [
        {
            "name": "cubic",
            "f": lambda x: x**3 - x - 2,
            "df": lambda x: 3*x**2 - 1,
            "g": lambda x: (x + 2)**(1/3),
            "bisection_args": (1, 2),
            "newton_args": (1.5,),
            "secant_args": (1.0, 2.0),
            "fixed_point_args": (1.5,),
        },
        {
            "name": "cos_x_minus_x",
            "f": lambda x: __import__("math").cos(x) - x,
            "df": lambda x: -__import__("math").sin(x) - 1,
            "g": lambda x: __import__("math").cos(x),
            "bisection_args": (0, 1),
            "newton_args": (0.5,),
            "secant_args": (0.0, 1.0),
            "fixed_point_args": (0.5,),
        },
        {
            "name": "exp_minus_3x",
            "f": lambda x: __import__("math").exp(x) - 3*x,
            "df": lambda x: __import__("math").exp(x) - 3,
            "g": lambda x: __import__("math").exp(x) / 3,
            "bisection_args": (0, 1),
            "newton_args": (0.5,),
            "secant_args": (0.0, 1.0),
            "fixed_point_args": (0.5,),
        },
    ]

    results = []

    for case in test_cases:
        f = case["f"]
        df = case["df"]

        # bisection
        start = time.perf_counter()
        result_b = bisection(f, *case["bisection_args"])
        time_b = time.perf_counter() - start

        results.append({
            "function": case["name"],
            "method": "bisection",
            "root": result_b.root,
            "iterations": result_b.iterations,
            "residual": result_b.residual,
            "converged": result_b.converged,
            "runtime_sec": time_b,
        })

        # newton
        start = time.perf_counter()
        result_n = newton(f, df, *case["newton_args"])
        time_n = time.perf_counter() - start

        results.append({
            "function": case["name"],
            "method": "newton",
            "root": result_n.root,
            "iterations": result_n.iterations,
            "residual": result_n.residual,
            "converged": result_n.converged,
            "runtime_sec": time_n,
        })

        # secant
        start = time.perf_counter()
        result_s = secant(f, *case["secant_args"])
        time_s = time.perf_counter() - start

        results.append({
            "function": case["name"],
            "method": "secant",
            "root": result_s.root,
            "iterations": result_s.iterations,
            "residual": result_s.residual,
            "converged": result_s.converged,
            "runtime_sec": time_s,
        })

        # fixed point
        start = time.perf_counter()
        result_fp = fixed_point(case["g"], *case["fixed_point_args"])
        time_fp = time.perf_counter() - start

        results.append({
            "function": case["name"],
            "method": "fixed_point",
            "root": result_fp.root,
            "iterations": result_fp.iterations,
            "residual": result_fp.residual,
            "converged": result_fp.converged,
            "runtime_sec": time_fp,
        })

    return results