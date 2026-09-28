# numerical-methods

A small educational Python library that implements scalar root finding, dense
LU/Gaussian elimination, barycentric polynomial interpolation, batched Monte Carlo,
and exact grid-point simulation of Brownian and geometric Brownian motion.
The algorithms live in this package; NumPy supplies array operations and random
numbers. SciPy is a reference dependency for tests and studies, not runtime code.

## Install

Requires Python 3.10+ and NumPy 1.24+. Local release verification uses Python 3.11
and 3.13; see [VALIDATION.md](VALIDATION.md) for the exact environments and results.

```powershell
python -m pip install .
# Development, tests, figures, and build tools:
python -m pip install -e '.[dev]'
```

On Windows, a project virtual environment avoids ambiguity about the interpreter:

```powershell
py -3.11 -m venv .venv
$py = '.\.venv\Scripts\python.exe'
& $py -m pip install -e '.[dev]'
```

The distribution name is `numerical-methods`; the import name is
`numerical_methods`. Installation here is from the local checkout or built wheel;
no package has been published by this work.

## Quick starts

### Scalar roots

```python
from numerical_methods import bisection, newton

r = bisection(lambda x: x*x - 2, 1, 2)
print(r.root, r.residual, r.converged, r.reason, r.nfev)
r = newton(lambda x: x*x - 2, lambda x: 2*x, 1.5,
           ftol=1e-12, record_history=True)
```

`bisection(f, a, b)`, `newton(f, df, x0)`, `secant(f, x0, x1)` and
`fixed_point(g, x0, f=None)` solve real scalar equations. Callbacks accept and
return real scalars. `scan_brackets(f, a, b, num=1001)` evaluates `f` once on a
NumPy grid and returns brackets of shape `(k, 2)`; its callback must accept an
array and return the same shape. It can miss even-multiplicity roots and multiple
roots within one grid cell. These are nonlinear **equations**, not a coupled
nonlinear system solver for F: Rⁿ → Rⁿ.

`RootResult` records the iterate, residual, status, termination reason, iteration
and evaluation counts, and optional history. Default tolerances are
`xtol=1e-12`, `rtol=4*eps`, `ftol=1e-12` where applicable. Only `residual` and
`bracket` mean convergence. Newton/secant return `stagnation` when the step is
small but the residual remains too large. Other failure reasons are `max_iter`,
`zero_derivative`, `zero_denominator`, and `nonfinite`.

Bisection needs continuity and an actual sign change or exact endpoint root;
its bracket termination gives an x-error bound under that assumption. `ftol=0`
still accepts an exactly zero computed function value. Newton and secant are
unsafeguarded local methods. A small residual can coexist with substantial root
error, especially at repeated roots or for badly scaled functions.

Fixed-point convergence tests `|g(x)-x| <= xtol + rtol*|x|` at the returned point.
It has no `ftol`. Passing `f` additionally reports `equation_residual=|f(x)|`;
this costs one extra call, excluded from `nfev`, and does not change the stopping
rule. A contraction assumption is needed to infer root error from this residual.

### Dense linear systems

```python
import numpy as np
from numerical_methods import lu_factor, residual_diagnostics

A = np.array([[0., 2.], [1., 3.]])
b = np.array([4., 7.])
fac = lu_factor(A)
x = fac.solve(b)
print(x, residual_diagnostics(A, x, b))
X = fac.solve(np.column_stack([b, 2*b]))  # reuse factors
assert np.allclose(A[fac.perm], fac.L @ fac.U)
```

`A` is nonempty, square, and real; `b` has shape `(n,)` or `(n, k)`, and the
solution preserves that shape. `solve`/`gaussian_elimination` factor and solve;
`lu_factor` lets you reuse the factorization. The permutation convention is
`P @ A = L @ U`, with `P = eye(n)[perm]`. No inverse is formed.

The default rejection threshold is `n*eps*max(abs(A))`. It is a scale-aware
pivot heuristic, not a condition estimate or a proof of exact singularity.
Badly scaled nonsingular systems can trigger it; `pivot_tol=0` rejects only
exactly zero computed pivots. A successful factorization does not establish
good conditioning. Partial pivoting can also exhibit large element growth.

`residual_diagnostics(A, x, b)` takes vectors and reports infinity-norm residual,
relative residual, and normwise backward error `||b-Ax||/(||A||||x||+||b||)`.
`forward_error(x, x_true)` compares to a known solution; `factorization_error`
checks reconstruction. A tiny backward error does not establish an accurate
solution to an ill-conditioned system. Triangular solves and a determinant
method are also available; the determinant can overflow or underflow.

### Polynomial interpolation

```python
import numpy as np
from numerical_methods import BarycentricInterpolator, chebyshev_nodes

nodes = chebyshev_nodes(21, -1, 1)
p = BarycentricInterpolator(nodes, 1/(1 + 25*nodes**2))
grid = np.linspace(-1, 1, 1001)
print(np.max(np.abs(p(grid) - 1/(1 + 25*grid**2))))
```

Nodes and values have shape `(n,)`; nodes must be distinct. A scalar query gives
a float; an array query preserves its shape. Exact nodes return the stored
values exactly. Weights use log scaling and evaluation runs in blocks.
`chebyshev_nodes(n, a, b, kind=2)` includes endpoints and needs n ≥ 2;
`kind=1` gives interior roots and permits n ≥ 1. Unrepresentable node spacing
or relative weights are rejected. Weight construction uses O(n²) work and
memory. Stable evaluation cannot repair a poorly conditioned choice of nodes;
high-degree equispaced interpolation and extrapolation can be inaccurate or
produce non-finite results.

### Monte Carlo and stochastic paths

```python
import numpy as np
from numerical_methods import monte_carlo, path_sampler, geometric_brownian_motion

result = monte_carlo(lambda rng, n: rng.random(n)**2, 100_000,
                     rng=np.random.default_rng(42), batch_size=10_000)
print(result.estimate, result.std_error, result.ci)  # E[U²] = 1/3

sampler = path_sampler(geometric_brownian_motion, np.linspace(0, 1, 33),
                       s0=1., mu=0.05, sigma=0.2)
terminal = monte_carlo(sampler, 20_000, rng=np.random.default_rng(43),
                       quantity=lambda paths: paths[:, -1], batch_size=2_000)
print(terminal.estimate, terminal.ci)  # true mean exp(0.05)
```

The sampler contract is `sampler(rng, n) -> array` with sample axis **0** of
length n. `quantity(samples)` must return exactly `(n,)`. Without a quantity,
the sampler itself must return `(n,)`. n ≥ 2, a positive batch size, and an
explicit `numpy.random.Generator` are required. Streaming two-pass batch moments
and Chan–Golub–LeVeque merges avoid the unstable `E[Y²]-E[Y]²` variance formula.
The result includes ddof=1 sample variance, standard error, a normal confidence
interval, batch counts and elapsed time. Constant samples give exactly zero
variance; they do not demonstrate correctness of the underlying model.

The normal interval assumes i.i.d. samples and finite variance. It approximates
**sampling uncertainty only**; finite-sample coverage is not guaranteed,
especially for heavy tails, and excludes model error and time-grid bias.
Reproducibility is limited to the same seed, sampler, batch size and recorded
environment. No stream equality across NumPy versions is promised.

`brownian_motion` and `geometric_brownian_motion` return
`(n_paths, len(times))`. Times must be finite, nonnegative, strictly increasing,
and contain at least two points. Column 0 is the initial value at `times[0]`.
They sample the exact Gaussian/GBM laws at grid points; continuous-path
functionals between points still have discretization error. `path_sampler`
generates batches without retaining all paths. Extreme parameter magnitudes
may exceed float64 range; finite-variance theory alone does not prevent
floating-point overflow.

## Validation and exceptions

Calculations use real float64. Real integer/float inputs are accepted and
converted to float64; boolean, complex, string, and object dtypes are rejected.
Unsupported shapes and non-finite API inputs raise `InvalidInputError`
(`ValueError`). Negligible LU pivots and zero triangular pivots raise
`SingularMatrixError` (`numpy.linalg.LinAlgError`). Invalid Monte Carlo samples
or quantity output raise `InvalidSampleError` (`ValueError`); no observations
are silently discarded. Root iteration failures are returned as status values.
Callback arithmetic `OverflowError` is treated as a non-finite root evaluation;
other exceptions from user callbacks propagate.

```powershell
python -m pytest -q --cov=numerical_methods --cov-report=term-missing
python -m ruff check .
python studies/run_all.py --smoke
python studies/run_all.py --seed 20260927
python -m build
```

Each study accepts `--smoke`, `--seed`, and `--out`. `run_all.py --out DIR`
creates one child directory per study. Individual default outputs are
`results/<area>/` (or `<area>_smoke`); all-study smoke outputs go to
`results/_tmp/smoke/`. Full results include JSON, CSV and PNGs, recorded versions,
CPU/BLAS/thread environment, configurations and seeds. Timings use warm-up and
repeated samples, recording median, IQR and raw samples. They are measurements
on one machine, not performance guarantees. See [VALIDATION.md](VALIDATION.md)
for the evidence table, measured findings, budgets and limitations, and
[COMPLETION.md](COMPLETION.md) for release checks and exact commands.

The four [examples](examples/) use only the installed public package and NumPy.
`scripts/verify_wheel.py` installs a wheel in fresh temporary virtual
environments, copies tests/examples outside the repository, verifies imports
come from site-packages, runs them, and removes those environments. A GitHub
Actions workflow is supplied for Python 3.11 and 3.13. The repository's Actions
tab shows the current remote check status.

The shareable validation summary is [results/verification_summary.json](results/verification_summary.json).
Raw local verification logs are excluded from Git and source distributions because
they can contain machine-specific paths. Build archives and caches are excluded
from Git as well.

The original `src/root_finding.py` is a deprecated compatibility shim with basic
regression tests. `src/benchmarks.py` and `notebooks/root_finding.ipynb` are legacy,
unvalidated exploratory artifacts and are not release evidence. The notebook
also needs pandas. Use `studies/` for reproducible experiments.

## Scope and license

This is an educational implementation, not a replacement for production
LAPACK/SciPy solvers. Complex arithmetic, sparse methods, QR, SVD, CG, coupled
nonlinear systems, general SDE solvers, Milstein, MLMC, PDEs, optimization,
GPU support, variance reduction and sequential stopping are out of scope.

Licensed under the [MIT License](LICENSE).
Copyright (c) 2026 ryanhami-lab.
