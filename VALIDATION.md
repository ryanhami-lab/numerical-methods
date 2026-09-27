# Numerical validation

This record distinguishes implementation tests, empirical studies, and packaging
checks. All numerical findings below come from the full recorded runs with base
seed **20260927**, not smoke runs. Displayed numbers are rounded; JSON/CSV files
retain the recorded precision. See [COMPLETION.md](COMPLETION.md) for commands
and release outcomes.

## Requirement → implementation → evidence

| Requirement | Implementation / defects addressed | Tests | Recorded study evidence |
|---|---|---|---|
| Scalar roots with honest termination | `src/numerical_methods/root_finding.py`: separate residual, bracket, stagnation, max-iteration and numerical-failure statuses; exact endpoint roots; evaluation counts; overflow-safe bisection midpoint; callback overflow and secant difference failures | `tests/test_root_finding.py`: analytic roots, local order, repeated roots, cycles, poor starts, invalid brackets, extreme endpoints, non-finite values, counters | `results/root_finding/results.json`, `runs.csv`, `convergence_cubic.png`, `convergence_repeated.png` |
| Array evaluation of nonlinear functions | Same module, `scan_brackets`; validates callback shape/dtype and evaluates one whole grid | `TestScanBrackets`, including one-call assertion and failures | 1,000,000 grid points; 15 sign-change brackets; grid evaluation timings in root JSON |
| Implement elimination and reusable partial-pivot LU | `src/numerical_methods/linear_systems.py`: packed LU, `A[perm]=L@U`, vector/matrix RHS, strict pivot threshold validation, no inverse | `tests/test_linear_systems.py`: SciPy comparisons, pivoting, singularity, scaling, permutation, reuse and diagnostics | `results/linear_systems/{random_systems,conditioning,factor_reuse}.csv`, `results.json`, three PNGs |
| Keep forward/backward error distinct | Infinity-norm residual and normwise backward error; separate known-solution forward error and factorization error; document threshold heuristic | Conditioning-aware tolerances and small-residual counterexample | Hilbert n=2…12, default and exact-zero pivot policies, SciPy reference and warnings |
| Stable barycentric evaluation | `src/numerical_methods/interpolation.py`: logarithmic relative weights, block evaluation, exact-node returns, range-safe distances and value scaling, representability checks, Chebyshev helper | `tests/test_interpolation.py`: polynomial recovery, SciPy comparison, duplicate/shape rejection, node order, blocks, extreme ranges and subnormal cases | `results/interpolation/{polynomial_recovery,runge}.csv`, `results.json`, three PNGs |
| Batched Monte Carlo with stable moments | `src/numerical_methods/monte_carlo.py`: explicit Generator, axis 0, two-pass batch moments plus Chan merges, invalid raw samples rejected before quantity, strict confidence validation and extreme-quantile fix | `tests/test_monte_carlo_stochastic.py`: exact constants, centered variance, determinism in environment, batch contract, invalid data and confidence | `results/monte_carlo/results.json`, `replications.csv`, `convergence.csv`, variance-stability demonstration |
| Exact BM/GBM on a validated time grid | `src/numerical_methods/stochastic.py`: Gaussian increments/exact GBM law, batched adapter, analytic moments with validated parameters | Same test module: time grids, deterministic limits, stream consistency, parameter validation and statistical tolerances | 120,000 paths per process at nine grid times; `process_moments.csv`, `process_moments.png`; scalar/array path timings |
| Installable package, examples and legacy boundary | `pyproject.toml`, `MANIFEST.in`, `examples/`, `scripts/verify_wheel.py`; repository-only shim | `tests/test_legacy_shim.py` locally; skipped outside source tree | `results/verification_summary.json`; full raw transcripts retained locally and excluded from distribution; `.github/workflows/ci.yml` is configured, not remotely run |

Coverage measures executed lines, not numerical correctness. Tests target
mathematical identities, failure contracts, independent references, scale/condition
effects and reproducibility. Stochastic tolerances account for sampling variation;
coverage experiments are studies, not pass/fail unit-test targets.

## Environment and experiment budgets

Full runs used Python 3.11.15, NumPy 2.4.3, SciPy 1.17.1, Matplotlib 3.10.8,
Windows build 26200, Intel Core Ultra 9 275HX, float64, and MKL 2025.
Every study records its own `metadata.json` with versions, seed, configuration,
CPU and BLAS details, and OMP/MKL/OPENBLAS/NUMEXPR thread environment variables.
The interpolation run explicitly used one thread for all four variables; the
other full runs left them unset. Unset variables do not imply a single thread.
Benchmark phases were run sequentially to avoid competing study loads.

| Study | Full budget |
|---|---|
| Roots | Three simple-root problems with all four methods, double/triple roots, failure cases; 30 timing repeats with 1 warm-up and 1 calibration call; 1,000,000-point array grid, 100,000-point Python loop with 6 repeats |
| Linear systems | Random n=10,25,50,100,200,400; Hilbert n=2…12; eight RHS at n=200; 9 timing repeats with 2 warm-ups and 1 calibration call |
| Interpolation | Degrees 4,6,…,40 for both node families; 10,001 evaluation points; polynomial degrees 0,1,2,5,8; 9 repeats with 1 warm-up and 1 calibration call |
| Monte Carlo | 400 independent replications at each N=128,512,2048,8192,32768; 200,000 samples for the standalone expectation and variance-stability demos; 120,000 paths per moment case; timing 2,000 paths × 128 increments, 7 repeats with 1 warm-up and 1 calibration call |

Timing helpers calibrate an inner-loop count to target 2 ms per repeat (3 ms for
linear systems), then report per-call median and IQR. JSON retains every timing
sample, repeat count and inner-loop count. Calibration is an estimate, not a
minimum-duration guarantee. Timings are local observations, not universal speedups.

## Root findings

For `x³-x-2`, Newton reached absolute error 7.33e-15 in 3 updates (4 function,
3 derivative evaluations); secant reached 3.33e-15 in 7 updates (9 function
evaluations). The final pre-saturation order estimates were **2.006** and
**1.627**, respectively. The error floor is 100 eps times the root scale;
reference-root rounding also limits the interpretation of the final digits.

Bisection's **bracket width**, not its oscillating point error, contracts by 1/2;
the study reports bracket-width order **1.0** for it. Plots still show the actual
point errors. Newton on `(x-1)²` also has order **1.0**, with point-error ratio
**0.5**, and on `(x-1)³` its ratio approaches **2/3**.

Comparisons use `xtol=ftol=1e-12`, `rtol=4*eps` where supported. Fixed point
tests its own map residual with `xtol+rtol*|x|`; it does not test the original
equation residual. Both residual definitions and actual root errors are saved.
For example, cubic fixed-point iteration has equation residual about 1.5e-12,
although its map residual is about 2.1e-13. A triple root lets Newton stop at
residual **7.07e-13** with root error **8.91e-5**: residual alone is not a universal
error bound.

Newton on atan from 1.5 fails with `zero_derivative` after its iterates grow;
the `x³-2x+2` start at zero cycles until `max_iter`. The divergent exponential
fixed-point map reports `nonfinite`; the logarithmic map for its upper root
converges. Zero derivatives, zero secant denominators, invalid brackets and
unattainable residual targets have distinct recorded outcomes.

Array evaluation found 15 sign changes of `sin(x)exp(-x/10)` and then solved
the corresponding scalar equations. Array/loop throughput uses different
grid lengths, explicitly recorded; it is not a coupled-system solver benchmark.

## Linear findings

Across the six random known-solution sizes, maximum package relative forward
error was **3.098735e-14**, maximum normwise backward error **2.985500e-16**.
At n=400, package solve median/IQR were **36.4548 / 1.2934 ms**; SciPy solve
was **10.7163 / 2.5726 ms**. Package/SciPy factor-only medians were
**35.8700 / 8.6158 ms**. The educational Python implementation is slower here.

For Hilbert(12), estimated 2-norm condition number was **1.760619e16**.
Package forward error was **1.218521**, with backward error **4.712639e-17**;
SciPy forward error was **0.621037**, with an ill-conditioning warning recorded.
These errors include float64 formation of `b=A@ones`; the condition estimate
itself is uncertain near the reciprocal of machine precision. Tiny residuals
describe nearby-problem accuracy, not an accurate solution of this sensitive
original system.

At n=200 for eight RHS, refactoring each column took median **39.0280 ms**,
while one factorization followed by eight column solves took **6.8993 ms**:
**5.6568×** faster for those explicitly matched scopes. Solve-only timings are
separately labeled and exclude factorization.

The tiny leading-pivot example returns `[1,1]` with pivoting; deliberately
unpivoted elimination produces `[0,1]`. An exactly singular matrix is rejected.
The nonsingular `diag(1e20,1)` demonstrates the default global threshold's
limitation; explicit `pivot_tol=0` solves it exactly for the chosen RHS.

## Interpolation findings

Polynomial degrees 0,1,2,5,8 were recovered with maximum observed absolute
error **3.552714e-15**. All exact-node values were returned exactly.

| Runge function at degree 40 | Equispaced | Chebyshev extrema |
|---|---:|---:|
| Maximum error on the 10,001-point grid | 104,668.422882 | 0.000339877500 |
| Estimated Lebesgue constant on that grid | 4,692,461,681.41 | 3.310467 |
| Evaluation median | 5.1089 ms | 4.9458 ms |

Stable barycentric arithmetic does not repair node-induced ill-conditioning.
These are grid estimates, not rigorous sup-norm bounds. The degree-20 plots use
different vertical scales to show each interpolant clearly.

## Monte Carlo and process findings

The fixed SeedSequence tree assigns independent streams to expectation, BM,
GBM, N sweeps, stability and timings. Each N and each replication gets its own
child stream. There is no seed selection or stopping on favorable results.

For E[U²]=1/3, N=200,000 gave estimate **0.3335756696**, SE **0.0006664690**
and 95% interval **[0.3322694144, 0.3348819249]**. The five-N log–log RMSE
slope was **−0.5012988**, consistent with the expected −1/2 rate in this study.

| N | RMSE | Empirical SD / mean reported SE | Coverage (400 repetitions) | Pointwise 95% Wilson interval |
|---:|---:|---:|---:|---:|
| 128 | 0.02687522 | 1.02353 | 383/400 = 95.75% | [93.30%, 97.33%] |
| 512 | 0.01231736 | 0.93811 | 385/400 = 96.25% | [93.91%, 97.71%] |
| 2,048 | 0.00702227 | 1.06683 | 374/400 = 93.50% | [90.65%, 95.53%] |
| 8,192 | 0.00324736 | 0.98706 | 387/400 = 96.75% | [94.52%, 98.09%] |
| 32,768 | 0.00162101 | 0.98518 | 385/400 = 96.25% | [93.91%, 97.71%] |

All five Wilson intervals include 95%; they are pointwise intervals, not a
simultaneous coverage guarantee. Raw replication outcomes and binomial standard
errors are retained. Mean/variance errors for the 120,000-path BM/GBM runs were
at most **1.0242 / 1.5852** analytic standard errors. Times within a process share
paths, so those moment errors are correlated.

For observations near 1e9 with noise scale 1e-3, stable sample variance was
**8.3330043e-8** versus centered same-observation reference **8.3329807e-8**
(relative difference **2.8245e-6**). The naive second-moment subtraction gave
**−384.0019200**, an impossible variance. Input quantization and rounding still
affect the stable method; it is not exact arithmetic.

| 2,000 paths × 128 increments | Scalar median / IQR | Array median / IQR | Median ratio |
|---|---:|---:|---:|
| BM | 150.3259 / 2.8293 ms | 4.8262 / 0.4013 ms | 31.1479× |
| GBM | 165.3766 / 1.6423 ms | 6.2298 / 0.9414 ms | 26.5461× |

Both implementations include RNG construction and allocation and consume
equivalent path-major streams. Paths matched exactly in this environment; that
bitwise agreement is not promised across environments. Exact grid-point laws
do not remove between-grid error in maxima, crossings or continuous integrals.

## References consulted

The following sources were opened for the numerical reasoning in this report.
They are background for the implementation and interpretation, not substitutes
for the package's recorded tests and experiments.

- Berrut and Trefethen (2004), [Barycentric Lagrange Interpolation](https://people.maths.ox.ac.uk/trefethen/barycentric.pdf), SIAM Review 46(3), 501–517: barycentric formulas, node choice and conditioning.
- Chan, Golub and LeVeque (1983), [Algorithms for Computing the Sample Variance: Analysis and Recommendations](https://math.pku.edu.cn/teachers/litj/notes/numer_anal/AmerStat_37_242_Chan_Variance.pdf), The American Statistician 37(3), 242–247: centered/shifted variance computations and cancellation.
- Higham (1990), [How Accurate Is Gaussian Elimination?](https://nhigham.com/wp-content/uploads/2023/08/high90g.pdf), Numerical Analysis 1989, 137–154: forward/backward error and the role of conditioning. This report does not claim a fresh consultation of the separate Higham textbook cited in a code docstring.
- Sigman (2013), [Simulating Brownian Motion and Geometric Brownian Motion](https://www.columbia.edu/~ks20/4404-Sigman/4404-Notes-sim-BM.pdf): independent Gaussian increments and exact grid simulation. The library's GBM `mu` is the SDE drift, so log drift is `mu-sigma²/2`.

## Remaining limits

The package is educational and real float64 only. There is no global root
safeguard, condition estimator in the runtime package, scaling/equilibration,
iterative refinement, adaptive interpolation, variance reduction or sequential
Monte Carlo stopping. Ordinary finite inputs can still create unrepresentable
intermediate/output magnitudes. Barycentric extrapolation and highly clustered
nodes can lose accuracy; quadratic weight storage limits scale. Normal intervals
are approximate and omit model/time-grid error. Experimental cases and full line
coverage do not prove universal numerical reliability.

No coupled nonlinear-system, QR, SVD, CG, sparse, general SDE, MLMC, Milstein,
PDE, optimization or GPU implementation is included. SciPy SVD is used only to
estimate condition numbers in the study. Python 3.10 is declared compatible but
was not locally verified; local wheel checks target 3.11 and 3.13. The original
validation record covers local checks; consult GitHub Actions for current remote
results. A license must be selected before public distribution.
