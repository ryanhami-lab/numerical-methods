# Completion record

The outdated handoff described five failing tests. On taking over, the current
checkout already had **123 passing tests and 100% line coverage**, an implemented
library, and a partial root study. The earlier five test-design corrections had
already been made. At that point the folder had no Git history; this record distinguishes the
original work from the completion and validation work.

This continuation reviewed the numerical contracts, fixed additional edge-case
defects, added regression coverage, completed all four studies and four public-API
examples, wrote installation/validation documentation, and added packaging,
CI configuration and a clean-wheel verification script. Public publication and
license selection are separate from these implementation checks.

The original project, according to the supplied handoff, consisted of four
unvalidated scalar root routines, single-shot timings, a notebook and six weak
tests. Problems included endpoint handling, overflow-prone sign tests,
redundant evaluations, missing termination reasons and a fixed-point step
misidentified as the residual. The present library implements scalar nonlinear
equations, dense pivoted LU, barycentric interpolation, Monte Carlo and BM/GBM;
see [VALIDATION.md](VALIDATION.md) for requirement-to-file evidence, measured
findings, references and limits, and [README.md](README.md) for installation
and usage.

## Reproduce locally

From the project root in PowerShell (with Python 3.11 and 3.13 installed):

```powershell
py -3.11 -m venv .venv
$py = '.\.venv\Scripts\python.exe'
$python311 = (py -3.11 -c 'import sys; print(sys.executable)').Trim()
$python313 = (py -3.13 -c 'import sys; print(sys.executable)').Trim()
& $py -m pip install -e '.[dev]'
& $py -m pytest -q --cov=numerical_methods --cov-report=term-missing
& $py -m ruff check .
& $py -W error::RuntimeWarning studies/run_all.py --smoke --seed 20260927 --out results/_tmp/final_smoke
& $py -m build
& $py scripts/verify_wheel.py dist/numerical_methods-1.0.0-py3-none-any.whl `
  --python $python311 `
  --python $python313
```

The full studies were run separately to coordinate timing load:

```powershell
& $py -W error::RuntimeWarning studies/study_root_finding.py --seed 20260927
& $py studies/study_linear_systems.py --seed 20260927 --out results/linear_systems
& $py studies/study_monte_carlo.py --seed 20260927 --out results/monte_carlo
```

Interpolation used the following environment in its own PowerShell process:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
& $py studies/study_interpolation.py --seed 20260927 --out results/interpolation
```

Alternatively, `& $py studies/run_all.py --seed 20260927` runs all four full
studies sequentially with the caller's environment. It replaces full artifacts;
use `--out results/_tmp/rerun` to retain the recorded results. Timings vary with
thread settings and system load. Statistical streams and budgets are specified
in each `metadata.json`.

## Evidence and checks

The shareable summary is `results/verification_summary.json`. Raw wheel
transcripts, absolute import paths, command logs and checkout outputs remain in
local `results/validation/`, excluded from Git and source archives. The summary
retains versions, outcomes, durations, import-isolation checks and cleanup
results without the local account name or filesystem paths.

| Check | Observed outcome |
|---|---|
| Checkout Python 3.11.15, NumPy 2.4.3, SciPy 1.17.1 | **176 passed in 1.27 s**; all **669/669** runtime statements covered (100% line coverage) |
| Ruff 0.6.9 | **All checks passed** |
| All-study smoke runner with RuntimeWarning as error | **All four completed**, exit code 0 |
| Four separate full studies | **All four completed**, JSON/CSV/metadata retained; all **11 PNGs visually inspected** |
| `python -m build` | **sdist and wheel built successfully**, with wheel built from the sdist |
| Fresh Python 3.11.15 wheel environment, NumPy 2.4.6, SciPy 1.17.1, pytest 9.1.1 | **172 passed, 1 skipped in 0.97 s**; all four copied examples passed |
| Fresh Python 3.13.9 wheel environment, NumPy 2.5.3, SciPy 1.18.1, pytest 9.1.1 | **172 passed, 1 skipped in 0.96 s**; all four copied examples passed |
| Import isolation | Both imports resolved under the fresh venv's `Lib/site-packages/numerical_methods/` while cwd, tests and examples were outside the repository |
| Cleanup | Both temporary wheel-test environments removed; local build archives and temporary output remain excluded from Git. |
| Remote GitHub Actions | Configured only; **not run** |

The one skip in each wheel environment is the entire repository-only legacy
shim module (four local tests), explaining 176 local versus 172 installed tests.
No numerical tests were skipped. A coverage JSON and checkout outputs are in
`results/validation/coverage.json`, `pytest.txt` and `ruff.txt`.

The source archive includes documentation, tests, examples, studies, full study
results and CI configuration. It excludes raw local logs, private review files,
caches and temporary experiments. Inspect a newly built archive with:

```powershell
& $py scripts/inspect_distribution.py
```

The optional `--compare PATH` argument compares every wheel entry against a
previously saved mapping of entry names to SHA256 hashes. Documentation changes
can legitimately change wheel metadata. Runtime source integrity is always
checked independently against the current source files.

The completion-stage wheel had 11 entries; every entry matched the build tested
in both clean environments. Privacy preparation subsequently changed documentation
and packaging exclusions without changing the numerical algorithms. Current
archive checks are recorded in local `results/validation/distribution_checks.json`.

During development, dotted submodule coverage selectors triggered a NumPy import
error before collection. The documented package-wide selector subsequently passed;
the failed invocation was not counted as a passed check. Use the commands above
for reproduction. Timing results are observations, not promised performance.

## Résumé wording grounded in recorded experiments

- Implemented a NumPy-based library for scalar nonlinear equations, pivoted LU,
  barycentric interpolation and batched Monte Carlo; demonstrated 5.66× faster
  handling of eight n=200 right-hand sides by reusing one LU factorization
  versus refactorizing each column on the recorded Windows/MKL environment.
- Implemented array-based BM/GBM simulation and stable batched variance
  estimation; measured 31.15×/26.55× speedups over scalar loops for 2,000 paths
  with 128 increments, and an empirical Monte Carlo RMSE slope of −0.5013
  over five sample sizes with 400 independent repetitions each.

Use **nonlinear equations**, not “nonlinear systems”: there is no coupled
F: Rⁿ → Rⁿ solver. The timing claims apply to the specified comparisons on one
recorded environment, not all workloads. No LICENSE file exists; select MIT,
BSD-3-Clause or another appropriate license before public distribution.
