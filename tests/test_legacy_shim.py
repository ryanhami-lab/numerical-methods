"""The original src/root_finding.py API (used by notebooks/root_finding.ipynb) keeps working."""

import sys
import warnings
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if not (SRC / "root_finding.py").exists():  # pragma: no cover - installed-wheel runs
    pytest.skip("legacy shim is repository-only", allow_module_level=True)
sys.path.insert(0, str(SRC))
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import root_finding as legacy
finally:
    sys.path.remove(str(SRC))

ROOT = 1.5213797068045676


def f(x):
    return x**3 - x - 2


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        yield


def test_emits_deprecation_warning():
    with pytest.warns(DeprecationWarning):
        legacy.bisection(f, 1, 2)


def test_legacy_methods_converge_with_history():
    for r in (legacy.bisection(f, 1, 2), legacy.newton(f, lambda x: 3 * x * x - 1, 1.5),
              legacy.secant(f, 1.0, 2.0), legacy.fixed_point(lambda x: (x + 2) ** (1 / 3), 1.5)):
        assert r.converged and abs(r.root - ROOT) < 1e-10
        assert isinstance(r.history, list) and len(r.history) == len(r.residuals) > 0


def test_legacy_bisection_invalid_bracket():
    with pytest.raises(ValueError):
        legacy.bisection(lambda x: x**2 + 1, -1, 1)


def test_legacy_newton_zero_derivative():
    # the original test used f=x^3 at x0=0, which *is* the root; this is a real zero-derivative case
    r = legacy.newton(lambda x: x**2 - 1, lambda x: 2 * x, 0.0)
    assert not r.converged
    assert legacy.newton(lambda x: x**3, lambda x: 3 * x**2, 0.0).converged
