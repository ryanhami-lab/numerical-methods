"""Shared helpers for the numerical studies: metadata, timing, output."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def parse_args(name: str, description: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--smoke", action="store_true", help="small, fast configuration")
    p.add_argument("--seed", type=int, default=20260927, help="base seed for numpy.random.default_rng")
    p.add_argument("--out", type=Path, default=None, help=f"output directory (default results/{name})")
    a = p.parse_args()
    if a.out is None:
        a.out = REPO / "results" / (f"{name}_smoke" if a.smoke else name)
    a.out.mkdir(parents=True, exist_ok=True)
    return a


def _cpu() -> str:
    name = platform.processor()
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            ) as k:
                name = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                name = line.split(":", 1)[1].strip()
                break
    return name


def _blas() -> str:
    try:
        cfg = np.show_config(mode="dicts")
        b = cfg.get("Build Dependencies", {}).get("blas", {})
        return f"{b.get('name', '?')} {b.get('version', '')}".strip()
    except Exception:  # pragma: no cover - older numpy
        return "unknown"


def metadata(config: dict, seed: int) -> dict:
    import numerical_methods

    meta = {
        "timestamp_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seed": seed,
        "config": config,
        "dtype": "float64",
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "numerical_methods": numerical_methods.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": _cpu(),
        "logical_cpus": os.cpu_count(),
        "blas": _blas(),
        "thread_env": {v: os.environ.get(v) for v in THREAD_VARS},
    }
    for mod in ("scipy", "matplotlib"):
        try:
            meta[mod] = __import__(mod).__version__
        except ImportError:
            meta[mod] = None
    return meta


def timeit(fn, *, repeats: int, warmup: int = 1, min_time: float = 2e-3) -> dict:
    """Median/IQR per call; calibrate inner loops to target ``min_time`` per sample."""
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    fn()
    single = max(time.perf_counter() - t0, 1e-9)
    inner = max(1, int(min_time / single))
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        samples.append((time.perf_counter() - t0) / inner)
    q1, med, q3 = np.percentile(samples, [25, 50, 75])
    return {"median_s": float(med), "iqr_s": float(q3 - q1), "repeats": repeats,
            "inner_loops": inner, "warmup_calls": warmup, "calibration_calls": 1,
            "target_sample_s": min_time, "samples_s": samples}


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if np.isfinite(f) else str(f)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(_clean(obj), indent=2), encoding="utf-8")


def plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    return pyplot
