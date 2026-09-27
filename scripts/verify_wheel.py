"""Install a wheel in clean venvs and run copied tests/examples outside the repository.

Usage: python scripts/verify_wheel.py dist/package.whl --python /path/to/python [--python ...]
The JSON transcript records every subprocess, version, import path and observed outcome.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--python", action="append", dest="interpreters", required=True)
    parser.add_argument("--out", type=Path, default=Path("results/validation/wheel_checks.json"))
    args = parser.parse_args()
    wheel = args.wheel.resolve(strict=True)
    repo = Path(__file__).resolve().parents[1]
    report = {"wheel": str(wheel), "runs": []}
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        for interpreter in args.interpreters:
            entry = {"interpreter": interpreter, "commands": []}
            report["runs"].append(entry)
            with tempfile.TemporaryDirectory(prefix="numerical-methods-wheel-") as folder:
                root = Path(folder).resolve()
                assert root.is_relative_to(Path(tempfile.gettempdir()).resolve())
                entry["temporary_directory"] = str(root)

                def run(command, root=root, entry=entry):
                    print(subprocess.list2cmdline(command), flush=True)
                    result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
                    entry["commands"].append({"command": command, "exit_code": result.returncode,
                                              "stdout": result.stdout, "stderr": result.stderr})
                    print(result.stdout, end="", flush=True)
                    if result.returncode:
                        print(result.stderr, file=sys.stderr)
                        raise RuntimeError(f"Verification failed: {command}")

                run([interpreter, "-m", "venv", str(root / "venv")])
                py = str(root / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
                run([py, "-m", "pip", "install", f"{wheel}[test]"])
                run([py, "-c", "import sys, numpy, scipy, numerical_methods as nm; "
                     "from pathlib import Path; "
                     "print(sys.version); print('numpy', numpy.__version__, 'scipy', scipy.__version__); "
                     "print(nm.__file__); "
                     "assert Path(nm.__file__).is_relative_to(Path(sys.prefix)); "
                     "assert 'site-packages' in str(nm.__file__)"])
                shutil.copytree(repo / "tests", root / "tests", ignore=shutil.ignore_patterns("__pycache__"))
                shutil.copytree(repo / "examples", root / "examples",
                                ignore=shutil.ignore_patterns("__pycache__"))
                run([py, "-m", "pytest", "-q", "-ra", "-W", "error::RuntimeWarning", str(root / "tests")])
                for example in sorted((root / "examples").glob("*.py")):
                    run([py, "-W", "error::RuntimeWarning", str(example)])
                entry["passed"] = True
            entry["temporary_directory_removed"] = not root.exists()
    finally:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
