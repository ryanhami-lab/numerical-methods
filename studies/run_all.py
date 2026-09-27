"""Run all four studies sequentially, with a shared seed and output root."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--out", type=Path, help="output root; each study gets its own subdirectory")
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    out = args.out or folder.parent / "results" / ("_tmp/smoke" if args.smoke else "")
    for name in ("root_finding", "linear_systems", "interpolation", "monte_carlo"):
        command = [sys.executable, "-W", "error::RuntimeWarning", str(folder / f"study_{name}.py"),
                   "--seed", str(args.seed), "--out", str(out / name)]
        if args.smoke:
            command.append("--smoke")
        print(f"Running {name} -> {out / name}", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

