"""Inspect archive contents and compare wheel payloads to the tested build."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", type=Path, help="JSON mapping of tested wheel entry names to SHA256")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    wheel = repo / "dist/numerical_methods-1.0.0-py3-none-any.whl"
    sdist = repo / "dist/numerical_methods-1.0.0.tar.gz"
    license_text = (repo / "LICENSE").read_bytes()
    with zipfile.ZipFile(wheel) as archive:
        contents = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()}
        metadata = BytesParser().parsebytes(archive.read("numerical_methods-1.0.0.dist-info/METADATA"))
        assert metadata["License-Expression"] == "MIT"
        assert "LICENSE" in metadata.get_all("License-File", [])
        assert archive.read("numerical_methods-1.0.0.dist-info/licenses/LICENSE") == license_text
    for source in (repo / "src/numerical_methods").glob("*.py"):
        assert contents[f"numerical_methods/{source.name}"] == hashlib.sha256(source.read_bytes()).hexdigest()
    if args.compare:
        assert contents == json.loads(args.compare.read_text()), "Wheel payload differs from tested build"
    with tarfile.open(sdist) as archive:
        files = [member.name for member in archive.getmembers() if member.isfile()]
        assert archive.extractfile("numerical_methods-1.0.0/LICENSE").read() == license_text
    relative = [name.split("/", 1)[1] for name in files]
    for required in ("LICENSE", "README.md", "VALIDATION.md", "COMPLETION.md", "pyproject.toml",
                     "MANIFEST.in", ".gitignore", ".github/workflows/ci.yml",
                     "scripts/verify_wheel.py", "scripts/inspect_distribution.py"):
        assert required in relative, f"Missing source artifact: {required}"
    counts = {}
    for folder, expected in (("src/numerical_methods", 7), ("tests", 5), ("examples", 4), ("studies", 6)):
        counts[folder] = sum(name.startswith(f"{folder}/") and name.endswith(".py") for name in relative)
        assert counts[folder] == expected, (folder, counts[folder])
    assert not any("__pycache__" in name or "/_tmp/" in name or "_smoke/" in name
                   or name.startswith(("results/validation/", ".local-review/")) for name in relative)
    assert sum(name.endswith(".png") for name in relative) == 11
    report = {
        "passed": True, "wheel_payload_matches_source": True,
        "license_expression": "MIT", "license_in_both_archives": True,
        "wheel_payload_matches_tested_build": True if args.compare else None,
        "wheel_entries": contents, "sdist_files": relative, "python_file_counts": counts,
        "full_study_figures": 11,
        "artifacts": [{"file": path.name, "bytes": path.stat().st_size,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in (wheel, sdist)],
    }
    output = repo / "results/validation/distribution_checks.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Archive checks passed: {len(relative)} source files, {len(contents)} wheel entries, 11 figures.")
    if args.compare:
        print("Every wheel entry matches the clean-environment tested build byte for byte.")


if __name__ == "__main__":
    main()
