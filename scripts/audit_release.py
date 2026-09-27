"""Check the staged Git snapshot for accidental secrets and private local artifacts.

Reports filenames and line numbers, never matched values. This is a focused
pre-upload check, not a guarantee that every possible secret can be recognized.
Run after staging the intended review files: python scripts/audit_release.py
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

PATTERNS = {
    "credential token": re.compile(
        r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
        r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16}|AIza[A-Za-z0-9_-]{30,}"
    ),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "personal filesystem path": re.compile(r"(?i)[A-Z]:[\\/]+Users[\\/]+[^\s/\\]+|/(?:Users|home)/[^\s/]+"),
    "email address": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "credential assignment": re.compile(
        r"(?i)(?:api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*[\"'][^\"'\n]{6,}"
    ),
    "authenticated URL": re.compile(r"(?i)https?://[^\s/:]+:[^\s/@]+@"),
}
FORBIDDEN_PARTS = {
    ".local-review", ".ipynb_checkpoints", "__pycache__", ".pytest_cache", ".ruff_cache",
    "dist", "build", "_tmp", "_smoke",
}


def main():
    root = Path(__file__).resolve().parents[1]
    names = subprocess.check_output(["git", "ls-files", "--cached", "-z"], cwd=root).decode().split("\0")
    names = [name for name in names if name]
    findings = []
    total_bytes = 0
    for name in names:
        path = Path(name)
        forbidden = (
            any(part in FORBIDDEN_PARTS or part.endswith((".egg-info", "_smoke")) for part in path.parts)
            or name.startswith("results/validation/") or path.name.startswith((".env", ".coverage"))
            or path.suffix.lower() in {".pem", ".key", ".p12", ".pfx", ".pyc"}
        )
        if forbidden:
            findings.append({"file": name, "kind": "local-only or credential file"})
        raw = subprocess.check_output(["git", "show", f":{name}"], cwd=root)
        total_bytes += len(raw)
        # Latin-1 also permits checking binary metadata without echoing data.
        text = raw.decode("utf-16") if raw[:2] in {b"\xff\xfe", b"\xfe\xff"} else raw.decode("latin-1")
        for kind, pattern in PATTERNS.items():
            for number, line in enumerate(text.splitlines(), 1):
                if pattern.search(line):
                    findings.append({"file": name, "line": number, "kind": kind})
        if path.suffix == ".ipynb":
            notebook = json.loads(raw)
            if any(cell.get("outputs") for cell in notebook.get("cells", [])):
                findings.append({"file": name, "kind": "saved notebook output"})
    report = {"files_checked": len(names), "bytes_checked": total_bytes, "findings": findings,
              "scope": "exact staged file bytes, including binary metadata", "passed": not findings}
    output = root / ".local-review/privacy_audit.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(bool(findings))


if __name__ == "__main__":
    main()
