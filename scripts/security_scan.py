#!/usr/bin/env python3
"""Lightweight static security scan for FlashDet.

This is not a replacement for Bandit/Semgrep; it catches the patterns that have
historically caused RCE / path traversal in this project so CI can fail fast.

Usage:
    python scripts/security_scan.py
    python scripts/security_scan.py --root flashdet --json report.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

CRITICAL = "CRITICAL"
HIGH = "HIGH"
MEDIUM = "MEDIUM"
INFO = "INFO"

# file-name allowlist for the one intentional fallback in the safe loader
ALLOWLIST = {
    # Intentional, explicit opt-in fallback inside the safe loader.
    "flashdet/utils/checkpoint.py": {"weights_only=false"},
    # extractall is wrapped by _safe_target() and covered by Zip/Tar Slip tests.
    "flashdet/data/download.py": {"extractall"},
}

PATTERNS = [
    (CRITICAL, re.compile(r"torch\.load\s*\([^)]*weights_only\s*=\s*False", re.S),
     "Unsafe torch.load: pickle RCE risk"),
    (CRITICAL, re.compile(r"pickle\.loads?\s*\("),
     "Unsafe pickle deserialization"),
    (CRITICAL, re.compile(r"(?<![.\w])eval\s*\("),
     "eval() on untrusted input"),
    (CRITICAL, re.compile(r"(?<![.\w])exec\s*\("),
     "exec() on untrusted input"),
    (HIGH, re.compile(r"\.extractall\s*\("),
     "Archive extractall: verify Zip Slip / Tar Slip protection"),
    (HIGH, re.compile(r"subprocess\.[A-Za-z_]+\([^)]*shell\s*=\s*True", re.S),
     "subprocess with shell=True"),
    (HIGH, re.compile(r"yaml\.load\s*\((?![^)]*SafeLoader)", re.S),
     "yaml.load without SafeLoader"),
    (MEDIUM, re.compile(r"urlretrieve\s*\([^)]*['\"]http://", re.S),
     "Dataset download over plain HTTP"),
    (MEDIUM, re.compile(r"requests\.(get|post|put|delete)\s*\([^)]*verify\s*=\s*False", re.S),
     "TLS verification disabled"),
    (INFO, re.compile(r"tempfile\.mktemp\s*\("),
     "Race-prone tempfile.mktemp"),
]


@dataclass
class Finding:
    severity: str
    rule: str
    path: str
    line: int
    snippet: str


def _is_allowlisted(relpath: str, rule: str, snippet: str) -> bool:
    text = f"{rule} {snippet}".lower()
    return any(token in text for token in ALLOWLIST.get(relpath, set()))


def scan(root: Path) -> List[Finding]:
    findings: List[Finding] = []
    for path in sorted(root.rglob("*.py")):
        relpath = path.as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for severity, pattern, rule in PATTERNS:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                snippet = text.splitlines()[line - 1].strip()[:160]
                if _is_allowlisted(relpath, rule, snippet):
                    continue
                findings.append(Finding(severity, rule, relpath, line, snippet))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="FlashDet static security scan")
    parser.add_argument("--root", default="flashdet", help="Directory to scan")
    parser.add_argument("--json", default=None, help="Write JSON report")
    parser.add_argument("--fail-on", default=CRITICAL,
                        choices=[INFO, MEDIUM, HIGH, CRITICAL],
                        help="Exit non-zero when this severity or worse is found")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"scan root not found: {root}", file=sys.stderr)
        return 2

    findings = scan(root)
    order = {INFO: 0, MEDIUM: 1, HIGH: 2, CRITICAL: 3}
    if findings:
        for f in findings:
            print(f"[{f.severity}] {f.path}:{f.line}: {f.rule}")
            print(f"    {f.snippet}")
    else:
        print("No findings.")

    if args.json:
        Path(args.json).write_text(
            json.dumps([asdict(f) for f in findings], indent=2),
            encoding="utf-8",
        )
        print(f"JSON report: {args.json}")

    threshold = order[args.fail_on]
    worst = max((order[f.severity] for f in findings), default=-1)
    return 1 if worst >= threshold else 0


if __name__ == "__main__":
    raise SystemExit(main())
