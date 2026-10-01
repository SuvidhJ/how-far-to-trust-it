"""Fail if a fresh harness run disagrees with the committed results.

    uv run python -m support_agent.eval.harness --offline --out runs/ci.json
    uv run python scripts/check_results.py reports/results.json runs/ci.json

`reports/results.json` and `reports/results_naive.json` are the source of the
evaluation tables in the README, the report and the demo's dashboard. CI reruns the harness from a clean checkout and compares every
leaf: integers and strings exactly, floats to `--tol`. A difference means a number
in the write-up is no longer what the code produces.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Run metadata, not results.
VOLATILE = {"run_at", "git_sha", "python", "elapsed_s"}


# Rows a fresh clone cannot compute (they need the gitignored raw-derived data) say so
# in their "n" field. They are listed, never counted as passing or failing.
SKIPPED: list[str] = []


def diff(expected, actual, tol: float, path: str = "") -> list[str]:
    if isinstance(actual, dict) and str(actual.get("n", "")).startswith("skipped:"):
        SKIPPED.append(f"{path}: {actual.get('step', '')} ({actual['n']})")
        return []
    if isinstance(expected, dict) and isinstance(actual, dict):
        out = []
        for key in sorted(set(expected) | set(actual)):
            if key in VOLATILE:
                continue
            where = f"{path}.{key}" if path else key
            if key not in actual or key not in expected:
                out.append(f"{where}: only in {'expected' if key in expected else 'actual'}")
            else:
                out += diff(expected[key], actual[key], tol, where)
        return out
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{path}: length {len(expected)} != {len(actual)}"]
        return [
            x
            for i, (e, a) in enumerate(zip(expected, actual, strict=True))
            for x in diff(e, a, tol, f"{path}[{i}]")
        ]
    if isinstance(expected, float) or isinstance(actual, float):
        if isinstance(expected, int | float) and isinstance(actual, int | float):
            if math.isnan(expected) and math.isnan(actual):
                return []
            if math.isclose(expected, actual, rel_tol=0, abs_tol=tol):
                return []
        return [f"{path}: expected {expected!r}, got {actual!r}"]
    return [] if expected == actual else [f"{path}: expected {expected!r}, got {actual!r}"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("expected", type=Path)
    ap.add_argument("actual", type=Path)
    ap.add_argument("--tol", type=float, default=1e-9, help="absolute tolerance on floats")
    args = ap.parse_args()

    expected = json.loads(args.expected.read_text(encoding="utf-8"))
    actual = json.loads(args.actual.read_text(encoding="utf-8"))
    problems = diff(expected, actual, args.tol)
    for p in problems[:50]:
        print(p)
    for skip in SKIPPED:
        print(f"not checked, {skip}")
    if problems:
        print(f"\n{len(problems)} differences: {args.actual} does not reproduce {args.expected}")
        return 1
    print(f"{args.actual} reproduces {args.expected}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
