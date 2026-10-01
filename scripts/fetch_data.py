#!/usr/bin/env python
"""Idempotently fetch the Kaggle twcs dataset into data/raw/.

Requires Kaggle credentials (`kaggle auth login`, or ~/.kaggle/kaggle.json, or
KAGGLE_USERNAME/KAGGLE_KEY). Re-running is a no-op once the CSV is present.

    uv run python scripts/fetch_data.py
    uv run python scripts/fetch_data.py --force
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
TARGET = RAW_DIR / "twcs" / "twcs.csv"
DATASET = "thoughtvector/customer-support-on-twitter"
EXPECTED_MIN_BYTES = 400 * 1024**2  # the real file is ~493MB


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    if TARGET.exists() and not args.force:
        size = TARGET.stat().st_size
        if size >= EXPECTED_MIN_BYTES:
            print(f"already present: {TARGET} ({size / 1024**2:.0f} MB) - nothing to do")
            return 0
        print(f"present but suspiciously small ({size / 1024**2:.0f} MB); re-downloading")

    if shutil.which("kaggle") is None:
        sys.exit(
            "kaggle CLI not found. Install it (`uv tool install kaggle`) and authenticate\n"
            "with `kaggle auth login`, or set KAGGLE_USERNAME / KAGGLE_KEY."
        )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(RAW_DIR), "--unzip"]
    if args.force:
        cmd.append("--force")

    print("running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode != 0:
        sys.exit(
            f"kaggle download failed (exit {proc.returncode}).\n"
            "Most common cause: not authenticated, or the dataset's terms have not been\n"
            "accepted on kaggle.com while signed in."
        )

    if not TARGET.exists():
        found = sorted(p.name for p in RAW_DIR.rglob("*.csv"))
        sys.exit(f"download finished but {TARGET} is missing. CSVs found: {found}")

    print(f"ok: {TARGET} ({TARGET.stat().st_size / 1024**2:.0f} MB)")
    print("Inspect it with: uv run python scripts/peek.py --schema")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
