#!/usr/bin/env python
"""Environment doctor: one screen that answers "is this repo ready to work in?"

Backs the /ctx slash command. Prints state only - never mutates anything, and
never prints a secret value (only whether one is set).

    uv run python scripts/env_report.py
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

OK, WARN, BAD = "OK  ", "WARN", "MISS"


def line(status: str, label: str, detail: str = "") -> None:
    print(f"[{status}] {label:<22} {detail}")


def mb(n: float) -> str:
    return f"{n / 1024**2:.0f} MB"


def git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def main() -> int:
    print(f"=== {ROOT.name} environment ===")
    print(f"python {platform.python_version()} on {platform.system()} {platform.release()}\n")

    # --- toolchain -----------------------------------------------------
    for tool in ("uv", "git", "kaggle", "ruff"):
        path = shutil.which(tool)
        line(OK if path else BAD, tool, path or "not on PATH")

    venv = ROOT / ".venv"
    line(OK if venv.exists() else BAD, ".venv", str(venv) if venv.exists() else "run `uv sync`")

    # --- data ----------------------------------------------------------
    print()
    csv = ROOT / "data" / "raw" / "twcs" / "twcs.csv"
    if csv.exists():
        line(
            OK,
            "raw dataset",
            f"{mb(csv.stat().st_size)} (never open directly; use scripts/peek.py)",
        )
    else:
        line(BAD, "raw dataset", "run `uv run python scripts/fetch_data.py`")

    for stage in ("interim", "processed"):
        d = ROOT / "data" / stage
        arts = sorted(p.name for p in d.glob("*") if p.suffix in {".parquet", ".jsonl", ".csv"})
        line(OK if arts else WARN, f"data/{stage}", ", ".join(arts) if arts else "(empty)")

    golden = sorted((ROOT / "data" / "golden").glob("*.jsonl"))
    if golden:
        counts = []
        for p in golden:
            with p.open("rb") as fh:
                counts.append(f"{p.name}={sum(1 for _ in fh)}")
        line(OK, "golden set", ", ".join(counts) + "  (target 150-250)")
    else:
        line(WARN, "golden set", "not started - highest-value deliverable")

    # --- llm config ----------------------------------------------------
    print()
    env_file = ROOT / ".env"
    line(
        OK if env_file.exists() else WARN,
        ".env",
        "present" if env_file.exists() else "copy from .env.example",
    )
    env_lines = env_file.read_text(encoding="utf-8").splitlines() if env_file.exists() else []
    for key in ("MISTRAL_API_KEY", "NVIDIA_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"):
        in_file = any(
            ln.startswith(f"{key}=") and ln.split("=", 1)[1].split("#")[0].strip()
            for ln in env_lines
        )
        is_set = bool(os.environ.get(key)) or in_file
        line(
            OK if is_set else WARN, key, "set" if is_set else "unset (only uncached calls need it)"
        )

    cache = ROOT / "data" / "cache" / "llm_cache.jsonl"
    if cache.exists():
        with cache.open("rb") as fh:
            n = sum(1 for _ in fh)
        line(OK, "llm cache", f"{n} cached responses - reruns are free")
    else:
        line(WARN, "llm cache", "empty - first run will cost real calls")

    # --- results -------------------------------------------------------
    print()
    metrics = ROOT / "reports" / "metrics_latest.json"
    if metrics.exists():
        try:
            m = json.loads(metrics.read_text(encoding="utf-8"))
            line(OK, "latest metrics", ", ".join(f"{k}={v}" for k, v in list(m.items())[:6]))
        except (OSError, ValueError) as exc:
            line(WARN, "latest metrics", f"unreadable: {exc}")
    else:
        line(WARN, "latest metrics", "no run recorded yet")

    # --- git -----------------------------------------------------------
    print()
    if (ROOT / ".git").exists():
        dirty = git("status", "--short")
        line(
            OK,
            "git",
            f"branch {git('rev-parse', '--abbrev-ref', 'HEAD') or '?'}, "
            f"{len(dirty.splitlines())} changed file(s)",
        )
    else:
        line(WARN, "git", "not a repository")
    return 0


if __name__ == "__main__":
    sys.exit(main())
