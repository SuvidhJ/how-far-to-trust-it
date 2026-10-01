"""Build the brand-scoped, thread-split exchange table from the raw dump.

    uv run python scripts/make_sample.py --brand AmericanAir

Writes `data/interim/<brand>_exchanges.parquet` and a sibling manifest. This is
the only script that touches `data/raw/`; everything downstream reads the
parquet and the manifest, never the CSV.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from support_agent.config import Paths, seed_for
from support_agent.data import (
    add_thread_splits,
    build_exchanges,
    filter_exchanges,
    write_manifest,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--brand", default="AmericanAir")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--dev-frac", type=float, default=0.1)
    args = ap.parse_args()

    t0 = time.time()
    raw = build_exchanges(args.brand)
    kept, counts = filter_exchanges(raw)
    seed = seed_for(f"split:{args.brand}")
    out = add_thread_splits(kept, seed, args.test_frac, args.dev_frac)

    Paths.interim.mkdir(parents=True, exist_ok=True)
    path = Paths.interim / f"{args.brand.lower()}_exchanges.parquet"
    out.to_parquet(path, index=False)

    by_split = out["split"].value_counts().to_dict()
    threads_by_split = out.groupby("split")["thread_id"].nunique().to_dict()
    write_manifest(
        path,
        brand=args.brand,
        rows=len(out),
        threads=int(out["thread_id"].nunique()),
        filters=counts,
        split_seed=seed,
        split_purpose=f"split:{args.brand}",
        test_frac=args.test_frac,
        dev_frac=args.dev_frac,
        rows_by_split=by_split,
        threads_by_split=threads_by_split,
        elapsed_s=round(time.time() - t0, 1),
    )

    print(f"{path}  {len(out):,} exchanges  {out['thread_id'].nunique():,} threads")
    print("  filters:", counts)
    print("  rows by split:", by_split)
    print(f"  {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
