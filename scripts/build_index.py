"""Build the retrieval corpus the drafter grounds in.

    uv run python scripts/build_index.py --n 3000

Drawn from the **train split only**. The golden set comes from dev and test
threads, so no message the system is evaluated on can appear in the evidence it
is grounded in - which would otherwise let the drafter retrieve the very reply it
is being scored against.

Subsampled on purpose. The size is a cost decision: embedding is
rate-limited on a free tier, and the vectors have to be committed for the
reproduce path to work without an API key. What the subsample costs in retrieval
quality is measurable and is reported rather than waved away.

Candidates are filtered to exchanges that could plausibly *teach* something: a
brand reply long enough to be a real answer, and not a pure DM deflection. An
index full of "please DM us" grounds the drafter in deflection, which is the
trap AppleSupport would have walked into (DECISIONS #16).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from support_agent.config import Paths, seed_for, settings
from support_agent.data import write_manifest
from support_agent.embed import MODEL, Embedder

# Known gap, kept so the committed corpus rebuilds identically: `\bdm\b` misses "DMs",
# and 81 of the 3,000 committed replies are DM deflections (2.7%, DECISIONS #51).
# `text.DM_HANDOFF` is the corrected pattern.
DEFLECT = r"(?i)\b(dm|d\.m\.|direct message|private message|pm us|inbox us)\b"
MIN_REPLY_CHARS = 60


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--brand", default=settings.brand)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--no-embed", action="store_true", help="write the corpus, skip vectors")
    args = ap.parse_args()

    df = pd.read_parquet(Paths.interim / f"{args.brand.lower()}_exchanges.parquet")
    train = df[df["split"] == "train"]
    before = len(train)

    keep = train[
        train["brand_clean"].str.len().ge(MIN_REPLY_CHARS)
        & ~train["brand_clean"].str.contains(DEFLECT, regex=True, na=False)
    ]
    # One exchange per thread: a chatty conversation would otherwise fill the
    # index with near-identical neighbours and starve the top-k of variety.
    keep = keep.drop_duplicates("thread_id")

    seed = seed_for(f"index:{args.brand}")
    corpus = keep.sample(n=min(args.n, len(keep)), random_state=seed).sort_values(
        "customer_tweet_id"
    )

    Paths.processed.mkdir(parents=True, exist_ok=True)
    path = Paths.processed / f"{args.brand.lower()}_index.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for r in corpus.itertuples():
            fh.write(
                json.dumps(
                    {
                        "id": int(r.customer_tweet_id),
                        "thread_id": int(r.thread_id),
                        "text": r.customer_clean,
                        "reply": r.brand_clean,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    elapsed, calls = 0.0, 0
    if not args.no_embed:
        emb = Embedder()
        t0 = time.time()
        emb.encode(corpus["customer_clean"].tolist(), input_type="passage")
        elapsed, calls = time.time() - t0, emb.api_calls

    write_manifest(
        path,
        brand=args.brand,
        rows=len(corpus),
        drawn_from="train split only (golden set is dev+test, so no overlap)",
        train_rows_before_filter=before,
        after_reply_length_filter=int(train["brand_clean"].str.len().ge(MIN_REPLY_CHARS).sum()),
        after_dm_filter=len(keep),
        min_reply_chars=MIN_REPLY_CHARS,
        seed=seed,
        embed_model=MODEL,
        embed_api_calls=calls,
        embed_seconds=round(elapsed, 1),
    )
    print(f"{path}  {len(corpus):,} exchanges from {before:,} train rows")
    print(f"  after reply-length + no-DM + one-per-thread filters: {len(keep):,} eligible")
    if not args.no_embed:
        print(f"  embedded in {elapsed:.0f}s ({calls} api calls)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
