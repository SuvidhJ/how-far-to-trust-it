"""Choose the retriever by measurement.

    uv run python scripts/bakeoff_retrieval.py

Retrieval has no ground truth here - nobody labelled which historical exchange
is the "right" one to ground a reply in. So the proxy is intent agreement:
retrieve over the golden set leave-one-out, and ask what fraction of the top-k
share the query's hand-labelled intent.

That proxy is imperfect and the report will say so. Two exchanges can share an
intent and still be useless to each other, and two with different intents can
still model the right tone. What it does measure is whether a retriever pulls
back the same *kind* of problem, which is the minimum a grounding step must do -
a retriever that fails this cannot possibly be grounding a draft usefully.

Reported: precision@k, and MRR of the first same-intent hit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from support_agent.config import Paths, settings
from support_agent.retrieve import build, search_batch

METHODS = ["bm25", "tfidf", "embed", "hybrid"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    path = Paths.golden / f"{settings.brand.lower()}_golden.jsonl"
    gold = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    gold = [r for r in gold if not r.get("excluded")]
    texts = [r["text"] for r in gold]
    intents = [r["intent"] for r in gold]
    meta = [{"id": r["id"], "reply": r["brand_reply"], "intent": r["intent"]} for r in gold]

    # The number to beat: pick k at random and see how often the intent matches.
    counts = np.array([intents.count(i) for i in set(intents)], dtype=float)
    chance = float(((counts / counts.sum()) ** 2).sum())
    print(f"n={len(gold)}  k={args.k}  chance precision@k = {chance:.3f}\n")

    print(
        f"{'retriever':<12}{'P@1':>8}{'P@3':>8}{f'P@{args.k}':>8}{'MRR':>8}{'index':>9}{'query':>9}"
    )
    print("-" * 62)
    for name in METHODS:
        t0 = time.time()
        r = build(name).index(texts, meta)
        index_s = time.time() - t0

        t0 = time.time()
        p_at = {1: [], 3: [], args.k: []}
        rr = []
        all_hits = search_batch(r, texts, k=max(args.k, 3), exclude_self=True)
        for hits, truth in zip(all_hits, intents, strict=True):
            match = [h.intent == truth for h in hits]
            for cut in p_at:
                window = match[:cut]
                p_at[cut].append(sum(window) / cut if window else 0.0)
            first = next((j for j, m in enumerate(match) if m), None)
            rr.append(1.0 / (first + 1) if first is not None else 0.0)
        query_s = time.time() - t0
        print(
            f"{name:<12}{np.mean(p_at[1]):>8.3f}{np.mean(p_at[3]):>8.3f}"
            f"{np.mean(p_at[args.k]):>8.3f}{np.mean(rr):>8.3f}{index_s:>8.2f}s{query_s:>8.2f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
