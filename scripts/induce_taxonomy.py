"""Induce a candidate intent taxonomy from the data, for a human to finalise.

    uv run python scripts/induce_taxonomy.py --k 20

Deliberately dumb and inspectable: TF-IDF over entity-masked customer messages,
KMeans at a chosen k, then one LLM call per cluster to name it. The output is a
*proposal*. I read it, merge clusters, write the rubric by hand, and freeze the
result in `src/support_agent/taxonomy.json` before any labelling happens. That
ordering is disclosed in the report because it is the difference between a
taxonomy induced from data and one hallucinated from a model's priors.

Sampling is from the **train split only** — a taxonomy fitted on test messages
would leak, subtly and unprovably, into every number downstream.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from support_agent.config import Paths, seed_for, settings
from support_agent.llm import LLM, parse_json

NAME_PROMPT = """You are naming a cluster of customer-support messages sent to an airline on Twitter.

Give the cluster a short intent name in verb + noun form, lowercase, 2-3 words
(for example: "report delay", "claim baggage", "request refund").

Reply with JSON only:
{"name": "...", "description": "one sentence describing what the customer wants", "distinct_from": "the nearest other intent this could be confused with"}"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--brand", default=settings.brand)
    ap.add_argument("--n", type=int, default=1200, help="messages to sample")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--sweep", type=str, default="6,8,10,12,15,20,25,30")
    ap.add_argument("--examples", type=int, default=8, help="examples shown per cluster")
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()

    df = pd.read_parquet(Paths.interim / f"{args.brand.lower()}_exchanges.parquet")
    train = df[df["split"] == "train"]
    seed = seed_for(f"taxonomy:{args.brand}")
    sample = train.sample(n=min(args.n, len(train)), random_state=seed)

    vec = TfidfVectorizer(
        min_df=3, max_df=0.5, ngram_range=(1, 2), sublinear_tf=True, stop_words="english"
    )
    X = vec.fit_transform(sample["customer_clean"])

    print(f"sample n={len(sample)} (train only, seed={seed})  vocab={X.shape[1]}")
    print("\nk sweep (silhouette, higher is better — but note it rewards k that split by topic):")
    for k in [int(x) for x in args.sweep.split(",")]:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)
        print(f"  k={k:<3} silhouette={silhouette_score(X, km.labels_):.4f}")

    km = KMeans(n_clusters=args.k, random_state=seed, n_init=10).fit(X)
    sample = sample.assign(cluster=km.labels_)
    terms = np.array(vec.get_feature_names_out())
    order = km.cluster_centers_.argsort()[:, ::-1]

    llm = None if args.no_llm else LLM.for_role("drafter")
    out = []
    for c in range(args.k):
        members = sample[sample["cluster"] == c]
        top = ", ".join(terms[order[c, :12]])
        ex = members["customer_clean"].head(args.examples).tolist()
        rec = {"cluster": c, "size": len(members), "top_terms": top, "examples": ex}
        if llm is not None:
            body = f"Top terms: {top}\n\nMessages:\n" + "\n".join(f"- {e}" for e in ex)
            try:
                r = llm.complete(NAME_PROMPT, body, json_mode=True, max_tokens=800)
                rec["proposed"] = parse_json(r.text)
            except Exception as exc:
                rec["proposed"] = {"error": f"{type(exc).__name__}: {exc}"}
        out.append(rec)

    path = Path("scratchpad") / f"taxonomy_proposal_k{args.k}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n{args.k} clusters, n={len(sample)}:\n")
    for rec in sorted(out, key=lambda r: -r["size"]):
        p = rec.get("proposed", {})
        print(
            f"  [{rec['size']:>4}] {p.get('name', '?'):<24} {p.get('description', rec['top_terms'])[:88]}"
        )
    print(f"\nfull proposal with examples -> {path}")
    if llm is not None:
        print("llm:", llm.stats())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
