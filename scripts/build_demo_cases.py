"""Precompute the demo's replayable cases: every golden message, run through the agent
out-of-fold, with the judge's scores for each drafter.

    uv run python scripts/build_demo_cases.py

The CLI agent fits its classifier on every golden row, so running it on a golden
message scores that message in-sample. The demo must not do that. Here each message
is classified by the fold model that never saw it (repeat 0 of the headline CV, the
same predictions the harness routes and drafts from), so every intent, margin,
draft and judgement shown in the demo is one the evaluation actually scored.

Needs no keys: embeddings, drafts and judgements replay from the committed caches.
A miss is recorded as an error, never filled in.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from support_agent.agent import check_draft
from support_agent.config import GLOBAL_SEED, Paths, settings
from support_agent.data import write_manifest
from support_agent.draft import build as build_drafter
from support_agent.embed import Embedder
from support_agent.eval.harness import DRAFTERS, corpus_retriever, load_golden
from support_agent.eval.judge import Judge
from support_agent.intents import oof_predictions
from support_agent.llm import LLM
from support_agent.retrieve import search_batch
from support_agent.route import OPERATING_THRESHOLD, Router
from support_agent.text import has_placeholder

K = 4  # the harness default


def main() -> int:
    gold = load_golden(settings.brand)
    texts = [r["text"] for r in gold]
    y = np.array([r["intent"] for r in gold])

    emb = Embedder(offline=True)
    preds, posteriors = oof_predictions(emb.encode(texts), y, 5, GLOBAL_SEED)
    hits = search_batch(corpus_retriever(emb), texts, k=K)

    router = Router(threshold=OPERATING_THRESHOLD, signal="margin")
    judge = Judge(llm=LLM.for_role("judge", offline=True))
    drafters = {
        name: build_drafter(
            name,
            **({"llm": LLM.for_role("drafter", offline=True)} if name.endswith("_llm") else {}),
        )
        for name in DRAFTERS
    }

    out = Paths.processed / f"{settings.brand.lower()}_demo_cases.jsonl"
    misses = 0
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for row, text, pred, post, h in zip(gold, texts, preds, posteriors, hits, strict=True):
            drafts = {}
            for name, drafter in drafters.items():
                d = drafter.draft(text, pred, h)
                s = None if d.error else judge.score(text, d.text, h)
                misses += bool(d.error) + bool(s and s.parse_failed)
                drafts[name] = {
                    "text": d.text,
                    "error": d.error,
                    "grounded_in": d.grounded_in,
                    "placeholder": bool(not d.error and has_placeholder(d.text)),
                    "judge": s.as_dict() if s else None,
                }
            decision = check_draft(router.decide(row["text"], pred, post), _as_draft(drafts))
            case = {
                "id": row["id"],
                "text": text,
                "stratum": row.get("stratum"),
                "gold_intent": row["intent"],
                "ambiguous": bool(row.get("ambiguous")),
                # What AmericanAir actually sent. The agent never sees it.
                "brand_reply": row.get("brand_reply", ""),
                "pred_intent": pred,
                "posterior": {k: round(v, 4) for k, v in post.items()},
                "decision": {
                    "action": decision.action,
                    "rule": decision.rule,
                    "reason": decision.reason,
                    "confidence": round(decision.confidence, 4),
                },
                "evidence": [
                    {"id": x.id, "score": round(x.score, 4), "text": x.text, "reply": x.reply}
                    for x in h
                ],
                "drafts": drafts,
            }
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    write_manifest(
        out,
        rows=len(gold),
        classifier="out-of-fold, repeat 0 of the headline 5-fold CV",
        router_threshold=OPERATING_THRESHOLD,
        k=K,
        drafters=list(DRAFTERS),
        judge=judge.llm.model,
        cache_misses=misses,
    )
    print(f"{len(gold)} cases -> {out}  (cache misses or parse failures: {misses})")
    return 0


def _as_draft(drafts: dict):
    """The grounded draft, rebuilt as the object `check_draft` expects."""
    from support_agent.draft import Draft

    g = drafts["grounded_llm"]
    return Draft(g["text"], "grounded_llm", g["grounded_in"], error=g["error"])


if __name__ == "__main__":
    raise SystemExit(main())
