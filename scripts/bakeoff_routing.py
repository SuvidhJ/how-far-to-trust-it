"""Choose the escalation signal by measurement.

    uv run python scripts/bakeoff_routing.py

There is no hand-labelled "should this have been escalated" column, and inventing
one after seeing model output would be circular. So escalation is evaluated as
**selective prediction**: the router's job is to hand off the cases the
classifier would have got wrong. Under that definition the ground truth is free -
it is whether the out-of-fold prediction was correct.

The headline is **AURC** (area under the risk-coverage curve): sweep the
threshold, and at each coverage level record the error rate on the auto-handled
remainder. Lower is better, and a perfect confidence signal would push all errors
into the escalated set. This is the standard metric for exactly this problem and
it needs no threshold to be chosen first.

Reported alongside: error rate at fixed coverage levels, which is what an
operations manager actually asks - "if I let it answer half the tickets, how
often is it wrong?"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from support_agent.config import GLOBAL_SEED, Paths, settings
from support_agent.embed import Embedder
from support_agent.intents import oof_predictions
from support_agent.route import Router, margin, normalised_entropy


def risk_coverage(correct: np.ndarray, confidence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sweep the threshold from answer-everything to answer-nothing.

    Ties are broken by the sort order, which is arbitrary but consistent across
    methods, so the comparison stays fair.
    """
    order = np.argsort(-confidence, kind="stable")
    ordered = correct[order]
    n = len(ordered)
    coverage = np.arange(1, n + 1) / n
    risk = 1.0 - np.cumsum(ordered) / np.arange(1, n + 1)
    return coverage, risk


def aurc(correct: np.ndarray, confidence: np.ndarray) -> float:
    coverage, risk = risk_coverage(correct, confidence)
    return float(np.trapezoid(risk, coverage))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    path = Paths.golden / f"{settings.brand.lower()}_golden.jsonl"
    gold = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    gold = [r for r in gold if not r.get("excluded")]
    texts = [r["text"] for r in gold]
    y = np.array([r["intent"] for r in gold])
    X = Embedder().encode(texts)

    # Out-of-fold posteriors from the chosen classifier.
    preds, posteriors = oof_predictions(X, y, args.folds, GLOBAL_SEED)

    correct = np.array([p == t for p, t in zip(preds, y, strict=True)], dtype=float)
    base_err = 1.0 - correct.mean()
    print(f"n={len(y)}  classifier accuracy={correct.mean():.3f}  base error={base_err:.3f}")
    print(f"random-confidence AURC would be ~{base_err:.3f}\n")

    signals = {
        "max probability": np.array([max(p.values()) if p else 0.0 for p in posteriors]),
        "top-2 margin": np.array([margin(p) for p in posteriors]),
        "negative entropy": np.array([1.0 - normalised_entropy(p) for p in posteriors]),
    }

    levels = [0.3, 0.5, 0.7, 0.9]
    header = "".join(f"{f'err@{int(c * 100)}%':>10}" for c in levels)
    print(f"{'signal':<20}{'AURC':>8}{header}")
    print("-" * (28 + 10 * len(levels)))
    for name, conf in signals.items():
        _, risk = risk_coverage(correct, conf)
        at = [risk[min(len(risk) - 1, int(c * len(risk)) - 1)] for c in levels]
        cells = "".join(f"{v:>10.3f}" for v in at)
        print(f"{name:<20}{aurc(correct, conf):>8.3f}{cells}")

    # The rule layer is not a confidence signal, so it is judged differently:
    # how much of the classifier's error does it catch, and at what cost in
    # unnecessary escalations?
    print("\nrule layer (hard rules only, no confidence threshold):")
    router = Router(threshold=0.0, use_rules=True)
    decisions = [
        router.decide(t, p, post) for t, p, post in zip(texts, preds, posteriors, strict=True)
    ]
    esc = np.array([d.escalated for d in decisions])
    wrong = correct == 0
    print(f"  escalated {esc.sum()}/{len(esc)} ({esc.mean():.0%} of traffic)")
    print(
        f"  caught {int((esc & wrong).sum())}/{int(wrong.sum())} of the classifier's errors "
        f"({(esc & wrong).sum() / max(1, wrong.sum()):.0%} recall on errors)"
    )
    print(f"  cost: {int((esc & ~wrong).sum())} correct answers escalated unnecessarily")
    remaining = ~esc
    if remaining.any():
        print(
            f"  error rate on what it still auto-handles: "
            f"{1 - correct[remaining].mean():.3f} (vs {base_err:.3f} overall)"
        )

    from collections import Counter

    print("\n  which rules fired:")
    for rule, n in Counter(d.rule for d in decisions if d.escalated).most_common():
        print(f"    {rule:<26}{n:>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
