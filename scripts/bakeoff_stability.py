"""Separate the top candidates properly, or admit they cannot be separated.

    uv run python scripts/bakeoff_stability.py

The deep sweep left five methods inside one another's confidence intervals. A
single 5-fold split cannot rank those; the ordering it produces is mostly an
artefact of which examples landed in which fold. Two things fix that:

* **Repeated cross-validation.** Five different fold assignments, so each method
  gets a mean and a spread rather than one number.
* **A paired test that respects the resampling.** Methods are compared fold by
  fold on the same examples. Summing McNemar's discordant counts over repeats
  is *not* a valid test: each example is counted once per repeat and folds
  share training data, so the pooled p-value was far too small. That test
  backed "nemotron worse, p<=0.022 on every head" in DECISIONS #23. The
  Nadeau-Bengio corrected resampled t-test on per-fold macro-F1 replaces it
  (#42); pooled counts are still printed, as description only.

The output is deliberately allowed to conclude "no significant difference". A
tie that is reported as a tie is worth more than a winner invented from noise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestCentroid
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from support_agent.config import Paths, settings
from support_agent.embed import Embedder
from support_agent.eval.metrics import corrected_resampled_ttest, macro_f1
from support_agent.intents import load_taxonomy, system_head

CANDIDATES = {
    "mistral + centroid": ("mistral", NearestCentroid),
    "mistral + logreg": (
        "mistral",
        lambda: system_head(),
    ),
    "mistral + svm": ("mistral", lambda: LinearSVC(class_weight="balanced", C=1.0, max_iter=5000)),
    "nemotron + centroid": ("nemotron", NearestCentroid),
    "nemotron + logreg": (
        "nemotron",
        lambda: system_head(),
    ),
    "nemotron + svm": (
        "nemotron",
        lambda: LinearSVC(class_weight="balanced", C=1.0, max_iter=5000),
    ),
}
EMBEDDERS = {
    "mistral": ("mistral", "mistral-embed"),
    "nemotron": ("nvidia", "nvidia/nemotron-3-embed-1b"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    tax = load_taxonomy()
    path = Paths.golden / f"{settings.brand.lower()}_golden.jsonl"
    gold = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    gold = [r for r in gold if not r.get("excluded")]
    texts = [r["text"] for r in gold]
    y = np.array([r["intent"] for r in gold])

    vecs = {}
    for tag, (provider, model) in EMBEDDERS.items():
        vecs[tag] = Embedder(model=model, provider=provider).encode(texts, input_type="passage")

    # preds[name][repeat] -> out-of-fold predictions for every example
    preds: dict[str, list[list[str]]] = {name: [] for name in CANDIDATES}
    fold_f1: dict[str, list[float]] = {name: [] for name in CANDIDATES}
    for rep in range(args.repeats):
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=1000 + rep)
        splits = list(skf.split(vecs["mistral"], y))
        for name, (tag, make) in CANDIDATES.items():
            x = vecs[tag]
            out = [""] * len(y)
            for tr, te in splits:
                m = make()
                m.fit(x[tr], y[tr])
                fold_pred = m.predict(x[te])
                for i, p in zip(te, fold_pred, strict=True):
                    out[i] = p
                fold_f1[name].append(macro_f1(list(y[te]), list(fold_pred), tax.ids))
            preds[name].append(out)

    print(f"n={len(y)}  {args.repeats} repeats x {args.folds} folds\n")
    print(f"{'method':<24}{'mean macro-F1':>15}{'sd':>8}{'min':>8}{'max':>8}")
    print("-" * 63)
    scores = {}
    for name, runs in preds.items():
        s = np.array([macro_f1(list(y), p, tax.ids) for p in runs])
        scores[name] = s
        print(f"{name:<24}{s.mean():>15.3f}{s.std():>8.3f}{s.min():>8.3f}{s.max():>8.3f}")

    ranked = sorted(scores, key=lambda k: -scores[k].mean())
    top = ranked[0]
    n_test = len(y) // args.folds
    print(f"\nagainst the leader ({top}): corrected resampled t-test on per-fold macro-F1")
    print(f"{'challenger':<24}{'only-leader':>12}{'only-chal':>11}{'delta':>8}{'p':>8}  verdict")
    print("-" * 78)
    for name in ranked[1:]:
        a_only = b_only = 0
        for ra, rb in zip(preds[top], preds[name], strict=True):
            for truth, pa, pb in zip(y, ra, rb, strict=True):
                if (pa == truth) and (pb != truth):
                    a_only += 1
                elif (pa != truth) and (pb == truth):
                    b_only += 1
        diffs = np.array(fold_f1[top]) - np.array(fold_f1[name])
        delta, _, p = corrected_resampled_ttest(diffs, len(y) - n_test, n_test)
        verdict = (
            "leader better"
            if p < 0.05 and delta > 0
            else ("challenger better" if p < 0.05 else "no significant difference")
        )
        print(f"{name:<24}{a_only:>12}{b_only:>11}{delta:>+8.3f}{p:>8.3f}  {verdict}")
    print("only-leader / only-chal: discordant items summed over repeats, descriptive only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
