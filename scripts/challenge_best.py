"""Try hard to beat the incumbent. Report honestly if nothing does.

    uv run python scripts/challenge_best.py

The incumbent is mistral-embed + logistic regression. Every arm here is a
genuine attempt to beat it, chosen because it plausibly could:

* **hyperparameter variation** - C in {0.5, 4, 16} and class_weight on/off, which
  were set by taste. (A proper search inside nested CV is in
  `scripts/nested_cv.py`.)
* **PCA fitted inside the fold** - the earlier PCA result was optimistic because
  the projection saw the evaluation rows. This is the honest version.
* **isotonic calibration** - changes the posterior the router depends on even
  when it leaves accuracy alone.

Not run, although an earlier draft of this docstring listed them: ensembles with
the LLM few-shot posterior, an LLM cascade, class-prior correction, and Platt
scaling.

Also reported: **prevalence-weighted metrics**. The golden set is stratified, so
per-class recall is an unbiased estimate but precision is not - the mix of
negatives a class competes against is wrong. Weighting each example by
(traffic prevalence / sample prevalence), with prevalence taken from the random
control stratum, gives the production-facing number. It is noisier, because the
random stratum is only 40 rows, and that is stated rather than hidden.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from support_agent.config import Paths, settings
from support_agent.embed import Embedder
from support_agent.eval.metrics import corrected_resampled_ttest, macro_f1
from support_agent.intents import load_taxonomy, system_head

BASE = "mistral + logreg (incumbent)"


def load_gold() -> list[dict]:
    path = Paths.golden / f"{settings.brand.lower()}_golden.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [r for r in rows if not r.get("excluded")]


def weighted_macro_f1(y_true, y_pred, weights, labels) -> float:
    """Macro-F1 where each example carries an importance weight.

    Weighting corrects the stratified sample back toward real traffic. A class
    absent from the traffic estimate gets weight 0 and drops out of the average,
    which is correct: an intent that never occurs cannot contribute to a
    production score.
    """
    scores = []
    for lab in labels:
        tp = sum(
            w for t, p, w in zip(y_true, y_pred, weights, strict=True) if t == lab and p == lab
        )
        fp = sum(
            w for t, p, w in zip(y_true, y_pred, weights, strict=True) if t != lab and p == lab
        )
        fn = sum(
            w for t, p, w in zip(y_true, y_pred, weights, strict=True) if t == lab and p != lab
        )
        if tp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    tax = load_taxonomy()
    gold = load_gold()
    texts = [r["text"] for r in gold]
    y = np.array([r["intent"] for r in gold])
    X = Embedder().encode(texts)
    print(f"n={len(y)}  dim={X.shape[1]}  {args.repeats}x{args.folds}-fold\n")

    # Traffic prevalence from the unbiased control stratum.
    rand = [r["intent"] for r in gold if r["stratum"] == "random"]
    traffic = Counter(rand)
    sampled = Counter(y)
    weights = np.array(
        [(traffic.get(t, 0) / len(rand)) / (sampled[t] / len(y)) for t in y], dtype=float
    )

    arms: dict[str, object] = {
        BASE: lambda: system_head(),
        "logreg C=0.5": lambda: LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5),
        "logreg C=16": lambda: LogisticRegression(max_iter=3000, class_weight="balanced", C=16.0),
        "logreg no class_weight": lambda: LogisticRegression(max_iter=3000, C=4.0),
        "PCA-64 in-fold + logreg": lambda: make_pipeline(
            PCA(n_components=64, random_state=0),
            system_head(),
        ),
        "PCA-32 in-fold + logreg": lambda: make_pipeline(
            PCA(n_components=32, random_state=0),
            system_head(),
        ),
        "isotonic calibrated logreg": lambda: CalibratedClassifierCV(
            system_head(),
            cv=3,
            method="isotonic",
        ),
    }

    preds: dict[str, list[list[str]]] = {k: [] for k in arms}
    fold_f1: dict[str, list[float]] = {k: [] for k in arms}
    for rep in range(args.repeats):
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=1000 + rep)
        splits = list(skf.split(X, y))
        for name, make in arms.items():
            out = [""] * len(y)
            for tr, te in splits:
                m = make()
                m.fit(X[tr], y[tr])
                fold_pred = m.predict(X[te])
                for i, p in zip(te, fold_pred, strict=True):
                    out[i] = p
                fold_f1[name].append(macro_f1(list(y[te]), list(fold_pred), tax.ids))
            preds[name].append(out)

    print(f"{'arm':<28}{'macro-F1':>10}{'sd':>7}{'weighted-F1':>13}{'delta':>8}")
    print("-" * 68)
    base_mean = float(np.mean([macro_f1(list(y), p, tax.ids) for p in preds[BASE]]))
    rows = []
    for name, runs in preds.items():
        s = np.array([macro_f1(list(y), p, tax.ids) for p in runs])
        w = np.mean([weighted_macro_f1(list(y), p, weights, tax.ids) for p in runs])
        rows.append((name, s.mean(), s.std(), w))
    for name, mean, sd, w in sorted(rows, key=lambda r: -r[1]):
        d = mean - base_mean
        print(f"{name:<28}{mean:>10.3f}{sd:>7.3f}{w:>13.3f}{d:>+8.3f}")

    best = max(rows, key=lambda r: r[1])[0]
    if best != BASE:
        a_only = b_only = 0
        for ra, rb in zip(preds[best], preds[BASE], strict=True):
            for truth, pa, pb in zip(y, ra, rb, strict=True):
                a_only += (pa == truth) and (pb != truth)
                b_only += (pa != truth) and (pb == truth)
        # Corrected resampled t-test, not McNemar pooled over repeats (#42).
        n_test = len(y) // args.folds
        diffs = np.array(fold_f1[best]) - np.array(fold_f1[BASE])
        delta, _, p = corrected_resampled_ttest(diffs, len(y) - n_test, n_test)
        verdict = "SIGNIFICANT" if p < 0.05 else "not significant - incumbent stands"
        print(
            f"\nbest challenger '{best}' vs incumbent: {delta:+.3f} per-fold macro-F1, "
            f"corrected resampled t-test p={p:.3f} -> {verdict}"
            f"\n(discordant items summed over repeats, descriptive only: {a_only} vs {b_only})"
        )
    else:
        print("\nno challenger beat the incumbent")

    print(f"\ntraffic prevalence from the random stratum (n={len(rand)}):")
    for k in tax.ids:
        pct = traffic.get(k, 0) / len(rand)
        print(f"  {k:<22}{traffic.get(k, 0):>3}{pct:>8.1%}   golden {sampled[k]:>3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
