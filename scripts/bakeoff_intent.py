"""Choose the intent-classification method by measurement.

    uv run python scripts/bakeoff_intent.py

Six candidates, one protocol, one number each. The supervised arms are scored by
stratified 5-fold cross-validation over the golden set, because the golden set is
the only labelled data that exists - there is no separate labelled training set
to hold out. The zero-shot arms need no training, so they are scored on exactly
the same fold assignments, which keeps every number comparable.

The row flagged `anchored` is excluded throughout: its label may have been
influenced by seeing a model's prediction before labelling.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from support_agent.config import GLOBAL_SEED, Paths, settings
from support_agent.eval.metrics import intent_report
from support_agent.intents import (
    FewShotLLMClassifier,
    LLMClassifier,
    MajorityClassifier,
    TfidfClassifier,
    load_taxonomy,
    system_head,
)


def load_golden(brand: str) -> list[dict]:
    path = Paths.golden / f"{brand.lower()}_golden.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [r for r in rows if not r.get("excluded")]


def cv_predict(make_model, texts: list[str], labels: list[str], folds: int, seed: int) -> list[str]:
    """Out-of-fold predictions, so every example is predicted by a model that
    never saw it."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    preds = [""] * len(texts)
    x, y = np.array(texts, dtype=object), np.array(labels)
    for train_idx, test_idx in skf.split(x, y):
        model = make_model()
        model.fit(list(x[train_idx]), list(y[train_idx]))
        for i, p in zip(test_idx, model.predict(list(x[test_idx])), strict=True):
            preds[i] = p.intent if hasattr(p, "intent") else p
    return preds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--brand", default=settings.brand)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--skip-embed", action="store_true")
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--fewshot-k", type=int, default=12)
    args = ap.parse_args()

    tax = load_taxonomy()
    gold = load_golden(args.brand)
    texts = [r["text"] for r in gold]
    labels = [r["intent"] for r in gold]
    print(f"golden n={len(gold)} (1 anchored row excluded), {len(set(labels))} classes present\n")

    results: dict[str, tuple[list[str], float]] = {}

    t = time.time()
    results["majority (trivial)"] = (
        cv_predict(MajorityClassifier, texts, labels, args.folds, GLOBAL_SEED),
        time.time() - t,
    )

    t = time.time()
    results["tfidf + logreg (simple)"] = (
        cv_predict(TfidfClassifier, texts, labels, args.folds, GLOBAL_SEED),
        time.time() - t,
    )

    if not args.skip_embed:
        from support_agent.embed import Embedder

        emb = Embedder()
        t = time.time()
        vectors = emb.encode(texts, input_type="passage")
        encode_s = time.time() - t
        print(
            f"embedded {len(texts)} texts in {encode_s:.1f}s "
            f"({emb.api_calls} api calls, {emb.cache_hits} cache hits), dim={vectors.shape[1]}\n"
        )

        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=GLOBAL_SEED)
        y = np.array(labels)
        for name, make in (
            ("embed + knn(k=5)", lambda: KNeighborsClassifier(n_neighbors=5, metric="cosine")),
            (
                "embed + logreg",
                lambda: system_head(),
            ),
        ):
            t = time.time()
            preds = [""] * len(texts)
            for train_idx, test_idx in skf.split(vectors, y):
                model = make()
                model.fit(vectors[train_idx], y[train_idx])
                for i, p in zip(test_idx, model.predict(vectors[test_idx]), strict=True):
                    preds[i] = p
            results[name] = (preds, time.time() - t)

    if not args.skip_llm:
        clf = LLMClassifier()
        t = time.time()
        preds = [p.intent for p in clf.predict(texts)]
        results["llm zero-shot"] = (preds, time.time() - t)
        print("llm zero-shot:", clf.llm.stats())

        # The fair arm: the prompt gets the same labels the linear model gets,
        # as the k nearest labelled examples from that fold's training half.
        t = time.time()
        fs = cv_predict(
            lambda: FewShotLLMClassifier(k=args.fewshot_k), texts, labels, args.folds, GLOBAL_SEED
        )
        results[f"llm few-shot (k={args.fewshot_k})"] = (fs, time.time() - t)
        print()

    print(f"{'method':<26}{'macro-F1':>10}{'95% CI':>18}{'acc':>7}{'time':>9}")
    print("-" * 72)
    best = None
    for name, (preds, secs) in results.items():
        rep = intent_report(labels, preds, tax.ids)
        ci = f"[{rep.macro_f1_ci[0]:.3f}, {rep.macro_f1_ci[1]:.3f}]"
        print(f"{name:<26}{rep.macro_f1:>10.3f}{ci:>18}{rep.accuracy:>7.3f}{secs:>8.1f}s")
        if best is None or rep.macro_f1 > best[1]:
            best = (name, rep.macro_f1, rep)
    print(f"\nbest: {best[0]}  macro-F1={best[1]:.3f}  n={best[2].n}")
    print("\nper class, best method:\n")
    print(best[2].table())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
