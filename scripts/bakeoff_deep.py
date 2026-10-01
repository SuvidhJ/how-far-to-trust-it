"""Exhaustive method sweep for intent classification.

    uv run python scripts/bakeoff_deep.py

`bakeoff_intent.py` compared six obvious candidates. This compares everything
that could plausibly change the design decision, along three axes that the first
sweep held fixed:

1. **Which embedding model.** The first sweep picked nemotron on a single
   three-sentence similarity probe. That is not evidence. Here all three free
   embedding models are scored on the actual task.
2. **Which classifier head.** Logistic regression was assumed. Linear SVM,
   centroid, kNN, MLP and a TF-IDF-union model are all cheap to try.
3. **Whether labels are needed at all.** Embedding the intent *descriptions* and
   taking the nearest one is a zero-shot method that needs no training data. If
   it were competitive it would change what the golden set is for.

Same protocol throughout: stratified 5-fold CV, out-of-fold predictions,
identical folds, n=219, the anchored row excluded.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from support_agent.config import GLOBAL_SEED, Paths, settings
from support_agent.embed import Embedder
from support_agent.eval.metrics import intent_report
from support_agent.intents import load_taxonomy, system_head

EMBED_MODELS = [
    ("nvidia", "nvidia/nemotron-3-embed-1b"),
    ("mistral", "mistral-embed"),
    ("gemini", "gemini-embedding-001"),
]


def short_name(model: str) -> str:
    return model.split("/")[-1].replace("-embed", "").replace("embedding-", "")[:12]


BY_SHORT = {short_name(m): (p, m) for p, m in EMBED_MODELS}


def load_golden(brand: str) -> list[dict]:
    path = Paths.golden / f"{brand.lower()}_golden.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [r for r in rows if not r.get("excluded")]


def cv_matrix(x, y, make, folds: int, seed: int) -> list[str]:
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    preds = [""] * len(y)
    for tr, te in skf.split(x, y):
        m = make()
        m.fit(x[tr], y[tr])
        for i, p in zip(te, m.predict(x[te]), strict=True):
            preds[i] = p
    return preds


HEADS = {
    "logreg": lambda: system_head(),
    "logreg C=1": lambda: LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0),
    "linear svm": lambda: LinearSVC(class_weight="balanced", C=1.0, max_iter=5000),
    "centroid": NearestCentroid,
    "knn k=5": lambda: KNeighborsClassifier(n_neighbors=5, metric="cosine"),
    "knn k=15": lambda: KNeighborsClassifier(n_neighbors=15, metric="cosine", weights="distance"),
    # LinearSVC has no predict_proba, and the escalation router needs a
    # probability. Platt scaling via CalibratedClassifierCV buys one back, at
    # the cost of an inner CV loop; whether it keeps the SVM's accuracy is
    # exactly the question this arm answers.
    "calibrated svm": lambda: CalibratedClassifierCV(
        LinearSVC(class_weight="balanced", C=1.0, max_iter=5000), cv=3, method="sigmoid"
    ),
    "mlp": lambda: MLPClassifier(
        hidden_layer_sizes=(256,), max_iter=800, random_state=GLOBAL_SEED, early_stopping=True
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--brand", default=settings.brand)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    tax = load_taxonomy()
    gold = load_golden(args.brand)
    texts = [r["text"] for r in gold]
    y = np.array([r["intent"] for r in gold])
    print(f"n={len(gold)}  classes={len(set(y))}  folds={args.folds}\n")

    rows: list[tuple[str, float, tuple[float, float], float, float]] = []

    def record(name: str, preds: list[str], secs: float) -> None:
        rep = intent_report(list(y), preds, tax.ids)
        rows.append((name, rep.macro_f1, rep.macro_f1_ci, rep.accuracy, secs))

    # ---- axis 1 and 2: embedding model x classifier head -------------------
    vectors: dict[str, np.ndarray] = {}
    for provider, model in EMBED_MODELS:
        short = short_name(model)
        try:
            emb = Embedder(model=model, provider=provider)
            t = time.time()
            vectors[short] = emb.encode(texts, input_type="passage")
            print(
                f"{short:<14} dim={vectors[short].shape[1]:<5} {time.time() - t:5.1f}s "
                f"({emb.api_calls} calls, {emb.cache_hits} hits)"
            )
        except Exception as exc:
            print(f"{short:<14} FAILED {type(exc).__name__}: {str(exc)[:70]}")
    print()

    for short, X in vectors.items():
        for head_name, make in HEADS.items():
            t = time.time()
            try:
                record(
                    f"{short} + {head_name}",
                    cv_matrix(X, y, make, args.folds, GLOBAL_SEED),
                    time.time() - t,
                )
            except Exception as exc:
                print(f"  {short} + {head_name} failed: {type(exc).__name__}")

    # ---- TF-IDF arms ------------------------------------------------------
    # These go through a Pipeline so the vectoriser is refitted inside every
    # fold. Fitting it once on all 219 texts leaks the test half's vocabulary
    # and idf weights into training, and it is the single easiest way to publish
    # an inflated baseline without noticing.
    from sklearn.pipeline import make_pipeline

    x_text = np.array(texts, dtype=object)
    word_vec = dict(
        sublinear_tf=True, ngram_range=(1, 2), min_df=1, max_df=0.9, strip_accents="unicode"
    )
    for name, make in (
        ("tfidf + logreg", lambda: make_pipeline(TfidfVectorizer(**word_vec), HEADS["logreg"]())),
        (
            "tfidf + linear svm",
            lambda: make_pipeline(TfidfVectorizer(**word_vec), HEADS["linear svm"]()),
        ),
        (
            "tfidf + complementNB",
            lambda: make_pipeline(TfidfVectorizer(**word_vec), ComplementNB()),
        ),
        (
            "char-tfidf + logreg",
            lambda: make_pipeline(
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True
                ),
                HEADS["logreg"](),
            ),
        ),
    ):
        t = time.time()
        record(name, cv_matrix(x_text, y, make, args.folds, GLOBAL_SEED), time.time() - t)

    if vectors:
        best_emb = max(
            vectors, key=lambda k: max((r[1] for r in rows if r[0].startswith(k)), default=0.0)
        )

        # ---- axis 3: zero-shot by label-description similarity -------------
        # No labels used at all: embed each intent's name and description, assign
        # the nearest one. If this came close to the supervised numbers, the
        # golden set would not be earning its cost.
        provider, model = BY_SHORT[best_emb]
        emb = Embedder(model=model, provider=provider)
        descriptions = [f"{i.name}: {i.description}" for i in tax.intents]
        t = time.time()
        D = emb.encode(descriptions, input_type="passage")
        sims = vectors[best_emb] @ D.T
        preds = [tax.intents[j].id for j in sims.argmax(axis=1)]
        record(f"{best_emb} zero-shot label similarity", preds, time.time() - t)

    rows.sort(key=lambda r: -r[1])
    print(f"{'method':<40}{'macro-F1':>10}{'95% CI':>18}{'acc':>7}{'time':>9}")
    print("-" * 86)
    for name, f1, ci, acc, secs in rows:
        print(f"{name:<40}{f1:>10.3f}{f'[{ci[0]:.3f}, {ci[1]:.3f}]':>18}{acc:>7.3f}{secs:>8.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
