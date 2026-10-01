"""Nested cross-validation: evaluate the *selection procedure*, not the winner.

    uv run python scripts/nested_cv.py

Every method choice in this project (embedder, head, C) was made by comparing
candidates on the same 218 golden rows the headline is reported on. Picking the
best of many configurations on the evaluation data inflates the winner's score.
Nested CV measures by how much: an inner 4-fold CV on each outer training split
picks the configuration, and only the outer test fold scores it.

The simple baseline gets the identical treatment (word or character TF-IDF,
logistic regression or SVM, C searched). A tuned system against an untuned
baseline would overstate the gap. That is exactly what the fixed
`tfidf-word + logreg C=4` baseline in the harness does (DECISIONS #43).

5 outer repeats x 5 folds, seeds from GLOBAL_SEED, embeddings replayed offline.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import collections
import json
import warnings

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from support_agent.config import GLOBAL_SEED, Paths
from support_agent.embed import Embedder
from support_agent.eval.metrics import macro_f1
from support_agent.intents import load_taxonomy

warnings.filterwarnings("ignore")
ids = load_taxonomy().ids
g = [
    json.loads(line)
    for line in (Paths.golden / "americanair_golden.jsonl").read_text(encoding="utf-8").splitlines()
]
g = [r for r in g if not r.get("excluded")]
texts = np.array([r["text"] for r in g], dtype=object)
y = np.array([r["intent"] for r in g])
V = {
    "mistral": Embedder(offline=True).encode(list(texts)),
    "nemotron": Embedder(
        model="nvidia/nemotron-3-embed-1b", provider="nvidia", offline=True
    ).encode(list(texts)),
}


def LR(C):
    return lambda: LogisticRegression(max_iter=3000, class_weight="balanced", C=C)


SYSTEM = {}
for e in V:
    for C in (0.5, 1, 4, 16):
        SYSTEM[f"{e}+logreg C={C}"] = (e, LR(C))
    SYSTEM[f"{e}+svm"] = (e, lambda: LinearSVC(class_weight="balanced", C=1.0, max_iter=5000))
    SYSTEM[f"{e}+centroid"] = (e, NearestCentroid)
    SYSTEM[f"{e}+knn15"] = (
        e,
        lambda: KNeighborsClassifier(15, metric="cosine", weights="distance"),
    )
WORD = dict(sublinear_tf=True, ngram_range=(1, 2), min_df=1, max_df=0.9, strip_accents="unicode")
CHAR = dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
SIMPLE = {}
for vn, vk in (("word", WORD), ("char", CHAR)):
    for C in (0.5, 1, 4, 16):
        SIMPLE[f"tfidf-{vn}+logreg C={C}"] = (
            None,
            (lambda vk=vk, C=C: make_pipeline(TfidfVectorizer(**vk), LR(C)())),
        )
    SIMPLE[f"tfidf-{vn}+svm"] = (
        None,
        (
            lambda vk=vk: make_pipeline(
                TfidfVectorizer(**vk), LinearSVC(class_weight="balanced", C=1.0, max_iter=5000)
            )
        ),
    )


def X_of(emb, idx):
    return V[emb][idx] if emb else list(texts[idx])


def cv_score(cands, name, idx, seed):
    emb, make = cands[name]
    ys = y[idx]
    out = np.empty(len(idx), dtype=object)
    for tr, te in StratifiedKFold(4, shuffle=True, random_state=seed).split(np.zeros(len(idx)), ys):
        m = make().fit(X_of(emb, idx[tr]), ys[tr])
        out[te] = m.predict(X_of(emb, idx[te]))
    return macro_f1(list(ys), list(out), ids)


def nested(cands, fixed):
    res, chosen, fixed_res = [], collections.Counter(), []
    for rep in range(5):
        pred = np.empty(len(y), dtype=object)
        fpred = np.empty(len(y), dtype=object)
        for k, (tr, te) in enumerate(
            StratifiedKFold(5, shuffle=True, random_state=GLOBAL_SEED + rep).split(
                np.zeros(len(y)), y
            )
        ):
            best = max(cands, key=lambda nm: cv_score(cands, nm, tr, GLOBAL_SEED + 100 * rep + k))
            chosen[best] += 1
            emb, make = cands[best]
            pred[te] = make().fit(X_of(emb, tr), y[tr]).predict(X_of(emb, te))
            emb, make = cands[fixed]
            fpred[te] = make().fit(X_of(emb, tr), y[tr]).predict(X_of(emb, te))
        res.append(macro_f1(list(y), list(pred), ids))
        fixed_res.append(macro_f1(list(y), list(fpred), ids))
    return np.array(res), np.array(fixed_res), chosen


for label, cands, fixed in (
    ("system (embeddings)", SYSTEM, "mistral+logreg C=4"),
    ("simple (tf-idf)", SIMPLE, "tfidf-word+logreg C=4"),
):
    r, f, ch = nested(cands, fixed)
    print(
        f"{label:<22} nested (selection inside the fold): {r.mean():.3f} sd {r.std():.3f} | fixed config {fixed}: {f.mean():.3f} sd {f.std():.3f}"
    )
    print(f"{'':<22} chosen in the 25 outer folds: {dict(ch.most_common())}")
