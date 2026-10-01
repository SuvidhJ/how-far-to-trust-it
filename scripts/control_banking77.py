"""Banking77 control: is 0.64 macro-F1 a data problem or a method problem?

    uv run python scripts/control_banking77.py

Runs the *identical* intent classifier - mistral-embed + logistic regression
(C=4, balanced), 5 repeats x 5-fold CV, same baselines - on Banking77, a clean,
crowd-labelled benchmark (Casanueva et al. 2020, PolyAI).

Two configurations, because a bigger label set and more data are both confounds:

* **matched** - 10 intents drawn at random, sampled to reproduce the AmericanAir
  golden set's exact class-size profile (n=218, largest class 39, smallest 9).
  Only the data differs. Repeated over several random intent draws, because which
  10 intents are drawn decides how confusable they are.
* **full** - all 77 intents, 20 examples each (n=1,540).

If the method is sound, it should score far higher here than on AmericanAir. The
gap is then task difficulty, not a broken pipeline: short, noisy, multi-intent
tweets; a single annotator whom an independent LLM annotator matches at only kappa
0.466 (#47); and intents that sit closer together than Banking77's (#45).
Random draws do not control for that last point, which is why the most confusable
10 intents are scored as well.

The data comes from the URL the Hugging Face `PolyAI/banking77` loader itself uses.
"""

from __future__ import annotations

import csv
import io
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from support_agent.config import GLOBAL_SEED, ROOT, Paths, seed_for, settings
from support_agent.data import write_manifest
from support_agent.embed import MODEL, Embedder
from support_agent.eval.metrics import macro_f1
from support_agent.intents import system_head

URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
# `Paths.raw` is the twcs CSV file itself, not a directory.
RAW = ROOT / "data" / "raw" / "banking77" / "train.csv"
FOLDS, REPEATS, DRAWS, FULL_PER_CLASS = 5, 5, 5, 20


def fetch() -> list[tuple[str, str]]:
    if not RAW.exists():
        RAW.parent.mkdir(parents=True, exist_ok=True)
        # A user-level SSLKEYLOGFILE crashes OpenSSL inside Python on this machine.
        os.environ.pop("SSLKEYLOGFILE", None)
        RAW.write_bytes(urllib.request.urlopen(URL, timeout=120).read())
    rows = list(csv.DictReader(io.StringIO(RAW.read_text(encoding="utf-8"))))
    return [(r["text"], r["category"]) for r in rows]


def cv(texts: list[str], X: np.ndarray, y: np.ndarray, seed_base: int) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {"majority": [], "tfidf+logreg": [], "embed+logreg": []}
    for rep in range(REPEATS):
        skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed_base + rep)
        preds = {k: [""] * len(y) for k in out}
        for tr, te in skf.split(X, y):
            vals, counts = np.unique(y[tr], return_counts=True)
            for i in te:
                preds["majority"][i] = str(vals[counts.argmax()])
            tf = make_pipeline(
                TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=1),
                LogisticRegression(
                    max_iter=3000, class_weight="balanced", C=4.0
                ),  # baseline's own head
            ).fit([texts[i] for i in tr], y[tr])
            for i, p in zip(te, tf.predict([texts[i] for i in te]), strict=True):
                preds["tfidf+logreg"][i] = p
            m = system_head().fit(X[tr], y[tr])
            for i, p in zip(te, m.predict(X[te]), strict=True):
                preds["embed+logreg"][i] = p
        labels = sorted(set(y))
        for k in out:
            out[k].append(macro_f1(list(y), preds[k], labels))
    return out


def main() -> int:
    t0 = time.time()
    data = fetch()
    by_class: dict[str, list[str]] = {}
    for text, cat in data:
        by_class.setdefault(cat, []).append(text)
    classes = sorted(by_class)
    print(f"Banking77 train: {len(data):,} texts, {len(classes)} intents  ({URL})\n")

    gold = [
        json.loads(line)
        for line in (Paths.golden / f"{settings.brand.lower()}_golden.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    gold = [r for r in gold if not r.get("excluded")]
    profile = sorted(
        (sum(r["intent"] == c for r in gold) for c in {r["intent"] for r in gold}), reverse=True
    )
    rng = np.random.default_rng(seed_for("banking77"))
    emb = Embedder()

    configs: list[tuple[str, list[str], list[str]]] = []
    for d in range(DRAWS):
        drawn = rng.choice(classes, size=len(profile), replace=False)
        texts, labels = [], []
        for cat, size in zip(drawn, profile, strict=True):
            for t in rng.choice(by_class[cat], size=size, replace=False):
                texts.append(str(t))
                labels.append(cat)
        configs.append((f"matched draw {d}", texts, labels))
    texts, labels = [], []
    for cat in classes:
        for t in rng.choice(by_class[cat], size=FULL_PER_CLASS, replace=False):
            texts.append(str(t))
            labels.append(cat)
    configs.append(("full 77 intents", texts, labels))

    results = {}
    print(
        f"{'configuration':<20}{'n':>6}{'classes':>9}{'majority':>10}{'tfidf+logreg':>16}{'embed+logreg':>16}"
    )
    print("-" * 77)
    for name, texts, labels in configs:
        X = emb.encode(texts)
        y = np.array(labels)
        r = cv(texts, X, y, GLOBAL_SEED)
        results[name] = {
            "n": len(y),
            "classes": len(set(y)),
            **{k: [float(v) for v in vs] for k, vs in r.items()},
        }
        print(
            f"{name:<20}{len(y):>6}{len(set(y)):>9}"
            + "".join(
                f"{np.mean(v):>10.3f}"
                if k == "majority"
                else f"{np.mean(v):>9.3f} ± {np.std(v):.3f}"
                for k, v in r.items()
            )
        )
    emb.flush()

    matched = [results[n]["embed+logreg"] for n in results if n.startswith("matched")]
    per_draw = [float(np.mean(m)) for m in matched]
    print(
        f"\nmatched, embed+logreg across {DRAWS} intent draws: mean {np.mean(per_draw):.3f}, "
        f"range [{min(per_draw):.3f}, {max(per_draw):.3f}]  (AmericanAir, same protocol: 0.640, n=218)"
    )

    # The random draws are not matched on *confusability*: ten intents drawn from 77
    # are usually far apart, while one brand's own taxonomy is ten neighbours. So the
    # ten most mutually similar Banking77 intents (greedy on centroid cosine, from the
    # full-77 sample, whose vectors are already cached) are scored too (DECISIONS #45).
    full_texts, full_labels = configs[-1][1], configs[-1][2]
    Xf, yf = emb.encode(full_texts), np.array(full_labels)

    def mean_centroid_cos(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, list[str], float]:
        cls = sorted(set(y))
        c = np.array([X[y == k].mean(0) for k in cls])
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        sim = c @ c.T
        off = sim[~np.eye(len(cls), dtype=bool)]
        return sim, cls, float(off.mean())

    sim, cls, _ = mean_centroid_cos(Xf, yf)
    np.fill_diagonal(sim, -1)
    i, j = np.unravel_index(sim.argmax(), sim.shape)
    chosen = [int(i), int(j)]
    while len(chosen) < len(profile):
        rest = [k for k in range(len(cls)) if k not in chosen]
        chosen.append(max(rest, key=lambda k: sim[k, chosen].mean()))
    names = [cls[k] for k in chosen]
    mask = np.isin(yf, names)
    hard_texts = [t for t, m in zip(full_texts, mask, strict=True) if m]
    r = cv(hard_texts, Xf[mask], yf[mask], GLOBAL_SEED)
    _, _, hard_cos = mean_centroid_cos(Xf[mask], yf[mask])
    results["most confusable 10"] = {
        "n": int(mask.sum()),
        "classes": len(names),
        "intents": names,
        "mean_centroid_cosine": hard_cos,
        **{k: [float(v) for v in vs] for k, vs in r.items()},
    }
    gX = emb.encode([r_["text"] for r_ in gold])
    _, _, gold_cos = mean_centroid_cos(gX, np.array([r_["intent"] for r_ in gold]))
    print(
        f"most confusable 10 of 77 (n={int(mask.sum())}, centroid cosine {hard_cos:.3f}): "
        f"embed+logreg {np.mean(r['embed+logreg']):.3f}, tfidf+logreg {np.mean(r['tfidf+logreg']):.3f}"
        f"\nAmericanAir golden centroid cosine: {gold_cos:.3f} (higher = intents closer together)"
    )

    out = Paths.processed / "banking77_control.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    write_manifest(
        out,
        source=URL,
        rows=sum(r["n"] for r in results.values()),
        seed=seed_for("banking77"),
        filters={"matched_profile": profile, "draws": DRAWS, "full_per_class": FULL_PER_CLASS},
        protocol=f"{REPEATS}x{FOLDS}-fold CV, logreg C=4 balanced",
        embed_model=MODEL,
        embed_api_calls=emb.api_calls,
    )
    print(f"\n{time.time() - t0:.0f}s, {emb.api_calls} embedding calls -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
