"""Intent classification: the frozen taxonomy, the deployed model, and the rivals.

* `system_head` + `oof_predictions` - **the deployed system**: logistic regression
  on `mistral-embed` vectors (DECISIONS #21, #24). Every script that needs the
  system's out-of-fold predictions calls `oof_predictions`, so predictions and
  the posteriors that gate them always come from the same fold models. The
  fold-mismatch bug in #40 came from five hand-copied versions of this loop.
* `MajorityClassifier`  - the trivial baseline. Always the most common class.
* `TfidfClassifier`     - the simple baseline. TF-IDF into logistic regression.
* `LLMClassifier`, `FewShotLLMClassifier` - evaluated and rejected as the system
  (#21: few-shot matched the linear model within noise at ~600x the latency).
  They remain in use as independent annotators (`scripts/annotator_agreement.py`).
  The taxonomy is shown as a lettered list and the model answers with one letter,
  so token 0 carries a posterior over the label set.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from support_agent.config import GLOBAL_SEED
from support_agent.llm import LLM

TAXONOMY_PATH = Path(__file__).resolve().parent / "taxonomy.json"


@dataclass(frozen=True)
class Intent:
    id: str
    letter: str
    name: str
    description: str
    positive: str
    near_miss: str
    near_miss_is: str
    near_miss_why: str


@dataclass(frozen=True)
class Taxonomy:
    version: str
    brand: str
    tie_break_rule: str
    other_rule: str
    intents: tuple[Intent, ...]

    @property
    def ids(self) -> list[str]:
        return [i.id for i in self.intents]

    def by_letter(self, letter: str) -> Intent | None:
        return next((i for i in self.intents if i.letter == letter), None)

    def by_id(self, intent_id: str) -> Intent | None:
        return next((i for i in self.intents if i.id == intent_id), None)

    def as_options(self) -> str:
        """The lettered list shown to the classifier, with its near-miss rubric.

        Each option carries one positive example and one near miss with the
        reason it belongs elsewhere. The near misses are the whole point: 27.6%
        of messages carry more than one intent, so the boundaries are where the
        errors live.
        """
        lines = []
        for i in self.intents:
            lines.append(f"{i.letter}. {i.name} - {i.description}")
            lines.append(f'   example: "{i.positive}"')
            lines.append(
                f'   NOT this: "{i.near_miss}" (that is {i.near_miss_is}: {i.near_miss_why})'
            )
        return "\n".join(lines)


@lru_cache(maxsize=1)
def load_taxonomy(path: Path = TAXONOMY_PATH) -> Taxonomy:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return Taxonomy(
        version=raw["version"],
        brand=raw["brand"],
        tie_break_rule=raw["tie_break_rule"],
        other_rule=raw["other_rule"],
        intents=tuple(Intent(**i) for i in raw["intents"]),
    )


# The deployed intent head. C=4 and balanced class weights were chosen by bake-off
# (#24); nested CV shows that choosing them on the golden set did not inflate the
# headline (#43). Change the system here, and every script follows.
SYSTEM_C = 4.0


def system_head() -> LogisticRegression:
    return LogisticRegression(max_iter=3000, class_weight="balanced", C=SYSTEM_C)


def oof_predictions(
    X: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = GLOBAL_SEED
) -> tuple[list[str], list[dict[str, float]]]:
    """Out-of-fold predicted intents and their posteriors, from the same fold models.

    Every example is predicted by a model that never saw it. The prediction and the
    posterior that gates it in the router come from one fitted model, so they cannot
    drift apart (DECISIONS #40).
    """
    preds: list[str] = [""] * len(y)
    posteriors: list[dict[str, float]] = [{} for _ in y]
    for tr, te in StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(X, y):
        m = system_head().fit(X[tr], y[tr])
        for i, label, row in zip(te, m.predict(X[te]), m.predict_proba(X[te]), strict=True):
            preds[i] = str(label)
            posteriors[i] = {str(c): float(p) for c, p in zip(m.classes_, row, strict=True)}
    return preds, posteriors


@dataclass
class Prediction:
    intent: str
    confidence: float
    posterior: dict[str, float] = field(default_factory=dict)
    raw: str = ""
    # True when the model returned nothing usable and `intent` is a fallback
    # rather than a decision. Silently folding these into `other` once made a
    # broken annotator look like a merely disagreeing one, at 7.5% agreement,
    # and nothing in the numbers said so.
    parse_failed: bool = False


class Classifier(Protocol):
    name: str

    def fit(self, texts: list[str], labels: list[str]) -> Classifier: ...

    def predict(self, texts: list[str]) -> list[Prediction]: ...


class MajorityClassifier:
    """Trivial baseline. Exists to make every other number interpretable."""

    name = "majority"

    def __init__(self) -> None:
        self.label = "other"

    def fit(self, texts: list[str], labels: list[str]) -> MajorityClassifier:
        vals, counts = np.unique(labels, return_counts=True)
        self.label = str(vals[counts.argmax()])
        return self

    def predict(self, texts: list[str]) -> list[Prediction]:
        return [Prediction(self.label, 1.0, {self.label: 1.0}) for _ in texts]


class TfidfClassifier:
    """Simple baseline. TF-IDF into a linear model.

    Sublinear term frequency matters on tweets: a word repeated for emphasis
    ("delayed delayed delayed") should not dominate a 20-token document.
    """

    name = "tfidf_logreg"

    def __init__(self, seed: int = GLOBAL_SEED) -> None:
        self.pipe = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        sublinear_tf=True,
                        ngram_range=(1, 2),
                        min_df=1,
                        max_df=0.9,
                        strip_accents="unicode",
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000, class_weight="balanced", C=4.0, random_state=seed
                    ),
                ),
            ]
        )
        self.classes_: list[str] = []

    def fit(self, texts: list[str], labels: list[str]) -> TfidfClassifier:
        self.pipe.fit(texts, labels)
        self.classes_ = list(self.pipe.named_steps["clf"].classes_)
        return self

    def predict(self, texts: list[str]) -> list[Prediction]:
        probs = self.pipe.predict_proba(texts)
        out = []
        for row in probs:
            post = {c: float(p) for c, p in zip(self.classes_, row, strict=True)}
            best = max(post, key=post.__getitem__)
            out.append(Prediction(best, post[best], post))
        return out


CLASSIFY_PROMPT = """You label customer messages sent to {brand} on Twitter with exactly one intent.

{options}

Rules:
- {tie_break}
- {other_rule}

Reply with the single option letter and nothing else."""


class LLMClassifier:
    """Zero-shot LLM classifier: one letter out, a posterior over letters from token 0.

    Rejected as the deployed system (#21); used as an independent annotator.
    """

    name = "llm_letter"

    def __init__(
        self,
        llm: LLM | None = None,
        taxonomy: Taxonomy | None = None,
        use_logprobs: bool = True,
        extra_body: dict | None = None,
        min_letter_mass: float = 0.5,
    ) -> None:
        # Not every provider exposes logprobs (Mistral returns 400 outright), and
        # a model can still be a useful annotator without a posterior. Without
        # them there is no confidence signal, so such a model can annotate but
        # must never be the deployed classifier, whose router needs one.
        self.use_logprobs = use_logprobs
        self.llm = llm or LLM.for_role("classifier")
        self.tax = taxonomy or load_taxonomy()
        self.system = CLASSIFY_PROMPT.format(
            brand=self.tax.brand,
            options=self.tax.as_options(),
            tie_break=self.tax.tie_break_rule,
            other_rule=self.tax.other_rule,
        )
        self._letters = {i.letter for i in self.tax.intents}
        self._bare = re.compile(rf"^\W*([{''.join(sorted(self._letters))}])\W*$")
        # A reasoning model's token 0 is "We" at ~0.98; the option letters share the
        # remaining 1-2%. Renormalising that tail manufactured labels the model never
        # gave (DECISIONS #51), so a posterior needs real mass on the letters.
        self.min_letter_mass = min_letter_mass
        self.extra_body = extra_body
        self.parse_failures = 0

    def fit(self, texts: list[str], labels: list[str]) -> LLMClassifier:
        return self  # nothing to fit: the taxonomy is frozen and lives in the prompt

    def _posterior(self, resp) -> dict[str, float]:
        """Token 0's top-k renormalised over the option letters, or {} if the letters
        hold less than `min_letter_mass` of the probability (the model did not answer)."""
        if not resp.logprobs:
            return {}
        cand = {
            tok.strip(): lp
            for tok, lp in resp.logprobs[0]["top"].items()
            if tok.strip() in self._letters
        }
        if not cand:
            return {}
        z = sum(math.exp(v) for v in cand.values())
        if z < self.min_letter_mass:
            return {}
        out: dict[str, float] = {}
        for letter, lp in cand.items():
            intent = self.tax.by_letter(letter)
            if intent is not None:
                out[intent.id] = math.exp(lp) / z
        return out

    def _decide(self, r) -> Prediction:
        post = self._posterior(r) if self.use_logprobs else {}
        # Only a reply that is a bare option letter counts: "I think..." must not
        # become the intent lettered I.
        m = self._bare.match(r.text or "")
        named = self.tax.by_letter(m.group(1)) if m else None
        if post:
            intent = max(post, key=post.__getitem__)
            return Prediction(intent, post[intent], post, raw=r.text)
        if named is not None:
            return Prediction(named.id, 1.0, {}, raw=r.text)
        # Nothing usable came back: counted as a failure, never a confident `other`.
        self.parse_failures += 1
        return Prediction("other", 0.0, post, raw=r.text, parse_failed=True)

    def predict(self, texts: list[str]) -> list[Prediction]:
        return [
            self._decide(
                self.llm.complete(
                    self.system,
                    text,
                    logprobs=self.use_logprobs,
                    max_tokens=4,
                    top_logprobs=12,
                    extra_body=self.extra_body,
                )
            )
            for text in texts
        ]


def build(name: str, **kw) -> Classifier:
    return {
        "majority": MajorityClassifier,
        "tfidf_logreg": TfidfClassifier,
        "llm_letter": LLMClassifier,
    }[name](**kw)


FEWSHOT_PROMPT = """You label customer messages sent to {brand} on Twitter with exactly one intent.

{options}

Rules:
- {tie_break}
- {other_rule}

Here are labelled examples from this dataset, most similar first:

{examples}

Reply with the single option letter and nothing else."""


class FewShotLLMClassifier(LLMClassifier):
    """The fair comparison to a supervised model.

    Zero-shot prompting loses to a linear model fitted on a couple of hundred
    labels, but that is not a like-for-like test: the linear model has seen the
    labels and the prompt has not. This arm gives the model the same evidence,
    as the k nearest labelled examples by embedding similarity, so the remaining
    difference is about the method rather than the information.
    """

    name = "llm_fewshot"

    def __init__(self, k: int = 12, **kw) -> None:
        super().__init__(**kw)
        self.k = k
        self._embedder = None
        self._train_vecs = None
        self._train: list[tuple[str, str]] = []

    def fit(self, texts: list[str], labels: list[str]) -> FewShotLLMClassifier:
        from support_agent.embed import Embedder

        self._embedder = Embedder()
        self._train = list(zip(texts, labels, strict=True))
        self._train_vecs = self._embedder.encode(texts, input_type="passage")
        return self

    def _examples_for(self, vec) -> str:
        sims = self._train_vecs @ vec
        order = sims.argsort()[::-1][: self.k]
        lines = []
        for idx in reversed(order):  # most similar last, closest to the question
            text, label = self._train[idx]
            intent = self.tax.by_id(label)
            letter = intent.letter if intent else "?"
            lines.append(f'"{text}" -> {letter}')
        return "\n".join(lines)

    def predict(self, texts: list[str]) -> list[Prediction]:
        if self._train_vecs is None or self._embedder is None:
            return super().predict(texts)
        vecs = self._embedder.encode(texts, input_type="query")
        out = []
        for text, vec in zip(texts, vecs, strict=True):
            system = FEWSHOT_PROMPT.format(
                brand=self.tax.brand,
                options=self.tax.as_options(),
                tie_break=self.tax.tie_break_rule,
                other_rule=self.tax.other_rule,
                examples=self._examples_for(vec),
            )
            r = self.llm.complete(
                system,
                text,
                logprobs=True,
                max_tokens=4,
                top_logprobs=12,
                extra_body=self.extra_body,
            )
            out.append(self._decide(r))
        return out
