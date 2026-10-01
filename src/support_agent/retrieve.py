"""Find historical exchanges to ground a draft in.

Four retrievers behind one interface so the harness can swap them with a flag:

* `TfidfRetriever`  - cosine over word TF-IDF. No API, rebuilt in seconds.
* `Bm25Retriever`   - the standard lexical baseline. Implemented here in ~25
                      lines rather than pulled in as a dependency: BM25 is short
                      enough to own, and owning it means it can be explained
                      and modified without reading someone else's library.
* `EmbedRetriever`  - cosine over `mistral-embed` vectors.
* `HybridRetriever` - reciprocal rank fusion of BM25 and embeddings.

RRF is used rather than a weighted score blend because BM25 scores and cosine
similarities are on incomparable scales; fusing *ranks* needs no tuning constant,
which matters when there is no held-out set to tune one on.

`intent_filter` restricts candidates to a predicted intent. That is the retrieval
half of the grounding contract: a draft about a lost bag should not be grounded
in how the brand answered a loyalty question.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class Hit:
    """One retrieved exchange. `id` travels with every draft so grounding can be
    checked mechanically rather than asserted."""

    index: int
    score: float
    id: int | None = None
    text: str = ""
    reply: str = ""
    intent: str | None = None


class Retriever(Protocol):
    name: str

    def index(self, texts: list[str], meta: list[dict] | None = None) -> Retriever: ...

    def search(self, query: str, k: int = 5, exclude: int | None = None) -> list[Hit]: ...


class _Base:
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.meta: list[dict] = []

    def _hits(self, scores: np.ndarray, k: int, exclude: int | None) -> list[Hit]:
        if exclude is not None and 0 <= exclude < len(scores):
            scores = scores.copy()
            scores[exclude] = -np.inf
        order = np.argsort(scores)[::-1][:k]
        out = []
        for i in order:
            if not np.isfinite(scores[i]):
                continue
            m = self.meta[i] if i < len(self.meta) else {}
            out.append(
                Hit(
                    index=int(i),
                    score=float(scores[i]),
                    id=m.get("id"),
                    text=self.texts[i],
                    reply=m.get("reply", ""),
                    intent=m.get("intent"),
                )
            )
        return out


class TfidfRetriever(_Base):
    name = "tfidf"

    def __init__(self, **kw) -> None:
        super().__init__()
        self.vec = TfidfVectorizer(
            sublinear_tf=True, ngram_range=(1, 2), min_df=1, strip_accents="unicode", **kw
        )
        self.matrix = None

    def index(self, texts: list[str], meta: list[dict] | None = None) -> TfidfRetriever:
        self.texts, self.meta = texts, meta or []
        self.matrix = self.vec.fit_transform(texts)
        return self

    def search(self, query: str, k: int = 5, exclude: int | None = None) -> list[Hit]:
        q = self.vec.transform([query])
        return self._hits(np.asarray((self.matrix @ q.T).todense()).ravel(), k, exclude)


class Bm25Retriever(_Base):
    """Okapi BM25.

    `k1` controls how fast term frequency saturates and `b` how strongly to
    normalise by document length. The defaults are the standard ones; on tweets
    length normalisation matters more than usual, because a 20-token complaint
    and a 3-token one are both whole documents.
    """

    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        super().__init__()
        self.k1, self.b = k1, b

    @staticmethod
    def _tok(text: str) -> list[str]:
        return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]

    def index(self, texts: list[str], meta: list[dict] | None = None) -> Bm25Retriever:
        self.texts, self.meta = texts, meta or []
        docs = [self._tok(t) for t in texts]
        self.vocab = {w: i for i, w in enumerate({w for d in docs for w in d})}
        self.lens = np.array([len(d) for d in docs], dtype=np.float32)
        self.avg_len = float(self.lens.mean()) if len(self.lens) else 1.0

        tf = np.zeros((len(docs), len(self.vocab)), dtype=np.float32)
        for i, d in enumerate(docs):
            for w in d:
                tf[i, self.vocab[w]] += 1
        df = (tf > 0).sum(axis=0)
        n = len(docs)
        # Robertson-Sparck-Jones idf with the +0.5 smoothing that keeps it
        # positive for terms appearing in more than half the corpus.
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        denom = tf + self.k1 * (1 - self.b + self.b * self.lens[:, None] / self.avg_len)
        self.weights = (tf * (self.k1 + 1) / denom) * self.idf
        return self

    def search(self, query: str, k: int = 5, exclude: int | None = None) -> list[Hit]:
        cols = [self.vocab[w] for w in self._tok(query) if w in self.vocab]
        if not cols:
            return []
        return self._hits(self.weights[:, cols].sum(axis=1), k, exclude)


class EmbedRetriever(_Base):
    name = "embed"

    def __init__(self, embedder=None) -> None:
        super().__init__()
        if embedder is None:
            from support_agent.embed import Embedder

            embedder = Embedder()
        self.embedder = embedder
        self.matrix = None
        # Populated by `search_batch` so repeated queries do not each cost a
        # round trip. Empty in normal single-query use.
        self._query_cache: dict[str, np.ndarray] = {}

    def index(self, texts: list[str], meta: list[dict] | None = None) -> EmbedRetriever:
        self.texts, self.meta = texts, meta or []
        self.matrix = self.embedder.encode(texts, input_type="passage")
        return self

    def search(self, query: str, k: int = 5, exclude: int | None = None) -> list[Hit]:
        q = self._query_cache.get(query)
        if q is None:
            q = self.embedder.encode([query], input_type="query")[0]
        return self._hits(self.matrix @ q, k, exclude)


class HybridRetriever(_Base):
    """Reciprocal rank fusion of BM25 and embeddings.

    score(d) = sum over retrievers of 1 / (rrf_k + rank(d)). `rrf_k=60` is the
    value from the original RRF paper and is left untuned deliberately: there is
    no held-out set here big enough to tune it on without overfitting.
    """

    name = "hybrid"

    def __init__(self, rrf_k: int = 60, embedder=None) -> None:
        super().__init__()
        self.rrf_k = rrf_k
        self.lex = Bm25Retriever()
        self.sem = EmbedRetriever(embedder)

    def index(self, texts: list[str], meta: list[dict] | None = None) -> HybridRetriever:
        self.texts, self.meta = texts, meta or []
        self.lex.index(texts, meta)
        self.sem.index(texts, meta)
        return self

    def search(self, query: str, k: int = 5, exclude: int | None = None) -> list[Hit]:
        pool = min(len(self.texts), max(k * 10, 50))
        fused: dict[int, float] = {}
        for sub in (self.lex, self.sem):
            for rank, hit in enumerate(sub.search(query, pool, exclude)):
                fused[hit.index] = fused.get(hit.index, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        scores = np.full(len(self.texts), -np.inf, dtype=np.float32)
        for i, s in fused.items():
            scores[i] = s
        return self._hits(scores, k, exclude)


def build(name: str, **kw) -> Retriever:
    return {
        "tfidf": TfidfRetriever,
        "bm25": Bm25Retriever,
        "embed": EmbedRetriever,
        "hybrid": HybridRetriever,
    }[name](**kw)


def search_batch(
    retriever: Retriever, queries: list[str], k: int = 5, exclude_self: bool = False
) -> list[list[Hit]]:
    """Run many queries, embedding them in one batch where that is possible.

    `EmbedRetriever.search` embeds one query per call. Over a few hundred
    queries that is a few hundred API round trips, which the free tiers
    rate-limit. Encoding all queries up front turns it into a handful.
    """
    inner = getattr(retriever, "sem", retriever)
    if isinstance(inner, EmbedRetriever):
        vectors = inner.embedder.encode(queries, input_type="query")
        inner._query_cache = dict(zip(queries, vectors, strict=True))
    try:
        return [
            retriever.search(q, k=k, exclude=i if exclude_self else None)
            for i, q in enumerate(queries)
        ]
    finally:
        if isinstance(inner, EmbedRetriever):
            inner._query_cache = {}
