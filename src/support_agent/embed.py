"""Embeddings, cached to disk so a rerun costs nothing.

Embeddings turned out to be the backbone of the classifier rather than an
ablation (DECISIONS #21), so which model is used is a real decision and is made
by scoring each one on the actual task, not on a toy similarity probe.

Two implementation details that were bugs before they were features:

* **One archive per model.** Vectors from different models have different
  dimensions; a single shared archive fails inside numpy with an error that
  mentions concatenation axes and not embeddings.
* **`input_type` is NVIDIA-only.** Sending it to Mistral returns 422 and to
  Gemini 400. An earlier comparison silently "showed" that both models were
  unusable, when in fact the request was malformed.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

from support_agent.config import ROOT, settings
from support_agent.llm import PROVIDERS

# One compressed archive, not one file per text. 438 vectors as separate JSON
# files came to 11 MB and could not sensibly be committed; the same vectors as
# float16 in a single npz are under 2 MB, which ships. Half precision is safe
# here because the vectors are L2-normalised before use and cosine similarity is
# insensitive to that rounding at four decimal places.
CACHE_DIR = ROOT / "data" / "cache"

# Chosen by repeated cross-validation on the golden set, not by a similarity probe.
# mistral-embed is never worse than nvidia/nemotron-3-embed-1b on any head tried, but
# with a test that respects the resampling the gap is significant only for the
# centroid head (logreg p=0.28). The pooled-McNemar p-values first quoted here were
# invalid (DECISIONS #23, #42).
PROVIDER = "mistral"
MODEL = "mistral-embed"
DIM = 1024


def cache_path(model: str) -> Path:
    """One archive per model. A single shared archive cannot work: vectors from
    different models have different dimensions, and stacking them raises deep
    inside numpy with an error that says nothing about embeddings."""
    return CACHE_DIR / f"embeddings.{model.split('/')[-1]}.npz"


def _key(text: str, model: str, input_type: str, asymmetric: bool) -> str:
    """Cache key. `input_type` only belongs in it for models that actually use
    it: mistral-embed is symmetric, so keying on it would send the identical
    request twice and double the calls against a rate-limited free tier."""
    tag = input_type if asymmetric else "-"
    return hashlib.sha256(f"{model}|{tag}|{text}".encode()).hexdigest()[:32]


class Embedder:
    """Batched, disk-cached embeddings.

    `input_type` matters for asymmetric retrieval models: the same sentence gets
    a different vector as a query than as a stored passage. Getting this wrong
    silently degrades retrieval, and it is the sort of thing that never shows up
    as an error.
    """

    def __init__(
        self,
        model: str = MODEL,
        provider: str = PROVIDER,
        batch_size: int = 32,
        offline: bool = False,
    ) -> None:
        self.model = model
        self.provider = provider
        self.batch_size = batch_size
        self.offline = offline
        self.cache_path = cache_path(model)
        # Only NVIDIA's endpoint distinguishes query from passage encodings.
        self.asymmetric = provider == "nvidia"
        self.api_calls = 0
        self.cache_hits = 0
        self._client = None
        self._store: dict[str, np.ndarray] = {}
        self._dirty = False
        if self.cache_path.exists():
            with np.load(self.cache_path, allow_pickle=False) as z:
                keys, vecs = z["keys"], z["vectors"]
            self._store = {str(k): v for k, v in zip(keys, vecs, strict=True)}

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            spec = PROVIDERS[self.provider]
            key = getattr(settings, spec["key_env"].lower(), "")
            if not key:
                raise RuntimeError(f"{spec['key_env']} is not set; see .env.example")
            self._client = OpenAI(
                base_url=spec["base_url"], api_key=key, timeout=120.0, max_retries=2
            )
        return self._client

    def _create_with_backoff(self, batch: list[str], extra: dict | None, attempts: int = 6):
        """Mistral's free tier rate-limits hard and its 429 carries no
        Retry-After, so back off exponentially rather than give up. Embedding a
        corpus is a long single-threaded loop; one unhandled 429 wastes the
        whole run."""
        delay = 2.0
        for i in range(attempts):
            try:
                return self.client.embeddings.create(
                    model=self.model, input=batch, extra_body=extra
                )
            except Exception as exc:
                if getattr(exc, "status_code", None) != 429 or i == attempts - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
        raise RuntimeError("unreachable")

    def flush(self) -> None:
        """Write the archive back. Cheap enough to call after every encode."""
        if not self._dirty:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(self._store)
        np.savez_compressed(
            self.cache_path,
            keys=np.array(keys),
            vectors=np.vstack([self._store[k] for k in keys]).astype(np.float16),
        )
        self._dirty = False

    def encode(self, texts: list[str], input_type: str = "passage") -> np.ndarray:
        keys = [_key(t, self.model, input_type, self.asymmetric) for t in texts]
        out: list[np.ndarray | None] = [self._store.get(k) for k in keys]
        self.cache_hits += sum(v is not None for v in out)

        todo = [i for i, v in enumerate(out) if v is None]
        if todo and self.offline:
            raise RuntimeError(
                f"offline=True and {len(todo)} texts are not in {self.cache_path.name}. "
                "Run once with NVIDIA_API_KEY set to populate it."
            )
        for start in range(0, len(todo), self.batch_size):
            chunk = todo[start : start + self.batch_size]
            # `input_type` is an NVIDIA extension. Sending it to Mistral gives a
            # 422 and to Gemini a 400, which silently disqualified both models
            # from an earlier comparison until this was fixed.
            extra = (
                {"input_type": input_type, "truncate": "END"} if self.provider == "nvidia" else None
            )
            resp = self._create_with_backoff([texts[i] for i in chunk], extra)
            self.api_calls += 1
            for i, item in zip(chunk, resp.data, strict=True):
                vec = np.asarray(item.embedding, dtype=np.float16)
                out[i] = vec
                self._store[keys[i]] = vec
                self._dirty = True
        self.flush()

        matrix = np.vstack([v for v in out if v is not None]).astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.clip(norms, 1e-9, None)
