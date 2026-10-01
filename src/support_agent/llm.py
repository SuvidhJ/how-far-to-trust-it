"""The only place in the project that talks to a language model.

Three properties the project depends on:

* **Reruns are free and identical.** Every response is cached on a hash of
  (provider, model, messages, decoding params) into a committed JSONL file, so
  `harness --offline` on a fresh clone replays recorded responses instead of
  re-billing and re-sampling. This is what makes the "<15 minutes" promise honest.
* **Roles are configuration, not code.** `classifier`, `drafter` and `judge`
  each name a provider and model. The judge must not be the drafter's model —
  self-preference bias is 10-25% in the literature — and `LLM.for_role`
  refuses a configuration that violates it.
* **There is a hard call ceiling.** A loop over a 2.8M-row dataset is the
  realistic way to burn a quota; the adapter raises rather than spend.

Providers are all reached through their OpenAI-compatible endpoint, so one code
path covers Gemini, NVIDIA NIM and Mistral.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from openai import OpenAI

from support_agent.config import ROOT, settings

CACHE_PATH = ROOT / "data" / "cache" / "llm_cache.jsonl"

Role = Literal["classifier", "drafter", "judge"]

PROVIDERS: dict[str, dict[str, str]] = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
    },
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "key_env": "NVIDIA_API_KEY",
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "key_env": "MISTRAL_API_KEY",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
    },
}

# Every one of these was chosen by probing all five free endpoints available to
# this project rather than by reputation (see DECISIONS #19 and #20).
ROLE_MODELS: dict[Role, tuple[str, str]] = {
    # The LLM classifier arm of the bake-offs, rejected as the deployed system (#21):
    # the router gates on the embedding model's posteriors, not on LLM logprobs.
    # Kept because it answers with a bare letter, so token 0 is a proper posterior
    # over the option set; reasoning models spend token 0 on a channel marker.
    "classifier": ("nvidia", "meta/llama-3.2-11b-vision-instruct"),
    # Was OpenRouter's 550B free model, which writes well but is throttled to
    # roughly one completion per 55 seconds - 90 minutes for a 100-draft
    # evaluation, which makes the whole eval loop unusable. Throughput is a
    # capability. See DECISIONS #31.
    "drafter": ("nvidia", "nvidia/nemotron-3-super-120b-a12b"),
    # Different *training family* from the drafter, which is what self-preference
    # bias is about — not merely a different endpoint. `for_role` enforces it.
    "judge": ("nvidia", "openai/gpt-oss-20b"),
}

# Request extensions per role. nemotron is a reasoning model and, with thinking on,
# wrote its deliberation into the reply: 9/218 grounded drafts began "We need
# to..." and 2 more were cut off at max_tokens. `enable_thinking=False` in the chat
# template took that to 0/217 on the same messages, while a "/no_think" prompt
# prefix did not work at all (DECISIONS #35). Measured, not assumed.
ROLE_EXTRA_BODY: dict[Role, dict[str, Any]] = {
    "drafter": {"chat_template_kwargs": {"enable_thinking": False}},
}

# The judge used only to *measure* self-preference bias. It is the drafter's own
# model, which is the strongest form of the test: whatever gap appears between
# this judge's scores and the real judge's is the full inflation a lazy
# same-model setup would have produced. Never the headline.
SELF_PREFERENCE_PROBE = ROLE_MODELS["drafter"]

# Which lineage a model comes from. Two models from the same lineage must never
# be drafter and judge for the same number.
FAMILY: dict[str, str] = {
    "meta/llama-3.2-11b-vision-instruct": "llama",
    "nvidia/nemotron-3-super-120b-a12b": "nemotron",
    "nvidia/nemotron-3-ultra-550b-a55b:free": "nemotron",
    "openai/gpt-oss-20b": "gpt-oss",
    "google/gemma-4-31b-it": "gemma",
    "gemini-3.6-flash": "gemini",
    "ministral-3b-latest": "mistral",
}


class CallBudgetExceeded(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    cached: bool
    latency_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # [{token, logprob, top: {tok: logprob}}] for the first N generated tokens.
    logprobs: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = ""


class _Cache:
    """Append-only JSONL keyed by request hash, loaded once per process."""

    def __init__(self, path: Path = CACHE_PATH) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._mem: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # a torn final line must not break a fresh clone
                    self._mem[rec["key"]] = rec["response"]

    def get(self, key: str) -> dict[str, Any] | None:
        return self._mem.get(key)

    def put(self, key: str, response: dict[str, Any]) -> None:
        with self._lock:
            if key in self._mem:
                return
            self._mem[key] = response
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "response": response}, sort_keys=True) + "\n")

    def __len__(self) -> int:
        return len(self._mem)


_CACHE: _Cache | None = None


def cache() -> _Cache:
    global _CACHE
    if _CACHE is None:
        _CACHE = _Cache()
    return _CACHE


def _request_key(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:32]


class LLM:
    """One role's model, with cache, budget and token accounting."""

    def __init__(
        self,
        provider: str,
        model: str,
        *,
        role: str = "adhoc",
        max_calls: int | None = None,
        offline: bool = False,
    ) -> None:
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider!r}; have {sorted(PROVIDERS)}")
        self.provider, self.model, self.role = provider, model, role
        self.max_calls = max_calls if max_calls is not None else settings.llm_max_calls_per_run
        self.offline = offline
        self.calls = 0
        self.cache_hits = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._client: OpenAI | None = None

    @classmethod
    def for_role(cls, role: Role, **kw: Any) -> LLM:
        provider, model = ROLE_MODELS[role]
        if role == "judge":
            _, d_model = ROLE_MODELS["drafter"]
            # Fail closed: a model missing from FAMILY cannot be shown to differ.
            unknown = [m for m in (model, d_model) if m not in FAMILY]
            if unknown:
                raise ValueError(f"no training family recorded for {unknown}; add it to FAMILY")
            if FAMILY[model] == FAMILY[d_model]:
                raise ValueError(
                    f"judge ({model}) and drafter ({d_model}) share a training family; "
                    "self-preference bias would inflate every reply-quality number."
                )
        return cls(provider, model, role=role, **kw)

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            spec = PROVIDERS[self.provider]
            key = getattr(settings, spec["key_env"].lower(), "") or os.environ.get(
                spec["key_env"], ""
            )
            if not key:
                raise RuntimeError(f"{spec['key_env']} is not set; see .env.example")
            headers = (
                {
                    "HTTP-Referer": "https://github.com/SuvidhJ/how-far-to-trust-it",
                    "X-Title": "support-agent",
                }
                if self.provider == "openrouter"
                else None
            )
            self._client = OpenAI(
                base_url=spec["base_url"],
                api_key=key,
                timeout=120.0,
                max_retries=0,
                default_headers=headers,
            )
        return self._client

    def complete(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
        logprobs: bool = False,
        max_tokens: int = 600,
        temperature: float = 0.0,
        top_logprobs: int = 5,
        extra_body: dict[str, Any] | None = None,
    ) -> LLMResponse:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        payload: dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "json_mode": json_mode,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs if logprobs else None,
        }
        # Only keyed when set, so every response cached before this parameter
        # existed still replays.
        if extra_body:
            payload["extra_body"] = extra_body
        key = _request_key(payload)

        hit = cache().get(key)
        if hit is not None:
            self.cache_hits += 1
            return LLMResponse(**{**hit, "cached": True, "latency_s": 0.0})

        if self.offline:
            raise RuntimeError(
                f"offline=True and no cached response for {self.role}/{self.model}. "
                "Run with network access once to populate data/cache/llm_cache.jsonl."
            )
        if self.calls >= self.max_calls:
            raise CallBudgetExceeded(
                f"{self.role}: {self.calls} calls reached the ceiling of {self.max_calls}"
            )

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = top_logprobs
        if extra_body:
            kwargs["extra_body"] = extra_body

        t0 = time.time()
        r = self._call_with_retries(kwargs)
        latency = time.time() - t0
        self.calls += 1

        # OpenRouter can answer HTTP 200 with `choices: null` when the upstream
        # provider errored, so the SDK raises nothing and `r.choices[0]` throws a
        # TypeError several frames away from the cause. Surface it here instead.
        if not getattr(r, "choices", None):
            err = getattr(r, "error", None) or getattr(r, "model_extra", {}).get("error")
            raise RuntimeError(
                f"{self.provider}/{self.model} returned no choices"
                + (f": {str(err)[:200]}" if err else " and no error detail")
            )

        choice = r.choices[0]
        lp: list[dict[str, Any]] = []
        if logprobs and choice.logprobs and getattr(choice.logprobs, "content", None):
            for tok in choice.logprobs.content[:24]:
                lp.append(
                    {
                        "token": tok.token,
                        "logprob": tok.logprob,
                        "top": {t.token: t.logprob for t in (tok.top_logprobs or [])},
                    }
                )
        usage = getattr(r, "usage", None)
        resp = LLMResponse(
            text=choice.message.content or "",
            provider=self.provider,
            model=self.model,
            cached=False,
            latency_s=round(latency, 3),
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            logprobs=lp,
            finish_reason=choice.finish_reason or "",
        )
        self.prompt_tokens += resp.prompt_tokens
        self.completion_tokens += resp.completion_tokens

        record = asdict(resp)
        record["latency_s"] = 0.0  # replayed responses have no network cost
        cache().put(key, record)
        return resp

    def _call_with_retries(self, kwargs: dict[str, Any], attempts: int = 6):
        """Retry rate limits, transient 5xx, timeouts and dropped connections.

        Backoff runs 2, 4, 8, 16, 32s. The first drafting bake-off lost 13/50 drafts
        of one arm to uncached exceptions under a shorter schedule (DECISIONS #34).
        """
        delay = 2.0
        last: Exception | None = None
        for i in range(attempts):
            try:
                return self.client.chat.completions.create(**kwargs)
            except Exception as exc:  # provider SDKs raise distinct types
                last = exc
                if not is_transient(exc) or i == attempts - 1:
                    raise
                time.sleep(delay)
                delay *= 2
        raise last  # unreachable, keeps type checkers happy

    def stats(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "provider": self.provider,
            "model": self.model,
            "api_calls": self.calls,
            "cache_hits": self.cache_hits,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


def is_transient(exc: Exception) -> bool:
    """Worth retrying: rate limits, 5xx, timeouts, connection drops."""
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__
    return status in (408, 409, 429, 500, 502, 503, 504) or "imeout" in name or "Connection" in name


_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def parse_json(text: str) -> dict[str, Any]:
    """Parse a model's JSON reply, tolerating a markdown fence and trailing prose.

    A reply cut off by `max_tokens` is *not* recoverable and raises, like any other
    malformed reply, because silently returning `{}` would turn a broken prompt
    into a quietly wrong metric. Callers count these as parse failures.
    """
    cleaned = _FENCE.sub("", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    if start >= 0:
        depth, in_str, esc = 0, False, False
        for i, ch in enumerate(cleaned[start:], start):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(cleaned[start : i + 1])
    raise ValueError(f"not JSON: {text[:200]!r}")
