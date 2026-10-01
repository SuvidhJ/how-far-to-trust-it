"""The agent end to end: one customer message in, intent + draft + decision out.

    uv run python -m support_agent.agent "my bag never showed up at DFW"
    uv run python -m support_agent.agent --golden 3 --offline     # replay a cached example

The evaluation harness scores each stage separately, under cross-validation. This
module strings the same stages together the way they would run in production:

0. normalise: the same `text.clean` the golden set and corpus went through
   (handles, flight numbers, URLs, phone numbers masked). Skipping it would score
   production text the model never saw in evaluation. Risk rules still read the
   raw message, because masking removes the amounts that make a message risky.
1. classify: logistic regression on `mistral-embed` vectors, fitted on every
   usable golden row (the harness never scores this in-sample model)
2. retrieve: the k nearest past exchanges from the 3,000-row train-split corpus
3. draft: the grounded LLM reply, citing the ids of the exchanges it used
4. route: hard rules, then the top-2 margin at the 0.10 operating point (#33)

A draft that failed (API error, leaked reasoning, truncation) is never returned as
a reply. It escalates with that failure as the reason.

A new message needs `MISTRAL_API_KEY` for its embedding and an NVIDIA key for the
draft. `--offline` makes no network call, so it only works on messages already in
the committed caches, such as golden rows.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from support_agent.config import Paths, settings
from support_agent.draft import Draft
from support_agent.draft import build as build_drafter
from support_agent.embed import Embedder
from support_agent.intents import system_head
from support_agent.llm import LLM
from support_agent.retrieve import EmbedRetriever, Hit
from support_agent.route import OPERATING_THRESHOLD, Decision, Router
from support_agent.text import clean, has_placeholder

# The drafting prompt tells the model to promise a colleague's follow-up rather than
# invent a policy. Nothing in this pipeline keeps that promise, so a draft making it
# must never be auto-sent: one of 50 judged grounded drafts did exactly that on an
# auto-handled message, and the judge still scored it safe (REPORT §3.5, DECISIONS #50).
UNKEPT_PROMISE = re.compile(
    r"\bcolleague\b|\bfollow[- ]?up\b|\breach out to you\b|\bwill (contact|be in touch)\b", re.I
)


@dataclass
class Result:
    message: str
    clean_text: str
    intent: str
    posterior: dict[str, float]
    evidence: list[Hit]
    draft: Draft
    decision: Decision
    timings_ms: dict[str, float] = field(default_factory=dict)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class Agent:
    def __init__(self, drafter: str = "grounded_llm", k: int = 4, offline: bool = False) -> None:
        brand = settings.brand.lower()
        gold = [
            r for r in _read_jsonl(Paths.golden / f"{brand}_golden.jsonl") if not r.get("excluded")
        ]
        self.golden_texts = [r["text"] for r in gold]
        self.embedder = Embedder(offline=offline)
        X = self.embedder.encode(self.golden_texts)
        y = np.array([r["intent"] for r in gold])
        self.classifier = system_head().fit(X, y)

        corpus = _read_jsonl(Paths.processed / f"{brand}_index.jsonl")
        self.retriever = EmbedRetriever(self.embedder).index(
            [c["text"] for c in corpus], [{"id": c["id"], "reply": c["reply"]} for c in corpus]
        )
        kw = {"llm": LLM.for_role("drafter", offline=offline)} if drafter.endswith("_llm") else {}
        self.drafter = build_drafter(drafter, **kw)
        self.k = k
        self.router = Router(threshold=OPERATING_THRESHOLD, signal="margin")

    def handle(self, message: str) -> Result:
        ms: dict[str, float] = {}
        tick = time.perf_counter()

        def lap(stage: str) -> None:
            nonlocal tick
            now = time.perf_counter()
            ms[stage] = round((now - tick) * 1000, 1)
            tick = now

        text = clean(message)
        vec = self.embedder.encode([text])
        lap("embed")
        probs = self.classifier.predict_proba(vec)[0]
        posterior = {str(c): float(p) for c, p in zip(self.classifier.classes_, probs, strict=True)}
        intent = max(posterior, key=posterior.get)
        lap("classify")
        hits = self.retriever.search(text, k=self.k)
        lap("retrieve")
        draft = self.drafter.draft(text, intent, hits)
        lap("draft")
        decision = check_draft(self.router.decide(message, intent, posterior), draft)
        lap("route")
        return Result(message, text, intent, posterior, hits, draft, decision, ms)


def check_draft(decision: Decision, draft: Draft) -> Decision:
    """Escalate a routed message whose draft cannot be sent as written.

    The router judges the message; this judges the reply. Shared with the demo's
    offline replay, so both apply the same rules.
    """
    if draft.error:
        return Decision(
            "escalate",
            f"no usable draft ({draft.error}), so a human must write the reply",
            "draft_failed",
            decision.confidence,
            decision.signals,
        )
    if not decision.escalated and has_placeholder(draft.text):
        return Decision(
            "escalate",
            "the draft still contains a placeholder (phone number, link or reference) "
            "that a human must fill in before it can be sent",
            "draft_placeholder",
            decision.confidence,
            decision.signals,
        )
    if not decision.escalated and (promise := UNKEPT_PROMISE.search(draft.text)):
        return Decision(
            "escalate",
            f"the draft promises a follow-up ('{promise.group(0)}') that nothing in "
            "this pipeline will keep, so a human must send or rewrite it",
            "draft_unkept_promise",
            decision.confidence,
            decision.signals,
        )
    return decision


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("message", nargs="?", help="a customer tweet")
    src.add_argument("--golden", type=int, help="use the i-th usable golden-set message")
    ap.add_argument(
        "--drafter", default="grounded_llm", choices=["grounded_llm", "nearest", "canned"]
    )
    ap.add_argument("--offline", action="store_true", help="replay caches, make no network call")
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = ap.parse_args()

    agent = Agent(drafter=args.drafter, offline=args.offline)
    message = args.message if args.golden is None else agent.golden_texts[args.golden]
    r = agent.handle(message)
    if args.json:
        print(json.dumps(asdict(r), indent=2, ensure_ascii=False, default=str))
        return 0

    top = sorted(r.posterior.items(), key=lambda kv: -kv[1])[:3]
    print(f"message:  {r.message}")
    if message in agent.golden_texts:
        print(
            "          (a golden-set row: the classifier was fitted on it, so its intent is in-sample)"
        )
    print(f"intent:   {r.intent}   top-3 " + ", ".join(f"{c} {p:.2f}" for c, p in top))
    print(f"decision: {r.decision.action.upper()} [{r.decision.rule}] {r.decision.reason}")
    print(f"draft:    {r.draft.text or '-'}")
    print(f"          by {r.draft.drafter}, grounded in exchange ids {r.draft.grounded_in or '-'}")
    for h in r.evidence:
        print(f"evidence: [{h.id} sim {h.score:.2f}] {h.reply[:110]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
