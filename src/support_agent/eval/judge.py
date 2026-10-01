"""LLM-as-judge for reply quality, with the biases designed against.

Three known judge failure modes and what is done about each:

* **Self-preference (10-25% in the literature).** The judge must come from a
  different training family than the drafter. `LLM.for_role` enforces it, and
  `SELF_PREFERENCE_PROBE` pins a same-family judge so the size of the effect can
  be *measured on this data* rather than cited from a paper.
* **Position bias (up to 75% in pairwise setups).** Avoided structurally: this
  is absolute scoring, one draft at a time, so there is no A/B order to bias.
  The evidence snippets are order-randomised from a fixed seed so a drafter
  cannot be rewarded for echoing whichever example happened to come first.
* **Verbosity bias.** Judges reward length. The rubric penalises it explicitly,
  and `length_chars` is recorded on every score so the correlation between
  length and score is checkable rather than assumed away.

A judge with no human-agreement number is decoration, so `agreement.py` scores a
blind human sample against it and reports Cohen's kappa.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

from support_agent.config import load_prompt
from support_agent.llm import LLM, parse_json
from support_agent.retrieve import Hit

PROMPT_VERSION = "judge_v1"
DIMENSIONS = ("grounded", "addresses", "tone", "safe")


def load_rubric(version: str = PROMPT_VERSION) -> str:
    """Prompts are versioned files, never inline f-strings: the report has to
    quote the exact rubric a number was produced under."""
    return load_prompt(version)


@dataclass
class Score:
    grounded: int = 0
    addresses: int = 0
    tone: int = 0
    safe: int = 0
    worst_problem: str = ""
    length_chars: int = 0
    judge_model: str = ""
    parse_failed: bool = False
    error: str = ""

    @property
    def total(self) -> int:
        return self.grounded + self.addresses + self.tone + self.safe

    @property
    def sendable(self) -> bool:
        """Operationally the only question that matters: could an agent send this
        without editing? Requires safety and grounding to be perfect, not just
        good on average - which is why the average is not the headline."""
        return self.safe == 2 and self.grounded == 2 and self.addresses >= 1

    def as_dict(self) -> dict:
        d = asdict(self)
        d["total"] = self.total
        d["sendable"] = self.sendable
        return d


USER = """Customer message:
{message}

Draft reply to audit:
{draft}

Replies this brand actually sent to similar messages:
{evidence}"""


@dataclass
class Judge:
    llm: LLM = field(default_factory=lambda: LLM.for_role("judge"))
    version: str = PROMPT_VERSION
    seed: int = 20250911
    # A reasoning model spends tokens before its JSON; the self-preference probe
    # completed only 56/77 calls at 1200 (DECISIONS #34), so it is given more room.
    max_tokens: int = 1200
    extra_body: dict | None = None

    def __post_init__(self) -> None:
        self.rubric = load_rubric(self.version)
        self.parse_failures = 0

    def evidence(self, draft: str, hits: list[Hit]) -> list[Hit]:
        """The evidence this judge shows for a draft, in the order it shows it.

        The human in the agreement study sees exactly this list (DECISIONS #51).
        """
        use = [h for h in hits if h.reply][:4]
        random.Random(self.seed + len(draft)).shuffle(use)
        return use

    def score(self, message: str, draft: str, hits: list[Hit]) -> Score:
        if not draft.strip():
            # An empty draft is a real result, worth zero on every dimension.
            # Skipping it would quietly raise the mean for whichever system
            # failed to produce output.
            return Score(judge_model=self.llm.model, worst_problem="empty draft", length_chars=0)

        use = self.evidence(draft, hits)
        evidence = (
            "\n\n".join(f'- customer: "{h.text}"\n  brand: "{h.reply}"' for h in use)
            or "(none retrieved)"
        )

        try:
            r = self.llm.complete(
                self.rubric,
                USER.format(message=message, draft=draft, evidence=evidence),
                json_mode=True,
                max_tokens=self.max_tokens,
                temperature=0.0,
                extra_body=self.extra_body,
            )
        except Exception as exc:
            self.parse_failures += 1
            return Score(
                judge_model=self.llm.model,
                parse_failed=True,
                error=f"{type(exc).__name__}: {exc}"[:200],
                length_chars=len(draft),
            )

        try:
            data = parse_json(r.text)
        except ValueError as exc:
            self.parse_failures += 1
            return Score(
                judge_model=self.llm.model,
                parse_failed=True,
                error=str(exc)[:200],
                length_chars=len(draft),
            )

        # A missing or out-of-scale dimension is a parse failure, not a zero: a
        # silent 0 would mark the draft unsendable for a reason the judge never gave.
        if not isinstance(data, dict) or any(data.get(d) not in (0, 1, 2) for d in DIMENSIONS):
            self.parse_failures += 1
            return Score(
                judge_model=self.llm.model,
                parse_failed=True,
                error=f"incomplete scores: {str(data)[:160]}",
                length_chars=len(draft),
            )

        def score_of(key: str) -> int:
            return int(data[key])  # already checked to be 0, 1 or 2

        return Score(
            grounded=score_of("grounded"),
            addresses=score_of("addresses"),
            tone=score_of("tone"),
            safe=score_of("safe"),
            worst_problem=str(data.get("worst_problem", ""))[:80],
            length_chars=len(draft),
            judge_model=self.llm.model,
        )
