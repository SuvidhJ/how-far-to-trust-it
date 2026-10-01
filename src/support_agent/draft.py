"""Draft a reply, grounded in what this brand actually said before.

Four drafters, so "grounding helps" is a measured claim rather than an
assumption:

* `CannedDrafter`    - one fixed reply for everything. The trivial baseline.
* `NearestDrafter`   - the brand's historical reply to the most similar past
                       message, copied verbatim. No LLM at all. This is a
                       surprisingly strong baseline for a channel where most
                       replies are near-templates, and if the LLM cannot beat it
                       the LLM is not earning its cost.
* `UngroundedDrafter`- LLM with the intent but no retrieved evidence. Isolates
                       how much the retrieval actually contributes.
* `GroundedDrafter`  - LLM conditioned on k retrieved exchanges. The system.

Every draft carries the ids of the exchanges it was grounded in, so groundedness
can be checked mechanically instead of asserted.

The framing decision that shapes the prompt: the dataset contains *responses*,
not verified *resolutions* - only 5-9% of brand replies get any acknowledgement
from the customer. So the drafter is told to imitate what this brand would say,
never to assert that something is resolved, and never to state a policy that is
not present in the retrieved evidence. That last rule is the Moffatt v. Air
Canada rule: an invented policy is binding on the brand.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from support_agent.config import load_prompt
from support_agent.llm import LLM, ROLE_EXTRA_BODY
from support_agent.retrieve import Hit

TWEET_LIMIT = 280

# How chain-of-thought opens when a reasoning model writes it into the message
# content instead of a separate channel. Every pattern here was seen in real
# nemotron drafts or is the documented think-tag. A reply that starts like this
# is internal deliberation, never a message to a customer.
_REASONING_OPENERS = re.compile(
    r"^\s*(<think>|we need to\b|we have to\b|we should\b|we must\b|let me\b|let's\b|"
    r"i need to\b|i should\b|i will\b|okay[,.]|ok[,.]|alright[,.]|first,|"
    r"the user\b|the customer (is|wants|asks|says|has|mentions)\b)",
    re.IGNORECASE,
)


def looks_like_reasoning(text: str) -> bool:
    return bool(_REASONING_OPENERS.match(text)) or "</think>" in text


# A real AmericanAir reply, used verbatim as the trivial baseline so the floor is
# a plausible human message rather than a strawman.
CANNED = "We're sorry for the trouble. Please DM us your confirmation number and we'll take a look."

SYSTEM = load_prompt("draft_system_v1")

GROUNDED_USER = load_prompt("draft_grounded_user_v1")

UNGROUNDED_USER = load_prompt("draft_ungrounded_user_v1")


@dataclass
class Draft:
    text: str
    drafter: str
    grounded_in: list[int] = field(default_factory=list)
    intent: str = ""
    cached: bool = True
    error: str = ""


class Drafter(Protocol):
    name: str

    def draft(self, message: str, intent: str, hits: list[Hit]) -> Draft: ...


class CannedDrafter:
    name = "canned"

    def draft(self, message: str, intent: str, hits: list[Hit]) -> Draft:
        return Draft(CANNED, self.name, intent=intent)


class NearestDrafter:
    """Copy the brand's historical reply to the nearest past message.

    No generation, no API, no hallucination risk by construction - the reply is
    something the brand demonstrably sent. Its failure mode is the opposite one:
    perfectly fluent, occasionally about the wrong problem.
    """

    name = "nearest"

    def draft(self, message: str, intent: str, hits: list[Hit]) -> Draft:
        if not hits or not hits[0].reply:
            return Draft(CANNED, self.name, intent=intent, error="no retrieval hit")
        top = hits[0]
        return Draft(top.reply, self.name, [top.id] if top.id else [], intent)


class _LLMDrafter:
    """Shared completion path.

    `system_prefix` and `extra_body` exist to switch a reasoning model's thinking
    off - "/no_think" in the prompt, or `chat_template_kwargs` in the request.
    Whether either actually works is measured by `bakeoff_draft.py --leak-test`,
    not assumed.
    """

    def __init__(
        self,
        llm: LLM | None = None,
        brand: str = "American Airlines",
        system_prefix: str = "",
        extra_body: dict[str, Any] | None = ROLE_EXTRA_BODY["drafter"],
    ) -> None:
        self.llm = llm or LLM.for_role("drafter")
        self.system = (system_prefix + SYSTEM).format(brand=brand)
        self.extra_body = extra_body

    def _complete(self, user: str, intent: str, ids: list[int]) -> Draft:
        try:
            r = self.llm.complete(
                self.system, user, max_tokens=1400, temperature=0.0, extra_body=self.extra_body
            )
        except Exception as exc:
            return Draft(
                "", self.name, ids, intent, error=f"api: {type(exc).__name__}: {exc}"[:200]
            )
        text = (r.text or "").strip().strip('"')
        # Invalid outputs keep their text for inspection, but carry an error so
        # no caller can score or send them as if they were replies.
        if not text:
            return Draft("", self.name, ids, intent, r.cached, error="empty completion")
        if looks_like_reasoning(text):
            return Draft(text, self.name, ids, intent, r.cached, error="reasoning leak")
        if r.finish_reason == "length":
            return Draft(text, self.name, ids, intent, r.cached, error="truncated at max_tokens")
        return Draft(text, self.name, ids, intent, r.cached)


class UngroundedDrafter(_LLMDrafter):
    name = "ungrounded_llm"

    def draft(self, message: str, intent: str, hits: list[Hit]) -> Draft:
        return self._complete(UNGROUNDED_USER.format(message=message, intent=intent), intent, [])


class GroundedDrafter(_LLMDrafter):
    name = "grounded_llm"

    def __init__(self, llm: LLM | None = None, brand: str = "American Airlines", k: int = 4, **kw):
        super().__init__(llm, brand, **kw)
        self.k = k

    def draft(self, message: str, intent: str, hits: list[Hit]) -> Draft:
        use = [h for h in hits if h.reply][: self.k]
        if not use:
            # No evidence means no grounding. Falling back to an ungrounded
            # generation here would quietly break the contract that every draft
            # cites what it was based on, so this is recorded as an error and
            # the router escalates on it.
            return Draft("", self.name, [], intent, error="no grounding evidence")
        evidence = "\n\n".join(
            f'{i + 1}. customer: "{h.text}"\n   brand: "{h.reply}"' for i, h in enumerate(use)
        )
        user = GROUNDED_USER.format(message=message, intent=intent, evidence=evidence)
        return self._complete(user, intent, [h.id for h in use if h.id])


def build(name: str, **kw) -> Drafter:
    return {
        "canned": CannedDrafter,
        "nearest": NearestDrafter,
        "ungrounded_llm": UngroundedDrafter,
        "grounded_llm": GroundedDrafter,
    }[name](**kw)
