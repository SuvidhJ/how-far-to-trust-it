"""Auto-handle or escalate, with a reason.

A routing decision is only useful with a stated reason, so the router is a small
ordered rule set over signals, not a bare threshold. Every escalation returns
the name of the rule that fired, which is what makes it explainable and what
makes the failure analysis possible.

Two design points worth defending.

**The costs are asymmetric.** An unnecessary escalation costs an agent a few
seconds. A missed one puts a wrong promise in public, under the brand's name. In
*Moffatt v. Air Canada* (2024) a tribunal held an airline to a refund policy its
chatbot invented. So the operating point is chosen on recall of the
should-escalate class, not on accuracy, and hard rules fire before the
confidence check rather than after it.

**Confidence comes from the classifier's posterior, never from asking a model
how sure it is.** Verbalised self-confidence saturates above 90% regardless of
correctness (DECISIONS #15). Which *function* of the posterior to threshold -
top probability, margin, or entropy - is an empirical question answered in
`scripts/bakeoff_routing.py`, not a guess.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Literal

Signal = Literal["max_prob", "margin", "entropy"]

# Intents that are never auto-handled regardless of confidence, and why. These are
# the ones where a wrong-but-fluent reply commits the brand to something: money,
# rebooking, and anything the customer will read as a promise.
# `ask_policy` was added after the robustness probe (#53): the agent has no policy
# source, only past replies, so every answer to a rules question is a policy claim
# no one checked - the Moffatt v. Air Canada failure.
ALWAYS_ESCALATE = {
    "refund_compensation": "commits the brand to money",
    "rebook_reroute": "commits the brand to a booking change",
    "ask_policy": "asks about rules or fees, and the agent has no policy source to answer from",
}

# The deployed operating point on the top-2 margin (#33, #40, #44). The agent and
# the harness both read it here.
OPERATING_THRESHOLD = 0.10

# Language that implies a policy, a legal threat, or a vulnerable customer. No
# pattern depends on masked entities, so the agent's raw tweet and the harness's
# already-masked golden text trigger them identically.
# Word forms matter: `\brefund\b` missed "refunded" and let a refund request be
# auto-handled (REPORT §3.2, DECISIONS #44). Stems take `\w*`, nouns their plural.
RISK_PATTERNS: dict[str, str] = {
    "legal_or_regulatory": r"\b(lawyers?|attorneys?|sue|sued|suing|lawsuits?|legal action|dot complaint|regulators?|ombudsman|small claims)\b",
    "compensation_claim": r"\b(compensat\w*|reimburs\w*|refund\w*|vouchers?|money back|owe me|pay for|out of pocket)\b",
    "vulnerable_customer": r"\b(wheelchairs?|disab\w*|service animals?|medical|unaccompanied minors?|funerals?|died|passed away|emergency)\b",
    "public_escalation": r"\b(press|media|journalists?|going viral|news|bbc|cnn)\b",
}


@dataclass
class Decision:
    action: Literal["auto_handle", "escalate"]
    reason: str
    rule: str
    confidence: float
    signals: dict[str, float] = field(default_factory=dict)

    @property
    def escalated(self) -> bool:
        return self.action == "escalate"


def entropy(posterior: dict[str, float]) -> float:
    """Shannon entropy in nats, 0 when the model is certain."""
    return -sum(p * math.log(p) for p in posterior.values() if p > 0)


def normalised_entropy(posterior: dict[str, float]) -> float:
    """Entropy scaled to [0, 1] so the threshold means the same thing whatever
    the number of classes."""
    if len(posterior) < 2:
        return 0.0
    return entropy(posterior) / math.log(len(posterior))


def margin(posterior: dict[str, float]) -> float:
    if len(posterior) < 2:
        return 1.0
    top2 = sorted(posterior.values(), reverse=True)[:2]
    return top2[0] - top2[1]


class Router:
    """Ordered rules. The first that fires decides, and names itself."""

    def __init__(
        self,
        threshold: float = 0.5,
        signal: Signal = "max_prob",
        use_rules: bool = True,
    ) -> None:
        self.threshold = threshold
        self.signal = signal
        self.use_rules = use_rules
        self._risk = {k: re.compile(v, re.IGNORECASE) for k, v in RISK_PATTERNS.items()}

    def score(self, posterior: dict[str, float]) -> float:
        """The confidence signal being thresholded. Higher means safer."""
        if not posterior:
            return 0.0
        if self.signal == "margin":
            return margin(posterior)
        if self.signal == "entropy":
            return 1.0 - normalised_entropy(posterior)
        return max(posterior.values())

    def decide(
        self,
        text: str,
        intent: str,
        posterior: dict[str, float],
    ) -> Decision:
        conf = self.score(posterior)
        signals = {
            "max_prob": max(posterior.values()) if posterior else 0.0,
            "margin": margin(posterior),
            "entropy_certainty": 1.0 - normalised_entropy(posterior),
        }

        if self.use_rules:
            for name, pattern in self._risk.items():
                if found := pattern.search(text):
                    # Quote the trigger: the agent picking this up should see why at a
                    # glance, and a false trigger ("news") is visible, not hidden.
                    return Decision(
                        "escalate",
                        f"message contains {name.replace('_', ' ')} language "
                        f"('{found.group(0)}'), which must not be answered from a template",
                        name,
                        conf,
                        signals,
                    )
            if intent in ALWAYS_ESCALATE:
                return Decision(
                    "escalate",
                    f"intent '{intent}' {ALWAYS_ESCALATE[intent]}, so it is never auto-handled",
                    "always_escalate_intent",
                    conf,
                    signals,
                )

        if conf < self.threshold:
            # Name the two readings the model is torn between, so the human knows
            # what to decide rather than only that the model was unsure.
            top = sorted(posterior.items(), key=lambda kv: -kv[1])[:2]
            torn = " vs ".join(f"'{k}' {v:.2f}" for k, v in top)
            return Decision(
                "escalate",
                f"unsure between {torn}: {self.signal} {conf:.2f} is below the "
                f"{self.threshold:.2f} operating point",
                "low_confidence",
                conf,
                signals,
            )

        return Decision(
            "auto_handle",
            f"intent '{intent}' predicted with {self.signal} {conf:.2f} and no risk language",
            "confident_and_safe",
            conf,
            signals,
        )
