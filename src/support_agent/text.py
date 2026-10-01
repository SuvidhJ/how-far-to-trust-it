"""Text normalisation shared by every stage.

Two jobs, both of which the data exploration proved necessary:

1. **Signature stripping.** Brands sign replies with an agent's initials
   (AmazonHelp `^QJ` on 89.5% of replies, Delta `*AOS`, SpotifyCares `/TF`).
   Left in, they are both noise and a leakage channel — a classifier can learn
   which human handled which intent.
2. **Entity masking.** Order ids, flight numbers, URLs and handles dominate
   TF-IDF weight. Unmasked, clusters form around entity names rather than
   intents, and retrieval returns "the tweet that mentions flight 1234".

Masking is applied *before* vectorisation and *before* the golden set is shown
to a labeller, so the labels describe intent rather than memorised strings.
"""

from __future__ import annotations

import re

import langid

langid.set_languages(["en", "es", "pt", "fr", "de", "it", "nl", "ja", "ar", "tr"])

# Trailing agent signature: "^QJ", "*AOS", "- JB", "/TF", "^ NL" — optionally
# followed by a split-reply part marker ("*AOS 1/2"), which we also drop.
SIGNATURE = re.compile(
    r"(?:\s*[\^*/~\-–]\s?[A-Z]{1,3}\d?)+(?:\s*\d{1,2}\s*/\s*\d{1,2})?\s*$"  # noqa: RUF001
)
# Leading/trailing handles, which carry no intent signal.
HANDLE = re.compile(r"@\w{1,15}")
URL = re.compile(r"https?://\S+|\bwww\.\S+")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
# Airline flight designators: "AA123", "AA 1234", "#AA123".
FLIGHT = re.compile(r"\b[A-Z]{2}\s?\d{2,4}\b")
# Booking refs / order ids: 6+ alphanumerics with at least one digit and one letter.
ORDERID = re.compile(r"\b(?=[A-Z0-9-]*\d)(?=[A-Z0-9-]*[A-Z])[A-Z0-9][A-Z0-9-]{5,}\b")
# A phone number needs phone-shaped punctuation or a country prefix. A bare run
# of digits is a case or booking number and becomes <num> instead.
PHONE = re.compile(r"\+\d[\d\s().-]{7,}\d|\b\d{3}[\s().-]\d{3}[\s.-]\d{4}\b")
NUMBER = re.compile(r"\b\d{3,}\b")
WS = re.compile(r"\s+")

# A mask token anywhere except a leading "<user>" (the @mention a reply opens with).
# A draft carrying one cannot be sent as-is: someone must fill in the phone number,
# link or reference first (DECISIONS #51).
_MASK_TOKEN = re.compile(r"<(?:user|url|email|flight|ref|phone|num)>")


def has_placeholder(draft: str) -> bool:
    body = re.sub(r"^\s*(?:<user>\s*)+", "", draft)
    return bool(_MASK_TOKEN.search(body))


# The brand moving a conversation to private messages. Matches "DM", "DMs", "D.M.";
# the first version missed "DMs" and undercounted handoffs 48 vs 60 (#51).
DM_HANDOFF = re.compile(r"(?i)\b(?:d\.?m\.?s?|direct messages?|private messages?|pm us|inbox us)\b")


def strip_signature(text: str) -> str:
    """Remove a trailing agent signature. Idempotent."""
    return SIGNATURE.sub("", text).rstrip()


def mask_entities(text: str) -> str:
    """Replace high-cardinality entities with type tokens.

    Order matters: URLs before handles (a URL can contain '@'), flight numbers
    before generic order ids, order ids before bare numbers.
    """
    t = URL.sub(" <url> ", text)
    t = EMAIL.sub(" <email> ", t)
    t = HANDLE.sub(" <user> ", t)
    t = FLIGHT.sub(" <flight> ", t)
    t = ORDERID.sub(" <ref> ", t)
    t = PHONE.sub(" <phone> ", t)
    t = NUMBER.sub(" <num> ", t)
    return WS.sub(" ", t).strip()


def clean(text: str) -> str:
    """Full normalisation for model input: signature stripped, entities masked."""
    return mask_entities(strip_signature(text))


def detect_lang(text: str) -> str:
    """ISO-639-1 language code. Masked text first — URLs skew detection."""
    stripped = WS.sub(" ", URL.sub(" ", HANDLE.sub(" ", text))).strip()
    if len(stripped) < 12:
        return "und"
    return langid.classify(stripped)[0]
