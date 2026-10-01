"""Tests for the pipeline itself, not the environment.

Deliberately weighted toward the properties that, if they silently broke, would
produce a plausible-looking number rather than a crash. A test that a function
returns a float is worthless here; a test that the golden set contains no two
messages from the same conversation is not.

No test calls a live model.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from support_agent.agent import Agent
from support_agent.config import Paths, seed_for, settings
from support_agent.data import add_thread_splits, filter_exchanges
from support_agent.draft import Draft, looks_like_reasoning
from support_agent.draft import build as build_drafter
from support_agent.eval.metrics import (
    cohens_kappa,
    corrected_resampled_ttest,
    gwet_ac1,
    intent_report,
    macro_f1,
    weighted_kappa,
    wilson,
)
from support_agent.intents import MajorityClassifier, TfidfClassifier, load_taxonomy
from support_agent.llm import (
    FAMILY,
    LLM,
    ROLE_MODELS,
    LLMResponse,
    is_transient,
    parse_json,
)
from support_agent.retrieve import Bm25Retriever, TfidfRetriever
from support_agent.route import Router, margin, normalised_entropy
from support_agent.text import clean, detect_lang, mask_entities, strip_signature


# --------------------------------------------------------------------------
# text normalisation
# --------------------------------------------------------------------------
def test_signature_stripping_handles_split_reply_markers():
    assert strip_signature("bag missing *AOS 1/2") == "bag missing"
    assert strip_signature("we can help ^QJ") == "we can help"
    assert strip_signature("no signature here") == "no signature here"


def test_signature_stripping_is_idempotent():
    once = strip_signature("we can help ^QJ")
    assert strip_signature(once) == once


def test_masking_distinguishes_phone_numbers_from_case_numbers():
    # A bare run of digits is a case reference, not a phone number. Getting this
    # backwards put <phone> on every Amazon order id.
    assert "<num>" in mask_entities("case 0012345678")
    assert "<phone>" in mask_entities("call 800-555-1234")


def test_masking_removes_high_cardinality_entities():
    out = clean("@AmericanAir AA 1234 delayed, ref ABC12X http://t.co/x")
    for token in ("<user>", "<flight>", "<ref>", "<url>"):
        assert token in out
    assert "ABC12X" not in out


def test_language_detection_separates_spanish():
    assert detect_lang("my flight was cancelled and nobody helped me at all") == "en"
    assert detect_lang("mi vuelo fue cancelado y necesito ayuda por favor") == "es"


# --------------------------------------------------------------------------
# splitting - the leakage guarantees
# --------------------------------------------------------------------------
def test_thread_splits_are_disjoint_by_thread():
    import pandas as pd

    df = pd.DataFrame(
        {
            "thread_id": [1, 1, 2, 2, 3, 4, 5, 6, 7, 8],
            "customer_clean": ["a"] * 10,
        }
    )
    out = add_thread_splits(df, seed=7)
    per_thread = out.groupby("thread_id")["split"].nunique()
    assert (per_thread == 1).all(), "a thread must land wholly in one split"


def test_thread_splits_are_stable_when_higher_thread_ids_are_appended():
    """The guarantee is narrow: appending threads keeps existing assignments.
    Inserting or removing threads does not (see `add_thread_splits`)."""
    import pandas as pd

    base = pd.DataFrame({"thread_id": list(range(50)), "customer_clean": ["a"] * 50})
    grown = pd.DataFrame({"thread_id": list(range(80)), "customer_clean": ["a"] * 80})
    a = add_thread_splits(base, seed=11).set_index("thread_id")["split"]
    b = add_thread_splits(grown, seed=11).set_index("thread_id")["split"]
    assert (a == b.loc[a.index]).all()


def test_filter_counts_everything_it_drops():
    import pandas as pd

    df = pd.DataFrame(
        {
            "lang": ["en", "es", "en", "en"],
            "customer_clean": [
                "a long enough customer message",
                "hola",
                "short",
                "another good one",
            ],
            "brand_clean": [
                "a long enough brand reply here",
                "x",
                "y",
                "a second good brand reply",
            ],
            "response_minutes": [1.0, 1.0, 1.0, -5.0],
        }
    )
    kept, counts = filter_exchanges(df)
    assert counts["input"] == 4
    assert counts["dropped_non_english"] == 1
    assert counts["output"] == len(kept)
    assert (
        sum(counts[k] for k in counts if k.startswith("dropped"))
        == counts["input"] - counts["output"]
    )


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def test_wilson_stays_inside_the_unit_interval_at_small_n():
    lo, hi = wilson(0, 3)
    assert 0.0 <= lo <= hi <= 1.0
    lo, hi = wilson(3, 3)
    assert 0.0 <= lo <= hi <= 1.0


def test_macro_f1_ignores_class_prevalence():
    """The point of macro-F1: getting the rare class wrong must hurt as much as
    getting the common one wrong."""
    y = ["a"] * 90 + ["b"] * 10
    all_common = ["a"] * 100
    assert macro_f1(y, all_common, ["a", "b"]) < 0.5


def test_intent_report_carries_n_for_every_class():
    y = ["a", "a", "b", "c"]
    p = ["a", "b", "b", "c"]
    rep = intent_report(y, p, ["a", "b", "c"])
    assert rep.n == 4
    assert {c.label: c.support for c in rep.per_class} == {"a": 2, "b": 1, "c": 1}


def test_kappa_is_zero_for_chance_agreement():
    a = ["x", "y"] * 50
    b = ["x", "x", "y", "y"] * 25
    k, n = cohens_kappa(a, b)
    assert n == 100
    assert abs(k) < 0.2


def test_kappa_matches_sklearn_unweighted_and_quadratic():
    from sklearn.metrics import cohen_kappa_score

    rng = np.random.default_rng(3)
    a = rng.integers(0, 3, 60).tolist()
    b = [x if rng.random() < 0.6 else int(rng.integers(0, 3)) for x in a]
    assert (
        abs(cohens_kappa([str(x) for x in a], [str(x) for x in b])[0] - cohen_kappa_score(a, b))
        < 1e-9
    )
    assert abs(weighted_kappa(a, b) - cohen_kappa_score(a, b, weights="quadratic")) < 1e-9


def test_ac1_survives_the_kappa_paradox():
    # 38 agreements on "yes", one disagreement each way: kappa is below zero,
    # AC1 reports the 95% agreement it actually is.
    a = ["yes"] * 38 + ["yes", "no"]
    b = ["yes"] * 38 + ["no", "yes"]
    assert cohens_kappa(a, b)[0] < 0
    assert abs(gwet_ac1(a, b) - (0.95 - 0.04875) / (1 - 0.04875)) < 1e-9


def test_corrected_ttest_is_more_conservative_than_naive():
    from scipy.stats import t as t_dist
    from scipy.stats import ttest_1samp

    d = np.array([0.03, 0.01, 0.05, -0.01, 0.04] * 5)
    mean, t, p_corrected = corrected_resampled_ttest(d, n_train=174, n_test=44)
    assert p_corrected > ttest_1samp(d, 0).pvalue
    # The exact Nadeau-Bengio statistic, so an inverted n_test/n_train ratio fails.
    expected_t = d.mean() / np.sqrt((1 / 25 + 44 / 174) * d.var(ddof=1))
    assert t == pytest.approx(expected_t) and mean == pytest.approx(d.mean())
    assert p_corrected == pytest.approx(2 * t_dist.sf(abs(expected_t), 24))


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------
def test_margin_is_the_gap_between_the_top_two():
    assert margin({"a": 0.5, "b": 0.3, "c": 0.2}) == pytest.approx(0.2)


def test_normalised_entropy_spans_zero_to_one():
    assert normalised_entropy({"a": 1.0, "b": 0.0}) == pytest.approx(0.0)
    assert normalised_entropy({"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}) == pytest.approx(1.0)


def test_risk_language_escalates_even_when_confident():
    r = Router(threshold=0.0)
    d = r.decide("my lawyer will be in touch", "praise_service", {"praise_service": 0.99})
    assert d.escalated and d.rule == "legal_or_regulatory"


def test_money_intents_never_auto_handle():
    r = Router(threshold=0.0)
    d = r.decide("where is my flight", "refund_compensation", {"refund_compensation": 0.99})
    assert d.escalated and d.rule == "always_escalate_intent"


def test_policy_questions_never_auto_handle():
    # The agent has no policy source, so a confident answer is an unchecked policy claim (#53).
    r = Router(threshold=0.0)
    d = r.decide("is there a bereavement fare?", "ask_policy", {"ask_policy": 0.99})
    assert d.escalated and d.rule == "always_escalate_intent" and "policy" in d.reason


def test_every_escalation_states_a_reason():
    r = Router(threshold=0.5)
    for text, intent, post in [
        ("i want compensation", "refund_compensation", {"refund_compensation": 0.9}),
        ("what time is boarding", "ask_policy", {"ask_policy": 0.3, "other": 0.29}),
    ]:
        d = r.decide(text, intent, post)
        assert d.escalated
        assert len(d.reason) > 20 and d.rule


def test_confident_safe_message_is_auto_handled():
    r = Router(threshold=0.2)
    d = r.decide(
        "thanks for the great flight", "praise_service", {"praise_service": 0.9, "other": 0.1}
    )
    assert not d.escalated and d.rule == "confident_and_safe"


# --------------------------------------------------------------------------
# retrieval
# --------------------------------------------------------------------------
def test_bm25_ranks_the_lexically_closest_document_first():
    docs = [
        "my bag is missing from baggage claim",
        "flight delayed three hours",
        "seat upgrade please",
    ]
    r = Bm25Retriever().index(docs, [{"id": i} for i in range(3)])
    assert r.search("bag missing baggage", k=1)[0].index == 0


def test_retriever_can_exclude_the_query_itself():
    docs = ["identical text here", "something else entirely", "identical text here too"]
    r = TfidfRetriever().index(docs, [{"id": i} for i in range(3)])
    hits = r.search(docs[0], k=2, exclude=0)
    assert all(h.index != 0 for h in hits)


def test_retriever_returns_ids_so_grounding_is_checkable():
    docs = ["bag missing", "flight late"]
    r = TfidfRetriever().index(docs, [{"id": 101, "reply": "sorry"}, {"id": 102, "reply": "ok"}])
    hit = r.search("bag missing", k=1)[0]
    assert hit.id == 101 and hit.reply == "sorry"


# --------------------------------------------------------------------------
# taxonomy, classifiers and the golden set
# --------------------------------------------------------------------------
def test_taxonomy_letters_and_ids_are_unique():
    tax = load_taxonomy()
    assert len({i.letter for i in tax.intents}) == len(tax.intents)
    assert len({i.id for i in tax.intents}) == len(tax.intents)


def test_taxonomy_prompt_contains_every_option_and_its_near_miss():
    tax = load_taxonomy()
    options = tax.as_options()
    for i in tax.intents:
        assert f"{i.letter}. {i.name}" in options
        assert i.near_miss in options


def test_majority_baseline_predicts_the_training_mode():
    m = MajorityClassifier().fit(["a", "b", "c"], ["x", "y", "y"])
    assert all(p.intent == "y" for p in m.predict(["anything"]))


def test_tfidf_classifier_returns_a_posterior_that_sums_to_one():
    texts = ["bag missing"] * 5 + ["flight delayed"] * 5
    labels = ["track_baggage"] * 5 + ["report_disruption"] * 5
    preds = TfidfClassifier().fit(texts, labels).predict(["my bag is gone"])
    assert sum(preds[0].posterior.values()) == pytest.approx(1.0)


def test_golden_set_has_one_row_per_thread():
    """The bug this catches shipped once: two messages from one conversation
    landed in different CV folds as train/test twins."""
    path = Paths.golden / f"{settings.brand.lower()}_golden.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    live = [r for r in rows if not r.get("excluded")]
    threads = [r["thread_id"] for r in live]
    assert len(set(threads)) == len(threads)


def test_golden_labels_are_all_in_the_frozen_taxonomy():
    tax = load_taxonomy()
    path = Paths.golden / f"{settings.brand.lower()}_golden.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert {r["intent"] for r in rows} <= set(tax.ids)


def test_golden_set_size_is_within_the_target():
    path = Paths.golden / f"{settings.brand.lower()}_golden.jsonl"
    rows = path.read_text(encoding="utf-8").strip().splitlines()
    assert 150 <= len(rows) <= 250


# --------------------------------------------------------------------------
# LLM adapter contracts
# --------------------------------------------------------------------------
def test_judge_and_drafter_never_share_a_training_family():
    _, judge = ROLE_MODELS["judge"]
    _, drafter = ROLE_MODELS["drafter"]
    assert FAMILY[judge] != FAMILY[drafter]


def test_for_role_refuses_a_same_family_judge(monkeypatch):
    monkeypatch.setitem(ROLE_MODELS, "judge", ROLE_MODELS["drafter"])
    with pytest.raises(ValueError, match="training family"):
        LLM.for_role("judge")


def test_for_role_refuses_a_sibling_model_and_unknown_models(monkeypatch):
    monkeypatch.setitem(
        ROLE_MODELS, "judge", ("openrouter", "nvidia/nemotron-3-ultra-550b-a55b:free")
    )
    with pytest.raises(ValueError, match="training family"):
        LLM.for_role("judge")
    monkeypatch.setitem(ROLE_MODELS, "judge", ("nvidia", "some/unlisted-model"))
    with pytest.raises(ValueError, match="no training family"):
        LLM.for_role("judge")


def test_parse_json_survives_fences_and_trailing_prose_but_not_truncation():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('{"a": 1} trailing noise') == {"a": 1}
    for bad in ("not json at all", '{"grounded": 2, "addresses": 1, "tone'):
        with pytest.raises(ValueError):
            parse_json(bad)


def test_offline_llm_refuses_rather_than_calling_out():
    llm = LLM("nvidia", "meta/llama-3.2-11b-vision-instruct", offline=True)
    with pytest.raises(RuntimeError, match="offline"):
        llm.complete("system", "a prompt that is certainly not cached " + "x" * 40)


def test_extra_body_changes_the_cache_key_only_when_set(monkeypatch):
    """Checked on the payload `complete()` really hashes, not a hand-built dict."""
    import support_agent.llm as llm_mod

    seen = []
    monkeypatch.setattr(llm_mod, "_request_key", lambda payload: seen.append(payload) or "k")
    llm = LLM("nvidia", "meta/llama-3.2-11b-vision-instruct", offline=True)
    for body in (None, {}, {"chat_template_kwargs": {"enable_thinking": False}}):
        with pytest.raises(RuntimeError, match="offline"):
            llm.complete("s", "u", extra_body=body)
    assert "extra_body" not in seen[0] and "extra_body" not in seen[1]
    assert seen[2]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_transient_errors_are_retried_and_client_errors_are_not():
    class Http(Exception):
        def __init__(self, status):
            self.status_code = status

    class APIConnectionError(Exception):
        pass

    assert is_transient(Http(429)) and is_transient(Http(503))
    assert is_transient(APIConnectionError())
    assert not is_transient(Http(404)) and not is_transient(Http(422))


# --------------------------------------------------------------------------
# drafting validity: an invalid draft must never be scored as a reply
# --------------------------------------------------------------------------
class _FakeLLM:
    def __init__(self, text="", finish_reason="stop", exc=None):
        self.text, self.finish_reason, self.exc = text, finish_reason, exc

    def complete(self, system, user, **kw):
        if self.exc:
            raise self.exc
        return LLMResponse(self.text, "fake", "fake", True, 0.0, finish_reason=self.finish_reason)


@pytest.mark.parametrize(
    "text",
    [
        "We need to respond to the customer about their bag.",
        "Okay, the customer is upset about a delay.",
        "The user wants a refund.",
        "<think>short</think> Sorry about that.",
    ],
)
def test_reasoning_openers_are_detected(text):
    assert looks_like_reasoning(text)


@pytest.mark.parametrize(
    "text",
    [
        "We're sorry your bag didn't make it. Please DM us your file reference.",
        "We understand the frustration. Our team will follow up via DM.",
        "Let us know your confirmation number via DM and we'll take a look.",
    ],
)
def test_real_replies_are_not_flagged_as_reasoning(text):
    assert not looks_like_reasoning(text)


@pytest.mark.parametrize(
    ("llm", "error"),
    [
        (_FakeLLM(exc=TimeoutError("read timed out")), "api:"),
        (_FakeLLM(text=""), "empty completion"),
        (_FakeLLM(text="We need to think about this."), "reasoning leak"),
        (_FakeLLM(text="Sorry about the delay, we", finish_reason="length"), "truncated"),
    ],
)
def test_invalid_drafts_carry_an_error(llm, error):
    d = build_drafter("ungrounded_llm", llm=llm).draft("my flight is late", "report_disruption", [])
    assert d.error.startswith(error)


def test_valid_draft_has_no_error():
    llm = _FakeLLM(text="Sorry for the delay. We'll get you on your way as soon as we can.")
    assert not build_drafter("ungrounded_llm", llm=llm).draft("late", "report_disruption", []).error


def _load_script(name: str):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_judge_agreement_scoring_end_to_end_on_a_synthetic_fixture():
    # Synthetic scores, built in memory: this checks the arithmetic, never a result.
    ja = _load_script("judge_agreement")
    human = ja.parse_scores("# comment\n0 2 2 2 2\n1 2 1 2 2\n2 0 1 1 0\n3 1 0 2 1\n")
    machine = {0: human[0], 1: human[1], 2: human[2], 3: {**human[3], "tone": 0}}
    rows = {r["dimension"]: r for r in ja.agreement_rows(human, machine)}
    assert rows["grounded"]["kappa"] == pytest.approx(1.0) and rows["grounded"]["n"] == 4
    assert rows["tone"]["exact"] == pytest.approx(0.75)
    s = rows["sendable (binary)"]
    assert (s["human_yes"], s["judge_yes"], s["n"]) == (2, 2, 4)


def test_repeated_cv_is_deterministic_and_reports_every_repeat():
    from support_agent.eval.harness import cv_macro_f1

    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(c, 1.0, size=(20, 4)) for c in (0, 3, 6)])
    y = np.array(["a"] * 20 + ["b"] * 20 + ["c"] * 20)
    p1, f1 = cv_macro_f1(X, y, ["a", "b", "c"], folds=5, repeats=3)
    p2, f2 = cv_macro_f1(X, y, ["a", "b", "c"], folds=5, repeats=3)
    assert len(f1) == 3 and f1 == f2 and p1 == p2
    assert all(p != "" for rep in p1 for p in rep)  # every row predicted out of fold


def test_judge_agreement_rejects_out_of_range_scores():
    ja = _load_script("judge_agreement")
    with pytest.raises(ValueError):
        ja.parse_scores("0 3 2 2 2\n")


def test_thinking_switch_reaches_the_request():
    seen = {}

    class Spy(_FakeLLM):
        def complete(self, system, user, **kw):
            seen.update(system=system, **kw)
            return super().complete(system, user, **kw)

    body = {"chat_template_kwargs": {"enable_thinking": False}}
    d = build_drafter(
        "ungrounded_llm", llm=Spy(text="Sorry."), system_prefix="/no_think\n", extra_body=body
    )
    d.draft("late", "report_disruption", [])
    assert seen["system"].startswith("/no_think\n") and seen["extra_body"] == body


# --------------------------------------------------------------------------
# seeds
# --------------------------------------------------------------------------
def test_sub_seeds_are_stable_and_distinct():
    assert seed_for("golden") == seed_for("golden")
    assert seed_for("golden") != seed_for("split")


def test_seeded_sampling_reproduces():
    a = np.random.default_rng(seed_for("x")).random(5)
    b = np.random.default_rng(seed_for("x")).random(5)
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------
# agent end to end (offline: committed embedding cache, no LLM)
# --------------------------------------------------------------------------
def test_agent_returns_intent_draft_and_reasoned_decision_offline():
    agent = Agent(drafter="nearest", offline=True)
    r = agent.handle(agent.golden_texts[0])
    assert r.intent in r.posterior and abs(sum(r.posterior.values()) - 1) < 1e-6
    assert r.draft.text and r.draft.grounded_in == [r.evidence[0].id]
    assert r.decision.action in ("auto_handle", "escalate") and r.decision.reason


def test_agent_escalates_when_the_draft_failed():
    agent = Agent(drafter="nearest", offline=True)

    class Broken:
        name = "broken"

        def draft(self, message, intent, hits):
            return Draft("", self.name, intent=intent, error="reasoning leak")

    agent.drafter = Broken()
    r = agent.handle(agent.golden_texts[0])
    assert r.decision.escalated and r.decision.rule == "draft_failed"
    assert "reasoning leak" in r.decision.reason


@pytest.mark.parametrize(
    "reply",
    [
        '{"grounded": 2, "addresses": 2, "tone": 2}',  # a dimension missing
        '{"grounded": 2, "addresses": 5, "tone": 2, "safe": 2}',  # off the 0-2 scale
    ],
)
def test_judge_counts_incomplete_scores_as_parse_failures_not_zeros(reply):
    from support_agent.eval.judge import Judge

    fake = _FakeLLM(text=reply)
    fake.model = "fake"
    s = Judge(llm=fake).score("where is my bag", "Sorry, we're on it.", [])
    assert s.parse_failed and s.total == 0


@pytest.mark.parametrize(
    "text",
    [
        "How do I get refunded for that???",
        "they gave me two vouchers that don't work",
        "my lawyers will hear about this",
        "wheelchairs were not waiting at the gate",
    ],
)
def test_risk_rules_match_inflected_word_forms(text):
    d = Router(threshold=0.0, signal="margin").decide(
        text, "manage_booking", {"manage_booking": 1.0}
    )
    assert d.escalated and d.rule != "low_confidence"


# --------------------------------------------------------------------------
# golden-set provenance
# --------------------------------------------------------------------------
def test_golden_set_rebuilds_byte_identically_from_the_hand_label_files(tmp_path, monkeypatch):
    """The committed labels are the hand-typed `<index> <letter> [?]` batches in
    data/golden/labels/. If the golden set ever drifts from them, this fails."""
    import shutil
    import sys

    sys.path.insert(0, str(Paths.golden.parents[1] / "scripts"))
    import build_golden

    shutil.copy(Paths.golden / "americanair_frame.jsonl", tmp_path / "americanair_frame.jsonl")
    labels = str(Paths.golden / "labels" / "labels_batch*.txt")
    monkeypatch.setattr(build_golden, "Paths", type("P", (), {"golden": tmp_path}))
    monkeypatch.setattr(build_golden, "write_manifest", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["build_golden.py", "--labels", labels])
    assert build_golden.main() == 0
    rebuilt = (tmp_path / "americanair_golden.jsonl").read_bytes()
    committed = (Paths.golden / "americanair_golden.jsonl").read_bytes()
    # Line endings are the platform's on write and the checkout's on read
    # (.gitattributes stores LF); everything else must match byte for byte.
    assert rebuilt.replace(b"\r\n", b"\n") == committed.replace(b"\r\n", b"\n")


def test_escalation_reasons_are_actionable():
    r = Router(threshold=0.10, signal="margin")
    unsure = r.decide(
        "my flight",
        "report_disruption",
        {"report_disruption": 0.41, "rebook_reroute": 0.37, "other": 0.22},
    )
    assert unsure.rule == "low_confidence"
    assert "report_disruption" in unsure.reason and "rebook_reroute" in unsure.reason
    money = r.decide(
        "how do I get refunded", "manage_booking", {"manage_booking": 0.9, "other": 0.1}
    )
    assert money.rule == "compensation_claim" and "'refunded'" in money.reason


def test_agent_normalises_input_like_the_golden_set_but_routes_on_raw_text():
    from support_agent.text import clean

    agent = Agent(drafter="canned", offline=True)
    golden = agent.golden_texts[0]
    assert clean(golden) == golden  # golden rows are already clean, so offline replay still hits
    r = agent.handle(golden)
    assert r.clean_text == golden and set(r.timings_ms) == {
        "embed",
        "classify",
        "retrieve",
        "draft",
        "route",
    }


def test_agent_never_auto_sends_a_draft_that_promises_a_follow_up():
    agent = Agent(drafter="canned", offline=True)

    class Promising:
        name = "promising"

        def draft(self, message, intent, hits):
            return Draft(
                "Sorry about that. A colleague will follow up shortly.", self.name, intent=intent
            )

    agent.drafter = Promising()
    # A router that auto-handles everything, so only the post-draft guard can escalate.
    agent.router = Router(threshold=0.0, signal="margin", use_rules=False)
    r = agent.handle(agent.golden_texts[0])
    assert r.decision.escalated and r.decision.rule == "draft_unkept_promise"
    assert "colleague" in r.decision.reason


@pytest.mark.parametrize(
    "text,top,expect",
    [
        ("I think this is a refund", {}, None),  # prose starting with a letter is not a label
        ("We need to label", {"We": -0.02, "B": -6.0, "C": -6.5}, None),  # letters hold <1% mass
        (" B ", {}, "B"),  # a bare letter is a label
        ("B", {"B": -0.05, "C": -3.0}, "B"),  # real mass on letters gives a posterior
    ],
)
def test_llm_classifier_only_accepts_real_answers(text, top, expect):
    from support_agent.intents import LLMClassifier

    tax = load_taxonomy()
    clf = LLMClassifier(llm=_FakeLLM(), taxonomy=tax, use_logprobs=bool(top))
    resp = LLMResponse(
        text,
        "fake",
        "fake",
        True,
        0.0,
        logprobs=[{"token": text, "logprob": 0.0, "top": top}] if top else [],
    )
    pred = clf._decide(resp)
    if expect is None:
        assert pred.parse_failed
    else:
        assert not pred.parse_failed and pred.intent == tax.by_letter(expect).id


def test_placeholders_and_dm_handoffs_are_detected():
    from support_agent.text import DM_HANDOFF, has_placeholder

    assert not has_placeholder("<user> We're sorry about the delay.")
    assert has_placeholder("<user> Please call us at <phone> for help.")
    assert has_placeholder("<user> Thanks, <user>! Details here: <url>")
    assert DM_HANDOFF.search("Please follow and meet us in DMs with your record locator")
    assert DM_HANDOFF.search("Send us a DM") and DM_HANDOFF.search("via direct message")
    assert not DM_HANDOFF.search("the admin team will help")


def test_agent_classifies_masked_text_and_routes_the_raw_message():
    agent = Agent(drafter="canned", offline=True)
    seen: dict = {}

    class SpyEmbedder:
        def encode(self, texts, input_type="passage"):
            seen.setdefault("embedded", []).extend(texts)
            v = np.ones((len(texts), agent.classifier.coef_.shape[1]), dtype=np.float32)
            return v / np.linalg.norm(v, axis=1, keepdims=True)

    decide = agent.router.decide

    def spy_decide(text, intent, posterior):
        seen["routed"] = text
        return decide(text, intent, posterior)

    agent.embedder = agent.retriever.embedder = SpyEmbedder()
    agent.router.decide = spy_decide
    raw = "@AmericanAir flight AA 1234 delayed, call me at 212-555-0199"
    r = agent.handle(raw)
    assert (
        seen["embedded"][0] == r.clean_text == "<user> flight <flight> delayed, call me at <phone>"
    )
    assert seen["routed"] == raw


def test_golden_set_and_retrieval_corpus_share_no_conversation():
    def rows(path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    gold = rows(Paths.golden / "americanair_golden.jsonl")
    corpus = rows(Paths.processed / "americanair_index.jsonl")
    assert not {r["thread_id"] for r in gold} & {c["thread_id"] for c in corpus}


def test_oof_predictions_are_deterministic_and_out_of_fold():
    from support_agent.intents import oof_predictions

    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(c, 1.0, size=(20, 4)) for c in (0, 3, 6)])
    y = np.array(["a"] * 20 + ["b"] * 20 + ["c"] * 20)
    p1, post1 = oof_predictions(X, y, folds=5, seed=7)
    p2, post2 = oof_predictions(X, y, folds=5, seed=7)
    assert p1 == p2 and post1 == post2 and all(p1)
    assert all(max(d, key=d.get) == p for d, p in zip(post1, p1, strict=True))


def test_harness_end_to_end_offline_smoke(tmp_path):
    """Runs every harness path, naive waterfall included, from the committed caches.
    A refactor once broke the judge-family step with an IndexError that no unit test saw."""
    from support_agent.eval import harness

    out = tmp_path / "run.json"
    assert (
        harness.main(
            ["--offline", "--naive", "--repeats", "1", "--draft-n", "8", "--out", str(out)]
        )
        == 0
    )
    manifest = json.loads(out.read_text(encoding="utf-8"))
    assert (
        manifest["routing"]["auto_handled"] + manifest["routing"]["escalated"]
        == manifest["golden_n"]
    )
    assert any(
        r["step"].startswith("judge from") and r["naive"] is not None
        for r in manifest["naive_waterfall"]
    )


def test_results_check_catches_moved_numbers_and_lists_skips():
    mod = _load_script("check_results")
    expected = {"run_at": "a", "auto": 48, "err": 0.0625, "rows": [{"n": 218, "v": 0.5}]}
    assert mod.diff(expected, {**expected, "run_at": "b", "err": 0.0625 + 1e-12}, 1e-9) == []
    assert mod.diff(expected, {**expected, "auto": 49}, 1e-9) == ["auto: expected 48, got 49"]
    skipped = {**expected, "rows": [{"n": "skipped: raw data not built"}]}
    assert mod.diff(expected, skipped, 1e-9) == [] and mod.SKIPPED
