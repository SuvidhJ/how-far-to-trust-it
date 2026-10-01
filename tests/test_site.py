"""The demo page and its replay cases: built offline, and consistent with the numbers
the harness commits to reports/results.json."""

import json
import re
import sys

import pytest

from support_agent.config import ROOT, Paths, settings

RESULTS = json.loads((ROOT / "reports" / "results.json").read_text(encoding="utf-8"))
CASES = [
    json.loads(line)
    for line in (Paths.processed / f"{settings.brand.lower()}_demo_cases.jsonl")
    .read_text(encoding="utf-8")
    .splitlines()
]
# The post-draft guards turn a router auto-handle into an escalation (#50, #51).
DRAFT_GUARDS = {"draft_placeholder", "draft_unkept_promise"}


def test_site_builds_offline_with_every_case_and_no_placeholder(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_site

    out = tmp_path / "index.html"
    monkeypatch.setattr(
        sys, "argv", ["build_site.py", "--repo", "someone/some-repo", "--out", str(out)]
    )
    assert build_site.main() == 0

    html = out.read_text(encoding="utf-8")
    assert "__DATA__" not in html and "__REPO__" not in html
    assert "https://github.com/someone/some-repo" in html
    m = re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S)
    assert m, "payload script tag missing"
    payload = json.loads(m.group(1))
    assert len(payload["cases"]) == len(CASES) == RESULTS["golden_n"] == 218
    assert payload["results"]["routing"] == RESULTS["routing"]


def test_replay_cases_agree_with_committed_results():
    n = RESULTS["golden_n"]
    assert len(CASES) == n

    # Repeat 0 of the headline CV: the same predictions the harness scores.
    correct = sum(c["pred_intent"] == c["gold_intent"] for c in CASES)
    assert correct == round(RESULTS["intent"]["system (embed+logreg)"]["accuracy"] * n)

    arm = RESULTS["drafting"]["arms"]["grounded_llm"]
    judged = [c["drafts"]["grounded_llm"]["judge"] for c in CASES]
    assert sum(bool(j and j["sendable"]) for j in judged) == arm["sendable"]
    assert sum(c["drafts"]["grounded_llm"]["placeholder"] for c in CASES) == arm["with_placeholder"]

    routing = RESULTS["routing"]
    router_auto = [
        c
        for c in CASES
        if c["decision"]["action"] == "auto_handle" or c["decision"]["rule"] in DRAFT_GUARDS
    ]
    assert len(router_auto) == routing["auto_handled"]
    wrong = sum(c["pred_intent"] != c["gold_intent"] for c in router_auto)
    assert wrong == pytest.approx(routing["error_on_auto_handled"] * routing["auto_handled"])
