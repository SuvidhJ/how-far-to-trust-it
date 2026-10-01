"""The hand-scoring page for judge agreement: blind, complete, and its export parses."""

import json
import re
import sys

from support_agent.config import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import build_judge_page
import judge_agreement


def _payload(tmp_path, monkeypatch) -> tuple[dict, str]:
    out = tmp_path / "judge.html"
    monkeypatch.setattr(sys, "argv", ["build_judge_page.py", "--out", str(out)])
    assert build_judge_page.main() == 0
    html = out.read_text(encoding="utf-8")
    m = re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S)
    return json.loads(m.group(1)), html


def test_page_shows_all_40_sheet_items_and_no_judge_output(tmp_path, monkeypatch):
    payload, html = _payload(tmp_path, monkeypatch)
    assert "__DATA__" not in html
    items = payload["items"]
    assert [i["idx"] for i in items] == list(range(40))
    assert all(set(i) == {"idx", "sha", "customer", "draft", "evidence"} for i in items)
    assert all(i["customer"] and i["draft"] and i["evidence"] for i in items)
    # Blind: the payload carries no judge score, verdict or reason of any kind.
    blob = json.dumps(items)
    for word in ("sendable", "worst_problem", "judge_model", '"total"'):
        assert word not in blob


def test_export_format_parses_with_the_scorer(tmp_path, monkeypatch):
    payload, _ = _payload(tmp_path, monkeypatch)
    assert tuple(payload["dimensions"]) == judge_agreement.DIMENSIONS
    assert payload["header"] and all(h.startswith("#") for h in payload["header"])
    # The page's Export joins the header, then "<idx> g a t s" for a scored item and
    # "<idx>" for an unscored one. These values are test data, not anyone's scores.
    lines = [*payload["header"], "0 2 1 2 2", "1", "2 0 0 1 2"]
    parsed = judge_agreement.parse_scores("\n".join(lines) + "\n")
    assert parsed == {
        0: {"grounded": 2, "addresses": 1, "tone": 2, "safe": 2},
        2: {"grounded": 0, "addresses": 0, "tone": 1, "safe": 2},
    }
