"""Build the static demo page: the agent replayed on every golden message, and the
evaluation that says how far to trust it.

    uv run python scripts/build_site.py            # -> site/dist/index.html
    uv run python scripts/build_site.py --out x.html

One self-contained HTML file with its data inlined, so it runs from disk or GitHub
Pages with no server and no API keys. It reads only committed artifacts:
`reports/results.json`, `reports/results_naive.json`, `reports/annotator_agreement.json`
and the out-of-fold replay cases from `scripts/build_demo_cases.py`. No number is typed
into the page by hand.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from support_agent.config import GLOBAL_SEED, ROOT, Paths, settings
from support_agent.data import git_sha
from support_agent.eval.metrics import cohens_kappa, kappa_ci

TEMPLATE = ROOT / "site" / "index.template.html"
OUT = ROOT / "site" / "dist" / "index.html"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _unescape(t: str) -> str:
    """Tweets carry HTML entities (&amp;, &gt;) as the dataset stored them."""
    return html.unescape(t or "")


def _slim(case: dict) -> dict:
    """Only what the page shows; keeps the inlined payload small."""
    top = sorted(case["posterior"].items(), key=lambda kv: -kv[1])[:3]
    drafts = {}
    for name, d in case["drafts"].items():
        j = d["judge"]
        drafts[name] = {
            "text": _unescape(d["text"]),
            "error": d["error"],
            "placeholder": d["placeholder"],
            "cited": d["grounded_in"],
            "judge": None
            if j is None
            else {
                k: j[k]
                for k in (
                    "grounded",
                    "addresses",
                    "tone",
                    "safe",
                    "sendable",
                    "worst_problem",
                    "parse_failed",
                )
            },
        }
    return {
        "id": case["id"],
        "text": _unescape(case["text"]),
        "gold": case["gold_intent"],
        "ambiguous": case["ambiguous"],
        "brand_reply": _unescape(case["brand_reply"]),
        "pred": case["pred_intent"],
        "top": top,
        "decision": case["decision"],
        "evidence": [
            {**e, "text": _unescape(e["text"]), "reply": _unescape(e["reply"])}
            for e in case["evidence"]
        ],
        "drafts": drafts,
    }


def _repo_from_remote() -> str | None:
    """owner/name from the origin remote, for the page's links to code and report."""
    try:
        url = subprocess.run(
            ["git", "remote", "get-url", "origin"], capture_output=True, text=True, cwd=ROOT
        ).stdout.strip()
    except OSError:
        return None
    m = re.search(r"github\.com[:/]([^/]+/[^/]+?)(?:\.git)?$", url)
    return m.group(1) if m else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--repo", help="GitHub owner/name for links; default: the origin remote")
    ap.add_argument("--out", type=Path, default=OUT, help="default: site/dist/index.html")
    args = ap.parse_args()
    repo = args.repo or _repo_from_remote()
    if not repo:
        print("warning: no --repo and no GitHub origin remote; links will point to OWNER/REPO")
        repo = "OWNER/REPO"

    brand = settings.brand.lower()
    results = json.loads((ROOT / "reports" / "results.json").read_text(encoding="utf-8"))
    naive = json.loads((ROOT / "reports" / "results_naive.json").read_text(encoding="utf-8"))
    cases = _read_jsonl(Paths.processed / f"{brand}_demo_cases.jsonl")
    annot = json.loads((ROOT / "reports" / "annotator_agreement.json").read_text(encoding="utf-8"))
    gold, pred = [c["gold_intent"] for c in cases], [c["pred_intent"] for c in cases]
    k, n = cohens_kappa(gold, pred)
    classifier = {
        "b": "the classifier, out of fold",
        "n": n,
        "raw": sum(g == p for g, p in zip(gold, pred, strict=True)) / n,
        "kappa": k,
        "ci": list(kappa_ci(gold, pred, seed=GLOBAL_SEED)),
    }
    tax = json.loads((ROOT / "src" / "support_agent" / "taxonomy.json").read_text(encoding="utf-8"))

    payload = {
        "brand": settings.brand,
        "git_sha": git_sha(),
        "built_at": datetime.now(UTC).strftime("%Y-%m-%d"),
        "results": {k: results[k] for k in ("intent", "routing", "routing_sweep", "per_class")},
        "drafting": {k: results["drafting"][k] for k in ("n", "judge", "arms", "sendable_mcnemar")},
        "waterfall": naive["naive_waterfall"],
        "agreement": {"annotators": annot["pairs"], "classifier": classifier},
        "intents": {i["id"]: i["description"] for i in tax["intents"]},
        "cases": [_slim(c) for c in cases],
    }
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # Inlined inside <script type="application/json">: a literal "</" would end it.
    data = data.replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("__REPO__", repo)
    html = html.replace("__DATA__", data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8", newline="\n")
    print(f"{len(cases)} cases, {len(html) / 1e6:.2f} MB -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
