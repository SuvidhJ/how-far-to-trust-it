"""A local page for scoring the 40 judge-agreement items by hand.

    uv run python scripts/build_judge_page.py      # -> runs/judge_scoring.html

The page shows each item exactly as `data/golden/<brand>_judge_sheet.txt` does
(customer, draft, the evidence the judge saw, in its order) and nothing a judge
produced. Four 0-2 inputs per item, keyboard-driven, progress kept in the
browser's localStorage. "Export" gives the text for
`data/golden/<brand>_judge_human_scores.txt`, which `scripts/judge_agreement.py
--score` reads. The scores must be the author's own: nothing here fills or
suggests one. The page goes to `runs/` (gitignored) and is never deployed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from support_agent.config import ROOT, Paths, settings
from support_agent.eval.judge import DIMENSIONS

BRAND = settings.brand.lower()
SHEET = Paths.golden / f"{BRAND}_judge_sheet.txt"
ITEMS = Paths.golden / f"{BRAND}_judge_items.json"
SCORES = Paths.golden / f"{BRAND}_judge_human_scores.txt"
RUBRIC = ROOT / "src" / "support_agent" / "prompts" / "judge_v1.txt"
OUT = ROOT / "runs" / "judge_scoring.html"


def parse_sheet(text: str) -> list[dict]:
    """The sheet's items: idx, sha, customer, draft and evidence lines, in order."""
    items: list[dict] = []
    for line in text.splitlines():
        m = re.match(r"=== (\d+) \(([0-9a-f]+)\) ===$", line)
        if m:
            items.append(
                {"idx": int(m[1]), "sha": m[2], "customer": "", "draft": "", "evidence": []}
            )
        elif items and line.startswith("customer: "):
            items[-1]["customer"] = line[len("customer: ") :]
        elif items and line.startswith("draft:    "):
            items[-1]["draft"] = line[len("draft:    ") :]
        elif items and line.startswith("evidence: "):
            items[-1]["evidence"].append(line[len("evidence: ") :])
    return items


def score_header() -> list[str]:
    """The comment lines of the committed scores file, so an export can replace it."""
    return [ln for ln in SCORES.read_text(encoding="utf-8").splitlines() if ln.startswith("#")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    items = parse_sheet(SHEET.read_text(encoding="utf-8"))
    shas = {r["idx"]: r["sha"] for r in json.loads(ITEMS.read_text(encoding="utf-8"))}
    if {i["idx"]: i["sha"] for i in items} != shas:
        raise SystemExit(f"{SHEET.name} and {ITEMS.name} disagree; re-emit the sheet")

    payload = {
        "items": items,
        "dimensions": DIMENSIONS,
        "header": score_header(),
        "rubric": RUBRIC.read_text(encoding="utf-8"),
        "storage_key": f"judge-human-scores:{BRAND}:" + "".join(sorted(shas.values()))[:40],
        "scores_file": f"data/golden/{SCORES.name}",
    }
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(PAGE.replace("__DATA__", data), encoding="utf-8", newline="\n")
    print(f"{len(items)} items -> {args.out}")
    return 0


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Judge agreement: score by hand</title>
<style>
body { font: 15px/1.5 system-ui, sans-serif; max-width: 860px; margin: 24px auto; padding: 0 16px; }
.item { border: 1px solid #ccc; border-radius: 8px; padding: 12px 16px; margin: 16px 0; }
.item.done { border-color: #2a7; }
.item:focus-within { outline: 2px solid #2a78d6; }
.label { font-size: 12px; text-transform: uppercase; color: #666; margin-top: 8px; }
.draft { background: #f4f6fa; padding: 6px 8px; border-radius: 4px; }
.ev { color: #444; font-size: 14px; margin: 2px 0 2px 12px; }
.dims { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 10px; }
fieldset { border: 1px solid #ddd; border-radius: 6px; margin: 0; padding: 4px 8px; }
#bar { position: sticky; top: 0; background: #fff; padding: 8px 0; border-bottom: 1px solid #ddd; }
textarea { width: 100%; height: 160px; font: 13px monospace; }
@media (prefers-color-scheme: dark) {
  body { background: #111; color: #eee; } #bar { background: #111; }
  .draft { background: #1d2230; } .ev { color: #bbb; } .label { color: #999; }
}
</style>
</head>
<body>
<h1>Judge agreement: score by hand</h1>
<p>Score every draft 0, 1 or 2 on each dimension, using only the rubric below and what
is on screen. Keys: <kbd>0</kbd>/<kbd>1</kbd>/<kbd>2</kbd> score the focused field and move
on; <kbd>Tab</kbd> moves too. Progress is saved in this browser. Nothing here shows or
suggests what the judge said.</p>
<details><summary>Rubric (judge_v1)</summary><pre id="rubric" style="white-space:pre-wrap"></pre></details>
<div id="bar"><span id="progress"></span>
  <button id="export">Export</button> <button id="clear">Clear all</button></div>
<div id="items"></div>
<div id="out" hidden><p>Paste this over <code id="file"></code>, then run
<code>uv run python scripts/judge_agreement.py --score</code>.</p>
<textarea id="text" readonly></textarea><p><a id="dl">Download</a></p></div>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById("data").textContent);
const S = JSON.parse(localStorage.getItem(D.storage_key) || "{}");
const save = () => localStorage.setItem(D.storage_key, JSON.stringify(S));
const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  e.append(...kids);
  return e;
};
document.getElementById("rubric").textContent = D.rubric;
document.getElementById("file").textContent = D.scores_file;
const inputs = [];
function refresh() {
  let done = 0;
  for (const it of D.items) {
    const full = D.dimensions.every(d => S[it.sha] && S[it.sha][d] !== undefined);
    document.getElementById("item-" + it.idx).classList.toggle("done", full);
    done += full;
  }
  document.getElementById("progress").textContent = `${done} of ${D.items.length} scored `;
}
for (const it of D.items) {
  const box = el("section", { class: "item", id: "item-" + it.idx },
    el("strong", {}, `Item ${it.idx}`),
    el("div", { class: "label" }, "customer"), el("div", {}, it.customer),
    el("div", { class: "label" }, "draft"), el("div", { class: "draft" }, it.draft),
    el("div", { class: "label" }, "evidence the judge saw, in its order"),
    ...it.evidence.map(e => el("div", { class: "ev" }, "• " + e)));
  const dims = el("div", { class: "dims" });
  for (const d of D.dimensions) {
    const fs = el("fieldset", {}, el("legend", {}, d));
    for (const v of [0, 1, 2]) {
      const r = el("input", { type: "radio", name: `${it.sha}-${d}`, value: String(v) });
      if (S[it.sha] && S[it.sha][d] === v) r.checked = true;
      r.addEventListener("change", () => { (S[it.sha] ||= {})[d] = v; save(); refresh(); });
      fs.append(el("label", {}, r, " " + v + " "));
    }
    inputs.push(fs);
    dims.append(fs);
  }
  box.append(dims);
  document.getElementById("items").append(box);
}
document.addEventListener("keydown", ev => {
  if (!["0", "1", "2"].includes(ev.key)) return;
  const i = inputs.findIndex(fs => fs.contains(document.activeElement));
  if (i < 0) return;
  const r = inputs[i].querySelector(`input[value="${ev.key}"]`);
  r.checked = true;
  r.dispatchEvent(new Event("change"));
  ev.preventDefault();
  const next = inputs[i + 1];
  if (next) (next.querySelector("input:checked") || next.querySelector("input")).focus();
});
document.getElementById("clear").addEventListener("click", () => {
  if (!confirm("Delete every score saved in this browser?")) return;
  for (const k of Object.keys(S)) delete S[k];
  save();
  location.reload();
});
document.getElementById("export").addEventListener("click", () => {
  const lines = D.items.map(it => {
    const s = S[it.sha] || {};
    const full = D.dimensions.every(d => s[d] !== undefined);
    return full ? [it.idx, ...D.dimensions.map(d => s[d])].join(" ") : String(it.idx);
  });
  const text = [...D.header, ...lines].join("\\n") + "\\n";
  document.getElementById("text").value = text;
  const a = document.getElementById("dl");
  a.href = URL.createObjectURL(new Blob([text], { type: "text/plain" }));
  a.download = D.scores_file.split("/").pop();
  document.getElementById("out").hidden = false;
});
refresh();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
