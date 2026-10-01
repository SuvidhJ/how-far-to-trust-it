"""Judge-human agreement. A judge without it is decoration.

Two steps, run separately so the scoring is genuinely blind.

    uv run python scripts/judge_agreement.py --emit    # write the blind sheet
    uv run python scripts/judge_agreement.py --score   # compare to my scores

`--emit` writes customer message, draft and evidence to
`data/golden/<brand>_judge_sheet.txt`, with the drafter's identity and the
judge's scores withheld, in an order shuffled by a fixed seed. A human scores each
item on the same four-dimension rubric the judge uses (`prompts/judge_v1.txt`) and
writes one line per item to `data/golden/<brand>_judge_human_scores.txt`:

    <idx> <grounded> <addresses> <tone> <safe>      each 0-2

`--score` then computes Cohen's kappa per dimension, with a bootstrap interval.

Only *valid* drafts are sampled. An API failure or a leaked chain-of-thought is
not a reply, so asking a human and a judge whether they agree on its quality
would measure nothing (DECISIONS #34).

Kappa on a 3-point ordinal scale is strict - it treats a 2-vs-1 disagreement as
badly as 2-vs-0 - so quadratic-weighted kappa is reported beside it. Kappa also
collapses when one score dominates (most drafts are `safe=2`), so Gwet's AC1 is
reported too. Within-one agreement and agreement on the binary decision that
matters operationally (is this draft sendable without an edit) complete it.

The scores must come from a person. A model filling in `human_scores.txt` would
turn this into judge-vs-model agreement and the report would be claiming
something it did not measure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from support_agent.config import GLOBAL_SEED, Paths, settings
from support_agent.draft import build as build_drafter
from support_agent.embed import Embedder
from support_agent.eval.harness import corpus_retriever, draft_sample
from support_agent.eval.judge import DIMENSIONS, Judge
from support_agent.eval.metrics import cohens_kappa, gwet_ac1, kappa_band, kappa_ci, weighted_kappa
from support_agent.intents import oof_predictions
from support_agent.retrieve import Hit, search_batch

# Committed beside the golden set, so anyone can replay the agreement numbers.
SCORES_PATH = Paths.golden / f"{settings.brand.lower()}_judge_human_scores.txt"
SHEET_PATH = Paths.golden / f"{settings.brand.lower()}_judge_sheet.txt"
# The exact items behind the sheet. --score reads these back instead of rebuilding
# the sample, so a changed cache or code path cannot re-pair scores with drafts.
ITEMS_PATH = Paths.golden / f"{settings.brand.lower()}_judge_items.json"


def item_hash(message: str, draft: str) -> str:
    return hashlib.sha256(f"{message}\n{draft}".encode()).hexdigest()[:10]


DRAFTERS = ("nearest", "grounded_llm")


def build_sample(n: int, k: int):
    gold_path = Paths.golden / f"{settings.brand.lower()}_golden.jsonl"
    gold = [json.loads(line) for line in gold_path.read_text(encoding="utf-8").splitlines()]
    gold = [r for r in gold if not r.get("excluded")]
    texts = [r["text"] for r in gold]
    y = np.array([r["intent"] for r in gold])

    emb = Embedder()
    X = emb.encode(texts)
    pred, _ = oof_predictions(X, y)
    retriever = corpus_retriever(emb)
    # The same 50 messages as the drafting bake-off, so the judge calls replay.
    pick, sample, _, hits = draft_sample(
        len(gold), 50, texts, lambda m: search_batch(retriever, m, k=k), pred
    )

    items, invalid = [], 0
    for name in DRAFTERS:
        d = build_drafter(name)
        for j, i in enumerate(pick):
            out = d.draft(sample[j], pred[i], hits[j])
            if out.error:
                invalid += 1
                continue
            items.append(
                {"drafter": name, "message": sample[j], "draft": out.text, "hits": hits[j]}
            )
    # Shuffled so the drafter cannot be inferred from position, and truncated to n.
    random.Random(4242).shuffle(items)
    return items[:n], invalid


def agreement_rows(
    human: dict[int, dict[str, int]],
    machine: dict[int, dict[str, int]],
    groups: dict[int, str] | None = None,
) -> list[dict]:
    """Per-dimension and binary-sendable agreement over the items both scored.

    `groups` maps an item to its customer message; intervals resample messages.
    """
    idxs = sorted(set(human) & set(machine))
    g = [groups[i] for i in idxs] if groups else None
    rows = []
    for dim in DIMENSIONS:
        h = [str(human[i][dim]) for i in idxs]
        m = [str(machine[i][dim]) for i in idxs]
        k, n = cohens_kappa(h, m)
        rows.append(
            {
                "dimension": dim,
                "n": n,
                "exact": sum(a == b for a, b in zip(h, m, strict=True)) / n if n else float("nan"),
                "within1": (
                    sum(abs(int(a) - int(b)) <= 1 for a, b in zip(h, m, strict=True)) / n
                    if n
                    else float("nan")
                ),
                "kappa": k,
                "wkappa": weighted_kappa([int(v) for v in h], [int(v) for v in m]),
                "ac1": gwet_ac1(h, m),
                "ci": kappa_ci(h, m, seed=GLOBAL_SEED, groups=g),
            }
        )

    def sendable(s: dict[str, int]) -> str:
        return "yes" if s["safe"] == 2 and s["grounded"] == 2 and s["addresses"] >= 1 else "no"

    hs = [sendable(human[i]) for i in idxs]
    ms = [sendable(machine[i]) for i in idxs]
    k, n = cohens_kappa(hs, ms)
    rows.append(
        {
            "dimension": "sendable (binary)",
            "n": n,
            "exact": sum(a == b for a, b in zip(hs, ms, strict=True)) / n if n else float("nan"),
            "within1": float("nan"),
            "kappa": k,
            "wkappa": float("nan"),
            "ac1": gwet_ac1(hs, ms),
            "ci": kappa_ci(hs, ms, seed=GLOBAL_SEED, groups=g),
            "human_yes": hs.count("yes"),
            "judge_yes": ms.count("yes"),
        }
    )
    return rows


def parse_scores(text: str) -> dict[int, dict[str, int]]:
    human: dict[int, dict[str, int]] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0].startswith("#"):
            continue
        vals = [int(x) for x in parts[1:5]]
        if any(v not in (0, 1, 2) for v in vals):
            raise ValueError(f"scores must be 0-2: {line!r}")
        human[int(parts[0])] = dict(zip(DIMENSIONS, vals, strict=True))
    return human


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()
    if not (args.emit or args.score):
        ap.error("pass --emit or --score")

    judge = Judge()

    if args.emit:
        items, invalid = build_sample(args.n, args.k)
        records = []
        SHEET_PATH.parent.mkdir(exist_ok=True)
        with SHEET_PATH.open("w", encoding="utf-8") as fh:
            fh.write(
                "# Score each draft 0-2 on grounded, addresses, tone, safe (rubric:\n"
                "# src/support_agent/prompts/judge_v1.txt). Do not look at judge output first.\n"
                "# The evidence lines are exactly the ones the judge saw, in its order.\n"
                f"# Write lines '<idx> <grounded> <addresses> <tone> <safe>' to data/golden/{SCORES_PATH.name}\n\n"
            )
            for idx, it in enumerate(items):
                sha = item_hash(it["message"], it["draft"])
                fh.write(
                    f"=== {idx} ({sha}) ===\ncustomer: {it['message']}\ndraft:    {it['draft']}\n"
                )
                for h in judge.evidence(it["draft"], it["hits"]):
                    fh.write(f"evidence: {h.reply}\n")
                fh.write("\n")
                records.append(
                    {
                        "idx": idx,
                        "sha": sha,
                        "message": it["message"],
                        "draft": it["draft"],
                        "hits": [asdict(h) for h in it["hits"]],
                    }
                )
        ITEMS_PATH.write_text(
            json.dumps(records, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"{len(items)} blind items -> {SHEET_PATH}  ({invalid} invalid drafts excluded)")
        print(f"score into {SCORES_PATH}, then run --score")
        return 0

    records = json.loads(ITEMS_PATH.read_text(encoding="utf-8"))
    sheet = SHEET_PATH.read_text(encoding="utf-8")
    stale = [r["idx"] for r in records if f"=== {r['idx']} ({r['sha']}) ===" not in sheet]
    if stale:
        raise SystemExit(f"sheet and {ITEMS_PATH.name} disagree on items {stale}; re-emit")
    items = [
        {"message": r["message"], "draft": r["draft"], "hits": [Hit(**h) for h in r["hits"]]}
        for r in records
    ]

    if not SCORES_PATH.exists():
        raise SystemExit(f"{SCORES_PATH} not found; run --emit and score by hand first")
    human = parse_scores(SCORES_PATH.read_text(encoding="utf-8"))
    if not human:
        raise SystemExit(f"{SCORES_PATH} has no scored lines yet")

    machine = {}
    failed = 0
    for idx, it in enumerate(items):
        if idx not in human:
            continue
        s = judge.score(it["message"], it["draft"], it["hits"])
        if s.parse_failed:
            failed += 1  # excluded, and counted, rather than scored as zeros
            continue
        machine[idx] = {d: getattr(s, d) for d in DIMENSIONS}
    print(
        f"judge: {judge.llm.model} rubric={judge.version}  human-scored={len(human)}  "
        f"judge failures excluded={failed}\n"
    )

    rows = agreement_rows(human, machine, groups={i: items[i]["message"] for i in machine})
    print(
        f"{'dimension':<20}{'n':>4}{'exact':>7}{'within1':>9}{'kappa':>8}{'95% CI':>18}"
        f"{'w-kappa':>9}{'AC1':>7}  band"
    )
    print("-" * 94)
    for r in rows:
        w = "" if np.isnan(r["within1"]) else f"{r['within1']:.2f}"
        ci = f"[{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]"
        print(
            f"{r['dimension']:<20}{r['n']:>4}{r['exact']:>7.2f}{w:>9}{r['kappa']:>8.3f}"
            f"{ci:>18}{'' if np.isnan(r['wkappa']) else format(r['wkappa'], '.3f'):>9}"
            f"{r['ac1']:>7.3f}  {kappa_band(r['kappa'])}"
        )
    s = rows[-1]
    print(
        f"\nhuman says sendable: {s['human_yes']}/{s['n']};  judge says: {s['judge_yes']}/{s['n']}"
    )

    diffs = [
        i for i in sorted(machine) if sum(abs(human[i][d] - machine[i][d]) for d in DIMENSIONS) >= 3
    ]
    print(f"\n{len(diffs)} drafts where human and judge differ by 3+ points in total:")
    for i in diffs[:6]:
        print(f"  [{i}] human={human[i]}  judge={machine[i]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
