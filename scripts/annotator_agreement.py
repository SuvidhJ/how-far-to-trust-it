"""How much of the remaining error is irreducible task ambiguity?

    uv run python scripts/annotator_agreement.py

A label set needs agreement evidence. The obvious route - re-label a slice and
compute intra-annotator kappa - was tried and produced kappa = 1.00 on n=45,
which is worthless: both passes happened in one working session, so it measures
recall of my own earlier decisions, not the rubric's clarity. Reporting 1.00 as
a reliability figure would be exactly the inflated claim this project exists
to avoid.

Agreement is triangulated instead. Independent models, none of them the deployed
classifier, label the same messages from the same written rubric I used. If they
agree with each other about as much as either agrees with me, the disagreement
lives in the task rather than in my idiosyncrasy, and that level is a realistic
ceiling for any classifier scored against these labels.

This is model-human agreement, not agreement between two people. It is a weaker
claim and is labelled as such wherever it appears.

Two implementation points that materially change the numbers:

* **Reasoning models do not emit a bare letter.** With thinking on,
  `nemotron-3-super-120b` opens "We need to label" and its token 0 puts ~98% on
  "We". An earlier version read the label off the letter tokens anyway, which held
  under 1% of the probability, and reported that as agreement (#51). Now a label
  needs >=50% of token-0 mass on the letters, or a reply that is a bare letter, and
  nemotron is asked with thinking off.
* **Agreement is pairwise-complete.** A model that fails to parse on 12% of rows
  is still a valid annotator on the other 88%. Dropping it entirely throws away
  most of the evidence; imputing its failures as labels would fabricate
  agreement. Each pair therefore reports its own n.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from support_agent.config import GLOBAL_SEED, Paths, settings
from support_agent.eval.metrics import cohens_kappa, kappa_band, kappa_ci
from support_agent.intents import LLMClassifier, load_taxonomy
from support_agent.llm import LLM, ROLE_EXTRA_BODY

HUMAN = "human (me)"

# tag -> (provider, model, read_label_from_logprobs)
# Deliberately excludes the deployed classifier: agreement with the system under
# test would measure the system, not the task.
# tag -> (provider, model, read_label_from_logprobs, extra_body)
# nemotron is a reasoning model: with thinking on it opens "We need to label..." and
# never emits a letter, so its 0.466 came from letter tokens holding <1% of the
# probability (#51). It is asked with thinking off, as the drafter is.
ANNOTATORS: dict[str, tuple[str, str, bool, dict | None]] = {
    "nemotron-120b": (
        "nvidia",
        "nvidia/nemotron-3-super-120b-a12b",
        True,
        ROLE_EXTRA_BODY["drafter"],
    ),
    "gpt-oss-20b": ("nvidia", "openai/gpt-oss-20b", True, None),
    "ministral-3b": ("mistral", "ministral-3b-latest", False, None),
    "gemini-3.6-flash": ("gemini", "gemini-3.6-flash", False, None),
}
MAX_FAIL_RATE = 0.5  # above this an annotator is broken, not merely lossy


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=None, help="first n rows; default all (#47)")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "reports" / "annotator_agreement.json",
        help="pairwise results as JSON (the demo page reads them)",
    )
    args = ap.parse_args()

    tax = load_taxonomy()
    path = Paths.golden / f"{settings.brand.lower()}_golden.jsonl"
    gold = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    gold = [r for r in gold if not r.get("excluded")][: args.n]
    texts = [r["text"] for r in gold]

    # None marks "this annotator produced nothing usable for this row".
    labels: dict[str, list[str | None]] = {HUMAN: [r["intent"] for r in gold]}
    for tag, (provider, model, use_lp, extra) in ANNOTATORS.items():
        clf = LLMClassifier(
            llm=LLM(provider, model, role=f"annotator:{tag}"),
            taxonomy=tax,
            use_logprobs=use_lp,
            extra_body=extra,
        )
        # Row by row: one 503 from a free tier must not discard every answer already
        # received. A failed call is counted and left unlabelled; rerunning fills it
        # from the cache plus the calls that failed.
        row_labels: list[str | None] = []
        api_errors = parse_fails = 0
        last_error = ""
        for text in texts:
            try:
                pred = clf.predict([text])[0]
            except Exception as exc:
                api_errors += 1
                last_error = f"{type(exc).__name__}: {str(exc)[:80]}"
                row_labels.append(None)
                if "quota" in str(exc).lower() or "not set" in str(exc):
                    break  # an exhausted quota or a missing key will not recover mid-run
                if api_errors >= 20 and api_errors == len(row_labels):
                    break  # the endpoint is down, not flaky
                continue
            parse_fails += pred.parse_failed
            row_labels.append(None if pred.parse_failed else pred.intent)
        row_labels += [None] * (len(texts) - len(row_labels))
        rate = (api_errors + parse_fails) / len(texts)
        print(
            f"{tag:<20} api_errors={api_errors} parse_failures={parse_fails} of {len(texts)}"
            + (f"  (last error {last_error})" if api_errors else "")
        )
        if rate > MAX_FAIL_RATE:
            print(f"{'':<20} dropped: mostly failures, not judgements")
            continue
        labels[tag] = row_labels

    print("\npairwise Cohen's kappa (complete pairs only):")
    print(f"{'pair':<44}{'n':>5}{'raw':>8}{'kappa':>8}  {'95% bootstrap':<16}band")
    print("-" * 94)
    rows: list[dict] = []
    for a, b in combinations(labels, 2):
        both = [
            (x, y)
            for x, y in zip(labels[a], labels[b], strict=True)
            if x is not None and y is not None
        ]
        if len(both) < 20:
            print(f"{a + ' vs ' + b:<44}{len(both):>5}  too few complete pairs")
            continue
        xs, ys = [p[0] for p in both], [p[1] for p in both]
        k, n = cohens_kappa(xs, ys)
        raw = sum(x == y for x, y in zip(xs, ys, strict=True)) / n
        lo, hi = kappa_ci(xs, ys, seed=GLOBAL_SEED)
        ci = f"[{lo:.2f}, {hi:.2f}]"
        print(f"{a + ' vs ' + b:<44}{n:>5}{raw:>8.3f}{k:>8.3f}  {ci:<16}{kappa_band(k)}")
        rows.append({"a": a, "b": b, "n": n, "raw": raw, "kappa": k, "ci": [lo, hi]})
    if args.n is None:
        args.out.write_text(json.dumps({"pairs": rows}, indent=2) + "\n", encoding="utf-8")

    models = [m for m in labels if m != HUMAN]
    if not models:
        print("\nno usable annotators")
        return 0

    complete = [i for i in range(len(texts)) if all(labels[m][i] is not None for m in labels)]
    if complete:
        unan = sum(len({labels[m][i] for m in labels}) == 1 for i in complete)
        print(f"\nrows all {len(labels)} annotators labelled: {len(complete)}")
        print(f"  unanimous: {unan}/{len(complete)} ({unan / len(complete):.0%})")

        flagged = [i for i in complete if gold[i]["ambiguous"]]
        rest = [i for i in complete if not gold[i]["ambiguous"]]
        if flagged and rest:
            uf = sum(len({labels[m][i] for m in labels}) == 1 for i in flagged) / len(flagged)
            ur = sum(len({labels[m][i] for m in labels}) == 1 for i in rest) / len(rest)
            print(f"  unanimity on rows I flagged ambiguous: {uf:.0%} (n={len(flagged)})")
            print(f"  unanimity on the rest:                 {ur:.0%} (n={len(rest)})")
            print("  (a meaningful flag makes the first number much lower)")

    print("\nmost common human/model disagreements:")
    pairs: Counter = Counter()
    for i in range(len(texts)):
        for m in models:
            if labels[m][i] is not None and labels[m][i] != labels[HUMAN][i]:
                pairs[(labels[HUMAN][i], labels[m][i])] += 1
    for (h, m), c in pairs.most_common(10):
        print(f"  human={h:<22} model={m:<22} {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
