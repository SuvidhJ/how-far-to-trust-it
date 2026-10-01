"""Merge hand-written labels into the committed golden set.

    uv run python scripts/build_golden.py      # labels in data/golden/labels/labels_batch*.txt

Labels are supplied as `<index> <LETTER> [?]` lines, where the index is the row
position in the sampling frame, the letter is the taxonomy option, and a trailing
`?` marks the example as ambiguous. They were written by hand, reading the
customer message only - the brand's reply was never displayed during labelling,
because seeing the answer is the fastest way to talk yourself into a label.

The output is append-only and committed. Rebuilding it from the same inputs is
byte-identical; changing a label requires changing the label file, which shows
up in `git diff` and belongs in the decision log.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from support_agent.config import Paths, settings
from support_agent.data import write_manifest
from support_agent.intents import load_taxonomy

LABELLER = "author (single annotator)"

# Integrity note. Before labelling, a smoke test printed the classifier's
# prediction for eight dev messages, and one of those later landed in the
# sampling frame. My label for it may have been anchored by what I had seen, so
# it is flagged here and excluded from the headline metric. The effect is 1 row
# in 220; the disclosure matters more than the magnitude.
ANCHORED_IDS = {2749716}

# Thread leakage. The sampler originally deduplicated by row, not by thread, so
# one conversation contributed two consecutive messages (2677499 and 2677500) via
# different strata. Split across cross-validation folds they are near-identical
# train/test twins. `sample_golden.py` now enforces one row per thread; this
# frame predates that fix and the golden set is append-only, so the later message
# is excluded here instead of being silently deleted.
THREAD_DUP_IDS = {2677500}
EXCLUDED = {
    **{i: "anchored" for i in ANCHORED_IDS},
    **{i: "thread_duplicate" for i in THREAD_DUP_IDS},
}


def read_labels(patterns: list[str]) -> dict[int, tuple[str, bool]]:
    out: dict[int, tuple[str, bool]] = {}
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if not parts:
                    continue
                idx, letter = int(parts[0]), parts[1].upper()
                ambiguous = len(parts) > 2 and parts[2] == "?"
                if idx in out:
                    raise ValueError(f"duplicate label for index {idx}")
                out[idx] = (letter, ambiguous)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--brand", default=settings.brand)
    ap.add_argument(
        "--labels", nargs="+", default=[str(Paths.golden / "labels" / "labels_batch*.txt")]
    )
    args = ap.parse_args()

    tax = load_taxonomy()
    frame_path = Paths.golden / f"{args.brand.lower()}_frame.jsonl"
    frame = [json.loads(line) for line in frame_path.read_text(encoding="utf-8").splitlines()]
    labels = read_labels(args.labels)

    missing = set(range(len(frame))) - set(labels)
    if missing:
        raise SystemExit(f"{len(missing)} rows unlabelled: {sorted(missing)[:20]}")

    rows = []
    for i, rec in enumerate(frame):
        letter, ambiguous = labels[i]
        intent = tax.by_letter(letter)
        if intent is None:
            raise SystemExit(f"row {i}: letter {letter!r} is not in the taxonomy")
        rows.append(
            {
                "id": rec["id"],
                "text": rec["text"],
                "intent": intent.id,
                "ambiguous": ambiguous,
                "anchored": rec["id"] in ANCHORED_IDS,
                "excluded": EXCLUDED.get(rec["id"], ""),
                "stratum": rec["stratum"],
                "split": rec["split"],
                "thread_id": rec["thread_id"],
                "brand_reply": rec["brand_reply"],
                "labelled_by": LABELLER,
                "taxonomy_version": tax.version,
            }
        )

    path = Paths.golden / f"{args.brand.lower()}_golden.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["intent"]] = counts.get(r["intent"], 0) + 1
    n_amb = sum(r["ambiguous"] for r in rows)

    write_manifest(
        path,
        brand=args.brand,
        rows=len(rows),
        taxonomy_version=tax.version,
        labeller=LABELLER,
        label_counts=counts,
        ambiguous=n_amb,
        excluded={str(k): v for k, v in sorted(EXCLUDED.items())},
        protocol=(
            "Labelled from the customer message alone; the brand reply was hidden. "
            "Taxonomy frozen before labelling began. Single annotator, so "
            "inter-annotator agreement is not available and is reported as a limitation."
        ),
    )

    n_ex = sum(bool(r["excluded"]) for r in rows)
    print(
        f"{path}  {len(rows)} labelled, {n_amb} ambiguous ({n_amb / len(rows):.0%}), "
        f"{n_ex} excluded from metrics -> n={len(rows) - n_ex} scored"
    )
    width = max(len(k) for k in counts)
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<{width}} {v:>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
