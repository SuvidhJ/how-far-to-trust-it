"""Draw the golden-set sampling frame.

    uv run python scripts/sample_golden.py --n-per-stratum 20 --n-random 40

Uniform random sampling is the obvious approach and it is wrong here: the rarest
intent is 2.6% of traffic, so 200 uniform draws give it about five examples and a
confidence interval wide enough to cover anything. We stratify instead.

The strata are **crude keyword rules**, not model predictions. That matters twice
over. Using a model to choose what to label would make the golden set agree with
the model by construction. Using keywords instead introduces a different and
*disclosable* bias: messages that phrase an intent without its obvious keywords
are under-represented. The `random` stratum exists to measure that bias — it is
drawn with no keyword filter at all, so comparing label distributions between the
keyword strata and the random stratum estimates how much the frame distorts.

Sampled from **dev and test threads only**. Train stays untouched so a supervised
baseline can never have seen a golden thread.

This script writes an *unlabelled* frame. Labelling is a separate, manual step
(`scripts/build_golden.py`), so that no label is ever produced by a model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from support_agent.config import Paths, seed_for, settings
from support_agent.data import write_manifest

# Deliberately crude. These pick *candidates* for a stratum; they never assign a
# label. Their only job is to make rare intents reachable.
STRATUM_RULES: dict[str, str] = {
    "report_disruption": r"delay|cancel|diverted|stuck on|tarmac|missed connect|grounded|stranded",
    "rebook_reroute": r"rebook|re-book|put me on|another flight|next flight|standby|change my flight|reroute",
    "track_baggage": r"\bbag\b|\bbags\b|baggage|luggage|suitcase|carry.?on",
    "refund_compensation": r"refund|compensat|reimburs|voucher|money back",
    "manage_booking": r"seat|upgrade|check.?in|boarding pass|name change|reservation",
    "ask_policy": r"policy|allowed|\bfee\b|rules|how do i|can i\b|do you allow",
    "complain_service": r"rude|awful|terrible|worst|unaccept|disgrace|never fly|poor service",
    "praise_service": r"thank|kudos|shout ?out|great job|amazing|excellent|best airline",
    "loyalty_program": r"aadvantage|\bmiles\b|elite|executive plat|\bgold\b|\bplatinum\b|group \d",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--brand", default=settings.brand)
    ap.add_argument("--n-per-stratum", type=int, default=20)
    ap.add_argument("--n-random", type=int, default=40)
    args = ap.parse_args()

    df = pd.read_parquet(Paths.interim / f"{args.brand.lower()}_exchanges.parquet")
    pool = df[df["split"].isin(["dev", "test"])].copy()
    text = pool["customer_clean"].str.lower()

    seed = seed_for(f"golden:{args.brand}")
    picked: list[pd.DataFrame] = []
    taken: set[int] = set()

    # Threads already represented. Two messages from one conversation are near
    # duplicates of each other; if they land in different cross-validation folds
    # the model is effectively tested on its own training data. Excluding by row
    # is not enough - the constraint is one row per *thread*.
    used_threads: set[int] = set()

    def take_from(cand: pd.DataFrame, n: int, name: str) -> None:
        cand = cand[~cand.index.isin(taken) & ~cand["thread_id"].isin(used_threads)]
        # Draw at most one row per thread before sampling, so a chatty thread
        # cannot crowd out a stratum.
        cand = cand.sample(frac=1.0, random_state=seed).drop_duplicates("thread_id")
        chosen = cand.sample(n=min(n, len(cand)), random_state=seed)
        taken.update(chosen.index)
        used_threads.update(chosen["thread_id"])
        picked.append(chosen.assign(stratum=name))

    # Keyword strata first, so rare intents are reachable at all.
    for name, pattern in STRATUM_RULES.items():
        take_from(pool[text.str.contains(pattern, regex=True, na=False)], args.n_per_stratum, name)

    # The control stratum: no filter, so it reflects real traffic and can be used
    # to measure how far the keyword strata distort the distribution.
    take_from(pool, args.n_random, "random")

    frame = pd.concat(picked).sort_values("customer_tweet_id").reset_index(drop=True)

    Paths.golden.mkdir(parents=True, exist_ok=True)
    path = Paths.golden / f"{args.brand.lower()}_frame.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for r in frame.itertuples():
            fh.write(
                json.dumps(
                    {
                        "id": int(r.customer_tweet_id),
                        "thread_id": int(r.thread_id),
                        "split": r.split,
                        "stratum": r.stratum,
                        "text": r.customer_clean,
                        "brand_reply": r.brand_clean,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    write_manifest(
        path,
        brand=args.brand,
        rows=len(frame),
        sampled_from="dev+test threads only",
        strata={k: int((frame["stratum"] == k).sum()) for k in frame["stratum"].unique()},
        seed=seed,
        seed_purpose=f"golden:{args.brand}",
        note="Unlabelled sampling frame. Labels are added by hand, never by a model.",
    )
    print(f"{path}  {len(frame)} rows")
    print(frame["stratum"].value_counts().to_string())
    print(
        f"threads: {frame['thread_id'].nunique()}  splits: {frame['split'].value_counts().to_dict()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
