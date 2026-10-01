"""Raw CSV to brand-scoped, thread-aware exchanges.

The one design decision worth defending here: **a conversation is a connected
component of the reply graph**, computed once over all 2.8M tweets, not a chain
walked upward from a single tweet. Two reasons.

* `in_response_to_tweet_id` chains fan out. 139,508 parents have exactly two
  replies and 32,969 have three or more, because brands split long answers
  across tweets. Walking up from each brand tweet independently would assign
  sibling replies to different "threads".
* Splitting train/test by tweet leaks: the customer's follow-up and the original
  complaint share vocabulary and often the same answer. Components give a
  grouping that a `GroupShuffleSplit` can honour, and a test asserts disjointness.

Every artifact written here has a sibling `.manifest.json` recording the source,
row count, seed, filters and git SHA. Downstream code reads the manifest.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from support_agent.config import GLOBAL_SEED, ROOT, Paths
from support_agent.text import clean, detect_lang, strip_signature

# `created_at` is stored as VARCHAR in Twitter's own format. Sorting it as text
# orders by weekday name ("Fri" < "Mon"), which silently corrupts any "first
# seen" or chronological split. Parse it, and pin the session to UTC so the same
# query returns the same bytes on every machine.
TS = "CAST(strptime(created_at, '%a %b %d %H:%M:%S %z %Y') AS TIMESTAMP)"


def connect(raw: Path | None = None) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    src = (raw or Paths.raw).as_posix()
    con.execute(f"CREATE VIEW twcs AS SELECT * FROM read_csv_auto('{src}', sample_size=200000)")
    return con


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Paths.raw.parents[2],
        ).stdout.strip()
    except Exception:
        return "unknown"


def write_manifest(path: Path, **fields) -> None:
    """Record how an artifact was produced, beside the artifact."""
    manifest = {
        "artifact": path.name,
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": Paths.raw.relative_to(ROOT).as_posix(),
        "global_seed": GLOBAL_SEED,
        "git_sha": git_sha(),
        **fields,
    }
    path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )


def assign_thread_ids(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Label every tweet with the connected component of the reply graph.

    Returns a frame of `tweet_id, thread_id`. Runs over the whole file once;
    the result is what makes a leak-free split possible.
    """
    edges = (
        con.execute("SELECT tweet_id, in_response_to_tweet_id FROM twcs")
        .fetch_arrow_table()
        .to_pandas()
    )

    ids = edges["tweet_id"].to_numpy(dtype=np.int64)
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]

    parent = edges["in_response_to_tweet_id"].to_numpy(dtype="float64")
    has_parent = ~np.isnan(parent)
    parent_i = np.searchsorted(sorted_ids, parent[has_parent].astype(np.int64))
    # A parent id can point outside the dump; drop those edges rather than
    # silently linking to whatever sits at that index.
    parent_i = np.clip(parent_i, 0, len(sorted_ids) - 1)
    valid = sorted_ids[parent_i] == parent[has_parent].astype(np.int64)

    child_pos = np.arange(len(ids))[has_parent][valid]
    child_i = np.searchsorted(sorted_ids, ids[child_pos])
    parent_i = parent_i[valid]

    n = len(ids)
    g = coo_matrix((np.ones(len(child_i), dtype=np.int8), (child_i, parent_i)), shape=(n, n))
    _, labels = connected_components(g, directed=False)

    out = pd.DataFrame({"tweet_id": sorted_ids, "thread_id": labels})
    return out


@dataclass(frozen=True)
class ExtractStats:
    """Counts dropped at each filter, so the report can state them honestly."""

    brand_replies: int = 0
    paired_with_customer: int = 0
    after_merge_split_replies: int = 0
    dropped_non_english: int = 0
    dropped_empty: int = 0
    final: int = 0


def build_exchanges(brand: str, con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    """Customer message -> brand response, one row per exchange.

    An *exchange* is a customer tweet together with every brand tweet that
    replies directly to it, concatenated in timestamp order. That merge is the
    fix for split replies: without it, a two-tweet answer becomes two training
    rows whose targets are each half an answer.
    """
    con = con or connect()
    threads = assign_thread_ids(con)
    con.register("threads", threads)

    raw = (
        con.execute(
            f"""
        WITH brand_reply AS (
            SELECT tweet_id, in_response_to_tweet_id AS parent_id, text AS brand_text,
                   {TS} AS brand_at
            FROM twcs
            WHERE author_id = ? AND NOT inbound AND in_response_to_tweet_id IS NOT NULL
        ),
        cust AS (
            SELECT tweet_id, author_id, text AS customer_text, {TS} AS customer_at,
                   in_response_to_tweet_id AS cust_parent
            FROM twcs WHERE inbound
        )
        SELECT c.tweet_id       AS customer_tweet_id,
               c.author_id      AS customer_id,
               c.customer_text,
               c.customer_at,
               c.cust_parent IS NULL AS is_first_contact,
               t.thread_id,
               list(b.tweet_id ORDER BY b.brand_at, b.tweet_id) AS brand_tweet_ids,
               list(b.brand_text ORDER BY b.brand_at, b.tweet_id) AS brand_parts,
               min(b.brand_at) AS brand_at
        FROM brand_reply b
        JOIN cust c ON b.parent_id = c.tweet_id
        JOIN threads t ON t.tweet_id = c.tweet_id
        GROUP BY ALL
        """,
            [brand],
        )
        .fetch_arrow_table()
        .to_pandas()
    )

    raw["n_reply_parts"] = raw["brand_parts"].map(len)
    raw["brand_text"] = raw["brand_parts"].map(
        lambda parts: " ".join(strip_signature(p) for p in parts)
    )
    raw["brand_tweet_ids"] = raw["brand_tweet_ids"].map(list)
    raw = raw.drop(columns=["brand_parts"])

    raw["customer_clean"] = raw["customer_text"].map(clean)
    raw["brand_clean"] = raw["brand_text"].map(clean)
    raw["lang"] = raw["customer_text"].map(detect_lang)
    raw["response_minutes"] = (
        (raw["brand_at"] - raw["customer_at"]).dt.total_seconds() / 60
    ).round(2)

    return raw.sort_values("customer_tweet_id", kind="stable").reset_index(drop=True)


# Filters applied before anything is modelled. Each one is counted and the
# counts go in the manifest, so the report can state what was thrown away.
MIN_CUSTOMER_CHARS = 15
MIN_BRAND_CHARS = 20


def filter_exchanges(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop rows no model should see, and say how many of each."""
    counts = {"input": len(df)}

    keep = df["lang"].eq("en")
    counts["dropped_non_english"] = int((~keep).sum())
    df = df[keep]

    keep = df["customer_clean"].str.len().ge(MIN_CUSTOMER_CHARS) & df["brand_clean"].str.len().ge(
        MIN_BRAND_CHARS
    )
    counts["dropped_too_short"] = int((~keep).sum())
    df = df[keep]

    # A handful of exchanges have a brand timestamp before the customer's,
    # which means the reply graph edge is wrong. They are rare and unusable.
    keep = df["response_minutes"].ge(0)
    counts["dropped_negative_latency"] = int((~keep).sum())
    df = df[keep]

    counts["output"] = len(df)
    return df.reset_index(drop=True), counts


def add_thread_splits(
    df: pd.DataFrame, seed: int, test_frac: float = 0.2, dev_frac: float = 0.1
) -> pd.DataFrame:
    """Assign train/dev/test **by thread**, never by row.

    One uniform draw per thread, in sorted thread-id order. This is *not* a hash
    of the thread id, and stability is narrower than that would give: appending
    threads with higher ids keeps every existing assignment, but a change that
    removes or inserts a thread (a different filter, say) reshuffles every thread
    after it. The same raw file and filters reproduce the split exactly, which is
    what the committed golden set and index rely on (checked 2026-09-15: stored
    and recomputed splits are identical; golden and index threads are disjoint).
    """
    rng = np.random.default_rng(seed)
    threads = np.sort(df["thread_id"].unique())
    draw = rng.random(len(threads))
    split = np.where(
        draw < test_frac, "test", np.where(draw < test_frac + dev_frac, "dev", "train")
    )
    return df.merge(
        pd.DataFrame({"thread_id": threads, "split": split}), on="thread_id", how="left"
    )
