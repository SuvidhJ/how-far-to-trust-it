#!/usr/bin/env python
"""Output-capped inspection of the raw twcs dump.

The raw CSV is 493 MB / ~2.8M rows, too large to open with a file tool or `cat`
without flooding a terminal or an editor. This script is the sanctioned way in:
DuckDB streams the file, and every result is capped by rows, cell width and total
characters so no single command can flood the output.

    uv run python scripts/peek.py --schema
    uv run python scripts/peek.py --brands 15
    uv run python scripts/peek.py --counts inbound
    uv run python scripts/peek.py --sample AppleSupport -n 5
    uv run python scripts/peek.py --sql "SELECT count(*) FROM twcs"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "twcs" / "twcs.csv"

MAX_ROWS = 20
MAX_CELL = 160
MAX_CHARS = 12_000

# created_at is a VARCHAR in Twitter's format. Comparing it as a string sorts
# "Fri" < "Mon" < "Sat", so min()/max() on the raw column silently returns
# nonsense. Always go through this expression. The cast to TIMESTAMP drops the
# (always +0000) offset, which also avoids DuckDB needing pytz to marshal a
# tz-aware value back into Python.
TS = "CAST(strptime(created_at, '%a %b %d %H:%M:%S %z %Y') AS TIMESTAMP)"


def source() -> str:
    """DuckDB expression standing in for the raw table."""
    if not RAW.exists():
        sys.exit(f"raw dataset missing at {RAW}\nRun: uv run python scripts/fetch_data.py")
    return f"read_csv_auto('{RAW.as_posix()}', sample_size=200000)"


def render(con: duckdb.DuckDBPyConnection, sql: str, max_rows: int = MAX_ROWS) -> None:
    """Run a query and print it as a truncated, aligned table."""
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchmany(max_rows + 1)
    truncated = len(rows) > max_rows
    rows = rows[:max_rows]

    def cell(v: object) -> str:
        s = "" if v is None else str(v).replace("\n", " ").replace("\r", " ")
        return s[: MAX_CELL - 1] + "…" if len(s) > MAX_CELL else s

    table = [cols] + [[cell(v) for v in r] for r in rows]
    widths = [min(MAX_CELL, max(len(r[i]) for r in table)) for i in range(len(cols))]

    out = [
        "  ".join(h.ljust(w) for h, w in zip(table[0], widths, strict=True)).rstrip(),
        "  ".join("-" * w for w in widths),
    ]
    out += [
        "  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip() for r in table[1:]
    ]
    text = "\n".join(out)
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + f"\n… output truncated at {MAX_CHARS} chars"
    print(text)
    if truncated:
        print(f"… more rows exist; showing first {max_rows}. Aggregate in SQL rather than paging.")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--schema", action="store_true", help="columns, types and row count")
    g.add_argument(
        "--brands",
        type=int,
        nargs="?",
        const=15,
        metavar="N",
        help="top N brands by support-message volume",
    )
    g.add_argument("--counts", metavar="COLUMN", help="value counts for a column")
    g.add_argument("--sample", metavar="BRAND", help="sample exchanges for one brand")
    g.add_argument("--sql", metavar="QUERY", help="arbitrary SQL; the raw file is table 'twcs'")
    p.add_argument("-n", "--rows", type=int, default=MAX_ROWS, help=f"max rows (cap {MAX_ROWS})")
    a = p.parse_args()

    rows = min(a.rows, MAX_ROWS)
    src = source()
    con = duckdb.connect()
    # Without this, CAST(timestamptz AS TIMESTAMP) resolves against the machine's
    # local timezone, so the same query returns different times on different
    # machines. Everything in this project is UTC.
    con.execute("SET TimeZone='UTC'")
    con.execute(f"CREATE VIEW twcs AS SELECT * FROM {src}")

    if a.schema:
        print("== columns ==")
        render(con, "DESCRIBE SELECT * FROM twcs")
        print("\n== size ==")
        render(con, "SELECT count(*) AS rows FROM twcs")
        print(f"\nfile: {RAW}  ({RAW.stat().st_size / 1024**2:.0f} MB)")

    elif a.brands is not None:
        render(
            con,
            f"""
            SELECT author_id                                  AS brand,
                   count(*)                                   AS support_msgs,
                   count(DISTINCT in_response_to_tweet_id)    AS threads_replied,
                   round(100.0 * count(in_response_to_tweet_id) / count(*), 1)
                                                              AS pct_msgs_that_are_replies,
                   min({TS})                                  AS first_seen,
                   max({TS})                                  AS last_seen
            FROM twcs
            WHERE lower(CAST(inbound AS VARCHAR)) = 'false'
            GROUP BY 1 ORDER BY support_msgs DESC LIMIT {a.brands}
        """,
            max_rows=a.brands,
        )
        print("\nNote: pct_msgs_that_are_replies is a crude proxy for how conversational a")
        print("brand is. Confirm with thread-shape analysis before committing to a brand.")

    elif a.counts:
        col = "".join(ch for ch in a.counts if ch.isalnum() or ch == "_")
        render(
            con,
            f"""
            SELECT {col} AS value, count(*) AS n,
                   round(100.0 * count(*) / sum(count(*)) OVER (), 2) AS pct
            FROM twcs GROUP BY 1 ORDER BY n DESC LIMIT {rows}
        """,
            max_rows=rows,
        )

    elif a.sample:
        brand = a.sample.replace("'", "''")
        render(
            con,
            f"""
            SELECT c.tweet_id, substr(c.text, 1, 180) AS customer_msg,
                   substr(b.text, 1, 180)             AS brand_reply
            FROM twcs c
            JOIN twcs b ON b.in_response_to_tweet_id = c.tweet_id
            WHERE lower(CAST(b.inbound AS VARCHAR)) = 'false'
              AND b.author_id = '{brand}'
              AND lower(CAST(c.inbound AS VARCHAR)) = 'true'
            LIMIT {rows}
        """,
            max_rows=rows,
        )

    elif a.sql:
        sql = a.sql.rstrip().rstrip(";")
        if " limit " not in f" {sql.lower()} ":
            sql = f"SELECT * FROM ({sql}) LIMIT {rows}"
        render(con, sql, max_rows=rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
