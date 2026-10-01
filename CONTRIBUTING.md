# Contributing

Issues and pull requests are welcome — especially ones that break a claim in the
README. A reproducible counter-example is the most useful contribution this project
can receive.

## Setup

```bash
uv sync                                                # Python 3.12, provisioned by uv
uv run pytest                                          # none of the tests call a live LLM
uv run python -m support_agent.eval.harness --offline  # the headline numbers, no API keys
```

Use `uv add <pkg>` for dependencies, never `pip install`. API keys are only needed for
uncached model calls; copy `.env.example` to `.env`.

## Rules a pull request has to keep

- **CI must stay green.** `uv run ruff format . && uv run ruff check .` and
  `uv run pytest` pass, and `scripts/check_results.py` finds no moved number.
- **A change that moves a headline number** regenerates `reports/results*.json`,
  adds an entry to `reports/DECISIONS.md`, and keeps the old number next to the new
  one. Numbers are never silently overwritten.
- **`data/golden/` is append-only.** Never regenerate, reorder or machine-relabel it.
- **Every sample, split and shuffle takes an explicit seed** derived from
  `GLOBAL_SEED` in `src/support_agent/config.py`.
- **Every LLM call goes through `support_agent.llm`**, which caches responses to
  `data/cache/llm_cache.jsonl` so reruns need no keys. Commit new cache lines with the
  change that needed them.
- **The model that drafts a reply never judges it** in a reported number.
- **Never open `data/raw/twcs/twcs.csv` directly** (493 MB). Use
  `uv run python scripts/peek.py`.
- Prompts live in `src/support_agent/prompts/` as versioned files, not inline strings.
- Every number you report carries the n it was computed over.

## Licensing

Code contributions are accepted under MIT. Anything added under `data/` is derived
from the Kaggle dataset and is CC BY-NC-SA 4.0 — see `DATA_LICENSE.md`.
