# Golden set — AmericanAir, 220 hand-labelled messages

| file | what it is |
| --- | --- |
| `americanair_frame.jsonl` | the unlabelled sampling frame (220 rows), from `scripts/sample_golden.py` |
| `americanair_golden.jsonl` | the frame plus hand labels, from `scripts/build_golden.py`; **the eval set** |
| `*.manifest.json` | seed, strata, label counts, git SHA, and the exclusions |
| `labels/` | the hand-typed label batches the golden set is rebuilt from (a test checks it is byte-identical), the in-session relabel (#29), the k=20 taxonomy proposal |
| `americanair_judge_sheet.txt` | 40 drafts for the judge-human study, drafter hidden (22 are verbatim evidence lines, so blinding is partial), with exactly the evidence the judge saw |
| `americanair_judge_items.json` | the same 40 items with a hash each; `--score` reads these back, never rebuilds them |
| `americanair_judge_human_scores.txt` | template for the author's scores, 0-2 on each rubric dimension (**not yet filled**) |

## How it was sampled

- **From dev and test threads only.** The train split, which the taxonomy and the
  retrieval corpus come from, never contributes a row. **One row per thread.**
- **Stratified, not uniform.** The rarest intent is about 2.6% of traffic, so 200
  uniform draws would give it about five examples. Instead: 20 rows for each of the 9
  intents a keyword can reach (`other` has no stratum),
  picked by **crude keyword rules** (never by a model, because a model-picked set
  agrees with that model by construction), plus **40 rows with no filter**. The
  keyword strata over-represent messages that use the obvious keyword. The random
  stratum shows how large that effect is.
- Seed `3854363812`, derived from `GLOBAL_SEED=20250911` via `seed_for("golden:AmericanAir")`.

## How it was labelled

- **By the author alone**, against the 10-intent taxonomy in `src/support_agent/taxonomy.json`. That taxonomy
  was frozen before labelling started, and every intent has a positive and a
  near-miss example.
- **From the customer message alone.** The brand's reply was hidden, so the
  label reflects what the customer asked for, not how the brand chose to answer.
- **Written tie-break rule:** if a message carries several intents (27.6% of
  training messages match two or more keyword rules), label the one the customer
  wants *action* on. Use `other` only when nothing else fits. Unclear messages still
  get their best-fit label and are flagged `ambiguous` (26 rows).
- **Two rows excluded, not deleted** (the file is append-only):
  - `2677500` is a thread twin of another row, from before one-row-per-thread was enforced.
  - `2749716` may be anchored: its model prediction was seen before labelling.

  Headline metrics use the remaining **n=218**.

## How far to trust the labels

A single annotator means the labels reflect one person's reading, and no strong
independent check exists. The only LLM annotator that gave valid answers
(ministral-3b, bare letters) agrees at **kappa 0.229** (n=182). Earlier figures of
0.517 and then 0.466 came from a reasoning model that never emitted a label: its
"answers" were read off letter tokens holding under 1% of the probability, so they
are withdrawn (DECISIONS #47, #51). Raw agreement tells the same story: that annotator
matches the labels on 34% of messages, while a classifier trained on them matches 62%.

Source text: Kaggle *Customer Support on Twitter* (Thought Vector), CC BY-NC-SA 4.0.
Customer handles are pre-anonymised by the dataset.
