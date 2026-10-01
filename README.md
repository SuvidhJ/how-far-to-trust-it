# A support agent for @AmericanAir — and how far to trust it

[![ci](https://github.com/SuvidhJ/how-far-to-trust-it/actions/workflows/ci.yml/badge.svg)](https://github.com/SuvidhJ/how-far-to-trust-it/actions/workflows/ci.yml)
[![Code: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Data: CC BY-NC-SA 4.0](https://img.shields.io/badge/data-CC%20BY--NC--SA%204.0-lightgrey.svg)](DATA_LICENSE.md)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](.python-version)

An AI agent for a real customer-support channel: it reads a customer's tweet,
classifies what they want, drafts a reply grounded in how the brand answered similar
messages before, and then either sends that reply or hands the ticket to a person
**with a stated reason**. It is built from the Kaggle *Customer Support on Twitter*
dump (2.8M tweets), on free-tier model APIs, for one brand.

The agent is the smaller half of the project. The larger half is the evidence: a
hand-labelled evaluation set, two baselines, an independent LLM annotator, an adversarial
probe, 53 logged decisions, and a written account of what the headline number hides.

**[Live demo](https://suvidhj.github.io/how-far-to-trust-it/)** — every one of the 218 evaluation
messages replayed through the agent, plus the charts behind the numbers below ·
**[Full report](reports/REPORT.md)** · **[Decision log](reports/DECISIONS.md)**

```bash
uv sync && uv run python -m support_agent.eval.harness --offline
```

That reproduces the classification, escalation and drafting tables below in about a
minute, from committed data and a committed response cache, with **no API keys**. CI
reruns it on every push, with the naive configuration (the deliberately
careless one described below) and the demo replay cases, and to fail if any number has
moved. The other numbers (nested CV, the Banking77 control,
annotator κ, live latency) come from the scripts listed under
[Reproduce](#reproduce), and their outputs are committed.

---

## What it does with one message

```console
$ uv run python -m support_agent.agent "my bag never showed up at DFW and nobody will talk to me"

message:  my bag never showed up at DFW and nobody will talk to me
intent:   track_baggage   top-3 track_baggage 0.59, refund_compensation 0.08, report_disruption 0.07
decision: AUTO_HANDLE [confident_and_safe] intent 'track_baggage' predicted with margin 0.51 and no risk language
draft:    <user> We're sorry your bag didn't arrive. Please file a report with our Baggage team at DFW so we can locate it for you.
          by grounded_llm, grounded in exchange ids [1498114, 273255, 2309781, 2763396]
evidence: [273255 sim 0.77] <user> We want to reunite you with your bag quickly. Did you file a report with our Baggage team at the airpor
evidence: [2763396 sim 0.77] <user> We're sorry for the delayed bag. Please make sure to file a claim with the airport Baggage team.
```

Ask it for money, a rebooking or a rule, and it refuses to answer alone — messages
it *classifies* as those intents never auto-handle, whatever the confidence. It does
not always recognise them: `ask_policy` (the policy-question intent) recall is 0.53
(n=15), and 1 of the 41 messages it sends alone is a misread policy question.
`--golden N --offline` replays the N-th message of the hand-labelled evaluation set
out of the committed cache, with no keys at all.

Four stages, each scored separately in the evaluation:

1. **Classify** — 10 intents induced from the training split and frozen before
   labelling, then `mistral-embed` + logistic regression.
2. **Retrieve** — the nearest past customer/brand exchanges from a 3,000-row corpus
   drawn from the training threads only.
3. **Draft** — an LLM writes the reply from those exchanges and cites the ids it used,
   so grounding is checkable rather than asserted.
4. **Route** — hard rules (money, rebooking, policy questions, legal language,
   vulnerable customers) and then the top-two probability margin. Every escalation
   names what the human has to decide.

A draft that fails, still contains an unfilled placeholder, or promises a follow-up
nothing in the pipeline will keep is never sent, whatever the classifier thought.

## Results

All numbers on n=218 hand-labelled messages unless stated.

**Intent classification** — mean of 5 repeats × 5-fold cross-validation:

| system | macro-F1 | sd |
| --- | --- | --- |
| trivial baseline (majority class) | 0.049 | 0.000 |
| simple baseline (TF-IDF + logistic regression) | 0.328 | 0.018 |
| simple baseline, tuned as fairly as the system (nested CV) | 0.406 | 0.014 |
| **system (mistral-embed + logistic regression)** | **0.640** | 0.030 |
| the same pipeline on Banking77 (a public intent benchmark), matched to 10 intents and n=218 | 0.965 | |
| the same pipeline on Banking77's 10 most confusable intents, n=200 | 0.800 | |

Model choice did not inflate it: with the choice made inside each fold, 0.639.

**Is 0.640 the model's ceiling or the labels'?** An independent 120B LLM annotator,
given the same written rubric, agrees with my labels at **κ 0.584 [0.51, 0.65]**
(Cohen's κ: agreement once chance agreement is removed, 0 = chance, 1 = perfect).
The classifier agrees at **κ 0.574 [0.50, 0.64]** on the same rows. The classifier
agrees with me about as well as an independent strong reader does. That is
consistent with much of the remaining headroom being disagreement about what these
tweets mean rather than model capacity, not proof of it: the annotator is a model
and shares the classifier's pull towards `report_disruption` (#52 — numbers like
this one are entries in [the decision log](reports/DECISIONS.md)).

**Escalation** — top-two margin below 0.10, plus the rules:

| router | sends without a human | wrong intent among those | classifier errors caught |
| --- | --- | --- | --- |
| trivial: send everything | 218 (100%) | 37.6% | 0 of 82 |
| simple: hard rules only | 148 (68%) | 31.8% | 35 of 82 |
| **system: rules + margin** | **48 (22%)** | **6.2%** [0.02, 0.17] | **79 of 82** |
| escalating the same share at random | 48 (22%) | 37.6% | 64 of 82 |
| threshold picked on dev rows, scored on unseen test rows | 34 of 135 | 2 wrong | |
| **the agent end to end**, with the post-draft guards | **41 (19%)** | **1 wrong** | 81 of 82 |

**Drafting** — all 218 messages per arm, scored against a written rubric by an LLM
judge, `gpt-oss-20b`, a different model family from the drafter. "As-is" also fails
drafts left holding an unfilled `<phone>` or `<url>` placeholder:

| drafter | judged | score /8 | judge would send | as-is |
| --- | --- | --- | --- | --- |
| canned template | 218 | 4.14 | 17% | 17% |
| copy the nearest past reply | 216 | 6.67 | 54% | 48% |
| LLM with no retrieval | 214 | 5.84 | 27% | 27% |
| **LLM grounded in 4 retrieved replies** | 214 | **6.92** | **61%** | **55%** |

Grounding beats copying on "sendable" — the judge-would-send column — only borderline
(p=0.053, n=212 paired). It is
clearly better at addressing what the customer actually asked (1.52 vs 0.83 of 2) and
clearly *less* grounded (1.59 vs 1.95), both p<0.0001. Against no retrieval, grounding
buys tone and safety rather than relevance.

**Cost and latency** — median 2.4 s per ticket end to end, p90 4.8 s, of which the
embedding call is 1.4 s; 519 prompt + 30 completion tokens per draft (n=20 live calls
on free-tier endpoints).

## What is misleading about the headline number

The long version is [section 4 of the report](reports/REPORT.md), which is the part
worth reading. In short:

- **0.640 measures agreement with one person — me.** I wrote every label and designed
  the taxonomy after reading the data. No second human has checked it.
- **The evaluation set is not the traffic.** Rare intents were deliberately
  over-sampled so they could be scored at all, and only messages the brand replied to
  were eligible.
- **A careless setup would have reported 0.725** for the same model: keep a duplicated
  thread, quote the best of 5 fold seeds, and report accuracy on the uniform stratum
  instead of macro-F1. `harness --naive --offline` runs that configuration and prints
  the gap; it is the first chart on the demo page.
- **The escalation number is optimistic.** Two rule changes were found or checked on
  this same data. Against the brand's own hand-offs to DM, the confidence margin carries no
  signal at all (AUC 0.53).
- **The drafting numbers are one model's opinion,** replayed from a cache, with no
  human agreement evidence yet — the one measurement this project still owes
  (`scripts/judge_agreement.py`).
- **"Sendable" is not "resolved".** See the first finding below.

## Three things the data actually showed

- **The dataset has responses, not resolutions.** Across all 2.8M rows, only 5–9% of
  brand replies get any customer acknowledgement and 46–69% get no reply at all. So
  the system learns a *response policy*, and no metric computable from this data can
  show a customer's problem was solved. Everything here says "this looks like a reply
  AmericanAir would send", never "this worked".
- **The annotation ceiling looks low, and the classifier is already near it** (the κ
  comparison above, model-human agreement only). Chasing a higher F1 would likely
  fit one annotator's reading of borderline tweets.
- **The expensive components do not pay for themselves — at classification.**
  In the bake-off (one split, n=219, #21), zero-shot LLM classification (0.305) ties
  TF-IDF from 2005 (0.313), and few-shot is within noise of embeddings + logistic
  regression (0.582 vs 0.571) at roughly 600× the compute. For
  *drafting*, retrieval does earn its place, but by a narrower margin than the
  composite score suggests.

The brand choice was made the same way: AppleSupport sends 52.6% of replies to DM, so
there is nothing to ground on; AmericanAir has the highest substantive reply rate at
73.7%.

## Reproduce

```bash
# uv: https://docs.astral.sh/uv/  (curl -LsSf https://astral.sh/uv/install.sh | sh)
uv sync                                                    # Python 3.12, provisioned by uv
uv run pytest -q                                           # 83 tests, none call a live LLM
uv run python -m support_agent.eval.harness --offline      # the results above
uv run python -m support_agent.eval.harness --naive --offline   # + the "careless setup" table
uv run python scripts/build_site.py                        # the demo page -> site/dist/
uv run python scripts/nested_cv.py                         # nested-CV rows (~5 min, offline)
```

Not rerun by CI, because they download data or call models:
`scripts/control_banking77.py` (Banking77 control), `scripts/annotator_agreement.py`
(annotator κ, `reports/annotator_agreement.json`) and `scripts/probe_agent.py`
(adversarial probe and latency, `reports/ROBUSTNESS.md`).

Last checked from a fresh clone on 2026-10-01 on Windows, running every step of the CI
workflow (`.github/workflows/ci.yml`): `uv sync --locked` 9 s (warm package cache; 134 s
with an empty one on 2026-09-16), 83 tests 60 s, the harness 64 s and the naive harness
91 s with every checked number identical, the demo replay cases rebuilt byte-identical
in 9 s, the site in 4 s. One number cannot be rebuilt from a clone: the naive table's
"retrieval index split by thread, not by tweet" row needs the raw dump, so the check
reports it as not checked. The same workflow runs on Linux on every push. Rebuilding
from the raw dump instead needs Kaggle credentials (`scripts/fetch_data.py`, ~170 MB)
and, for uncached model calls, free API keys in `.env` (see `.env.example`).

Python 3.12 is pinned deliberately: parts of the ML stack still have no 3.14 wheels.

## Layout

```
data/raw/          Kaggle dump — gitignored; query it with scripts/peek.py
data/interim/      brand-scoped threads (parquet) — gitignored
data/processed/    retrieval corpus + the demo's replay cases — committed
data/golden/       hand-labelled evaluation set and the raw label files — committed, append-only
data/cache/        embeddings and LLM responses — committed, so every rerun needs no keys
src/support_agent/ agent.py (end to end), intents, retrieve, draft, route, eval/ (harness, judge, metrics)
scripts/           data build, the bake-off behind each decision, controls, probes, site build
site/              the demo page template and its build
reports/           REPORT.md, DECISIONS.md, ROBUSTNESS.md, and the results JSON CI checks
tests/             fast tests; none may call a live LLM
```

Every built dataset under `data/` (the golden set, the retrieval corpus, the replay
cases, the Banking77 control) is written with a sibling `<name>.manifest.json`
recording its source, row count, seed, filters and git SHA; the caches and the
hand-typed label files have none. Downstream code reads the manifest, never a
directory listing.

## The dataset

`thoughtvector/customer-support-on-twitter` (`twcs` throughout) →
`data/raw/twcs/twcs.csv`, **493 MB, ~2.8M rows**. Threads are reconstructed by
following `in_response_to_tweet_id`;
`response_tweet_id` may hold a comma-separated list (`"5,7"`), which is the usual
source of silent bugs here, because a tweet with several replies is exactly the
escalation-shaped conversation you care about.

The raw CSV is never opened directly. Inspect it with
`uv run python scripts/peek.py --help` (DuckDB-backed, row- and width-capped).

## Citations — what was borrowed

- **Data.** *Customer Support on Twitter*, Thought Vector, Kaggle (CC BY-NC-SA 4.0).
  Banking77: Casanueva et al., *Efficient Intent Detection with Dual Sentence
  Encoders*, NLP4ConvAI 2020 (CC BY 4.0).
- **Models, all free-tier.** Embeddings `mistral-embed`; drafter
  `nemotron-3-super-120b-a12b`; judge `gpt-oss-20b`; independent annotators
  `nemotron-3-super-120b-a12b` (thinking off) and `ministral-3b`; `gemini-3.6-flash`
  was tried and gave no usable labels (quota, #52).
- **Methods.** Wilson interval (Wilson 1927), McNemar (1947), Cohen's kappa (1960),
  Gwet's AC1 (2008), corrected resampled t-test (Nadeau & Bengio 2003; Bouckaert &
  Frank 2004), nested CV (Varma & Simon 2006), risk–coverage curves (Geifman &
  El-Yaniv, NeurIPS 2017), BM25 (Robertson & Zaragoza 2009) and reciprocal rank fusion
  (Cormack et al. 2009), taxonomy induction framed after DSTC11 Track 2 (Gung et al.
  2023), self-preference bias in LLM judges (Panickssery et al., NeurIPS 2024), no
  BLEU/ROUGE (Liu et al., EMNLP 2016). Stakes: *Moffatt v. Air Canada*, 2024 BCCRT 149.
  Wider background in `docs/RESEARCH.md`.

## Licence

Code is MIT (`LICENSE`). Everything under `data/` is derived from the Kaggle dataset
and stays CC BY-NC-SA 4.0 — see [`DATA_LICENSE.md`](DATA_LICENSE.md), which lists each
file and what it contains.

*The proof is worth more than the system: `reports/DECISIONS.md` keeps the whole
trail of decisions, corrections included.*
