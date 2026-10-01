# Decision log

Non-obvious decisions with their reasons. Entries are added as decisions are made,
not reconstructed at the end. Entries 4, 9 and 12 concerned local development tooling
that is not part of this repository; they are withdrawn and their numbers kept, so
every cross-reference below still resolves.

Entries 1–10 are environment and methodology decisions made before any modelling
began. Modelling decisions follow from 11 onward.

## The 14 that mattered

These are the non-obvious ones, each with its reason. The
numbered entries below record every decision in the order it was made, with the
alternatives and the evidence.

1. **AmericanAir, not AppleSupport or AmazonHelp.** Chosen on substantive-reply rate,
   73.7% vs 26.9%, not on volume. A brand that answers "DM us" gives the agent
   nothing to ground on (#16).
2. **Responses, not resolutions.** Only 5-9% of brand replies get any customer
   acknowledgement, so the agent learns how the brand *responds*, and no resolution
   is claimed (#17).
3. **Cut research features to the scope.** Binary routing, no torch, no NLI checks:
   a 15-minute reproduce and code a reader can follow outrank capability (#13).
4. **Split by thread, and run an audit designed to find problems.** It found a thread twin in the
   golden set and a silent `other` fallback that faked an annotator result (#7, #28).
5. **Linear model on embeddings, not an LLM classifier.** Few-shot matched it within
   noise at ~600x the latency (#21).
6. **The first embedding choice was wrong, and so was the test that reversed it.** A
   three-sentence probe and a malformed request picked nemotron. Repeated CV moved the
   choice to mistral, which is never worse, but McNemar pooled over repeats overstated
   significance. Corrected, the gap is significant only for one head (#23, #42).
7. **Dense retrieval only.** Hybrid RRF was *worse* than dense (P@1 0.434 vs 0.511),
   because BM25 is weak on tweets (#25).
8. **Escalate on the top-2 margin.** It beat max probability and entropy on AURC
   (0.201 vs 0.227 vs 0.274, n=218), narrowly against max probability (#27).
9. **The operating point is swept, not asserted, and it was corrected.** The sweep
   prints every run. A fold mismatch was fixed, and the old numbers are kept (#33, #40).
10. **Check the annotator check.** The LLM-annotator kappa first quoted beside the
    headline (0.517, then 0.466) was read off a reasoning model that never answered;
    withdrawn. Asked with thinking off, the same model answers every row and agrees
    at 0.584 (n=218), level with the classifier's 0.574 (#30, #47, #51, #52).
11. **Repeated-CV mean as the headline.** One split swings 0.591-0.674, more than any
    method difference (#32).
12. **Invalid drafts are a separate failure rate, and thinking is off.** `/no_think`
    still leaked; the chat-template flag took leaks from 9/218 to 0/217 (#34, #35).
13. **Banking77 as a control, instead of more classifier tuning.** 0.965 at matched
    size, but 0.800 on its 10 most confusable intents: part of the gap is how close this
    task's intents are; label noise is not separated out (#36, #45).
14. **Run the naive setup for real, then audit the honest one.** The naive setup
    measured +0.085 inflation and a same-family judge at 100% sendable (#37, #39).
    The audit found an untuned baseline (0.328, 0.406 when tuned), a composite metric
    that hid the LLM's gain, and a regex bug in escalation (#43, #44, #46).

Corrections kept in the log rather than overwritten: #26 by #41 (in-fold PCA does not
help), #33 by #40 and #44, #23 by #42, #36 by #45, #39 by #46, #47 by #51,
#51's annotator gap by #52.

---

### 1. Pin Python 3.12 rather than use the system 3.14
- **Decision:** `.python-version` pins 3.12; `uv` provisions it. All commands go
  through `uv run`.
- **Alternatives:** use the system Python 3.14.7 already on PATH.
- **Why:** 3.14 is new enough that parts of the ML stack still lack wheels, which
  would mean source builds — slow, and a hostile first experience for anyone
  who has 15 minutes. `uv` already had 3.12 cached locally, so pinning cost nothing.
- **Cost:** one more thing that must be right in the README. Mitigated by `uv`
  installing the interpreter itself.
- **Date:** 2026-09-11

### 2. Query the raw CSV with DuckDB; never load it with pandas
- **Decision:** all raw-file access goes through DuckDB, wrapped by `scripts/peek.py`.
- **Alternatives:** `pandas.read_csv` with chunking; pre-converting to parquet first.
- **Why:** the file is 493 MB / 2,811,774 rows. DuckDB streams it from disk and
  aggregates in SQL, so profiling queries stay near-instant and use little memory.
  Pre-converting hides the original as the source of truth.
- **Cost:** SQL rather than DataFrame idiom for exploration. pandas is still used
  downstream, once a brand extract has been written to parquet.
- **Date:** 2026-09-11

### 3. Make the raw data hard to open by accident
- **Decision:** `data/raw/**` is never opened with a file viewer or shell pager. Local
  development tooling blocks the shell equivalents (`cat`/`head`/`awk`/`Get-Content`)
  and unbounded `pandas.read_csv`; `scripts/peek.py` is the access path.
- **Alternatives:** a written convention asking not to read it.
- **Why:** conventions get ignored under pressure. One accidental `head` of a 493 MB
  file floods whatever is reading it. A guard makes the failure impossible rather
  than unlikely.
- **Cost:** the guard can false-positive on prose that quotes the pattern. Fixed
  by anchoring the regex to command position.
- **Date:** 2026-09-11

### 4. (Withdrawn)
Concerned local development tooling, which is not part of this repository.

### 5. Treat `data/golden/` as append-only, committed, and machine-validated
- **Decision:** the golden set is the only committed data. A local write-time check
  validates JSONL structure, required keys and duplicate ids on every write;
  deletion and truncating redirects are blocked.
- **Alternatives:** regenerate labels on demand; keep them out of git.
- **Why:** hand labels are the most expensive and least reproducible artifact in
  the project. Silent corruption discovered late invalidates every number
  measured against it.
- **Cost:** label corrections need a new entry plus a note, rather than an edit
  in place. That is the intent — relabelling history should be visible.
- **Date:** 2026-09-11

### 6. Build the evaluation harness and both baselines before the LLM system
- **Decision:** milestones 1–5 produce a working harness and trivial/simple
  baselines with no LLM involved.
- **Alternatives:** build the agent first, evaluate afterwards.
- **Why:** a harness written after the system tends to be shaped to flatter it.
  Fixing the measuring stick first makes every later number comparable, and the
  baselines are themselves a required deliverable.
- **Cost:** the impressive-looking part of the project starts later.
- **Date:** 2026-09-11

### 7. Split by conversation thread, never by tweet
- **Decision:** all train/test splits are at thread granularity, enforced by a test.
- **Alternatives:** row-level random split, which is the default in most tutorials.
- **Why:** consecutive tweets in a thread share customer, vocabulary and problem.
  A row-level split puts the same conversation on both sides and inflates every
  metric — the most likely route to a headline number that is quietly fake.
- **Cost:** fewer effective examples and a slightly more complex splitter.
- **Date:** 2026-09-11

### 8. The judge model must differ from the drafting model
- **Decision:** drafting and judging use different models, ideally different
  families; the pairing is stated in the report.
- **Alternatives:** use one strong model for both, which is simpler and cheaper.
- **Why:** LLM judges show measurable self-preference for their own outputs. A
  quality score produced by the same model that wrote the text is not evidence.
- **Cost:** a second provider/model to configure, and more cost per eval run.
- **Date:** 2026-09-11

### 9. (Withdrawn)
Concerned local development tooling, which is not part of this repository.

### 10. Keep `sentence-transformers`/`torch` behind an optional extra
- **Decision:** the default dependency set has no deep-learning framework;
  local embeddings install via `uv sync --extra local-embed`.
- **Alternatives:** include it by default for retrieval quality.
- **Why:** the README must reproduce headline results in under 15 minutes, and
  that budget includes dependency installation. Torch alone can consume most of
  it. Retrieval starts with TF-IDF and only moves to embeddings if measurement
  justifies it.
- **Cost:** retrieval quality may be left on the table. That becomes a
  "what I'd do next" item rather than an unmeasured assumption.
- **Date:** 2026-09-11

### 11. Develop on WSL2 Linux rather than native Windows
- **Decision:** the project lives in WSL2 Ubuntu 24.04 in the home directory, on
  ext4 rather than under `/mnt/`.
- **Alternatives:** stay on Windows, where the environment was already built,
  tested and working.
- **Why:** three things Windows could not give. The OS-level shell sandbox used
  during development is unsupported on native Windows. Most people who run this
  will do so on Linux or macOS, and the README promises a 15-minute reproduce — which
  cannot be verified from a platform nobody will use. And the whole class of
  CRLF, encoding and path-separator bugs disappears rather than being papered
  over. Keeping the repo on ext4 rather than `/mnt/e` avoids the 9p filesystem
  penalty; the test suite went from 1.32s to 0.51s.
- **Cost:** a one-time migration, and local tooling scripts had to change from
  `python` to `python3` because Ubuntu ships no bare `python`.
- **Date:** 2026-09-11

### 12. (Withdrawn)
Concerned local development tooling, which is not part of this repository.

### 13. Cut research-driven features back to the core scope
- **Decision:** after a literature and production-system survey, deliberately cut
  a three-tier action space, DSTC-style clustering metrics on the taxonomy,
  intent-name evaluation, NLI-based groundedness, SetFit, and pair-ranking. The
  default reproduce path is scikit-learn, DuckDB and an LLM API — nothing else.
- **Alternatives:** build the research-maximal system, which would have been more
  technically impressive in isolation.
- **Why:** three constraints of the project's scope dominate. It is a **binary**
  auto-handle/escalate decision, not three tiers. It promises reproduction in
  **under 15 minutes**, and `sentence-transformers`/`torch`/DeBERTa-MNLI are
  hundreds of megabytes against that budget. And every component has to be one I
  can explain and modify line by line — machinery I cannot defend and edit is a
  liability, not an asset. DSTC clustering metrics were
  cut for a separate reason: they score clusters against a *reference schema*,
  and we have none, so measuring against our own golden set would be circular.
- **Cost:** lower ceiling on retrieval and classification quality. Each cut item
  is named in the report's "what I chose not to build" section with its reason.
- **Date:** 2026-09-11

### 14. Induce the taxonomy with TF-IDF + KMeans rather than sentence embeddings
- **Decision:** cluster with TF-IDF + KMeans over a k sweep, then have an LLM
  summarise each group into a candidate intent, then merge by hand to 6–10.
- **Alternatives:** the literature-standard SBERT → UMAP → HDBSCAN pipeline.
- **Why:** the embedding pipeline is better at finding semantic structure, but it
  adds torch to the install budget and three more components to defend live. The
  semantic work here is done by the LLM naming and my manual merge, so the
  clustering only has to surface rough structure — which TF-IDF does adequately
  on a 500-message sample.
- **Cost:** rougher initial groups, so more manual merging. If the groups are too
  poor to be useful, the embedding path is the documented fallback.
- **Date:** 2026-09-11

### 15. Escalate on logprobs or self-consistency, never verbalized confidence
- **Decision:** derive the escalation confidence signal from token logprobs or
  2-sample self-consistency.
- **Alternatives:** ask the model to rate its own confidence 0–100, which is the
  obvious approach and what most implementations do.
- **Why:** verbalized confidence is a documented trap. Instruct-tuned models show
  ceiling rates above 90% — they report near-maximum confidence regardless of
  whether they are correct — and are consistently outperformed by plain token
  probabilities. Self-consistency beats Platt-scaled logprobs on Brier and AUROC
  with as few as two samples.
- **Cost:** two samples doubles inference cost for the escalation signal, and
  logprobs are not exposed by every provider. Both are acceptable at our volume.
- **Date:** 2026-09-11

### 16. Build for AmericanAir, and exclude the two most popular brands on evidence
- **Decision:** AmericanAir, on the highest *substantive* reply rate (73.7%),
  with British_Airways as the documented alternative.
- **Alternatives:** AppleSupport and AmazonHelp — the largest and most famous,
  and the likely default choice.
- **Why:** volume is the wrong criterion. What matters is whether the brand's
  public replies contain anything worth grounding on. AppleSupport deflects
  52.6% of replies to DM and only 26.9% are substantive, so an agent trained on
  it would mostly learn to say "please DM us". AmazonHelp deflects least (0.7%)
  but 89.5% of its replies carry an agent-initial signature and its intent space
  is diffuse. AmericanAir has none of those problems and a naturally crisp
  airline intent space.
- **Cost:** 36,531 exchanges rather than AmazonHelp's 168,814. Ample for a
  subsample, which is all this project needs.
- **Date:** 2026-09-11

### 17. Reframe the task as learning a response policy, not a resolution strategy
- **Decision:** state explicitly that the system imitates how the brand
  *responds*, and that no metric computed here demonstrates that a customer's
  problem was *solved*.
- **Alternatives:** take the obvious framing at face value and claim the agent is
  grounded in "how the brand historically resolved similar issues".
- **Why:** measured across all 2.8M rows, only **5–9%** of brand replies are
  followed by any customer acknowledgement and 46–69% receive no reply at all.
  The outcome of nearly every exchange is unobserved. The public Twitter channel
  is a triage layer, not a resolution layer. Claiming resolution would be an
  unfalsifiable claim about data that cannot support it.
- **Cost:** a less impressive-sounding pitch. It is also the honest one, and it
  is the backbone of the mandatory "what is misleading" section.
- **Date:** 2026-09-11

### 18. Ship the inflated number alongside the audited one
- **Decision:** the harness carries flags that reproduce the *dishonest*
  configuration — tweet-level split, no dedup, judge = drafter, uniform golden
  sample — so the report can show a waterfall from the naive number to the
  audited one, with the cost of each honesty step. Add a Banking77 control run of
  the identical classifier to separate data difficulty from method error.
- **Alternatives:** report only the audited number with a paragraph of caveats,
  which is what the section normally gets.
- **Why:** the honest number will look worse than competitors' inflated ones, and
  a reviewer skimming cannot tell the difference. Showing both, with the
  decomposition, turns that risk into the most memorable part of the report —
  and it is nearly free, since it is the same harness under different flags.
- **Cost:** a small amount of extra harness plumbing, and the risk that a reader
  quotes the naive number out of context. Mitigated by labelling.
- **Date:** 2026-09-11

---

<!-- Working log. The report presents the most significant entries; the rest
     stay here as the full record. -->

### 19. Provider stack chosen by measurement, not by reputation

**Decision.** Classifier `meta/llama-3.2-11b-vision-instruct` on NVIDIA NIM;
drafter `gemini-3.6-flash` on Google AI Studio; judge
`nvidia/nemotron-3-super-120b-a12b` on NVIDIA NIM. Every call is free-tier.

**Why.** Four providers were available. I probed all of them rather than picking
by name (`scratchpad/probe_*.py`):

- **Cerebras returns 402 Payment Required** on every model. It is not free, so it
  cannot be a dependency of a project that must run on anyone's machine.
- **Only NVIDIA NIM exposes token logprobs.** Decision #15 rules out verbalized
  confidence, so the escalation router needs logprobs, so the classifier must
  live on NVIDIA. That single fact determined the architecture.
- Roughly **two thirds of NVIDIA's advertised model list returns 404** when
  actually called. The working set is much smaller than `/v1/models` suggests.
- **Reasoning models cannot be used for logprob confidence.** `gpt-oss-20b` spends
  token 0 on a `<|channel|>` marker, so the first-token distribution carries no
  label signal. `llama-3.2-11b-vision-instruct` answers with a bare letter, so
  token 0 *is* the posterior over the option set.
- `nemotron-3-super-120b` returns HTTP 500 when JSON mode and logprobs are
  requested together — a combination that is fine on other models.
- **Mistral's free tier 429s** on everything above `ministral-3b`, so it is kept
  only as a spare.

**Consequence.** The taxonomy is presented to the classifier as a lettered list
and the model answers with one letter. The confidence used by the router is the
softmax over the letter tokens at position 0 — a genuine posterior, not a number
the model was asked to invent about itself.

**What I gave up.** Larger and probably more accurate models. The report will
state that the headline is achieved with an 11B model on a free tier, which
bounds the claim honestly and makes the cost-per-1k comparison meaningful.

### 20. OpenRouter added for the drafter; it does not replace NVIDIA for the classifier

**Decision.** Drafter moves to `nvidia/nemotron-3-ultra-550b-a55b:free` on
OpenRouter. Classifier stays on NVIDIA NIM. Judge stays `openai/gpt-oss-20b`.

**Why.** OpenRouter exposes 443 models, of which 19 are free. Probing them:

- Most free models are **reasoning models**, so token 0 is a thinking marker and
  the letter-posterior trick does not work. `nemotron-3.5-lightning` and
  `nemotron-3-ultra` both emit reasoning before an answer.
- The one free OpenRouter model that *did* return usable logprobs,
  `nex-agi/nex-n2.5-pro`, produced a near-uniform posterior (0.31/0.29/0.16/0.12/0.12)
  on an unambiguous baggage message. A flat posterior is a broken confidence
  signal, which is worse than none. NVIDIA's llama-3.2-11b returns 0.97 on the
  same case and 0.63 on a deliberately ambiguous one — graded, not saturated.
- `nemotron-3-ultra-550b` is a **550B model on a free tier**, completed 12/12
  requests at a 5.2s median, and produced a grounded, in-tone, length-compliant
  draft on the first try. For drafting — where no logprobs are needed — it is the
  strongest free option available.

**Consequence.** The three roles now sit on three different training families
(llama / nemotron / gpt-oss) across two providers, so judge-drafter independence
is structural rather than nominal.

**The bonus this unlocks.** `SELF_PREFERENCE_PROBE` pins a *same-family* judge
(`nemotron-3-super-120b`) alongside the real judge. Running both over the same
drafts turns self-preference bias from a cited literature range into a number
measured on this project's own data. That number goes in the "what is
misleading about my headline" section.

### 21. The classifier is a linear model on embeddings, not an LLM

**Decision.** The intent classifier is logistic regression over
`nvidia/nemotron-3-embed-1b` embeddings. The LLM arms are kept as comparisons.

**Why.** Six methods, one protocol - stratified 5-fold CV over the golden set,
out-of-fold predictions, identical folds for every arm, n=219:

| method | macro-F1 | 95% CI | accuracy | time |
| --- | --- | --- | --- | --- |
| majority (trivial) | 0.048 | [0.036, 0.062] | 0.164 | 0.0s |
| TF-IDF + logreg (simple) | 0.313 | [0.254, 0.364] | 0.420 | 0.3s |
| embed + kNN (k=5) | 0.374 | [0.303, 0.433] | 0.461 | 0.2s |
| **embed + logreg** | **0.571** | **[0.492, 0.635]** | 0.598 | **0.4s** |
| LLM zero-shot | 0.305 | [0.249, 0.355] | 0.388 | 325s |
| LLM few-shot (k=12) | 0.582 | [0.504, 0.644] | 0.626 | 248s |

Two things decide it.

**Zero-shot prompting is not competitive.** At 0.305 it is statistically
indistinguishable from TF-IDF + logistic regression (0.313), a method from 2005.
The intuition that a modern LLM must beat a linear bag of words is simply wrong
on this task, and it would have been the default choice.

**Few-shot prompting wins by 0.011 macro-F1 and costs 600x more.** The intervals
[0.504, 0.644] and [0.492, 0.635] overlap almost entirely; the gap is far inside
the noise of a 219-row golden set. Against that, the linear model classifies the
whole set in 0.4s with one cached embedding call per message, while few-shot takes
248s and an API call per prediction, forever. Picking the LLM here would be
buying a rounding error at six hundred times the price.

**What I gave up.** Few-shot is genuinely better on `praise_service` (0.85 vs
0.74) and `report_disruption` (0.70 vs 0.68). It is much worse on
`manage_booking` (0.43 vs 0.72) and `rebook_reroute` (0.17 vs 0.31). The linear
model is the more *balanced* classifier, which is what macro-F1 is measuring and
what matters when rare intents are the ones that need escalating.

**Consequence for the report.** The headline finding is not "we built an LLM
agent". It is that on this task, with 219 labelled examples, supervision on good
embeddings matches a large language model at 1/600th the cost - and that the
labelling, not the modelling, was the work that moved the number.

### 22. Embeddings ship as a committed float16 archive

**Decision.** Vectors are cached in `data/cache/embeddings.npz` (float16, single
compressed archive, committed) rather than per-text JSON files.

**Why.** Once decision #21 put embeddings on the critical path, the "<15 minutes
on a fresh clone" promise depended on them being reproducible without an API key.
The first implementation wrote one JSON file per text: 438 files, 11 MB, absurd
to commit. The same vectors as float16 in one npz are 816 KB. Half precision is
safe because the vectors are L2-normalised before use and cosine similarity is
unchanged to four decimal places.

**Consequence.** `Embedder(offline=True)` refuses rather than silently calling
the API, so a reviewer can prove the headline was replayed and not recomputed.

**The limit, stated honestly.** This works because the golden set is 219 rows.
Embedding the full 24,477-row training corpus at 2048 dimensions would be ~100 MB
even at float16, so the *retrieval* index stays TF-IDF unless an ablation shows
embeddings are worth a dimensionality reduction step.

### 23. The embedding model was chosen wrong the first time, and a toy probe was why

**Decision.** The embedding model is `mistral-embed`, not
`nvidia/nemotron-3-embed-1b`.

**What went wrong.** Nemotron was originally chosen because on a three-sentence
probe it separated a related pair from an unrelated one better than the
alternatives (0.503 vs 0.185, a gap of 0.318, against Mistral's 0.133). That
reasoning sounds right and is worthless: a wide similarity gap on one hand-picked
triple says nothing about whether a classifier fitted on those vectors will
generalise.

There was also a bug hiding the truth. `input_type` is an NVIDIA extension;
sending it to Mistral returns HTTP 422 and to Gemini HTTP 400. The first
comparison therefore reported both as "FAILED" and I nearly concluded nemotron
was the only usable option. It was a malformed request, not a bad model.

**The measurement that settled it.** 5 repeats x 5-fold CV, mean macro-F1:

| method | mean | sd |
| --- | --- | --- |
| mistral + logreg | **0.647** | 0.008 |
| mistral + centroid | 0.645 | 0.012 |
| mistral + svm | 0.627 | 0.018 |
| nemotron + logreg | 0.616 | 0.013 |
| nemotron + svm | 0.611 | 0.020 |
| nemotron + centroid | 0.549 | 0.022 |

McNemar on pooled paired predictions against the leader: nemotron+logreg
p=0.010, nemotron+svm p=0.022, nemotron+centroid p<0.001. Mistral wins on every
head, and the effect is significant rather than a fold artefact.

**What this changes about the report.** It is the cleanest example in the project
of a plausible proxy metric giving the wrong answer, and it will be written up as
one - including that a single 5-fold split had ranked `mistral + centroid` first
at 0.612 while repeated CV puts it second at 0.645, which is exactly the
instability that makes single-split comparisons untrustworthy.

### 24. Logistic regression over centroid and SVM, decided on a tie-break

**Decision.** The classifier head is logistic regression.

**Why, given it is a statistical tie.** McNemar puts `mistral + centroid`
(p=0.826) and `mistral + svm` (p=0.314) level with logistic regression; the F1
difference is 0.002 and 0.020 respectively, well inside the noise. Since accuracy
cannot separate them, three other criteria do:

1. **The router needs a probability.** Nearest-centroid and LinearSVC produce a
   distance and a decision-function margin, not a posterior. Calibrating the SVM
   with Platt scaling was tried and *costs* accuracy - it fell from 0.607 to
   0.528 on nemotron vectors, because the inner CV loop starves an already small
   training set.
2. **Stability.** Logistic regression has the lowest spread across seeds
   (sd 0.008, against 0.012 for centroid and 0.018 for SVM).
3. **Explainability under questioning.** Per-class coefficients over embedding
   dimensions are less interpretable than a centroid, but `predict_proba` output
   is directly inspectable, and a confidence number is the first thing anyone
   will ask about.

**What I gave up.** Nearest-centroid is five lines of numpy and would have been
the better answer to "explain this to me" if the router did not need calibrated
scores. It stays in the comparison table for exactly that reason.

### 25. Retrieval is dense-only; hybrid RRF made it worse

**Decision.** `EmbedRetriever` (cosine over mistral-embed) is the retriever.
BM25, TF-IDF and a hybrid stay as comparison arms.

**How it was measured.** Retrieval has no ground truth here - nobody labelled
which historical exchange is the right one to ground a reply in. The proxy is
intent agreement: leave-one-out over the golden set, asking what fraction of the
top-k share the query's hand-labelled intent. Chance is 0.118.

| retriever | P@1 | P@3 | P@5 | MRR |
| --- | --- | --- | --- | --- |
| BM25 | 0.274 | 0.292 | 0.264 | 0.453 |
| TF-IDF | 0.333 | 0.292 | 0.268 | 0.473 |
| **embeddings** | **0.511** | **0.467** | **0.424** | **0.642** |
| hybrid (RRF) | 0.434 | 0.414 | 0.367 | 0.594 |

**The interesting result is the hybrid losing.** Reciprocal rank fusion is the
standard recommendation and it is *worse* than the dense retriever alone
(0.434 vs 0.511 P@1), because RRF weights both retrievers equally and BM25 is
much the weaker of the two here. On tweets, lexical matching is close to useless:
"where is my luggage" and "bag never arrived" share no content words.

**The proxy's limits, stated.** Two exchanges can share an intent and be useless
to each other; two with different intents can still model the right tone. What
this measures is whether a retriever returns the same *kind* of problem, which
is a necessary condition for grounding rather than a sufficient one.

**What this overturns.** The plan had TF-IDF as the default retriever on the
grounds that embeddings could not be committed. That was a shipping constraint
masquerading as a design decision; see #26.

### 26. Compressing embeddings improves quality (provisional)

**Finding.** PCA over the golden-set embeddings, measuring both tasks:

| dims | MB per 24k rows | macro-F1 | retrieval P@1 |
| --- | --- | --- | --- |
| 1024 | 50.1 | 0.588 | 0.511 |
| 128 | 6.3 | 0.604 | 0.507 |
| 32 | **1.6** | **0.633** | **0.534** |

Compression does not cost quality, it *adds* it. 219 training examples in 1024
dimensions is badly under-determined, and projecting down regularises the linear
model. It also collapses the index from 50 MB to 1.6 MB, which is the difference
between a retrieval index that can be committed and one that cannot.

**Why this is marked provisional.** The PCA here is fitted on the same 219 rows
it is scored on. It uses no labels, so this is not label leakage, but the
projection has still seen the evaluation points and the numbers are optimistic.
The honest version fits PCA on the training corpus, which is disjoint from the
golden set. That is a prerequisite before the number goes in the report, and it
is exactly the kind of "technically unsupervised, still leaky" step that is easy
to publish without noticing.

### 27. The escalation signal is the top-2 margin, not the top probability

**Decision.** The router thresholds the gap between the top two class
probabilities.

**Why.** Escalation is evaluated as selective prediction, so the ground truth is
free: the router should hand off the cases the classifier got wrong. AURC (area
under the risk-coverage curve, lower better) over out-of-fold predictions,
n=219, base error 0.384:

| signal | AURC | err@30% coverage | err@50% | err@70% |
| --- | --- | --- | --- | --- |
| **top-2 margin** | **0.201** | **0.092** | 0.220 | 0.248 |
| max probability | 0.239 | 0.185 | 0.229 | 0.307 |
| negative entropy | 0.309 | 0.262 | 0.284 | 0.333 |

Max probability is the obvious choice and loses. Entropy, which uses the *whole*
posterior and therefore looks like it should be strictly more informative, is the
worst of the three - the tail of eight near-zero classes is noise, and averaging
it in dilutes the signal that matters. The margin asks the only question the
decision depends on: was this close?

**The rule layer is evaluated separately, because it has a different job.** Hard
rules escalate 24% of traffic and catch 30% of the classifier's errors, costing
27 unnecessary escalations, and they only move error on the remainder from 0.384
to 0.353. As an error-catcher that is weak. But the rules do not exist to catch
statistical errors, they exist to catch *consequential* ones - money, legal
threats, vulnerable customers - where a fluent wrong answer is a liability rather
than a mistake. Reporting them against AURC would be measuring the wrong thing.

### 28. Four bugs found by an adversarial audit, one of which was faking a result

**Decision.** Run an integrity audit written to *fail* rather than to confirm,
before adding any more methods.

**What it caught.**

1. **Thread leakage in the golden set.** The sampler deduplicated by row, not by
   thread, so one conversation contributed two consecutive messages (2677499,
   2677500) through different strata. Split across CV folds they are train/test
   twins. `sample_golden.py` now enforces one row per thread; the affected row is
   flagged `excluded` rather than deleted, because the golden set is append-only.
   Scored n is 218, not 220.
2. **A silent fallback that faked an experimental result.** `LLMClassifier`
   returned `other` with confidence 0.0 whenever it could not parse a label. In
   the agreement study `gpt-oss-20b` then scored 0.075 raw agreement, which reads
   as a wildly dissenting annotator. It was not dissenting - **86% of its calls
   returned nothing parseable**, because it is a reasoning model whose token 0 is
   a channel marker. `Prediction.parse_failed` now records this and every caller
   reports the rate.
3. **`choices: null` from OpenRouter.** The provider answers HTTP 200 with a null
   choices list when its upstream errors, so the SDK raises nothing and the
   failure surfaces as a `TypeError` several frames later. Now raised with the
   provider's own error text.
4. **A doubled embedding bill.** `mistral-embed` is symmetric, so keying the
   cache on `input_type` sent the identical request twice for every text.

**What passed, and is now asserted:** no taxonomy example appears in the golden
set (which would have leaked labelled test items into the LLM prompts), no
duplicate or near-duplicate texts at cosine > 0.95, no golden thread in the train
split, and float16 storage preserves nearest-neighbour ranking exactly (1.0000
agreement against float64).

### 29. Intra-annotator kappa of 1.00 is not evidence, and is not reported as such

**Decision.** Do not report the intra-annotator agreement figure as reliability
evidence. Report why.

**What happened.** 45 golden rows were re-labelled blind, in shuffled order, with
the original labels hidden. Result: 45/45 identical, kappa = 1.00. That is not
reliability, it is memory - both passes happened inside one working session, and
the standard advice is to re-label "a day later". A write-up that quotes
kappa = 1.00 as a strength is making exactly the kind of unexamined claim the
"what is misleading about your headline number" section is asking about.

### 30. The annotation ceiling, and why chasing macro-F1 past it is chasing noise

**Finding.** Independent annotators labelling the same 120 messages from the same
written rubric:

| pair | n | raw agreement | Cohen's kappa | band |
| --- | --- | --- | --- | --- |
| me vs nemotron-120b | 120 | 0.567 | **0.517** | moderate |
| me vs ministral-3b | 106 | 0.321 | 0.226 | fair |
| nemotron-120b vs ministral-3b | 106 | 0.208 | 0.139 | slight |

**Two things follow.**

*My labels are the coherent reference, not an outlier.* Had they been
idiosyncratic, the two models would agree with each other more than either agrees
with me. They agree with each other far *less* (0.139 against 0.517). The weakest
annotator is the 3B model, which agrees with nobody.

*The ceiling is low, and the classifier is already at it.* The best independent
annotator matches my labels 56.7% of the time. The deployed classifier, scored
out-of-fold, matches them 63.9% of the time. It is not that the classifier is
superhuman - it was trained on my labels and the annotator was not - but it does
mean **the remaining error is dominated by genuine boundary ambiguity rather than
by model capacity**, and that pushing macro-F1 from 0.65 to 0.70 would mostly be
fitting my personal reading of borderline tweets.

**Corroboration that this is task ambiguity, not model incompetence.** The top
disagreements are precisely the boundaries the frozen taxonomy documents as near
misses: complain_service vs track_baggage (15), refund_compensation vs
report_disruption (13), report_disruption vs rebook_reroute (8). And the
`ambiguous` flag I applied at labelling time predicts it - unanimity is 11% on
flagged rows against 20% on the rest, so the flag was tracking something real.

**Consequence for the report.** The headline macro-F1 is quoted against this
ceiling, not in isolation. This is the single most important honest number in the
project: it converts "the classifier is 65% accurate" into "the classifier
agrees with its annotator about as often as a competent independent annotator
does, on a task where two annotators agree 57% of the time".

### 31. Throughput is a capability: the drafter moved off the best model

**Decision.** The drafter is `nvidia/nemotron-3-super-120b-a12b` on NVIDIA NIM,
not `nemotron-3-ultra-550b:free` on OpenRouter.

**Why.** The 550B model writes better - measurably so on the smoke test, and it
was chosen for that in #20. But OpenRouter's free tier throttles it to roughly
one completion every 55 seconds. A 100-draft evaluation with a self-preference
probe is 200 generations; at that rate the evaluation loop takes an hour and a
half, so it gets run once instead of after every change. A model I cannot afford
to re-run is a model I cannot iterate against, and this project stands on
evaluation rigour rather than on generation quality.

**What I gave up, stated plainly.** Probably some draft quality. The report will
say the drafter is a 120B model on a free tier and that a stronger writer was
available but not affordable to evaluate against.

**What it bought.** The self-preference probe becomes the strongest version of
that test: the probe judge is now the drafter's *own model*, not merely one from
the same family, so the measured gap is the full inflation a lazy same-model
setup would produce.

### 32. The headline is a repeated-CV mean, because one split is not a result

**Decision.** The harness reports mean macro-F1 over 5 repeats of 5-fold CV, with
its standard deviation, not the number from a single split.

**Why.** On 218 rows the single-split macro-F1 for the same model ranges from
0.588 to 0.660 depending only on the fold seed. That swing is **larger than every
method difference measured in this project** - larger than logreg vs SVM, larger
than C=4 vs C=16, larger than PCA-64 vs full dimensionality. A write-up that
quotes one split is quoting whichever seed flattered it, and would have "shown"
that several of the rejected methods won.

Headline: **0.640 ± 0.030** against 0.328 ± 0.018 for the simple baseline and
0.049 for the trivial one. The bootstrap CI is still reported alongside, but it
answers a different question - how much the number moves if the *examples* had
been different, rather than if the *folds* had been.

### 33. The escalation operating point is swept, not asserted

**Decision.** Default margin threshold 0.10, with the full sweep printed by the
harness every run.

| threshold | coverage | error on auto-handled | errors caught | wasted escalations |
| --- | --- | --- | --- | --- |
| 0.00 (rules only) | 77% | 0.347 | 24/82 | 27 |
| 0.05 | 41% | 0.180 | 66/82 | 63 |
| **0.10** | **26%** | **0.123** | **75/82** | **86** |
| 0.15 | 11% | 0.087 | 80/82 | 115 |
| 0.25 | 4% | 0.000 | 82/82 | 128 |

Base error on all traffic is 0.376. At 0.10 the system auto-handles a quarter of
messages with a third of the base error rate and hands off 91% of what it would
have got wrong, at a cost of 86 unnecessary escalations.

**The honest reading.** Coverage is low. A quarter of traffic auto-handled is not
an impressive automation rate, and the report will say so rather than quoting the
0.123 error rate on its own. The reason it is the right point anyway is the cost
asymmetry: a needless escalation costs a shared-inbox agent a few seconds, while
a wrong public reply under the brand's name is the *Moffatt v. Air Canada*
failure. Anyone who prefers a different trade-off can read it off this table -
which is why the table is printed rather than a single chosen number.

### 34. The first drafting bake-off is invalid, and why its numbers are not reported

**Decision.** None of the n=50 drafting numbers go into the report until three
contaminations are fixed. They are recorded here only so the correction is
traceable.

**What the run printed** (judge `gpt-oss-20b`, rubric `judge_v1`, n=50):

| drafter | total /8 | sendable |
| --- | --- | --- |
| canned | 3.78 | 14% |
| nearest (verbatim past reply) | 6.90 | 64% |
| ungrounded_llm | 4.17 | 15% |
| grounded_llm | 7.08 | 69% |

and a self-preference "inflation" of +0.37 (grounded) and -0.73 (ungrounded).

**Why none of it stands.** An audit of the response cache, separating drafts from
judgements, found:

1. **`ungrounded_llm` was scored on API failures.** 13 of its 50 drafts are
   errors, and the judge scored each empty draft 0 on every dimension. The cache
   holds *zero* empty drafts, so these were uncached exceptions during the run -
   not the model running out of tokens, which was my stated hypothesis in the
   handoff and was wrong. The +2.9 "value of grounding" is therefore confounded
   with a 26% outage rate on one arm.
2. **The drafter leaks its chain-of-thought into the reply.**
   `nemotron-3-super-120b` is a reasoning model. 11 of 117 cached drafts (9%)
   begin "We need to...", 14 hit `max_tokens`, and the longest is 5,344
   characters of reasoning. Sent to a customer, that is not a weak reply, it is a
   leak of internal deliberation under the brand's name. 11/117 also break the
   280-character limit the prompt sets.
3. **The self-preference comparison was unpaired.** The probe judge (the drafter's
   own model) returned a complete score on only 56 of 77 calls - 23 truncated, 22
   with reasoning before the JSON. Parse failures were excluded from each judge's
   mean independently, so the two means were computed over *different drafts*.
   A difference between means of different subsets measures the subsets.

**What is still suggestive, and only suggestive.** Grounding the LLM in retrieved
replies moves `grounded` from 0.66 to 1.60 and `safe` from 1.36 to 1.88, and the
verbatim `nearest` baseline is competitive with the LLM - it wins on `grounded`
(1.96) and `safe` (1.94) while the LLM wins on `addresses` (1.69 vs 1.00). That is
the expected trade-off between a copied reply that is safe but generic and a
generated one that is specific but occasionally invents. Whether it survives the
fixes and a paired significance test is open.

**A second correction.** At n=6, `nearest` led the grounded LLM (7.33 vs 6.00).
At n=50 the order flipped. That was reported upward as an "early signal" and it
did not hold - another instance of #32's lesson, at a sample size where it should
have been obvious.

### 35. A reasoning model drafts with thinking off, and only after measuring that the switch works

**Decision.** The drafter stays `nemotron-3-super-120b`, called with
`chat_template_kwargs: {enable_thinking: false}`. Any draft that is empty, cut off
at `max_tokens`, an API failure, or opens with reasoning ("We need to...") is an
*invalid draft*: counted as a failure rate of its own and never sent to the judge.

**Why.** Same grounded prompt, same 218 golden messages, per configuration
(`scripts/bakeoff_draft.py --leak-test --n 218`):

| configuration | valid | reasoning leak | truncated | API error |
| --- | --- | --- | --- | --- |
| nemotron, thinking on | 207 | 9 | 2 | 0 |
| nemotron, `enable_thinking=false` | **217** | **0** | **0** | 1 (429 after 6 retries) |
| nemotron, `/no_think` prompt prefix | not run to n=218 | 2/50 at n=50 | | |

The obvious switch - the `/no_think` prompt tag - does not work on this endpoint:
it still leaked on 2 of the first 50 messages. Had it been assumed rather than
measured, the report would have claimed a fix that ships reasoning to customers.
The chat-template flag removes the leak entirely on the same messages (9/218 →
0/217, Fisher exact p≈0.004).

**The non-reasoning alternative could not be tested properly.** Every candidate
non-reasoning model on NVIDIA NIM - llama-3.3-70b, llama-3.1-70b,
llama-4-maverick, gemma-3-27b - now returns 410 Gone. The only one still served
is llama-3.2-11b, the classifier model; it is in the leak table as the
non-reasoning arm, but it is a much smaller writer, so it is a control for the
leak, not a candidate drafter.

**Why invalid drafts are excluded from quality rather than scored zero.** The first
bake-off scored 13 API failures as 0/8 drafts, which made one arm look 2.9 points
worse for a reason that had nothing to do with writing. Scoring them zero conflates
availability with quality; dropping them silently hides an outage. Reporting
validity and quality as two numbers keeps both visible.

### 36. Banking77 as a control: the ceiling is the data, not the method

**Decision.** Run the identical intent classifier on Banking77 and report it beside
the AmericanAir headline, instead of tuning the classifier further.

**What was run** (`scripts/control_banking77.py`): mistral-embed + logistic
regression (C=4, balanced), 5 repeats x 5-fold CV, the same two baselines. Two
configurations, because a larger label set and more data are both confounds:

| data | n | intents | majority | tfidf+logreg | embed+logreg |
| --- | --- | --- | --- | --- | --- |
| **AmericanAir golden** | 218 | 10 | 0.049 | 0.328 | **0.640** |
| Banking77, matched (5 random intent draws) | 218 | 10 | 0.030-0.047 | 0.890-0.948 | **0.938-0.981** (mean 0.965) |
| Banking77, all intents, 20 each | 1,540 | 77 | 0.000 | 0.723 | 0.887 |

"Matched" reproduces the AmericanAir class-size profile exactly (largest class 39,
smallest 9), so only the text and the labels differ.

**What it shows.** On clean, crowd-labelled data with the same n, the same number
of classes and the same imbalance, the same pipeline scores 0.94-0.98. Even with
77 intents it scores 0.89. The 0.33-point gap to AmericanAir is therefore not a
broken embedding, a bad head or a CV bug - it is the data: short multi-intent
tweets, labelled by one annotator, on a task where two independent annotators
agree at kappa 0.517 (#30). That is the evidence behind not spending the remaining
time on classifier methods.

**What it does not show.** Banking77 queries are single-intent by construction and
written to be classified; nothing here says the AmericanAir taxonomy is *good*,
only that the method is not what caps the number. Randomly drawn Banking77 intents
are also easier to separate than a taxonomy of neighbouring airline problems -
draw 4 (0.938) is the hardest of five, and still far above 0.640.

### 37. The naive configuration is actually run, and most honesty steps cost less than expected

**Decision.** `harness --naive` runs each dishonest choice for real and prints a
waterfall, one step undone at a time, instead of the report asserting that the
audited number is "more honest".

**Measured** (`uv run python -m support_agent.eval.harness --naive --offline --draft-n 0`,
embed+logreg, 5x5-fold CV):

| honesty step, undone | metric | naive | audited | delta | n |
| --- | --- | --- | --- | --- | --- |
| one row per thread (keep thread twin + taxonomy anchor) | macro-F1 | 0.632 | 0.640 | -0.009 | 220 vs 218 |
| repeated-CV mean, not the best single split | macro-F1 | 0.674 | 0.640 | +0.034 | 218 |
| macro-F1 on stratified set, not accuracy on a uniform sample | accuracy vs macro-F1 | 0.675 | 0.640 | +0.035 | 40 vs 218 |
| **all three intent steps together** | the quotable number | **0.725** | **0.640** | **+0.085** | 40 vs 218 |
| retrieval index split by thread, not tweet | top-1 hit from own conversation | 4.6% | 0.0% | +4.6 pts | 10/218 vs 0/218 |
| judge from a different family than the drafter (added after #39) | mean total /8, grounded drafts | 7.771 | 7.188 | +0.583 | 48 paired |
| same | sendable rate, grounded drafts | 1.000 | 0.729 | +0.271 | 48 paired |

**What is surprising, and reported as such.** The thread-level guard on the golden
set bought nothing measurable: the one leaking pair it removed is too few to move
a 218-row macro-F1, and keeping it scored *lower*. The guard is still right - it is
free and the failure it prevents scales with how many twins a sample happens to
draw - but claiming it "removed inflation" here would be false. Leakage matters
where there are many near neighbours: 188 thread-mate exchanges in the retrieval
corpus hand 10 of 218 messages their own conversation as top-1 evidence.

**Why run it rather than describe it.** The inflation a careless write-up would
report is +0.085 on the headline and comes mostly from two reporting choices
(seed-picking and accuracy on a uniform sample), not from modelling. That is a
more useful finding than the generic claim that leakage inflates scores, and it is
only available because the naive path exists as code.

### 38. Commit the 3,000-row retrieval corpus, breaking the "only golden data is committed" rule

**Decision.** `data/processed/americanair_index.jsonl` (866 KB) and its manifest are
committed, alongside the Banking77 control result.

**Why.** The corpus was gitignored like all derived data, and its vectors were
already in the committed embedding archive - so the archive was useless without
it. On a fresh clone the harness drafting section and `bakeoff_draft.py` stopped
with "run build_index.py first", and `build_index.py` needs the 493 MB Kaggle dump
plus thread reconstruction. That silently broke the README's "reproduce the
headline in under 15 minutes" promise for every drafting and retrieval number.
The rule existed to keep bulky, regenerable data out of git; a sub-megabyte file
that gates reproducibility is the case the rule was not written for.

**Cost.** Redistributes 3,000 customer tweets and brand replies. The golden set
already does the same for 220, the Kaggle dataset is published under CC BY-NC-SA
4.0 with the attribution in the README, and `@mentions` are anonymised upstream.

### 39. Grounded LLM drafting is not shown to beat copying the nearest past reply

**Decision.** The report says the grounded LLM drafter is *not significantly better*
than sending the nearest historical reply verbatim, and presents grounding - not
the LLM - as what earns its place. It does not claim LLM drafting is proven.

**Measured** after the #34 fixes (`uv run python -u scripts/bakeoff_draft.py --n 50`,
drafter nemotron-3-super-120b thinking off, judge `gpt-oss-20b` rubric `judge_v1`,
a different model family from the drafter):

| drafter | valid | judged n | total /8 | sendable [95% CI] |
| --- | --- | --- | --- | --- |
| canned template | 50/50 | 50 | 3.78 | 14% [0.07, 0.26] |
| nearest past reply, verbatim | 50/50 | 50 | 6.90 | 64% [0.50, 0.76] |
| ungrounded LLM | 50/50 | 49 | 5.82 | 31% [0.20, 0.45] |
| grounded LLM | 50/50 | 49 | **7.14** | **71%** [0.58, 0.82] |

Paired `sendable`, McNemar exact, on drafts both arms had scored completely:

| comparison | n | both | A only | B only | neither | p |
| --- | --- | --- | --- | --- | --- | --- |
| grounded LLM vs nearest | 49 | 23 | 12 | 8 | 6 | **0.503** |
| grounded LLM vs ungrounded LLM | 48 | 12 | 23 | 2 | 11 | <0.001 |

Zero invalid drafts in 200 (no API failures, leaks or truncation), so the
"invalid = not sendable" basis gives the same table; the two missing judgements
are judge parse failures, excluded pairwise. The McNemar counts were recomputed
from `scratchpad/bakeoff_draft_results.jsonl` by a separate script and match.

**What it shows.** Retrieval is the component that works: the same LLM goes from
31% to 71% sendable when given past replies (23 discordant pairs to 2). Whether the
LLM adds anything over the retrieved reply itself is unresolved - 12 vs 8
discordant pairs at n=49 is a 7-point gap with p=0.50. A 3,000-reply lookup that
copies text is a fair incumbent and has not been beaten.

**Self-preference, now paired.** Re-scoring the same drafts with the drafter's own
model as judge (thinking off, 4,096-token cap): it called **48/48** drafts sendable
for both arms, against 35/48 (grounded) and 15/48 (ungrounded) from the
cross-family judge, and raised the mean total by +0.58 (grounded, 19 up / 27 equal
/ 2 down) and +1.85 (ungrounded, 35 / 13 / 0), sign test p<0.001 for both. A
same-family judge would have reported 100% sendable. That is the size of the
error the "drafter never judges itself" rule prevents.

**What it does not show.** The headline judge has no human-agreement evidence yet
(PENDING, `scripts/judge_agreement.py`), so 71% is a judge's opinion, not a
measured send rate. n=50 cannot detect a gap under ~20 points. And `nearest` can
copy a reply written to a different customer: message 32 (a question about tipping
curbside staff) got a gate-check reply ending "We're so sorry, Steve." The judge
marked it not sendable, but for not addressing the question - its stated reason
never mentions the wrong name. One case in 50; the rubric has no check for it.

### 40. The escalation numbers were computed on mismatched folds; corrected, old numbers kept

**Decision.** The router now gates each out-of-fold prediction with the posterior
from the *same* fold model. Headline escalation moves from 26% / 0.123 / 75 of 82
to **25% / 0.109 / 76 of 82**.

**What was wrong.** The harness took predictions from repeat 0 of the repeated CV
but refitted the posteriors on the loop variable left over from repeat 4. Each
prediction was therefore paired with a margin from a model trained on different
folds, whose argmax need not even be the predicted label. Found while pulling
quoted examples for the report, by recomputing routing independently
(`scratchpad/report_examples.py`).

| margin 0.10 | coverage | error on auto-handled | errors caught |
| --- | --- | --- | --- |
| before (#33, mismatched folds) | 26% (57/218) | 0.123 | 75/82 |
| **after (same folds)** | **25% (55/218)** | **0.109** | **76/82** |
| after, repeats 0-4 separately | 25-28% | 0.085-0.131 | 59/65 to 76/82 |

**What it shows.** The correction is small and inside the repeat-to-repeat spread,
so #27 and #33 stand. But the spread itself is the finding: error on the
auto-handled quarter ranges 0.085-0.131 across fold seeds, on 55-62 messages. The
report quotes the range, not a single value. The sweep is no longer monotone
(0.15 gives 0.115 on 26 messages), which is what small-n noise looks like.

### 41. Resolving #26: PCA fitted inside each fold does not help, so embeddings stay uncompressed
- **Decision:** no PCA. Classify on full 1024-d `mistral-embed` vectors. #26 is left as written, and this entry corrects it.
- **Alternatives:** PCA-32 or PCA-64 before logistic regression, fitted inside each training fold (`scripts/challenge_best.py`). Also tried: `C=16`, isotonic calibration.
- **Why:** 5x5-fold CV, n=218: incumbent 0.645 (sd 0.015), PCA-64 0.643, PCA-32 **0.616**. #26's gain (0.588 to 0.633 at 32 dims) existed only because PCA saw the evaluation rows. Best challenger `C=16` scored 0.654, but McNemar was 34 vs 22, p=0.141, so the incumbent stays.
- **Cost:** the index stays at full size, which is fine at 3,000 rows. The in-script incumbent (0.645) differs from the harness headline (0.640) only by fold seeds, which is within the #32 spread.
- **Date:** 2026-09-15

### 42. Correcting #23: repeated-CV comparisons use a corrected t-test, and "nemotron is worse" mostly does not survive it
- **Decision:** compare classifiers with the Nadeau–Bengio corrected resampled t-test on per-fold macro-F1 (`metrics.corrected_resampled_ttest`). McNemar counts summed over repeats are printed only as description.
- **Alternatives:** the previous test, exact McNemar on discordant pairs summed over 5 repeats of the same 218 rows; per-repeat McNemar.
- **Why:** summing repeats counts each row five times and treats folds that share training data as independent, which understates p. At n=218, mistral vs nemotron: logreg pooled p=0.001 vs corrected **p=0.278**; SVM 0.137 vs **0.507**; centroid <0.001 vs **0.029**. Mistral is never worse and stays, but "p≤0.022 on every head" (#23, REPORT §1) was wrong. #41's C=16 comparison becomes p=0.708, and the incumbent still stands.
- **Cost:** a far less powerful test at n=218. Most method differences in this project are now formally ties, which is what they always were.
- **Date:** 2026-09-15

### 43. Nested CV: model selection did not inflate the system, but the simple baseline was under-tuned
- **Decision:** report the nested-CV estimate beside the fixed-configuration headline, and quote the tuned TF-IDF baseline too (`scripts/nested_cv.py`).
- **Alternatives:** keep quoting the fixed configurations chosen by bake-offs on the same 218 rows (winner's curse for the system; an untuned simple baseline).
- **Why:** the configuration is picked in an inner 4-fold CV and scored only on the outer fold (5×5). System: **0.639** (sd 0.027) nested vs 0.640 fixed, so selection bias is negligible. Simple baseline: **0.406** (sd 0.014) when char n-grams and C are searched fairly, vs 0.325 for the fixed word TF-IDF. The honest system-over-simple gap is about 0.23, not 0.31.
- **Cost:** the headline table keeps 0.328 for continuity, so the report has to show the tuned figure next to it or the gap is overstated.
- **Date:** 2026-09-15

### 44. Risk rules match word forms; escalation numbers move, old ones kept
- **Decision:** `refund\w*`, `vouchers?`, `lawyers?`, `attorneys?`, `lawsuits?`, `sued`, `wheelchairs?`, `funerals?`, `journalists?`, and a few more.
- **Alternatives:** leave the bug documented as failure mode §3.2 and unfixed.
- **Why:** `\brefund\b` missed "refunded" and auto-handled a refund request. The rule fired on only 16/30 hand-labelled `refund_compensation` messages (18/30 after the fix). Margin 0.10, same folds: auto-handled **53 (24%) at 0.094 [0.04, 0.20], 77/82 errors escalated**, was 55 (25%) at 0.109, 76/82. Across fold seeds: 24–28% coverage, 0.067–0.102 error, was 0.085–0.131. Rules only: 164 auto-handled at 0.341, was 167 at 0.347.
- **Cost:** the fix came from reading errors on the evaluation set, so the post-fix number is optimistic by construction. It is reported as such.
- **Date:** 2026-09-15

### 45. The Banking77 control was not matched on confusability; a harder control is added
- **Decision:** add a "10 most confusable intents" control (greedy on centroid cosine, 20 per class, n=200) and report it beside the random draws.
- **Alternatives:** keep only the 5 random 10-intent draws (0.965), matched on n and class sizes but not on how close the intents are.
- **Why:** mean intent-centroid cosine is 0.81 for random Banking77 draws, 0.91 for the hardest 10, and **0.96 for AmericanAir's taxonomy**. The same pipeline scores 0.982, **0.800** and 0.640. Much of the gap to Banking77 is how close this task's intents are to each other, not only label noise, so "the ceiling is the data, not the method" overstated what the control shows.
- **Cost:** centroid cosine depends on the embedding model (the same one throughout), and the hardest-10 set was chosen using all of its labels. It is a stress test, not an estimate.
- **Date:** 2026-09-15

### 46. The sendable composite hides the one dimension where the LLM wins
- **Decision:** report per-dimension paired tests beside `sendable`. Grounded LLM vs verbatim nearest: **addresses 1.61 vs 1.00, 24 better / 5 worse, Wilcoxon p=0.0008** (n=49, survives Bonferroni over 4). Grounded 1.67 vs 1.96 (p=0.017) and tone 1.92 vs 2.00 (p=0.046) do not survive it.
- **Alternatives:** keep "the LLM on top of retrieval has not been shown to add anything" (#39), based on sendable alone.
- **Why:** `sendable` requires grounded==2, and a `nearest` draft is by construction one of the evidence lines the judge reads, so it scores grounded≈2 automatically. The composite rewards copying a reply and hides whether the reply answers *this* customer.
- **Cost:** judge opinion only, still with no human agreement (§2.4). "Addresses the message" is precisely the dimension the human study must confirm before it is believed.
- **Date:** 2026-09-15

### 47. Annotator agreement is computed on all 218 rows; the 120-row figure was easier than the set
- **Decision:** `annotator_agreement.py` now scores every usable golden row by default. Agreement is **kappa 0.466, raw 0.518, n=218** (me vs nemotron-120b), and it is framed as a reference point, not a ceiling.
- **Alternatives:** keep `--n 120`, which took the *first* 120 rows in file order rather than a random sample.
- **Why:** the first 120 rows have half the ambiguity of the other 98 (8% vs 15% flagged), so 0.517 (raw 0.567) overstated agreement. On all 218, the classifier's out-of-fold accuracy (0.624) is *above* the independent annotator's raw agreement. A supervised model learns one annotator's conventions, so model-vs-author agreement cannot cap it.
- **Cost:** it is still model-human agreement. gpt-oss-20b was dropped (77% parse failures), ministral-3b agrees at kappa 0.214 (n=190), and gemini hit a quota. A second human remains the missing evidence.
- **Date:** 2026-09-15

### 48. Escalation is also checked against the brand's own DM handoffs, and the agreement is weak
- **Decision:** the harness reports how the router's escalations line up with AmericanAir's actual reply to each golden message: did the brand move the conversation to DM? The router never sees that reply.
- **Alternatives:** score escalation only as "did it catch the classifier's errors against my labels" (#33). That is circular: it measures agreement with the author, not whether a case needed a person.
- **Why:** it is the one escalation signal in the data that I did not produce. Result (n=218): the brand DMed 48 (22%). The router escalates 41/48 of those (85% [0.73, 0.93]) but also 124/170 of the rest (73%), Fisher p=0.088. Low margin predicts a DM at AUC 0.51, i.e. not at all. Only the money and booking rules line up with it: `refund_compensation` 37% DM and `rebook_reroute` 46% DM, both escalated 100%.
- **Cost:** DM is a noisy proxy (routine requests get DMed, some hard cases are answered publicly). But the conclusion stands: the confidence threshold escalates what the *model* finds hard, not what the *brand* treated as needing a person, and the report says so.
- **Date:** 2026-09-15

### 49. Second audit: one definition of the classifier, input normalisation in the agent, label provenance
- **Decision:** (a) `intents.system_head` and `intents.oof_predictions` replace 16 hand-copied model literals and 5 copies of the out-of-fold loop; the harness asserts that routing posteriors and predictions agree. (b) `agent.py` runs `text.clean` on incoming tweets, as the golden set and corpus were. (c) The hand-typed label batches are committed, with a test that rebuilds the golden set byte-identically from them. (d) Drafter prompts moved to versioned files, byte-identical.
- **Alternatives:** leave working code alone because the numbers are right.
- **Why:** (a) the #40 fold-mismatch bug was exactly a drifted copy of that loop. (b) raw tweets (handles, flight numbers, URLs unmasked) were being classified by a model that only ever saw masked text, a train/serve skew no test caught. (c) without the batch files, "hand-labelled" was an unverifiable claim. All evaluation outputs were verified unchanged: harness `--naive --offline`, the judge sheet, and cache replay.
- **Cost:** none to the numbers. The skew in (b) only affected the new-message CLI path, which no reported number used.
- **Date:** 2026-09-15

### 50. A post-draft guard: never auto-send a draft that promises a follow-up
- **Decision:** `agent.py` escalates any draft the router would auto-handle if it promises a colleague, follow-up or contact (`draft_unkept_promise`).
- **Alternatives:** trust the router plus the prompt rule, as REPORT §3.5 did ("these messages are escalated anyway").
- **Why:** re-checking §3.5 found that claim false. Of the 3 grounded drafts (n=50) that fall back to "a colleague will follow up", one was auto-handled (id 563596, predicted `track_baggage`): *"A colleague will follow up to assist you further."* Nothing in the pipeline keeps that promise, and it would have been published. The judge scored it safe=2 even though the rubric says 0 for claiming something is being handled, so the judge would not have caught it either.
- **Cost:** the harness routing numbers (24% auto-handled) are computed without drafts, so they do not include this guard. In deployment the auto-handled share can only be lower. The judge's miss is a finding for the judge-human study.
- **Date:** 2026-09-15

### 51. An independent review of the evaluation: six findings, all fixed or disclosed
- **Decision:** an independent adversarial audit of the code and methodology, run with no stake in the results. Every finding was re-verified on the data before acting on it.
- **Findings, verified:** (1) the nemotron annotator never answered, since thinking-on opens "We need to label", so kappa 0.466 was read off letter tokens with <1% mass. It is withdrawn; labels now need a bare letter or ≥50% letter mass. (2) The judge ignores unfilled `<phone>`/`<url>` placeholders: 28 `nearest` and 17 grounded drafts; "as-is" sendable is 48% and 55%. (3) Grounded-vs-ungrounded "sendable" is partly built in; per dimension, retrieval raises tone and safety, not addressing the customer (1.52 vs 1.60, p=0.20). (4) `\bdm\b` missed "DMs": 60 handoffs, not 48, and 81 of the 3,000 corpus replies are DM deflections. (5) The judge-agreement sheet showed different evidence from the judge and was re-paired by rebuilding; it now reads back hashed items. (6) The escalation threshold was picked on the evaluation rows; on held-out test rows it gives 30/135 auto-handled, 2 wrong, vs 62/82 errors caught by random escalation.
- **Also:** drafting now evaluated at n=218 (the n=50 sample was 4-10 points optimistic), a harness smoke test, and fixes to stale comments, a fail-open family guard and weak tests.
- **Cost:** a strong independent annotator and the latency/adversarial probe (`scripts/probe_agent.py`) could not run: NVIDIA 503, Mistral-large not on plan, Mistral-small rate-limited on 2026-09-15/16.
- **Date:** 2026-09-16
- **Erratum (same day, from a final read of the writing):** the DM proxy with "DMs" matched is 52/60 escalated vs 113/158, Fisher p=0.022, margin AUC 0.53. The valid annotator is ministral-3b at kappa 0.229 (n=182, bare letters only; #47's 0.214 used lenient parsing). Point (6) set held-out test rows (30/135 auto-handled, 2 wrong) against random escalation on all 218 rows (62 of 82 errors): different quantities, both valid, not a comparison.

### 52. A valid strong independent annotator: the classifier sits at its agreement level
- **Decision:** report nemotron-3-super-120b with thinking off as the independent annotator: kappa vs the author 0.584 [bootstrap 0.51, 0.65], raw 0.628, n=218, 0 parse failures. Every label is a bare letter (one reply "other"), token-0 letter mass min 0.516, median 0.98. The classifier's out-of-fold agreement with the author on the same 218 rows (repeat 0) is kappa 0.574 [0.50, 0.64], raw 0.624.
- **Alternatives:** keep ministral-3b (0.229, n=182) as the only valid annotator, as #51 left it; or reinstate the thinking-on 0.466, which #51 withdrew and which stays withdrawn.
- **Why:** the #51 run was blocked by provider outages, not by method. With thinking off the model answers instead of reasoning, and the #51 validity rule (≥50% letter mass or a bare letter) passes on every row, so this is a real label set. It is the first valid evidence for the "annotation ceiling" claim: an independent 120B model and the deployed classifier agree with the author equally.
- **Cost:** model-human, not human-human agreement. Nemotron over-predicts `report_disruption` (24 of its disagreements are refund→disruption, the classifier's own top confusion, §3.1), so shared bias may inflate it. gpt-oss-20b (reasoning, never a bare letter) and Gemini (quota) gave no usable pairs.
- **Date:** 2026-09-18

### 53. `ask_policy` is never auto-handled: the agent has no policy source
- **Decision:** add `ask_policy` to the always-escalate intents, with a per-intent reason string. Routing at margin 0.10 moves from 53 auto-handled, 5 wrong (0.094) and 77/82 errors caught to 48, 3 wrong (0.062 [0.02, 0.17]) and 79/82. Held out: threshold 0.11 -> 0.09, 30/135 -> 34/135 auto-handled, still 2 wrong. The agent end to end (with #50/#51 guards), replayed out of fold: 41/218 auto-sent, 1 wrong intent, 31/41 judged sendable.
- **Alternatives:** a regex for policy claims in the draft (the probe's regexes already missed both); escalating `loyalty_program` too (it also holds elite-member praise, a safe auto-handle); leaving the rule set alone and reporting the failure.
- **Why:** the first live robustness probe auto-sent 4 of 14 drafts, 2 of them policy stated as fact (bereavement fares, an elite upgrade claim), the *Moffatt v. Air Canada* failure. On the golden set, 5 `ask_policy` messages were auto-handled, and the one draft that got past the post-draft guards stated an unsourced procedure (id 2160895, judge grounded 0, safe 0). Retrieval supplies past replies, not policy, so the prompt rule is the only defence and the model breaks it.
- **Cost:** 5 fewer auto-handles (24% -> 22% coverage). Motivated by a 14-message probe and checked on the evaluation data, so the new routing error is optimistic. The loyalty policy claim remains an auto-send; a policy source, not another rule, is the fix.
- **Date:** 2026-09-18
