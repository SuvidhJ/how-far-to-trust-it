# Research notes

> **Scope note.** This is background, not a plan. It informs *how* the project is
> built; `reports/REPORT.md` and `reports/DECISIONS.md` record *what* was built.
> Several techniques catalogued here are deliberately not built — see decision 13.
> The purpose of this file is that anything borrowed is cited and understood.

What already exists for this problem, what the literature says works, and what
each finding changes about our plan. Every claim here is traceable to a source
so the report can cite rather than assert.

---

## 1. Prior work on this exact dataset

**TweetSumm** (Feigenblat et al., EMNLP Findings 2021, [arXiv:2111.11894](https://arxiv.org/abs/2111.11894))
built a dialogue-summarisation corpus directly on twcs. Their reconstruction is
the citable recipe we should follow:

- traverse `in_response_to_tweet_id` recursively → **49,155 unique dialogs**
- drop dialogs with **<6 or >20 utterances** → 45,547
- drop dialogs with **more than two speakers** → **32,081**
- when several tweets reply to the same parent, sort by `created_at` — this is
  usually **one message split across tweets**, not multiple replies

Also on twcs: *Towards Automated Customer Support* ([arXiv:1809.00303](https://arxiv.org/abs/1809.00303)),
*Evaluating Empathetic Chatbots in Customer Service* ([arXiv:2101.01334](https://arxiv.org/abs/2101.01334)).
Public Kaggle work is almost entirely EDA, VADER sentiment and LDA topic
modelling — no rigorous intent taxonomy with a held-out evaluation. **The
evaluation rigour is the gap, and it is the gap this project sets out to fill.**

## 2. Intent induction — there is a real benchmark

**DSTC11 Track 2** (Gung et al., AWS AI Labs, [arXiv:2304.12982](https://arxiv.org/abs/2304.12982),
[code](https://github.com/amazon-science/dstc11-track2-intent-induction))
is the standard benchmark for inducing intents from human-human support
conversations. 34 teams participated.

**Metrics** (worth adopting wholesale — they make our taxonomy defensible):

- **ACC** — clustering accuracy after optimal cluster→label matching via the
  Hungarian algorithm. Primary ranking metric. Penalises both too many and too
  few clusters.
- **Clustering F1** — harmonic mean of purity (precision) and inverse purity
  (recall). Too-coarse clusters hurt precision; too-granular hurt recall.
- **NMI** and **ARI** as secondary.

**Calibration of expectations:** the winning system reached **69.8 ACC**; the
k-means baseline reached **55.8**. Intent induction is not close to solved, so a
mid-60s number on our own data is respectable, not a failure. Quoting this
guards the report against "why is your accuracy only X".

**What top teams did:** SCCL / deep embedded clustering dominated the top ten;
one used HDBSCAN. Encoders were `all-mpnet-base-v2` or `DSE-RoBERTa-large`.
Silhouette score was the common way to choose k.

**The single most transferable trick:** two HDBSCAN teams left noise points
unassigned and were punished by the metric. Propagating labels to noise points
(train a classifier on the assigned points, apply to the rest) moved one team
**from 33rd to 7th**. If we use HDBSCAN, we must label-propagate.

**Unsupervised pipeline** (Costa et al., [arXiv:2307.15410](https://arxiv.org/abs/2307.15410)):
SBERT → **UMAP** dimensionality reduction → **HDBSCAN** (no k needed, `-1` for
noise). Critically, they apply **NER masking before embedding** — otherwise
clusters form around entity names rather than intents. For twcs the analogue is
masking order numbers, URLs, handles and product names.

**Dial-In LLM** (Hong et al., [arXiv:2412.09049](https://arxiv.org/abs/2412.09049))
uses an LLM inside the clustering loop for refinement and for generating
human-readable cluster names — the practical way to turn clusters into a
taxonomy a human can read.

**Taxonomy design** (industry consensus): start with high-volume,
high-operational-significance intents; overlap is the most common failure mode,
so write sharp boundary definitions with near-miss examples; support multi-label
or explicit tie-break rules for genuinely mixed tickets; a two-level hierarchy
helps with confusable intents.

## 3. Classification — the baseline is stronger than it looks

**SetFit** (Tunstall et al., [arXiv:2209.11055](https://arxiv.org/abs/2209.11055)):
contrastive fine-tuning of a Sentence Transformer plus a logistic head. With
**8 labelled examples per class** it is competitive with fine-tuning RoBERTa
Large on 3k examples, and beats standard fine-tuning by **+19.3 points** at
N=8. This is the right third system for a project whose labelled set is 150–250
examples.

**Cost/latency evidence** ([arXiv:2602.06370](https://arxiv.org/abs/2602.06370)):
fine-tuned encoders cost **~$12–33 per 1M requests** vs **~$843–1,175** for LLM
prompting — two orders of magnitude — while the best encoder (RoBERTa, 94.84
macro-F1) trailed the best LLM by **under 2 F1**. Encoder p50 latency 234–622 ms
vs LLM p95 approaching 2 s. Few-shot prompting roughly doubled input tokens
without materially improving accuracy.

> **Implication for the report.** The honest headline may well be that a cheap
> encoder is within noise of the LLM at 1/30th the cost. That is a *finding*,
> not a failure, and it is exactly the kind of result a "results vs
> baselines" section exists to surface.

**Does intent classification even help?** *Do LLMs Need Intent?*
([arXiv:2509.05006](https://arxiv.org/abs/2509.05006)) finds explicit intent
prediction does **not** consistently improve response generation; direct
generation is often comparable or better. We should therefore **run this as an
ablation** (intent-conditioned drafting vs direct drafting) rather than assume
the pipeline shape. This is a cheap, high-credibility experiment.

## 4. Grounded reply drafting

- Hallucination in production support agents is **primarily a retrieval
  failure**, not a generation failure. Fixes: better chunking, confidence
  thresholds, and explicit instructions to decline when evidence is weak. So
  **retrieval must be measured separately** from generation.
- **RAGAS faithfulness** decomposes a response into atomic claims and checks
  each against retrieved context: *score = supported claims / total claims*.
  This gives a mechanical groundedness number that does not depend on a holistic
  judge opinion. ([docs](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/))
- **Do not use BLEU/ROUGE as a headline.** Liu et al., *How NOT To Evaluate Your
  Dialogue System* ([arXiv:1603.08023](https://arxiv.org/abs/1603.08023)), showed
  n-gram metrics correlate weakly or not at all with human judgement of dialogue
  responses. Report them at most as a secondary sanity check, with that citation.
- RAG over historical tickets specifically: knowledge-graph RAG for customer
  service QA (SIGIR 2024, [arXiv:2404.17723](https://arxiv.org/abs/2404.17723))
  and RAG4Tickets ([arXiv:2510.08667](https://arxiv.org/abs/2510.08667)).

## 5. Escalation — there is a named task and a proper metric

**Machine-Human Chatting Handoff (MHCH)**, Liu et al., AAAI 2021
([arXiv:2012.07610](https://arxiv.org/abs/2012.07610),
[code](https://github.com/WeijiaLau/MHCH-DAMI)) frames handoff as per-utterance
sequence labelling (transferable vs normal) and proposes **GT-T (Golden Transfer
within Tolerance)** — a metric that accepts a *slightly early* handoff as
near-correct, because escalating early is far cheaper than escalating late.
That asymmetry is exactly our escalation cost structure and is worth citing when
we justify our operating point.

**Selective prediction** is the other half. Rather than defending a single
threshold, plot the **risk–coverage curve** and report **AURC** — the standard
way to present an abstention system. It lets a reader pick their own operating
point instead of trusting ours, which is a strong honesty signal. (See
*Overcoming Common Flaws in the Evaluation of Selective Classification Systems*,
NeurIPS 2024.)

**Signals used in practice:** low model confidence, repeated/looping intent,
negative sentiment or frustration, unusually long conversation, conflicting
knowledge, high customer value.

## 6. LLM-as-judge — known failure modes we must control

| Bias | Magnitude reported | Control |
| --- | --- | --- |
| Position | up to **75%** preference for first position | randomise order, evaluate both orders |
| Self-preference | **10–25%** for own generations | judge from a different provider |
| Verbosity | favours longer answers | penalise length explicitly in the rubric |

Strong judges reach **~80% agreement with humans**, about the same as
human-human agreement (MT-Bench / Chatbot Arena), but agreement is ~70% on some
benchmarks — so the agreement number must be measured on *our* data, not assumed.

**Agreement thresholds.** Cohen's κ (Landis & Koch): 0.41–0.60 moderate,
0.61–0.80 substantial, 0.81+ almost perfect. Krippendorff's α: **≥0.667** for
tentative conclusions, **≥0.800** for firm ones. Use κ for nominal labels with
two raters; α when the scale is ordinal or ratings are missing.

## 7. What "good" means commercially — and the metric trap

The realistic deployment for this kind of system is a **shared-inbox helpdesk**:
email, chat, voice and social messages routed to human agents, with AI for routing,
drafting and insights. That makes the right product framing **agent-assist**, not an
autonomous bot: the system drafts, a human sends. Escalation means *do not auto-send*.

**The industry metric trap, which belongs in our "misleading number" section:**
Intercom's Fin markets a **76%** resolution rate, but independent production
figures are **45–53%**, and a 500-ticket small-business test landed at **38%**.
The gap exists because "resolution" is counted as *conversation closed without a
human*, not *customer problem actually solved*. Deflection and resolution are
different things. We must define which one we measure and not conflate them.

**Legal stakes.** In *Moffatt v. Air Canada* (2024) a tribunal held Air Canada
liable for a bereavement-fare policy its chatbot invented, rejecting the argument
that the chatbot was a separate entity. Reported hallucination rates for support
chatbots run **3–27%**. This is the concrete justification for a hard rule:
**never state policy that is not present in retrieved evidence — escalate
instead.**

---

## 8. What this changes about our plan

1. **Adopt DSTC11 metrics** (ACC, clustering-F1, NMI, ARI) to evaluate the intent
   taxonomy itself, and quote the 69.8 / 55.8 benchmark to frame our number.
2. **Pipeline for induction:** mask entities → SBERT → UMAP → HDBSCAN →
   label-propagate noise → LLM names the clusters → human freezes the taxonomy.
3. **Three systems, not two:** trivial (majority + canned), simple (TF-IDF +
   linear), and SetFit — then the LLM must beat *all three* to justify its cost.
   Report cost-per-1k-tickets alongside accuracy.
4. **Run the intent-necessity ablation** (intent-conditioned vs direct drafting).
   Published evidence says it may not help; testing it is cheap and credible.
5. **Escalation reported as a risk–coverage curve + AURC**, with a stated
   operating point and an asymmetric-cost argument borrowed from GT-T.
6. **Judge from a different provider than the drafter**, randomise order, mask
   identity, penalise verbosity; report Cohen's κ against our own labels.
7. **Groundedness measured mechanically** via claim-level faithfulness, with
   retrieval quality measured separately from generation quality.
8. **Frame as agent-assist**, define resolution vs deflection explicitly, and use
   the Fin 76%-vs-45% gap and Air Canada as the motivation for conservative
   escalation.

## 9. Dataset facts established by our own exploration

Measured on the full 2,811,774 rows (see `scripts/peek.py`):

| Brand | support msgs | "DM us" deflection | has link | signed by agent | latin-script pairs |
| --- | --- | --- | --- | --- | --- |
| AmazonHelp | 169,840 | **0.7%** | 41.3% | **89.5%** | 158,790 (94.1%) |
| AppleSupport | 106,860 | 52.5% | 75.4% | – | 106,502 (99.9%) |
| Uber_Support | 56,270 | 35.9% | 51.3% | – | 56,150 |
| SpotifyCares | 43,265 | 30.8% | 50.5% | 0.0% | 43,056 |
| Delta | 42,253 | 16.5% | 15.3% | 0.2% | 42,105 (100%) |
| TMobileHelp | 34,317 | **81.9%** | 48.2% | – | 34,205 |
| comcastcares | 33,031 | 71.5% | 4.0% | – | 32,913 |

- **Deflection varies 100×.** TMobileHelp deflects 81.9% of replies to DM and
  AppleSupport 52.5% — those brands' public replies contain almost no resolution
  content to ground on. AmazonHelp (0.7%) and Delta (16.5%) do.
- **Replies are split across tweets.** 139,508 parents have 2 replies and 32,969
  have 3+. Delta splits with `*AOS 1/2` … `2/2`. A naive parent-child join
  therefore **double-counts exchanges** — merge before counting or modelling.
- **Agent signatures are brand-specific and must be stripped**: AmazonHelp `^QJ`
  (89.5% of replies), Delta `*AOS`, SpotifyCares `/TF`, British_Airways 18.1%.
  They are also a leakage risk if an agent correlates with an intent.
- **Language contamination is real**: ~5.9% of AmazonHelp exchanges contain
  CJK/Arabic/Cyrillic script; a naive non-ASCII test says 31.8% because it also
  catches emoji and smart quotes. Use proper language ID, not a character test.
- Thread structure: 787,346 customer-rooted threads; 875,292 first replies;
  395,310 second-level replies.

**Brand recommendation: Delta** (or AmericanAir/British_Airways as alternates).
Not the largest, but it has the properties that matter: low deflection (16.5%),
effectively no signature noise (0.2%), ~100% latin-script, substantive on-channel
answers containing real policy ("it can take up to 7 business days"), and an
airline intent space that is naturally crisp — delay, baggage, refund, seat
change, miles, check-in. AmazonHelp has the lowest deflection but pays for it
with 89.5% signature noise, heavy link-only replies and a much more diffuse
intent space. **Confirm with the thread-shape scoring in milestone 1 before
committing.**

---

# Second sweep — competitions, products, production systems

## 10. DSTC12: the newer, closer benchmark

**DSTC12 Track 2, Controllable Conversational Theme Detection**
([arXiv:2508.18783](https://arxiv.org/abs/2508.18783),
[code](https://github.com/amazon-science/dstc12-controllable-conversational-theme-detection))
supersedes DSTC11 for our purposes in two ways:

- **Themes, not intents.** A theme is a *user-facing summary of the core
  inquiry*, not a fixed label for downstream routing logic. That is closer to
  what this project needs than a classic intent schema.
- **Controllable granularity.** The input includes user preferences on how coarse
  or fine the clusters should be — directly addressing the "how many intents?"
  problem that DSTC11 left to silhouette heuristics.

**It also evaluates the cluster *names*, which DSTC11 did not**: ROUGE-1/2/L,
cosine similarity, BERTScore P/R/F1 and an LLM-based score over generated theme
labels, alongside the clustering metrics. Worth adopting — our taxonomy's labels
are read by a human, so label quality is part of the deliverable.

**Winner, KSTC** ([ACL](https://aclanthology.org/2025.dstc-1.5/)): embed dialogue
contexts, coarse cluster, LLM generates a theme label per cluster, **LLM refines
the clusters using that generated label**, then relocate utterances to match the
requested granularity. It also proposes *Task Independent Slots*: build the label
from an extracted **verb + noun slot-value** pair. Same shape as Banking77/CLINC
labels ("track order", "request refund") — a cheap, concrete naming convention.

## 11. Production systems worth copying

### Uber COTA ([engineering blog](https://www.uber.com/blog/cota/))

The closest production analogue to this project, and the most transferable.

- Ranks **over 1,000 candidate solutions** across hundreds of ticket types and
  surfaces the **top three** to a human agent, who picks. Agent-assist, not
  auto-send.
- Stack: preprocessing, TF-IDF + **LSA topic modelling**, then **cosine
  similarity between the ticket vector and each candidate-solution vector as
  features**, then a random-forest *pointwise ranker* over (ticket, solution)
  pairs.
- **The key result:** the cosine-similarity-feature formulation beat direct
  multi-class classification on topic vectors by **25% relative accuracy**, and
  trained 70% faster. Reframing "which of N intents" as "score these
  (ticket, candidate) pairs" is an architectural win, not a tuning detail.
- Evaluated by **online A/B test** across thousands of agents: ~10% accuracy
  gain, **~10% reduction in ticket handling time**, CSAT up a few points, and
  coverage of more than 90% of inbound tickets.

### Gmail Smart Reply (Kannan et al., KDD 2016, [paper](https://research.google.com/pubs/pub45189.html))

Solves our biggest safety problem *architecturally* rather than with prompting.

- Responses are drawn from a **curated response set**, not freely generated,
  specifically to prevent inappropriate responses. You cannot hallucinate a
  refund policy if the system can only emit a vetted, brand-authored reply.
- Semantic clustering of candidate responses by graph-based **semi-supervised
  label propagation** from a small set of manually defined clusters.
- **Diversity enforcement**: cluster to drop near-duplicate suggestions, and
  ensure the suggestion set is not all-affirmative or all-negative.
- Later work scores with a factorized dot-product model plus approximate nearest
  neighbour search over precomputed vectors
  ([arXiv:1705.00652](https://arxiv.org/abs/1705.00652)).
- Shipped scale: roughly 10% of all mobile Gmail replies.

### Intercom Fin ([AI engine](https://fin.ai/ai-engine))

Three sequential phases per message: **Refine Query, Generate Response, Validate
Accuracy**. Proprietary `fin-cx-retrieval` and `fin-cx-reranker` models handle
retrieval and reranking. Claimed ~0.1% hallucination rate.

The load-bearing detail: **if confidence is too low, Fin asks a clarifying
question or hands off — it does not guess.** So the real action space is
three-way, not two:

> auto-handle · **ask a clarifying question** · escalate to a human

This project scopes routing to auto-handle vs escalate. Naming the third action and
deliberately scoping it in or out is exactly the kind of framing decision the
report should surface.

### Klarna — the full arc, success *and* reversal

- **Feb 2024**: the AI assistant handled **2.3M conversations in its first
  month**, two-thirds of all chats, work equivalent to **700 agents**; resolution
  time **11 min to under 2 min**; 25% drop in repeat inquiries; ~$40M profit
  impact.
- **May 2025**: Klarna began rehiring humans. CEO Siemiatkowski: *"We went too
  far... We focused too much on cost. The result was lower quality."* The system
  handled volume fine but not complexity — edge cases, emotionally charged
  contacts and multi-step resolution degraded CSAT.
- **The diagnosis that matters to us:** they measured and scaled on throughput
  and cost *before* measuring customer-outcome quality with equal rigour.

That is this project's thesis, stated by a real company at real cost. It
belongs in the problem framing: **the reason the proof is worth more than the
system is that Klarna shipped the system without the proof and had to walk it
back.**

## 12. Escalation is out-of-scope detection

"Should this be auto-handled?" is the well-studied **out-of-scope (OOS) intent
detection** problem. **CLINC150** is purpose-built for it: 150 in-scope intents
plus **1,200 explicitly out-of-scope queries**.

Methods: DROID ([arXiv:2510.14110](https://arxiv.org/abs/2510.14110)), DETER
(dual encoders plus threshold re-classification,
[arXiv:2405.19967](https://arxiv.org/abs/2405.19967)), multi-cluster boundary
learning. The consistent finding is that **a well-calibrated threshold is a
strong baseline** that rejects OOS without hurting in-domain accuracy. Our
escalation baseline should therefore be an honestly calibrated threshold, and
anything fancier has to beat it.

## 13. Do not ask the model how confident it is

The most directly actionable finding of this sweep, because it is the thing most
support-agent evaluations get wrong.

- **Verbalized confidence** ("rate your confidence 0-100") is a **weak signal**,
  consistently outperformed by simple token probabilities. Instruct-tuned models
  at 3-9B show **ceiling rates above 90%** — they report near-maximum confidence
  regardless of whether they are right.
- **Self-consistency** (sample twice, measure agreement) beats Platt-scaled token
  probabilities on both Brier score and AUROC, and **two samples suffice**
  ([OpenReview](https://openreview.net/forum?id=66D3rZrNjV)).
- Combining verbalized confidence with self-consistency does best of all.

So the escalation signal should be logprob- or self-consistency-based, and we can
*demonstrate* the gap against verbalized confidence on our golden set. Cheap
experiment, genuinely interesting result, and it directly contradicts the obvious
approach.

## 14. Groundedness without paying for a judge

NLI entailment is the standard cheap baseline for faithfulness: decompose the
draft into claims, then check each against retrieved evidence with an off-the-
shelf entailment model (**DeBERTa-v3-large fine-tuned on MNLI** is the usual
backbone). See *With a Little Push, NLI Models can Robustly and Efficiently
Predict Faithfulness* ([arXiv:2305.16819](https://arxiv.org/abs/2305.16819)) and
Luna ([arXiv:2406.00975](https://arxiv.org/abs/2406.00975)).
**SelfCheckGPT** ([arXiv:2303.08896](https://arxiv.org/abs/2303.08896)) is a
zero-resource black-box alternative using sampled-response consistency.

NLI works best exactly where we are — a clear source document, i.e. retrieved
prior exchanges. It yields a groundedness number that costs no API calls and does
not inherit judge bias.

## 15. Smaller practical notes

- **Embeddings**: the MTEB average is misleading — a model strong at
  classification may be weak at retrieval. Pick per task and say so.
  `all-mpnet-base-v2` is the DSTC11 reference point; `bge-small-en-v1.5` plus a
  linear head is a well-regarded accuracy/speed compromise that runs on CPU.
- **Banking77 SOTA** is ~94% (ModernBERT-base 93.99 acc / 94.01 macro-F1;
  SPACE 2.0 94.77). Context worth quoting: Banking77 is clean, single-sentence
  and balanced. twcs is none of those, so our numbers will be lower, and that is
  a property of the data rather than of the method.
- **Active learning**: uncertainty sampling is the standard query strategy, but
  it **skews class balance** and causes label shift — a real hazard when
  hand-labelling only 150-250 examples. Stratify explicitly instead of letting
  uncertainty choose the set.
- **Eval tooling**: DeepEval is fully LLM-judged; promptfoo's assertions often
  need no judge call. We write our own thin harness — fewer dependencies, and the
  15-minute reproduce budget rewards it — but should cite the convention followed.
- **Prompt optimisation**: DSPy **MIPROv2** jointly optimises instructions and
  few-shot demonstrations by Bayesian search. A credible "with one more week"
  item; not worth the dependency now.

## 16. The architecture this sweep implies

Combining Smart Reply's curated response set, COTA's pair-ranking and top-3
surfacing, and Fin's validate-then-handoff gives a **three-tier action space**
that is safer and more defensible than free generation:

| Tier | Action | Mechanism | Hallucination risk |
| --- | --- | --- | --- |
| 1 | **Auto-send** | Select from a curated set of response templates mined and clustered from the brand's own historical replies | ~zero by construction |
| 2 | **Draft for agent** | LLM writes a reply grounded in retrieved historical exchanges; a human reviews before sending | bounded, human-gated |
| 3 | **Escalate** | Route to a human with a reason string and the retrieved evidence | n/a |

Tier 1 is the direct answer to *Moffatt v. Air Canada*: a system that can only
emit brand-authored text cannot invent a refund policy. Tier 2 is where the LLM
earns its cost. Tier 3 is where calibrated abstention lives.
