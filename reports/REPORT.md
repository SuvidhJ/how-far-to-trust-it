# Report — a Twitter support agent for AmericanAir, and how far to trust it

Every number below carries its n and the command that produces it. Classification,
escalation and drafting replay offline from committed data and a committed LLM
cache: `uv sync && uv run python -m support_agent.eval.harness --offline` took 3m24s
from a fresh clone with empty package caches and no API keys. The agent itself runs
on one message with `uv run python -m support_agent.agent "<tweet>"`. Decision
numbers (#n) refer to `reports/DECISIONS.md`.

## Summary

| question | answer | evidence |
| --- | --- | --- |
| Does the intent classifier beat the baselines? | Yes: macro-F1 0.640 vs 0.049 (trivial) and 0.328 (simple); 0.406 if the simple baseline is tuned as fairly as the system, which scores 0.639 under the same nested CV | n=218, 5x5-fold CV, #43 |
| Is 0.640 low because the method is weak? | Partly the task: the same pipeline scores 0.965 on random Banking77 intents, 0.800 on its 10 most confusable, and this taxonomy's intents are closer still. And partly the labels: an independent 120B LLM annotator agrees with me at kappa 0.584 [0.51, 0.65], and the classifier at 0.574 [0.50, 0.64], on the same 218 rows. The classifier is about as close to my labels as a strong independent reader, which is consistent with a label ceiling but does not prove one (§4.1) | #36, #45, #52 |
| Can it decide when to escalate? | It routes likely classifier errors to people, not need: 22% auto-handled at 6.2% intent error vs 37.6% overall (2 of 34 on held-out rows); with the post-draft guards the agent sends 19% alone, but its margin does not predict the cases the brand took to DM (AUC 0.53) | n=218, #44, #48, #51, #53 |
| Does grounding help drafting? | On tone and safety, yes (p<0.005); on addressing the customer, no. Sendable 61% vs 27%, but that composite partly rewards seeing the evidence by construction | n=214 each, paired tests n=210, #51 |
| Does the LLM beat copying the nearest past reply? | Borderline on "sendable": 61% vs 54%, p=0.053, or 55% vs 48%, p=0.050 once drafts with unfilled placeholders count as unsendable. Clearly better at addressing the message (1.52 vs 0.83 of 2), clearly less grounded (1.59 vs 1.95) | n=212 paired, #46, #51 |
| Does the judge agree with a human? | **PENDING** — sheet prepared, human scores not yet collected | see §2.4 |

**Verdict.** Trust it as a triage-and-draft assistant for a human agent, not as an
autonomous responder. It routes the messages it is likely to misread to people, and it
never auto-sends money, rebooking, legal, vulnerable-customer or placeholder-bearing
replies. What it would send alone is a fifth of this sample (41/218, 1 with the wrong
intent), mostly thanks and simple acknowledgements; the router alone, before the
post-draft guards, errs on 5-8% of what it auto-handles across fold seeds. Messages it
classifies as questions about rules or fees are never sent alone, because it has no
policy source (#53); one misread policy question is among the 41. Its drafts are sendable
as-is about half the time, per a judge that no human has validated yet.

## 1. Problem framing

**What the data can support.** The task asks for replies grounded in how the brand
*resolved* issues. Measured across all 2.8M rows, only 5-9% of brand replies are
followed by any customer acknowledgement and 46-69% get no further customer tweet
at all (#17). Outcomes are unobserved. The system therefore learns AmericanAir's
**response policy** — how it triages and answers in public — and no number here
shows a customer's problem was solved.

**Why AmericanAir.** Highest substantive-reply rate of the large brands, 73.7%
(36,531 exchanges). AppleSupport, the obvious pick, sends 52.6% of replies to DM
(26.9% substantive), so an agent grounded in it learns "please DM us" (#16).

**What "good" means for this brand.** On public Twitter a wrong reply is worse than
a slow one: an airline was held to a refund policy its chatbot invented
(*Moffatt v. Air Canada*, 2024). So good means: (1) the right intent, so the right
team sees it; (2) auto-handling only what is safe and routine, with a reason
string on every escalation; (3) replies in the brand's register that never state a
policy, refund or compensation absent from retrieved history.

**The system.**

| stage | chosen | beaten alternatives (same protocol, paired test) |
| --- | --- | --- |
| taxonomy | 10 intents induced from a 1,200-message *train* split (TF-IDF + KMeans k=20, LLM-named, merged by hand), frozen before labelling: report_disruption, rebook_reroute, track_baggage, refund_compensation, manage_booking, ask_policy, complain_service, praise_service, loyalty_program, other | — |
| embeddings | `mistral-embed` | nemotron-3-embed-1b: never better, but significantly worse only with a centroid head (corrected p=0.029; logreg p=0.28, #42) |
| classifier | logistic regression on embeddings | centroid, SVM, kNN, MLP, LLM zero-shot, LLM few-shot (#21, #24) |
| escalation | top-2 probability margin < 0.10, plus rules for money, legal threats, vulnerable customers | entropy (AURC 0.274 vs 0.201); max probability narrowly (0.227; bootstrap 95% CI of the difference [-0.054, -0.000], n=218) (#27) |
| retrieval | dense cosine over 3,000 past exchanges | BM25, TF-IDF, hybrid RRF, compared by intent match in golden-set leave-one-out retrieval, not on the 3,000-row corpus itself (#25) |
| drafter | nemotron-3-super-120b, thinking **off** | thinking on leaks reasoning; 550B model too slow to evaluate (#31, #35); llama-3.2-11b leaks nothing (0/218) but its quality was not judged |
| judge | gpt-oss-20b, rubric `prompts/judge_v1.txt`, 4 dimensions x 0-2 + sendable | the drafter's own model (#39) |

**What I chose not to build, and why.**

- **An LLM classifier.** In the bake-off (n=219, one split) few-shot scored 0.582 vs
  0.571 for the linear model on the same folds, intervals overlapping, at ~600x the latency (#21).
- **A three-tier action space, NLI groundedness checks, SetFit**: the task is a binary
  decision, and the project keeps a 15-minute reproduce, which torch-class dependencies
  break (#13).
- **Multi-intent labels.** 27.6% of training messages match two or more intent
  keyword rules; each gets one label under a written tie-break (the intent the
  customer wants *action* on). This costs accuracy on exactly those messages (§3.1).
- **Conversation state.** Each inbound message is classified alone.
- **Sending anything.** Drafts are suggestions; there is no Twitter integration.
- **Cost and latency per ticket** (`scripts/probe_agent.py`, n=20 live, free tier, sequential):
  median 2.4 s end to end, p90 4.8 s; the Mistral embedding call is 1.4 s of it and the
  draft 0.9 s (p90 3.4 s). 519 prompt + 30 completion tokens per draft, so 0.55M tokens per
  1,000 tickets; no price is quoted because the free tier has none. `reports/ROBUSTNESS.md`.

**Golden set** (`data/golden/`, 220 rows, 218 scored). Sampled from dev/test threads
only, seed-fixed, and only messages the brand replied to: 20 per intent from crude keyword strata plus 40 with no filter,
so rare intents get enough examples for an interval. One row per thread (#28).
Labelled by the author from the customer message alone, brand reply hidden,
against a rubric with a positive and a near-miss example per intent; 26 rows
flagged `ambiguous`. Two rows excluded, not deleted (append-only).

## 2. Results

### 2.1 Intent classification (n=218, 10 intents, 5 repeats x 5-fold CV)

| system | macro-F1 mean | sd over repeats | 95% bootstrap CI, one split* | accuracy* |
| --- | --- | --- | --- | --- |
| trivial: majority class | 0.049 | 0.000 | [0.035, 0.062] | 0.165 |
| simple: TF-IDF + logistic regression | 0.328 | 0.018 | [0.258, 0.374] | 0.431 |
| **system: mistral-embed + logistic regression** | **0.640** | 0.030 | [0.511, 0.658] | 0.624 |
| simple, tuned by nested CV (char TF-IDF chosen in 25/25 folds) | 0.406 | 0.014 | | |
| system, nested CV (embedder, head and C chosen inside each fold) | 0.639 | 0.027 | | |
| control: same pipeline, Banking77 matched n=218 / 10 intents (5 draws) | 0.965 | range 0.938-0.981 | | |
| control: Banking77, its 10 most confusable intents, n=200 | 0.800 | | | |
| reference: independent LLM annotator (nemotron-120b, thinking off) vs author (n=218) | kappa 0.584 | | [0.51, 0.65] | 0.628 |
| reference: the system's out-of-fold predictions (repeat 0) vs author (n=218) | kappa 0.574 | | [0.50, 0.64] | 0.624 |
| reference: small LLM annotator (ministral-3b) vs author, bare-letter answers (n=182) | kappa 0.229 | | [0.17, 0.29] | 0.341 |

\*From the first split alone. On that split the system scores 0.591, so the interval
describes that split, not the 0.640 repeated mean. Per class (first split): F1 from 0.79 (`praise_service`, n=21) down to 0.34
(`other`, n=18). With 9-38 examples per class, per-class recall intervals are
±0.15-0.20 wide; no per-class ranking below that resolution is claimed.

### 2.2 Escalation (n=218, margin threshold 0.10)

| router | auto-handled | error on auto-handled [Wilson 95%] | classifier errors sent to a human |
| --- | --- | --- | --- |
| trivial: auto-handle everything | 218 (100%) | 0.376 [0.31, 0.44] | 0/82 |
| simple: hard rules only | 148 (68%) | 0.318 [0.25, 0.40] | 35/82 |
| **system: rules + margin < 0.10** | **48 (22%)** | **0.062** [0.02, 0.17] | **79/82** |
| random escalation of the same 170 messages (expected) | 48 (22%) | 0.376 | 64/82 |
| held out: threshold 0.09 chosen on dev rows (n=83) by a fixed rule, scored on test rows | 34 of 135 | 0.059 [0.02, 0.19] | vs test base 0.363 |
| the agent end to end: router plus post-draft guards (#50, #51) | 41 (19%) | 0.024 [0.00, 0.13] | 81/82 |

These numbers follow two rule changes found during failure analysis: a word-form fix (§3.2, #44; before it, 55 at 0.109 and 76/82) and `ask_policy` joining the never-auto-handled intents after the robustness probe (§3.6, #53; before it, 53 at 0.094 and 77/82). Both were checked on this data, so the figures are optimistic. The cost is 91 escalations of messages the classifier had right. The full sweep
is printed every run. 0.10 was read off that sweep; the held-out row tests the choice
with a rule fixed in advance (the largest coverage whose dev error is at most a third of
the dev base error), which picked 0.09 from 18 dev auto-handles with 2 errors. The first four rows exclude
the agent's post-draft guards (unkept promises, placeholders, #50, #51); the last row is
the agent itself, replayed out of fold (`scripts/build_demo_cases.py`). Of its 41 sent
drafts, the judge calls 31 sendable (0.76 [0.61, 0.86]). Every escalation carries a
reason naming what to decide (the two intents it is torn between, or the phrase
that triggered a rule): `low_confidence` 100, `compensation_claim` 34,
`always_escalate_intent` 33, `public_escalation` 2, `vulnerable_customer` 1.

**Checked against a signal I did not produce.** Every golden row stores AmericanAir's
actual reply, which the router never sees. The brand moved 60 of 218 to DM. The router
escalates 54 of those 60, and 116 of the other 158 (Fisher p=0.010), but the margin
predicts a DM at AUC 0.53: the association comes from the money and booking rules.
The threshold escalates what the model finds hard, not what the brand needed a person
for (#48). The rules overlap the keywords used to sample the golden set, so on the
uniform stratum alone (n=40, replied-to messages only): 11 auto-handled, 1 wrong (#51).

### 2.3 Drafting (all 218 golden messages per arm, in the harness)

Judge gpt-oss-20b, a different family from the drafter; 0/872 drafts invalid after
re-running 9 free-tier 503s. "As-is" also fails drafts that still contain an unfilled
`<phone>`/`<url>` placeholder, which the judge ignores (#51).

| drafter | judged n | mean /8 | judged sendable [Wilson 95%] | placeholders | sendable as-is |
| --- | --- | --- | --- | --- | --- |
| trivial: one canned reply | 218 | 4.14 | 17% [0.13, 0.23] | 0 | 17% |
| simple: nearest past reply, verbatim | 216 | 6.67 | 54% [0.47, 0.60] | 28 | 48% [0.41, 0.54] |
| LLM, no retrieval | 214 | 5.84 | 27% [0.22, 0.33] | 0 | 27% |
| **LLM grounded in 4 retrieved replies** | 214 | **6.92** | **61%** [0.55, 0.67] | 17 | **55%** [0.48, 0.62] |

Paired McNemar, grounded vs nearest: 43 vs 26 discordant, n=212, **p=0.053** (as-is
42 vs 25, p=0.050); grounded vs ungrounded: 87 vs 14, n=210, p<0.001. The composite
flatters two arms by construction: a `nearest` draft *is* one of the evidence lines, and
only the grounded LLM saw the evidence it is graded against. Per dimension (Wilcoxon,
paired, Bonferroni over 4): against `nearest`, the grounded LLM addresses the message
far better (1.52 vs 0.83, 105 better / 10 worse) and is less grounded (1.59 vs 1.95),
both p<0.0001; tone and safety do not differ. Against no retrieval, it is better on
tone (1.90 vs 1.63) and safety (1.90 vs 1.78, p=0.0045) but *not* on addressing the
customer (1.52 vs 1.60, p=0.20). At n=50 the same arms read 71% / 64% / 31%: the
smaller sample was optimistic.

### 2.4 Judge-human agreement — PENDING

| agreement | n | kappa |
| --- | --- | --- |
| judge vs human, per rubric dimension | 40 drafts | **PENDING** |
| judge vs human, binary sendable | 40 drafts | **PENDING** |

The blind sheet (`data/golden/americanair_judge_sheet.txt`, 40 valid drafts, arm
identity hidden) shows exactly the evidence the judge saw, in its order, and scoring
reads the items back by hash. Blinding is partial: 22 of the 40 drafts are verbatim
copies of an evidence line, which marks them as `nearest`. Scoring needs the author's own scores in
`data/golden/americanair_judge_human_scores.txt`, then `uv run python scripts/judge_agreement.py --score`.
No model was used to stand in for the human. Until this row is filled, every
drafting percentage above is a model's opinion.

### 2.5 What a careless setup would have reported (`harness --naive --offline`)

| honesty step, undone | naive | audited | n |
| --- | --- | --- | --- |
| best single CV split, not the repeated mean | 0.674 | 0.640 | 218 |
| accuracy on the uniform sample, not stratified macro-F1 | 0.675 | 0.640 | 40 vs 218 |
| **all intent steps together** (thread twin kept too) | **0.725** | **0.640** | 40 vs 218 |
| retrieval index split by tweet: own conversation as top-1 (needs `data/interim`) | 10/218 | 0/218 | 218 |
| drafter's own model as judge: grounded sendable (n=50 sample) | 100% | 72.9% | 48 |

The inflation comes mostly from *reporting* choices; the thread guard bought nothing
on the golden set (0.632 with the twin kept), but matters in the retrieval index (#37).

## 3. Failure analysis — top 6

**3.1 A refund request is classified as the disruption that caused it.**
`refund_compensation` recall is 0.53 (n=30); 5 of its 30 go to `report_disruption`.
> "How can I get compensation for a 10 hour delayed flight that has resulted in my vacation being cancelled?" (id 2250493)

*Hypothesis:* a sentence embedding is dominated by the long disruption description;
the action ("compensation") is one token, and the tie-break rule "label the action"
is not something a similarity space encodes. *Test:* re-score these rows with the
action clause alone; if the prediction flips, add an action-verb feature and compare
under the same repeated CV. Operational impact is limited: all five are escalated,
three by the `compensation_claim` rule and two by low margin.

**3.2 A money request was auto-handled because the rule regex missed a word form.**
Before the fix, 6 of the 55 auto-handled messages were misclassified, and one of them
should never have been auto-handled at all:
> "I paid for upgraded seats but you bumped to a regular seat. How do I get refunded for that???" (id 845973; predicted `manage_booking`, margin 0.22)

*Cause, confirmed:* the rule was `\brefund\b`, which does not match "refunded". The
margin was confident, so nothing else caught it. *Measured since then:* the money
rule fired on only 16 of the 30 messages labelled `refund_compensation`, and on 18
after the word-form fix, which is now applied (§2.2, #44). The router still escalates
29 of 30, because the intent rule and the margin catch the rest. *Hypothesis:*
keyword rules have low recall on paraphrase ("I want my money back"). *Test:* label
should-escalate cases directly, which this project has not done.

**3.3 `other` is not a class, it is a residue.** F1 0.34 (n=18); 5 of 18 are
predicted `praise_service`.
> "Living life at the crossroads! On runway in Philly. <user> <url>" (id 1742998)

*Hypothesis:* `other` has no shared meaning, and upbeat travel chatter sits next to
praise in embedding space. *Test:* drop `other` as a training class, treat low margin
to all nine real intents as `other`, and compare macro-F1 over the same folds.

**3.4 The verbatim baseline sends another customer's reply.** A question about tipping
curbside staff (id 2286575) received the nearest past reply, a gate-check answer
ending *"We're so sorry, Steve."* The judge marked it unsendable for not addressing the
question; its reason never mentions the wrong name, and the rubric has no check for
it. *Hypothesis:* past replies carry case-specific names and details that similarity
does not penalise. *Test:* count agent-addressed names in `nearest` drafts, and add a
"wrong customer details" rubric item before re-running the judge-human study.

**3.5 The LLM deflects instead of answering when money is mentioned.** The drafting
prompt forbids stating any policy or compensation not in the evidence and says to
promise a colleague follow-up instead. Without retrieval, 20/50 drafts fall back
to that sentence; grounded, 3/50.
> customer: "Except it will cost us more to get home from there. <user> issues vouchers for its customers in such situations."
> grounded LLM: "We're sorry for the inconvenience. A colleague will follow up to discuss your concerns about additional costs." (judge: grounded 0, addresses 0, safe 2)

*Hypothesis:* the safety rule works, but it produces a promise nothing in the
pipeline keeps. *Tested* on the n=50 sample: one of the 3 grounded fallbacks was auto-handled (id 563596,
"thanks for not having available baggage service in LIT"), so it would have
published *"A colleague will follow up to assist you further."* The judge scored
17 of the 23 follow-up drafts safe=2, although the rubric says 0 for claiming
something is being handled. It caught them only through `grounded`. *Fix, applied:*
the agent escalates any auto-handled draft that promises a follow-up (#50).

**3.6 The agent auto-answered policy questions it had no source for.** The robustness
probe (`reports/ROBUSTNESS.md`, 14 adversarial messages written before the agent ran)
auto-sent 4 drafts, and its mechanical flags passed all 4. Read by hand, 2 of the 4 state
policy as fact, the *Moffatt* failure:
> customer: a bereavement-fare question. agent (auto-sent): "We don't have a specific bereavement fare policy. Please contact our Reservations team at 1-800-433-7300..."

*Confirmed on the golden set:* 5 `ask_policy` messages were auto-handled, and the one whose
draft survived the post-draft guards (id 2160895, "How do I add priority boarding if I've
already checked in?") was answered with an unsourced procedure the judge scored grounded 0,
safe 0. *Cause:* retrieval supplies past replies, not policy, so the prompt rule "state no
policy absent from the evidence" is the only defence, and the model breaks it. *Fix,
applied:* `ask_policy` is never auto-handled (#53); coverage 24% -> 22%. *Not fixed:* the
probe's loyalty-status message ("Executive Platinum members receive complimentary
upgrades...") is still auto-sent, because `loyalty_program` also covers praise from elite
members. A policy source is the real fix, not more rules.

## 4. What is misleading about my headline number?

The headline is **macro-F1 0.640**. Specifically:

1. **It measures agreement with me, and no second human has checked me.** One
   annotator, who also designed the taxonomy after reading the data. Earlier drafts
   quoted an LLM annotator at kappa 0.517, then 0.466. Both were artefacts: that
   reasoning model never answered, and its "labels" were read off letter tokens holding
   under 1% of the probability (#47, #51). Asked with thinking off, the same model
   answers every row and agrees with me at kappa 0.584 [0.51, 0.65] (n=218, #52), the
   same as the classifier (0.574). That is model-human agreement, and the model shares
   the classifier's pull towards `report_disruption`, so it bounds the ceiling loosely. A same-session re-label (kappa 1.00, n=45)
   measures memory (#29). Gains above 0.64 may mostly be fitting my own reading of
   borderline tweets.
2. **The sample is not the traffic.** Only messages the brand replied to were eligible, and 180 of 220 rows were drawn from keyword strata,
   which over-represent rare intents and under-represent messages that phrase an intent
   without its keyword. Macro-F1 weights `loyalty_program` (n=9) the same as
   `report_disruption` (n=38). On production traffic accuracy is what users feel, and
   on the 40-row uniform stratum it is 0.675 [Wilson 0.52, 0.80].
3. **Macro-F1 and accuracy differ, and the headline is the lower one.** Accuracy on
   the same predictions is 0.624 (per-class F1 0.34-0.79). Accuracy on the uniform
   stratum (0.675) or the best of five fold seeds (0.674) each inflate it; all the
   careless choices together give 0.725 (§2.5).
4. **The fold seed moves it more than any method choice.** A single split ranges
   0.591-0.674; the sd over repeats is 0.030 (#32). Every method comparison in this
   project is inside that band unless a paired test says otherwise.
5. **Escalation error is against my labels, on 48 messages.** 6.2% means "the classifier
   disagreed with the author", not "a support lead would have escalated". Against the
   brand's own DM handoffs, the margin carries no information (AUC 0.53, #48, #51). Across fold
   seeds it ranges 5.3-7.8% (Wilson [0.02, 0.17]), after two rule changes checked on this
   data and a threshold picked from a sweep over it. Earlier versions printed 12.3% (#40),
   10.9% (#44) and 9.4% (#53).
6. **The drafting numbers are a model's opinion, replayed from a cache.** No human
   agreement yet (§2.4). The drafter's own model as judge calls 48/48 sendable; this judge
   scores unkeepable follow-up promises safe=2 in 17 of 23 cases (n=50 sample, §3.5) and
   ignores unfilled placeholders. Cached responses make the variance zero by construction,
   and the first n=50 sample overstated the LLM and verbatim arms by 4-10 points.
7. **"Sendable" is not "resolved".** Even a perfect judge-human kappa would show the
   reply looks like AmericanAir's, not that it helped (§1).
8. **The comparisons were weaker than first reported**: an untuned simple baseline
   (0.406 when tuned, #43), a Banking77 control unmatched on confusability (#45), and
   p-values from McNemar counts pooled over CV repeats (#42).
9. **One brand, one collection window.** Nothing here transfers to another brand, or
   to AmericanAir's current policies, without re-labelling.

## 5. With one more week

1. **Judge-human agreement at n>=100 with a second human**, after adding a
   wrong-customer-details rubric item (§3.4).
2. **A second human annotator** on the golden set, and a day-later self re-label.
3. **Fill placeholders from the brand's contact data and mask agent names**, then
   re-judge: both copying and grounded arms lose 6 points to placeholders today.
4. **Rule recall**: label the money, legal and vulnerable cases the rules exist for.
5. **An outcome proxy**: test whether drafts resembling the 5-9% of replies that got a
   customer thank-you score differently.
6. **A policy source for the drafter** (the brand's published rules), so `ask_policy` can
   be answered instead of escalated (§3.6).

## 6. Decision log

`reports/DECISIONS.md` opens with the 14 decisions that mattered, each with its reason,
followed by all 53 entries in the order they were made. The alternatives and the evidence
are kept, and so are the corrections, which are added as new entries rather than
overwriting old ones (#23 by #42, #26 by #41, #33 by #40 and #44, #36 by #45, #39 by #46,
#47 and #48 by #51, #51's annotator gap by #52).

## Reproduce

```
uv sync
uv run python -m support_agent.eval.harness --offline          # §2.1, §2.2, §2.3 (~1 min)
uv run python -m support_agent.eval.harness --naive --offline  # §2.5
uv run python -m support_agent.agent --golden 3 --offline      # the agent on one message
uv run python scripts/nested_cv.py                             # §2.1 nested rows
uv run python scripts/annotator_agreement.py                   # §2.1 reference row (live calls)
uv run python scripts/judge_agreement.py --score               # §2.4, after human scores
uv run python scripts/control_banking77.py                     # needs network
```

## Citations

Data: Kaggle *Customer Support on Twitter* (Thought Vector, CC BY-NC-SA 4.0); Banking77
(Casanueva et al. 2020). Models via free-tier NVIDIA NIM, Mistral and OpenRouter APIs.
Methods: Wilson, McNemar, Cohen's kappa, Gwet's AC1, Nadeau-Bengio corrected t-test,
nested CV, risk-coverage (Geifman & El-Yaniv 2017), BM25 and RRF. Legal stakes: *Moffatt
v. Air Canada*, 2024 BCCRT 149. Full
list with versions and roles: README § Citations.
