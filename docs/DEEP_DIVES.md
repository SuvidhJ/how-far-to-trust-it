# Four candidate extensions, costed

**Recommendation: brand #2 transfer (candidate 1).** The argument is in the last
section; the costing that supports it is below.

This document decides what to build next. It exists because four ideas survived the
project's own backlog (`reports/REPORT.md` §5) and they cannot
all be built. Each is costed against the constraints that actually bind here:

- **Free-tier LLM providers only.** NVIDIA NIM, Mistral, OpenRouter. No price per token,
  but hard rate limits: the free NVIDIA tier 429s at roughly 4 concurrent requests and
  503s for hours at a time.
- **`LLM_MAX_CALLS_PER_RUN` defaults to 1200** (`src/support_agent/config.py:70`). A run
  that needs more than that is not one run.
- **One person's evenings.** No parallel labelling, no annotation vendor.
- **`data/golden/` is append-only and hand-labelled.** New labels cost the author's own
  time and cannot be machine-generated without destroying the thing the project is
  built on. Today it holds 220 rows, 218 scored.
- **Reproduce stays offline.** `harness --offline` replays from
  `data/cache/llm_cache.jsonl`; anything added has to survive that.

Every number below carries its n and the file it came from. Numbers I measured today are
marked **[measured today]**; numbers I could not verify are marked as such.

---

## 1. Brand #2 transfer

### What it produces

A **labels-vs-macro-F1 curve** for a second brand: macro-F1 at 0, 40, 80, 120, 200 hand
labels, against the same two baselines and the same 5x5 CV protocol. Plus a zero-shot
transfer point — the AmericanAir classifier applied to brand #2 with no new labels at all.

The claim it unlocks: the project can today only say the opposite. `reports/REPORT.md` §4
caveat 9 reads *"One brand, one collection window. Nothing here transfers to another
brand, or to AmericanAir's current policies, without re-labelling."* That is an admission,
not a measurement. This candidate converts it into a number: *how much does it degrade,
and how many labels buy it back*. No other candidate here touches a caveat the report has
already conceded in writing.

### Cost

- **LLM calls, intent-only version: ~150.** The AmericanAir retrieval index of 3,000 rows
  cost **94 embedding API calls** in 71.1 s (`data/processed/americanair_index.manifest.json`),
  and the golden rows plus queries add roughly another 20-50. A labels-vs-F1 curve needs
  embeddings and a logistic regression — nothing else. Comfortably inside the 1200 ceiling.
- **LLM calls, full replication including drafting: ~1,310** — over the ceiling, so it
  would need two runs or a raised limit. Arithmetic from `reports/results.json`, not
  measured: 2 LLM drafting arms x 218 = 436 draft calls (the `canned` and `nearest` arms
  make none), plus 4 arms x ~218 = ~872 judge calls.
- **Hand labelling: the whole real cost.** 200 rows at the existing rubric. *Estimate,
  not recorded anywhere in the repo:* 30-45 s per message, so ~30-45 min for the first 60
  rows and 2-3 hours for a full 200. The curve is built incrementally, so labelling can
  stop at any point and still yield a point on it.
- **New dependencies: none.** The entire data path is already brand-parameterised:
  `scripts/make_sample.py` (`--brand`, line 30), `scripts/build_index.py` (line 46),
  `scripts/sample_golden.py` (line 55), `scripts/build_golden.py` (line 70), all
  defaulting to `settings.brand` (`config.py:60`).

### Rough time

**2-3 evenings (estimate).** One evening for the pipeline run and the curve harness, one
to two for labelling. The first curve point is reachable in a single evening.

### Main risk

**Picking a brand whose taxonomy does not fit, which turns a label-efficiency measurement
into an uninterpretable taxonomy mismatch.** The 10 intents (`report_disruption`,
`rebook_reroute`, `track_baggage`, ...) are airline-specific by construction — induced
from a 1,200-message AmericanAir train split (`REPORT.md` §1). Run on SpotifyCares, a low
F1 would mean "wrong taxonomy", not "needs more labels", and the headline number would be
unreadable. Mitigation is to pick a second airline deliberately and say which question is
being answered: same-industry transfer measures **label efficiency**; cross-industry
transfer measures **taxonomy portability**. They are different experiments and should not
be run as one.

Secondary risk: the second brand's golden set is labelled by the same single annotator, so
it inherits caveat 1 of `REPORT.md` §4 intact. It does not make that caveat worse, but it
does not fix it either.

### Evidence today

- The brand-selection analysis behind DECISIONS #16 (a private working note, not in
  this repository) holds the brand table: **British_Airways 70.7% substantive replies, 9.2% customer thanks after,
  61.1% no follow-up**, against AmericanAir's 73.7% / 7.9% / 55.7%. BA is the closest
  match on channel behaviour of any brand measured, which is what makes it the clean
  choice for a label-efficiency reading.
- Excluded on evidence and still excluded: AppleSupport (52.6% deflects to DM, 26.9%
  substantive) and AmazonHelp (89.5% agent-signature noise) — DECISIONS #16.
- The AmericanAir pipeline reproduces from a clean clone offline in ~3m24s
  (`REPORT.md` header), so the second brand is a re-run, not a rebuild.
- Baseline to beat: macro-F1 **0.640** (sd 0.030 over 5 repeats, n=218) vs simple 0.328
  and trivial 0.049 (`reports/results.json`).

---

## 2. A policy source + tools, so `ask_policy` can be answered

### What it produces

An **auto-handle rate and a groundedness score for `ask_policy` specifically**, where both
are currently zero by rule. Today the router never auto-handles a message it classifies as
`ask_policy` (DECISIONS #53) — with the honest exception the report already records: 1 of
the 41 drafts the agent sends alone is a *misread* `ask_policy`, classified as something
else (`REPORT.md` §2.2, Verdict).

The claim it unlocks: *the agent can answer a policy question with a citation to the
policy it came from*, which is the fix `REPORT.md` §3.6 names explicitly — *"A policy
source is the real fix, not more rules."*

### Cost

- **LLM calls: moderate, and the eval is the expensive half.** Embedding a small policy
  corpus is tens of calls. Re-drafting and re-judging the affected rows is ~15 draft +
  ~15 judge calls per arm on the golden `ask_policy` rows. Cheap on the free tier.
- **Hand labelling: the real cost, and it is the wrong shape.** Scoring a policy answer
  needs *correct* answers, not intent labels. The author would be writing reference
  answers, not picking from 10 classes — slower per item and less mechanical.
  *Estimate: 3-5 min per item.*
- **New dependency: yes, and it is the awkward part.** Real airline policy text has to be
  acquired (fetch + HTML parse, or manual collection), licensed, and committed. The repo
  currently commits data under CC BY-NC-SA 4.0 (`DATA_LICENSE.md`); scraped airline policy
  pages are not covered by that and would need their own answer.

### Rough time

**3 evenings (estimate).** One to collect and clean the policy corpus, one for retrieval
plus tool-call plumbing, one for evaluation — and the evaluation is the weakest of the four.

### Main risk

**The number lands on n=15 and cannot support any claim.** `ask_policy` has **support 15**
in the golden set, with F1 0.516 and a 95% interval of **[0.301, 0.752]**
(`reports/results.json`, `per_class`). An interval that wide already fails to distinguish
"works" from "does not". Measuring a *new* capability on those same 15 rows produces a
percentage with an interval spanning most of the unit line — exactly the kind of number
this project's own rules call worthless. Getting to a usable n means appending ~40 more hand-labelled
`ask_policy` rows first, which is a labelling project in its own right and is not in this
candidate's cost above.

Second risk, structural: the twcs dump is from 2017. Today's published AA policy is not
the policy the historical replies were written under, so a "correct" answer judged against
current policy is wrong by construction for the historical cases. The corpus and the
evaluation set would disagree about what year it is.

### Evidence today

- Supporting: the failure is real and documented twice. The live probe auto-sent 4 of 14
  adversarial drafts and **2 of the 4 state policy as fact** — a bereavement-fare answer
  and an Executive Platinum upgrade claim (`reports/ROBUSTNESS.md` §1, §3). The mechanical
  flag column passed both.
- Supporting: on the golden set, 5 `ask_policy` messages were auto-handled pre-#53, and the
  one whose draft survived the post-draft guards (id 2160895) was answered with an
  unsourced procedure the judge scored **grounded 0, safe 0** (`REPORT.md` §3.6).
- Supporting: the legal stake is named and cited — *Moffatt v. Air Canada*, 2024 BCCRT 149.
- Undercutting: rules already contain the damage at a known cost. #53 moved routing from
  53 auto-handled / 5 wrong (0.094) to **48 auto-handled / 3 wrong (0.062 [0.02, 0.17])**,
  79/82 classifier errors caught, for 2 points of coverage (24% -> 22%).
- Undercutting: the residual hole is *not* `ask_policy`. The probe's loyalty-status policy
  claim is still auto-sent, because `loyalty_program` also carries elite-member praise
  (`ROBUSTNESS.md` §3). A policy source aimed at `ask_policy` does not close that one.

---

## 3. An outcome proxy from thank-you replies

### What it produces

A **weak resolution label**: for each historical brand reply, whether the customer's next
message in the thread thanked them. With it, the project could ask whether drafts
resembling thank-you-earning replies score differently — `REPORT.md` §5 item 5.

The claim it reaches for: something about *outcomes*, which the project currently lacks
entirely. `REPORT.md` §4 caveat 7: *"'Sendable' is not 'resolved'."* This is the deepest
caveat in the report, which is what makes this candidate tempting.

### Cost

- **LLM calls: near zero for the labels** (a regex over a committed parquet), moderate for
  any re-judging that follows.
- **Hand labelling: required, and not optional.** A lexical thank-you regex is not a
  resolution label until someone checks it. *Estimate: 100-150 rows to establish precision.*
- **New dependency: none.** DuckDB over `data/interim/americanair_exchanges.parquet` is
  already how I measured this.

### Rough time

**2 evenings (estimate)** — but see the risk; the second evening likely ends in a result
that cannot be published as a resolution claim.

### Main risk

**It is not measurable on the evaluation set, and the label does not mean what it needs to
mean.** Two problems, the first fatal on its own:

1. **[measured today]** Of the **220** golden rows, only **65** have any later customer
   message in the same thread (dev 25, test 40), and only **5** have a follow-up containing
   a thanks token (dev 1, test 4). A second, independent query with a looser token set
   (`thank`/`thx`/`ty `) reproduced the 65 exactly and found 9 rather than 5, so the count
   is single-digit either way. An outcome proxy cannot be evaluated on this project's
   evaluation set. It would need a purpose-built sample from the wider pool — where the
   positives do exist: **[measured today]** train 954 / 24,477 rows, test 288 / 7,062, dev
   109 / 3,480.
2. **Thanks is not resolution, and sometimes means the opposite.** The committed retrieval
   corpus contains, verbatim, `data/processed/americanair_index.jsonl` id 285815:
   *"we missed our flight because your rude staff didn't think to inform their customers
   about changing the boarding gate so thank you very much"*. A lexical proxy scores that
   as a positive outcome. Beyond sarcasm, the proxy is confounded by intent —
   `praise_service` messages are thanked-adjacent by definition — so any correlation with
   draft quality may be intent composition, not resolution.
3. **The causal direction is unavailable.** The label attaches to *AmericanAir's* reply,
   not to the agent's draft. Nothing here observes the outcome of a reply the agent wrote.

### Evidence today

- **[measured today]**, on `data/interim/americanair_exchanges.parquet` (35,019 exchanges,
  25,706 threads, English-only, per its manifest), via a DuckDB `lead()` over `customer_at`
  within `thread_id`, with a case-insensitive `thank|thx|tysm|apprecia|kudos|grateful`
  match on the following customer message: **9,313 of 35,019 brand replies (26.6%) have any
  later customer message, and 1,351 (3.9% of all replies, 14.5% of observed follow-ups)
  are followed by a thanks token.**
- **On the "5-9%" figure the project quotes.** It is real and I found its source:
  the brand-selection working note behind DECISIONS #16 (not in this repository), which
  gives AmericanAir **7.9%** "customer thanks after" and 55.7% "no follow-up at all",
  measured across all 2.8M raw rows. It is quoted in
  DECISIONS #17, and in `REPORT.md` §1 and §5. **I did not reconcile 7.9% with my 3.9%.**
  Two plausible reasons, neither verified: the interim parquet only contains exchanges the
  brand *replied to*, so a thank-you the brand never answered creates no row and is
  invisible to my query (making 3.9% a lower bound); and my regex is not their
  acknowledgement definition. The 5-9% figure is **not** invented — but it was measured on
  the raw dump under a definition I could not read, so treat it as sourced, not reproduced.
- Undercutting: DECISIONS #17 already drew the conclusion this candidate would rediscover —
  outcomes are unobserved, so the system learns a *response policy*. That reframing is the
  backbone of the misleading-number section. A weak proxy risks eroding a piece of honesty
  the project currently earns credit for.

---

## 4. A human-in-the-loop review queue

### What it produces

**No new number on its own.** It produces a mechanism: escalations and low-confidence
auto-handles routed to a reviewer, corrections captured, corrections becoming the next
round of labels. Its output is labels, which is an input to candidates 1, 2 and 3 — not a
result.

The claim it unlocks, eventually: *the escalation reasons were validated by a reviewer*,
which would attack `REPORT.md` §4 caveat 5 — *"Escalation error is against my labels, on 48
messages. 6.2% means 'the classifier disagreed with the author', not 'a support lead would
have escalated'."*

### Cost

- **LLM calls: zero.** It is a local page over already-committed artifacts.
- **Hand labelling: unbounded and open-ended, which is the point and the problem.**
- **New dependency: none, and the build is mostly done.** `scripts/build_judge_page.py`
  already generates `runs/judge_scoring.html` — a local, keyboard-driven, localStorage-backed
  scoring page with an Export button in the scorer's exact text format, with 2 tests
  (`tests/test_judge_page.py`). A review queue is that page pointed at different rows.

### Rough time

**1 evening to build (estimate)**, then unbounded reviewer time before it yields anything.

### Main risk

**It produces a tool instead of a result, and the reviewer is the same person whose labels
are already the ceiling.** Corrections captured from the author do not break caveat 1 of
§4 — one annotator, who also designed the taxonomy after reading the data. They make the
golden set bigger, not more independent. `REPORT.md` §5 item 2 asks for *a second human
annotator*, which is a different thing and a review queue does not supply it.

Second risk: it competes for the same evening as the **one measurement the project already
owes**. Judge-human agreement is the blocking PENDING row
(`REPORT.md` §2.4, two table cells reading **PENDING**), with the tooling already built and
the job scoped at ~20 minutes. Until that row is filled, *every* drafting percentage in
the report is a model's opinion with no human agreement evidence — which this project's own rules name
as an anti-pattern in so many words. Building a second review UI before using the first
one is the wrong order.

### Evidence today

- Supporting: the queue has real volume to work on. At margin 0.10, **170 of 218** are
  escalated with reasons `low_confidence` 100, `compensation_claim` 34,
  `always_escalate_intent` 33, `public_escalation` 2, `vulnerable_customer` 1
  (`reports/results.json`). Of the 91 escalations of messages the classifier had right,
  a reviewer would immediately see whether "wasted escalation" is the right name for them.
- Supporting: the escalation signal is known to be misaligned with real need — the router
  escalates 54 of 60 messages the brand took to DM and 116 of the other 158 (Fisher
  p=0.010), but margin predicts a DM at **AUC 0.53**, n=218 (DECISIONS #48, #51). A human
  reviewer is the only way to find out what the right signal is.
- Undercutting: the agreement machinery already exists and is idle. `scripts/judge_agreement.py --score`,
  the blind 40-item sheet, and the scoring page are all built and waiting on a human.

---

## Comparison

| candidate | produces | free-tier cost | time (est.) | main risk |
| --- | --- | --- | --- | --- |
| **1. Brand #2 transfer** | labels-vs-macro-F1 curve + zero-shot transfer point for a 2nd brand | ~150 embed calls (intent-only); ~1,310 calls if drafting is replicated, over the 1200 ceiling | 2-3 evenings | wrong 2nd brand turns label efficiency into taxonomy mismatch |
| **2. Policy source + tools** | `ask_policy` auto-handle rate with citations | tens of embed + ~30 draft/judge calls per arm | 3 evenings | lands on n=15 (F1 0.516 [0.30, 0.75]); 2017 data vs today's policy |
| **3. Outcome proxy** | weak "did it resolve" label from thank-you follow-ups | ~0 for labels | 2 evenings | only 5 of 220 golden rows have one [measured today]; sarcasm and intent confounds |
| **4. HITL review queue** | a mechanism, not a number | 0 | 1 evening + unbounded review time | tool instead of result; same single annotator; competes with the owed §2.4 measurement |

---

## Recommendation

**Build the brand #2 transfer, on a second airline, starting with the intent-only
labels-vs-F1 curve.** It is the only one of the four that produces a number the project
cannot state today and has already conceded in writing — §4 caveat 9 currently says
nothing transfers without re-labelling, which is a hypothesis the repo has never tested.
It costs almost nothing on the free tier in its useful form (~150 embedding calls against a
1200 ceiling, an arithmetic estimate anchored on the 94 calls the AmericanAir index
actually took), it adds no dependency because every data script is already
`--brand`-parameterised, and its expensive half — hand labelling — is the one cost that
degrades gracefully: stop after 60 rows and you still have two points on a curve and a
publishable zero-shot transfer number. It also fits the project's thesis more cleanly than
the alternatives, because a label-efficiency curve is *evaluation*, not capability.

It beats the runner-up, **candidate 2 (policy source)**, on measurability. Candidate 2
fixes a real and legally-flavoured failure that the report names as the genuine fix, and I
nearly recommended it. But its headline would be computed on the 15 `ask_policy` rows in
the golden set, whose existing F1 interval is already [0.301, 0.752] — wide enough that a
new capability could not be distinguished from noise. Adding a policy corpus scraped from
2017-era-mismatched published rules, under an unresolved licence, to produce a number with
an interval that wide, is three evenings spent to say nothing confidently. Candidate 1
produces a defensible number; candidate 2 produces a better agent and a worse proof, and
this project has stated which of those it values.

Candidates 3 and 4 are not close. Candidate 3 is **not measurable on the evaluation set**:
5 of 220 golden rows have a thank-you follow-up [measured today]. Candidate 4 builds a
second scoring UI while the first one sits unused and `REPORT.md` §2.4 still reads PENDING.

**First concrete step**, before labelling anything:

```
uv run python scripts/make_sample.py --brand British_Airways
```

then read the resulting `data/interim/british_airways_exchanges.manifest.json` for row,
thread and language-drop counts, and confirm the thread shape looks like AmericanAir's
before committing a single hand label. If BA's exchange volume or substantive-reply
profile does not hold up at the thread level, the honest move is to stop there and
reconsider the second brand — not to label 200 rows and find out afterwards.

**Do the owed 20 minutes first.** Whatever is built next, `REPORT.md` §2.4 is two PENDING
cells with the tooling already written (`uv run python scripts/build_judge_page.py`). Every
drafting number in the report — grounded sendable 61% [0.55, 0.67], n=214 included — is a
model's opinion until that row is filled. No new candidate improves the report as much as
removing the word PENDING from it.
