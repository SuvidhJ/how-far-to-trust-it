"""Does grounding actually help, and does the LLM beat copying a past reply?

    uv run python scripts/bakeoff_draft.py --leak-test --n 50   # which drafter config is valid
    uv run python scripts/bakeoff_draft.py --n 50               # the comparison

Four drafters, judged on the same messages by a model from a different training
family than the drafter:

    canned          one fixed reply           - the trivial floor
    nearest         a real past reply, copied - no LLM, no hallucination possible
    ungrounded_llm  LLM with intent only      - isolates retrieval's contribution
    grounded_llm    LLM with k retrieved      - the system

`nearest` is the arm that matters. Most replies on this channel are near
templates, so copying the brand's answer to the most similar past message is a
genuinely strong baseline. If the LLM cannot beat it, the LLM is not earning its
latency or its dependency.

The first run of this script was void (DECISIONS #34), and its three faults are
designed out here:

* **Invalid drafts are not scored.** API exceptions, empty completions, replies
  cut off at `max_tokens`, and chain-of-thought leaked into the reply are each
  counted and reported as a failure rate. They never reach the judge, so they
  cannot drag down (or be hidden inside) a quality mean.
* **Reasoning leaks are errors.** `--leak-test` measures, on the same messages,
  whether switching thinking off actually removes them, rather than assuming it.
* **Judges are compared on the same drafts.** The self-preference comparison uses
  only drafts both judges scored completely, and prints that n.

The headline is not the mean score. It is `sendable` - safe=2 and grounded=2 and
addresses>=1 - compared per message with McNemar's exact test, because an average
over four dimensions can look healthy while every third reply invents a policy.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from support_agent.config import Paths, settings
from support_agent.draft import TWEET_LIMIT, Draft
from support_agent.draft import build as build_drafter
from support_agent.embed import Embedder
from support_agent.eval.harness import corpus_retriever, draft_sample
from support_agent.eval.judge import Judge, Score
from support_agent.eval.metrics import mcnemar, wilson
from support_agent.intents import oof_predictions
from support_agent.llm import LLM
from support_agent.retrieve import search_batch

DRAFTERS = ["canned", "nearest", "ungrounded_llm", "grounded_llm"]
LLM_ARMS = {"ungrounded_llm", "grounded_llm"}

NEMOTRON = ("nvidia", "nvidia/nemotron-3-super-120b-a12b")
THINKING_OFF = {"chat_template_kwargs": {"enable_thinking": False}}

# Drafter configurations measured by --leak-test. Same prompt, same messages.
VARIANTS: dict[str, dict] = {
    "nemotron": {"model": NEMOTRON},
    "nemotron+no_think_prompt": {"model": NEMOTRON, "system_prefix": "/no_think\n"},
    "nemotron+thinking_off": {"model": NEMOTRON, "extra_body": THINKING_OFF},
    # The only non-reasoning chat model still served: llama-3.3-70b, llama-3.1-70b,
    # llama-4-maverick and gemma-3-27b all return 410 Gone as of 2026-09-13.
    "llama-3.2-11b": {"model": ("nvidia", "meta/llama-3.2-11b-vision-instruct")},
}


def pmap(fn, items, workers: int = 4) -> list:
    """Order-preserving parallel map. The cache is thread-safe; the free tier
    tolerates a handful of concurrent requests."""
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))


def setup(n: int, k: int):
    gold_path = Paths.golden / f"{settings.brand.lower()}_golden.jsonl"
    gold = [json.loads(line) for line in gold_path.read_text(encoding="utf-8").splitlines()]
    gold = [r for r in gold if not r.get("excluded")]
    texts = [r["text"] for r in gold]
    y = np.array([r["intent"] for r in gold])

    # Out-of-fold intents, so the drafter is never handed a label the classifier
    # could not have produced on unseen data.
    emb = Embedder()
    X = emb.encode(texts)
    pred_intent, _ = oof_predictions(X, y)
    retriever = corpus_retriever(emb)
    pick, msgs, intents, hits = draft_sample(
        len(gold), n, texts, lambda m: search_batch(retriever, m, k=k), pred_intent
    )
    print(f"n={len(pick)} messages, corpus={len(retriever.texts):,}, k={k}")
    print(f"mean top-1 retrieval similarity: {np.mean([h[0].score for h in hits]):.3f}\n")
    return msgs, intents, hits


def make_drafter(name: str, variant: str):
    if name not in LLM_ARMS:
        return build_drafter(name)
    v = VARIANTS[variant]
    provider, model = v["model"]
    llm = LLM(provider, model, role=f"drafter:{variant}")
    return build_drafter(
        name, llm=llm, system_prefix=v.get("system_prefix", ""), extra_body=v.get("extra_body")
    )


def category(d: Draft) -> str:
    if not d.error:
        return "valid"
    return "api error" if d.error.startswith("api:") else d.error


def run_drafts(name: str, variant: str, msgs, intents, hits) -> list[Draft]:
    d = make_drafter(name, variant)
    return pmap(lambda j: d.draft(msgs[j], intents[j], hits[j]), range(len(msgs)))


def leak_test(msgs, intents, hits, variants: list[str]) -> None:
    cats = ["valid", "api error", "empty completion", "reasoning leak", "truncated at max_tokens"]
    print("LEAK TEST - grounded prompt, identical messages, per drafter configuration\n")
    print(
        f"{'variant':<26}{'n':>4}" + "".join(f"{c.split()[0]:>11}" for c in cats) + f"{'>280ch':>9}"
    )
    print("-" * (30 + 11 * len(cats) + 9))
    for variant in variants:
        t0 = time.time()
        out = run_drafts("grounded_llm", variant, msgs, intents, hits)
        c = Counter(category(o) for o in out)
        long_ = sum(len(o.text) > TWEET_LIMIT for o in out if not o.error)
        print(
            f"{variant:<26}{len(out):>4}"
            + "".join(f"{c.get(k, 0):>11}" for k in cats)
            + f"{long_:>9}   ({time.time() - t0:.0f}s)"
        )
        for o in out:
            if o.error.startswith("api:"):
                print(f"    api error: {o.error}")
                break
        for o in out:
            if o.error == "reasoning leak":
                print(f"    leak example: {o.text[:120]!r}")
                break


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--variant", default="nemotron+thinking_off", choices=sorted(VARIANTS))
    ap.add_argument("--leak-test", action="store_true")
    ap.add_argument("--leak-variants", default=",".join(VARIANTS))
    ap.add_argument("--skip-selfpref", action="store_true")
    args = ap.parse_args()

    msgs, intents, hits = setup(args.n, args.k)
    if args.leak_test:
        leak_test(msgs, intents, hits, args.leak_variants.split(","))
        return 0

    n = len(msgs)
    print(f"LLM drafter configuration: {args.variant} {VARIANTS[args.variant]}")
    drafts: dict[str, list[Draft]] = {}
    for name in DRAFTERS:
        t0 = time.time()
        drafts[name] = run_drafts(name, args.variant, msgs, intents, hits)
        print(f"  drafted {name:<16} {time.time() - t0:>5.0f}s")

    # ---- validity: its own number, never folded into quality ----------------
    print(f"\nDRAFT VALIDITY (n={n} per arm)")
    cats = ["api error", "empty completion", "reasoning leak", "truncated at max_tokens"]
    print(
        f"{'drafter':<16}{'valid':>7}{'rate [95% CI]':>22}"
        + "".join(f"{c.split()[0]:>11}" for c in cats)
    )
    print("-" * (45 + 11 * len(cats)))
    for name in DRAFTERS:
        c = Counter(category(o) for o in drafts[name])
        lo, hi = wilson(c["valid"], n)
        other = n - c["valid"] - sum(c[k] for k in cats)
        print(
            f"{name:<16}{c['valid']:>7}{c['valid'] / n:>8.0%} [{lo:.2f}, {hi:.2f}]"
            + "".join(f"{c.get(k, 0):>11}" for k in cats)
            + (f"   other={other}" if other else "")
        )
        for o in drafts[name]:
            if o.error:
                print(f"    {o.error[:150]!r}")
    for name in LLM_ARMS & set(DRAFTERS):
        valid = [o for o in drafts[name] if not o.error]
        long_ = sum(len(o.text) > TWEET_LIMIT for o in valid)
        print(f"  {name}: {long_}/{len(valid)} valid drafts exceed {TWEET_LIMIT} characters")

    # ---- judge only valid drafts --------------------------------------------
    judge = Judge()
    print(f"\njudge: {judge.llm.model} (rubric {judge.version})")
    scores: dict[str, list[Score | None]] = {}
    for name in DRAFTERS:
        t0 = time.time()
        scores[name] = pmap(
            lambda j, name=name: (
                None
                if drafts[name][j].error
                else judge.score(msgs[j], drafts[name][j].text, hits[j])
            ),
            range(n),
        )
        judged = [s for s in scores[name] if s is not None]
        print(
            f"  scored {name:<16} {time.time() - t0:>5.0f}s  judged={len(judged)}  "
            f"judge_failures={sum(s.parse_failed for s in judged)}"
        )

    def complete(name: str, j: int) -> bool:
        s = scores[name][j]
        return s is not None and not s.parse_failed

    print("\nQUALITY over valid drafts the judge scored completely")
    print(
        f"{'drafter':<16}{'n':>4}{'grounded':>10}{'addresses':>11}{'tone':>7}{'safe':>7}"
        f"{'total':>8}{'sendable':>10}{'95% CI':>15}"
    )
    print("-" * 88)
    for name in DRAFTERS:
        s = [scores[name][j] for j in range(n) if complete(name, j)]
        if not s:
            print(f"{name:<16}   no completely scored drafts")
            continue
        k = sum(x.sendable for x in s)
        lo, hi = wilson(k, len(s))
        print(
            f"{name:<16}{len(s):>4}{np.mean([x.grounded for x in s]):>10.2f}"
            f"{np.mean([x.addresses for x in s]):>11.2f}"
            f"{np.mean([x.tone for x in s]):>7.2f}{np.mean([x.safe for x in s]):>7.2f}"
            f"{np.mean([x.total for x in s]):>8.2f}{k / len(s):>9.0%}  [{lo:.2f}, {hi:.2f}]"
        )

    # ---- paired significance -------------------------------------------------
    print("\nPAIRED sendable, McNemar exact test")
    print(
        f"{'comparison':<32}{'basis':<22}{'n':>4}{'both':>6}{'A only':>8}{'B only':>8}{'neither':>9}{'p':>8}"
    )
    print("-" * 97)
    for a, b in (("grounded_llm", "nearest"), ("grounded_llm", "ungrounded_llm")):
        # complete-case: quality when both produced a scorable reply.
        cc = [j for j in range(n) if complete(a, j) and complete(b, j)]
        # operational: an invalid draft is simply not sendable; only judge
        # failures on valid drafts are unknown and excluded.
        op = [
            j
            for j in range(n)
            if (drafts[a][j].error or complete(a, j)) and (drafts[b][j].error or complete(b, j))
        ]
        for basis, idx in (("both valid+scored", cc), ("invalid = not sendable", op)):
            sa = [
                bool(scores[a][j] and not drafts[a][j].error and scores[a][j].sendable) for j in idx
            ]
            sb = [
                bool(scores[b][j] and not drafts[b][j].error and scores[b][j].sendable) for j in idx
            ]
            both, ao, bo, nei, p = mcnemar(sa, sb)
            print(
                f"{a + ' vs ' + b:<32}{basis:<22}{len(idx):>4}{both:>6}{ao:>8}{bo:>8}{nei:>9}{p:>8.3f}"
            )

    # `sendable` needs grounded == 2, and a verbatim `nearest` draft is by
    # construction a line of the evidence the judge is shown, so the composite
    # favours `nearest`. Each dimension is therefore also tested on its own:
    # Wilcoxon signed-rank, paired by message, Bonferroni over the four dimensions.
    from scipy.stats import wilcoxon

    print("\nPAIRED per dimension, grounded_llm vs nearest (Wilcoxon; Bonferroni alpha 0.0125)")
    print(f"{'dimension':<12}{'n':>4}{'llm':>7}{'nearest':>9}{'llm>':>6}{'<':>4}{'=':>4}{'p':>9}")
    print("-" * 55)
    cc = [j for j in range(n) if complete("grounded_llm", j) and complete("nearest", j)]
    for dim in ("grounded", "addresses", "tone", "safe"):
        a = np.array([getattr(scores["grounded_llm"][j], dim) for j in cc])
        b = np.array([getattr(scores["nearest"][j], dim) for j in cc])
        d = a - b
        p = wilcoxon(a, b).pvalue if (d != 0).any() else 1.0
        print(
            f"{dim:<12}{len(cc):>4}{a.mean():>7.2f}{b.mean():>9.2f}"
            f"{int((d > 0).sum()):>6}{int((d < 0).sum()):>4}{int((d == 0).sum()):>4}{p:>9.4f}"
        )

    print("\nverbosity check (judges reward length; does score track it?):")
    for name in DRAFTERS:
        s = [scores[name][j] for j in range(n) if complete(name, j)]
        if len(s) > 5:
            lens = np.array([x.length_chars for x in s], dtype=float)
            tot = np.array([x.total for x in s], dtype=float)
            r = np.corrcoef(lens, tot)[0, 1] if lens.std() > 0 and tot.std() > 0 else float("nan")
            print(
                f"  {name:<16} mean_len={lens.mean():>5.0f}  corr(len, total)={r:>6.2f}  n={len(s)}"
            )

    print("\nworst problems named by the judge:")
    for name in DRAFTERS:
        top = Counter(
            scores[name][j].worst_problem
            for j in range(n)
            if complete(name, j) and scores[name][j].worst_problem
        ).most_common(3)
        print(f"  {name:<16}{'; '.join(f'{p} ({c})' for p, c in top)}")

    probe_scores: dict[str, list[Score | None]] = {}
    if not args.skip_selfpref:
        # The probe is the drafter's own model - the strongest form of the test -
        # with the same thinking setting, so its JSON is not buried in deliberation.
        provider, model = VARIANTS[args.variant]["model"]
        extra = VARIANTS[args.variant].get("extra_body")
        probe = Judge(
            llm=LLM(provider, model, role="judge:self-preference-probe"),
            max_tokens=4096,
            extra_body=extra,
        )
        print(f"\nSELF-PREFERENCE: probe judge {model} is the drafter's own model")
        print(f"  probe max_tokens=4096, extra_body={extra}")
        print(
            f"{'drafter':<16}{'probe done':>11}{'paired n':>10}{'real':>7}{'probe':>7}"
            f"{'inflation':>11}{'probe>':>8}{'=':>4}{'<':>4}{'sign p':>8}{'send real/probe':>18}"
        )
        print("-" * 104)
        for name in ("ungrounded_llm", "grounded_llm"):
            probe_scores[name] = pmap(
                lambda j, name=name: (
                    None
                    if drafts[name][j].error
                    else probe.score(msgs[j], drafts[name][j].text, hits[j])
                ),
                range(n),
            )
            ps = probe_scores[name]
            asked = [s for s in ps if s is not None]
            done = [s for s in asked if not s.parse_failed]
            pair = [
                j
                for j in range(n)
                if complete(name, j) and ps[j] is not None and not ps[j].parse_failed
            ]
            if not pair:
                print(f"{name:<16}{len(done):>5}/{len(asked):<5}   no paired drafts")
                continue
            real = np.array([scores[name][j].total for j in pair], dtype=float)
            prb = np.array([ps[j].total for j in pair], dtype=float)
            up, eq, down = (
                int((prb > real).sum()),
                int((prb == real).sum()),
                int((prb < real).sum()),
            )
            sp = binomtest(up, up + down, 0.5).pvalue if up + down else 1.0
            sr = sum(scores[name][j].sendable for j in pair)
            spb = sum(ps[j].sendable for j in pair)
            print(
                f"{name:<16}{len(done):>5}/{len(asked):<5}{len(pair):>10}{real.mean():>7.2f}{prb.mean():>7.2f}"
                f"{prb.mean() - real.mean():>+11.2f}{up:>8}{eq:>4}{down:>4}{sp:>8.3f}"
                f"{f'{sr}/{len(pair)} vs {spb}/{len(pair)}':>18}"
            )

    out_dir = Path("scratchpad")
    out_dir.mkdir(exist_ok=True)
    with (out_dir / "bakeoff_draft_results.jsonl").open("w", encoding="utf-8") as fh:
        for j in range(n):
            for name in DRAFTERS:
                s = scores[name][j]
                p = probe_scores.get(name, [None] * n)[j]
                fh.write(
                    json.dumps(
                        {
                            "j": j,
                            "drafter": name,
                            "variant": args.variant if name in LLM_ARMS else "",
                            "message": msgs[j],
                            "intent": intents[j],
                            "draft": drafts[name][j].text,
                            "error": drafts[name][j].error,
                            "judge": s.as_dict() if s else None,
                            "probe": p.as_dict() if p else None,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    with (out_dir / "draft_examples.md").open("w", encoding="utf-8") as fh:
        for j in range(min(8, n)):
            fh.write(f"## {msgs[j]}\n\npredicted intent: `{intents[j]}`\n\n")
            for name in DRAFTERS:
                s = scores[name][j]
                tag = (
                    f"(g{s.grounded} a{s.addresses} t{s.tone} s{s.safe})"
                    if s
                    else f"(INVALID: {drafts[name][j].error})"
                )
                fh.write(f"- **{name}** {tag}: {drafts[name][j].text or '(empty)'}\n")
            fh.write("\n")
    print(f"\nper-draft results -> {out_dir / 'bakeoff_draft_results.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
