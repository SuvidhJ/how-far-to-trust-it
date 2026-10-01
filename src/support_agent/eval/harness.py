"""The one command. Reproduces every headline number in the README.

    uv run python -m support_agent.eval.harness

Runs the full pipeline over the golden set - classify, retrieve, draft, route -
against both required baselines, and writes a run manifest so a number can always
be traced back to the code, seed and models that produced it.

Two switches exist because the report needs them, not because anyone would run
them by choice:

* `--naive` reproduces the *dishonest* configuration: tweet-level splitting
  instead of thread-level, no deduplication, and the judge drawn from the
  drafter's own family. It exists so the report can quantify its own inflation
  rather than gesture at it.
* `--offline` refuses to make any network call and replays the committed caches.
  It is what makes the "reproduce in under 15 minutes" claim checkable by someone
  with no API keys.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from support_agent.config import GLOBAL_SEED, Paths, settings
from support_agent.data import git_sha
from support_agent.draft import build as build_drafter
from support_agent.embed import MODEL as EMBED_MODEL
from support_agent.embed import Embedder
from support_agent.eval.metrics import intent_report, mcnemar, top_confusions, wilson
from support_agent.intents import load_taxonomy, oof_predictions, system_head
from support_agent.llm import ROLE_MODELS
from support_agent.retrieve import EmbedRetriever, TfidfRetriever, search_batch
from support_agent.route import OPERATING_THRESHOLD, Router
from support_agent.text import DM_HANDOFF, has_placeholder


def load_golden(brand: str, include_excluded: bool = False) -> list[dict]:
    path = Paths.golden / f"{brand.lower()}_golden.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return rows if include_excluded else [r for r in rows if not r.get("excluded")]


def load_index(brand: str) -> list[dict]:
    path = Paths.processed / f"{brand.lower()}_index.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path}; run scripts/build_index.py first")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def draft_sample(
    n_gold: int, n: int, texts: list[str], hits_for, preds: list[str]
) -> tuple[list[int], list[str], list[str], list]:
    """The messages every drafting evaluation uses: `n` golden rows drawn with
    GLOBAL_SEED, their out-of-fold intents, and their retrieval hits.

    One definition, because cached drafts and judgements only replay while the
    harness, `bakeoff_draft.py` and `judge_agreement.py` ask for identical requests.
    """
    pick = sorted(
        np.random.default_rng(GLOBAL_SEED).choice(n_gold, size=min(n, n_gold), replace=False)
    )
    msgs = [texts[i] for i in pick]
    return pick, msgs, [preds[i] for i in pick], hits_for(msgs)


def corpus_retriever(emb: Embedder, kind: str = "embed"):
    corpus = load_index(settings.brand)
    retriever = TfidfRetriever() if kind == "tfidf" else EmbedRetriever(emb)
    return retriever.index(
        [c["text"] for c in corpus], [{"id": c["id"], "reply": c["reply"]} for c in corpus]
    )


def cv_macro_f1(X: np.ndarray, y: np.ndarray, labels: list[str], folds: int, repeats: int):
    """Embed+logreg under the headline protocol: per-repeat out-of-fold predictions."""
    preds_by_rep = []
    for rep in range(repeats):
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=GLOBAL_SEED + rep)
        preds = [""] * len(y)
        for tr, te in skf.split(X, y):
            m = system_head().fit(X[tr], y[tr])
            for i, p in zip(te, m.predict(X[te]), strict=True):
                preds[i] = p
        preds_by_rep.append(preds)
    f1s = [intent_report(list(y), p, labels).macro_f1 for p in preds_by_rep]
    return preds_by_rep, f1s


def naive_waterfall(args: argparse.Namespace, emb: Embedder, labels: list[str]) -> list[dict]:
    """Each honesty step, undone one at a time, with the number it changes.

    Every row is measured, not simulated: the naive configuration is actually run.
    Steps whose inputs are not on disk (the interim thread data is gitignored) are
    reported as skipped rather than silently dropped.
    """
    rows: list[dict] = []
    gold = load_golden(settings.brand)
    gold_all = load_golden(settings.brand, include_excluded=True)
    y = np.array([r["intent"] for r in gold])
    X = emb.encode([r["text"] for r in gold])
    preds, f1s = cv_macro_f1(X, y, labels, args.folds, args.repeats)
    audited = float(np.mean(f1s))

    # 1. No thread dedup: keep the thread twin and the taxonomy anchor.
    y_all = np.array([r["intent"] for r in gold_all])
    preds_all, f1s_all = cv_macro_f1(
        emb.encode([r["text"] for r in gold_all]), y_all, labels, args.folds, args.repeats
    )
    rows.append(
        {
            "step": "one row per thread (drop thread twin + taxonomy anchor)",
            "metric": "macro-F1, repeated CV",
            "naive": float(np.mean(f1s_all)),
            "audited": audited,
            "n": f"{len(gold_all)} vs {len(gold)}",
        }
    )

    # 2. Quote one split: the best of the repeats is what a seed search finds.
    rows.append(
        {
            "step": f"mean of {args.repeats}x{args.folds}-fold CV, not the best single split",
            "metric": "macro-F1",
            "naive": float(np.max(f1s)),
            "audited": audited,
            "n": f"{len(gold)} (sd {np.std(f1s):.3f})",
        }
    )

    # 3. Uniform sample + accuracy. The `random` stratum is a uniform draw from
    # the same frame, so accuracy on it is what a uniformly sampled golden set
    # would have reported.
    uni = [i for i, r in enumerate(gold) if r.get("stratum") == "random"]
    acc_uni = float(np.mean([np.mean([p[i] == y[i] for i in uni]) for p in preds]))
    acc_all = float(np.mean([np.mean([a == b for a, b in zip(p, y, strict=True)]) for p in preds]))
    rows.append(
        {
            "step": "macro-F1 on a stratified set, not accuracy on a uniform sample",
            "metric": "accuracy (uniform stratum) vs macro-F1",
            "naive": acc_uni,
            "audited": audited,
            "n": f"{len(uni)} vs {len(gold)} (accuracy on all {len(gold)}: {acc_all:.3f})",
        }
    )

    # All three intent steps undone together: the number a careless write-up could quote.
    uni_all = [i for i, r in enumerate(gold_all) if r.get("stratum") == "random"]
    best_uni = max(float(np.mean([p[i] == y_all[i] for i in uni_all])) for p in preds_all)
    rows.append(
        {
            "step": "ALL intent steps undone: 220 rows, best split, uniform accuracy",
            "metric": "headline a careless setup would quote",
            "naive": best_uni,
            "audited": audited,
            "n": f"{len(uni_all)} vs {len(gold)}",
        }
    )

    # 4. Tweet-level split for the retrieval index: the other exchanges from each
    # evaluated conversation land in the corpus the drafter grounds in.
    rows.append(_retrieval_leak_row(args, emb, gold))

    # 5. Judge from the drafter's own model.
    rows.extend(_judge_family_rows(args, emb, gold, X, y))
    return rows


def _judge_family_rows(
    args, emb: Embedder, gold: list[dict], X: np.ndarray, y: np.ndarray
) -> list[dict]:
    """Score the same grounded drafts with the real judge and with the drafter's own
    model, paired on drafts both scored completely.

    Replays exactly the requests `scripts/bakeoff_draft.py --n 50` makes, so it runs
    offline once that script has populated the cache.
    """
    from support_agent.eval.judge import Judge
    from support_agent.llm import LLM, ROLE_EXTRA_BODY, SELF_PREFERENCE_PROBE

    step = "judge from a different family than the drafter"
    preds, _ = oof_predictions(X, y)
    retriever = corpus_retriever(emb)
    pick, msgs, intents, hits = draft_sample(
        len(gold),
        50,
        [r["text"] for r in gold],
        lambda m: search_batch(retriever, m, k=args.k),
        preds,
    )

    drafter = build_drafter("grounded_llm", llm=LLM.for_role("drafter", offline=args.offline))
    judge = Judge(llm=LLM.for_role("judge", offline=args.offline))
    probe = Judge(
        llm=LLM(*SELF_PREFERENCE_PROBE, role="judge:self-preference-probe", offline=args.offline),
        max_tokens=4096,
        extra_body=ROLE_EXTRA_BODY["drafter"],
    )
    real, same = [], []
    for j in range(len(pick)):
        d = drafter.draft(msgs[j], intents[j], hits[j])
        if d.error:
            continue
        a, b = judge.score(msgs[j], d.text, hits[j]), probe.score(msgs[j], d.text, hits[j])
        if not (a.parse_failed or b.parse_failed):
            real.append(a)
            same.append(b)
    n = len(real)
    if not n:
        return [
            {
                "step": step,
                "metric": "",
                "naive": None,
                "audited": None,
                "n": "skipped: no cached paired judgements; run scripts/bakeoff_draft.py --n 50",
            }
        ]
    note = f"{n} grounded drafts both judges scored completely (of {len(pick)})"
    return [
        {
            "step": step,
            "metric": "mean total /8, grounded_llm drafts",
            "naive": float(np.mean([s.total for s in same])),
            "audited": float(np.mean([s.total for s in real])),
            "n": note,
        },
        {
            "step": step,
            "metric": "sendable rate, grounded_llm drafts",
            "naive": float(np.mean([s.sendable for s in same])),
            "audited": float(np.mean([s.sendable for s in real])),
            "n": note,
        },
    ]


def _retrieval_leak_row(args: argparse.Namespace, emb: Embedder, gold: list[dict]) -> dict:
    step = "retrieval index split by thread, not by tweet"
    metric = "top-1 hit from the message's own conversation"
    interim = Paths.interim / f"{settings.brand.lower()}_exchanges.parquet"
    if not interim.exists():
        return {
            "step": step,
            "metric": metric,
            "naive": None,
            "audited": None,
            "n": f"skipped: {interim} not built",
        }
    import pandas as pd

    corpus = load_index(settings.brand)
    gold_ids = {r["id"] for r in gold}
    gold_threads = {r["thread_id"] for r in gold}
    df = pd.read_parquet(
        interim, columns=["customer_tweet_id", "thread_id", "customer_clean", "brand_clean"]
    )
    mates = df[df["thread_id"].isin(gold_threads) & ~df["customer_tweet_id"].isin(gold_ids)]
    mates = mates[mates["customer_clean"].str.len() > 0]

    def own_thread_rate(items: list[dict]) -> tuple[int, float]:
        r = EmbedRetriever(emb).index(
            [c["text"] for c in items], [{"id": c["id"], "reply": c["reply"]} for c in items]
        )
        thread_of = {c["id"]: c["thread_id"] for c in items}
        hits = search_batch(r, [g["text"] for g in gold], k=1)
        own = sum(
            bool(h) and thread_of.get(h[0].id) == g["thread_id"]
            for h, g in zip(hits, gold, strict=True)
        )
        return own, float(np.mean([h[0].score for h in hits if h]))

    naive_items = corpus + [
        {
            "id": int(m.customer_tweet_id),
            "thread_id": int(m.thread_id),
            "text": m.customer_clean,
            "reply": m.brand_clean,
        }
        for m in mates.itertuples()
    ]
    own_a, sim_a = own_thread_rate(corpus)
    own_n, sim_n = own_thread_rate(naive_items)
    n = len(gold)
    return {
        "step": step,
        "metric": metric,
        "naive": own_n / n,
        "audited": own_a / n,
        "n": f"{own_n}/{n} vs {own_a}/{n}; +{len(mates)} leaked exchanges; top-1 sim {sim_n:.3f} vs {sim_a:.3f}",
    }


def held_out_threshold(
    gold: list[dict],
    texts: list[str],
    preds: list[str],
    posteriors: list[dict],
    correct: np.ndarray,
) -> dict:
    """Pick the operating point on dev rows by a rule fixed in advance; score it on test rows.

    The headline threshold (0.10) was read off a sweep over all 218 rows. Here the golden
    set's own split tags keep choice and measurement apart. Rule: the largest auto-handle
    coverage whose error on auto-handled dev messages is at most a third of the dev base
    error rate, the target the 0.10 choice was justified by.
    """
    split = np.array([r.get("split") for r in gold])
    grid = [round(x, 2) for x in np.arange(0.0, 0.51, 0.01)]

    def at(thr: float, mask: np.ndarray) -> tuple[int, int]:
        r = Router(threshold=thr, signal="margin")
        auto = np.array(
            [
                not r.decide(t, p, q).escalated
                for t, p, q in zip(texts, preds, posteriors, strict=True)
            ]
        )
        sel = auto & mask
        return int(sel.sum()), int((sel & ~correct).sum())

    dev, test = split == "dev", split == "test"
    target = (1 - correct[dev].mean()) / 3
    best = None
    for thr in grid:
        n_auto, n_err = at(thr, dev)
        if n_auto and n_err / n_auto <= target and (best is None or n_auto > best[1]):
            best = (thr, n_auto, n_err)
    if best is None:
        return {"dev_n": int(dev.sum()), "test_n": int(test.sum()), "chosen": None}
    n_auto, n_err = at(best[0], test)
    return {
        "dev_n": int(dev.sum()),
        "test_n": int(test.sum()),
        "rule": "max dev coverage with dev error on auto-handled <= dev base error / 3",
        "chosen": best[0],
        "dev_auto": best[1],
        "dev_err": best[2],
        "test_auto": n_auto,
        "test_err": n_err,
        "test_err_ci": list(wilson(n_err, n_auto)) if n_auto else None,
        "test_base_error": float(1 - correct[test].mean()),
    }


def brand_dm_proxy(gold: list[dict], escalated: np.ndarray, posteriors: list[dict]) -> dict:
    """Escalation checked against something other than my labels: the brand's own reply.

    The router never sees `brand_reply`. When AmericanAir moved the conversation to a
    DM, the case needed account-level handling by a person. That is a proxy, not
    ground truth: the brand also DMs routine requests, and answers some hard ones in
    public (DECISIONS #48).
    """
    from scipy.stats import fisher_exact, mannwhitneyu

    from support_agent.route import margin

    dm = np.array([bool(DM_HANDOFF.search(r.get("brand_reply", ""))) for r in gold])
    a, b = int((escalated & dm).sum()), int((escalated & ~dm).sum())
    c, d = int((~escalated & dm).sum()), int((~escalated & ~dm).sum())
    mg = np.array([margin(p) for p in posteriors])
    u = mannwhitneyu(mg[dm], mg[~dm]).statistic if dm.any() and (~dm).any() else np.nan
    return {
        "dm_cases": int(dm.sum()),
        "escalated_dm": a,
        "non_dm_cases": int((~dm).sum()),
        "escalated_non_dm": b,
        "fisher_p": float(fisher_exact([[a, b], [c, d]])[1]),
        # P(margin on a DM case < margin on a non-DM case): 0.5 means no signal.
        "low_margin_auc": float(1 - u / (dm.sum() * (~dm).sum())),
    }


DRAFTERS = ("canned", "nearest", "ungrounded_llm", "grounded_llm")
OFFLINE_MISS = "offline=True"


def drafting_eval(sample: list[str], intents: list[str], hits: list, offline: bool) -> dict:
    """Reply quality: the LLM-as-judge rubric over all four drafters, plus paired tests.

    Makes the same requests as `scripts/bakeoff_draft.py --n N`, so `--offline`
    replays them from the committed cache. A draft or judgement missing from the
    cache is counted as `uncached` and reported, never scored as a model failure.
    """
    from scipy.stats import wilcoxon

    from support_agent.eval.judge import DIMENSIONS, Judge
    from support_agent.llm import LLM

    judge = Judge(llm=LLM.for_role("judge", offline=offline))
    out: dict = {"n": len(sample), "judge": judge.llm.model, "rubric": judge.version, "arms": {}}
    scores: dict[str, list] = {}
    as_is: dict[str, list[bool]] = {}
    for name in DRAFTERS:
        kw = {"llm": LLM.for_role("drafter", offline=offline)} if name.endswith("_llm") else {}
        drafter = build_drafter(name, **kw)
        drafts = [drafter.draft(m, i, h) for m, i, h in zip(sample, intents, hits, strict=True)]
        judged = [
            None if d.error else judge.score(m, d.text, h)
            for d, m, h in zip(drafts, sample, hits, strict=True)
        ]
        uncached = sum(OFFLINE_MISS in d.error for d in drafts) + sum(
            bool(s and s.parse_failed and OFFLINE_MISS in s.error) for s in judged
        )
        ok = [s for s in judged if s is not None and not s.parse_failed]
        k = sum(s.sendable for s in ok)
        placeholders = [not d.error and has_placeholder(d.text) for d in drafts]
        strict = [
            bool(s and not s.parse_failed and s.sendable and not ph)
            for s, ph in zip(judged, placeholders, strict=True)
        ]
        k_strict = sum(strict)
        out["arms"][name] = {
            "valid": sum(not d.error for d in drafts),
            "judged": len(ok),
            "uncached": uncached,
            "mean_total": float(np.mean([s.total for s in ok])) if ok else None,
            "sendable": k,
            "sendable_ci": list(wilson(k, len(ok))),
            "with_placeholder": int(sum(placeholders)),
            "sendable_as_is": k_strict,
            "sendable_as_is_ci": list(wilson(k_strict, len(ok))),
            **{
                f"mean_{d}": float(np.mean([getattr(s, d) for s in ok])) if ok else None
                for d in DIMENSIONS
            },
        }
        scores[name] = judged
        as_is[name] = strict

    def paired(a: str, b: str) -> list[int]:
        return [
            j
            for j in range(len(sample))
            if all(scores[x][j] is not None and not scores[x][j].parse_failed for x in (a, b))
        ]

    out["sendable_mcnemar"] = {}
    for a, b in (("grounded_llm", "nearest"), ("grounded_llm", "ungrounded_llm")):
        idx = paired(a, b)
        _, a_only, b_only, _, p = mcnemar(
            [scores[a][j].sendable for j in idx], [scores[b][j].sendable for j in idx]
        )
        _, sa_only, sb_only, _, p_strict = mcnemar(
            [as_is[a][j] for j in idx], [as_is[b][j] for j in idx]
        )
        out["sendable_mcnemar"][f"{a} vs {b}"] = {
            "n": len(idx),
            "as_is_a_only": sa_only,
            "as_is_b_only": sb_only,
            "as_is_p": p_strict,
            "a_only": a_only,
            "b_only": b_only,
            "p": p,
        }

    # `sendable` needs grounded=2, and a verbatim `nearest` draft is one of the evidence
    # lines the judge reads, so the composite favours copying. Test each dimension (#46).
    # The same for grounded vs ungrounded: only the grounded drafter saw the evidence
    # the judge checks `grounded` against, so that dimension favours it by construction.
    for rival in ("nearest", "ungrounded_llm"):
        idx = paired("grounded_llm", rival)
        key = f"dimensions_llm_vs_{rival}"
        out[key] = {}
        for d in DIMENSIONS:
            x = np.array([getattr(scores["grounded_llm"][j], d) for j in idx])
            z = np.array([getattr(scores[rival][j], d) for j in idx])
            p = float(wilcoxon(x, z).pvalue) if len(idx) and (x != z).any() else 1.0
            out[key][d] = {
                "n": len(idx),
                "llm": float(x.mean()) if len(idx) else None,
                "rival": float(z.mean()) if len(idx) else None,
                "p": p,
            }
    return out


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    tax = load_taxonomy()
    gold = load_golden(settings.brand)
    texts = [r["text"] for r in gold]
    y = np.array([r["intent"] for r in gold])

    emb = Embedder(offline=args.offline)
    X = emb.encode(texts)

    # --- intent: system against both required baselines --------------------
    # Repeated CV, not one split. A single 5-fold split of 218 rows moves the
    # headline from 0.59 to 0.66 depending only on the seed - a bigger swing than
    # any method choice in this project. Quoting one split would be quoting
    # whichever seed flattered the system.
    systems = {
        "trivial (majority)": None,
        "simple (tfidf+logreg)": "tfidf",
        "system (embed+logreg)": "embed",
    }
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline

    def fold_predict(kind: str | None, tr, te) -> list[str]:
        if kind is None:
            vals, counts = np.unique(y[tr], return_counts=True)
            return [str(vals[counts.argmax()])] * len(te)
        if kind == "tfidf":
            model = make_pipeline(
                TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=1),
                LogisticRegression(
                    max_iter=3000, class_weight="balanced", C=4.0
                ),  # baseline's own head
            )
            model.fit([texts[i] for i in tr], y[tr])
            return list(model.predict([texts[i] for i in te]))
        model = system_head()
        model.fit(X[tr], y[tr])
        return list(model.predict(X[te]))

    repeats: dict[str, list[list[str]]] = {k: [] for k in systems}
    for rep in range(args.repeats):
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=GLOBAL_SEED + rep)
        splits = list(skf.split(X, y))
        for name, kind in systems.items():
            preds = [""] * len(y)
            for tr, te in splits:
                for i, p in zip(te, fold_predict(kind, tr, te), strict=True):
                    preds[i] = p
            repeats[name].append(preds)

    # The reported per-class table comes from the first repeat; the headline
    # macro-F1 is the mean across repeats, with its spread.
    intent_preds = {k: v[0] for k, v in repeats.items()}
    reports = {k: intent_report(list(y), v, tax.ids) for k, v in intent_preds.items()}
    spread = {
        k: (
            float(np.mean([intent_report(list(y), p, tax.ids).macro_f1 for p in v])),
            float(np.std([intent_report(list(y), p, tax.ids).macro_f1 for p in v])),
        )
        for k, v in repeats.items()
    }

    # --- routing on the system's posteriors --------------------------------
    # Posteriors must come from the same folds as the predictions they gate
    # (repeat 0). Using the last repeat's folds paired each prediction with a
    # margin from a differently-trained model (DECISIONS #40).
    oof_preds, posteriors = oof_predictions(X, y, args.folds, GLOBAL_SEED)
    sys_preds = intent_preds["system (embed+logreg)"]
    # Repeat 0 of the headline CV uses the same seed, so the two must agree exactly;
    # if they ever do not, the router would gate predictions it did not see.
    assert oof_preds == [str(p) for p in sys_preds], "routing posteriors drifted from predictions"
    correct_all = np.array([p == t for p, t in zip(sys_preds, y, strict=True)])

    # Sweep the operating point rather than asserting one. The costs are
    # asymmetric - a needless escalation costs an agent seconds, a missed one
    # puts a wrong promise under the brand's name - so the point is chosen for
    # low error on what is auto-handled, not for high coverage.
    sweep = []
    for thr in (0.0, 0.05, 0.10, 0.15, 0.25, 0.40, 0.60):
        r = Router(threshold=thr, signal="margin")
        d = [r.decide(t, p, post) for t, p, post in zip(texts, sys_preds, posteriors, strict=True)]
        a = np.array([not x.escalated for x in d])
        sweep.append(
            {
                "threshold": thr,
                "coverage": float(a.mean()),
                "error_on_auto": float(1 - correct_all[a].mean()) if a.any() else None,
                "errors_caught": int((~a & ~correct_all).sum()),
                "wasted_escalations": int((~a & correct_all).sum()),
            }
        )

    router = Router(threshold=args.threshold, signal="margin")
    decisions = [
        router.decide(t, p, post) for t, p, post in zip(texts, sys_preds, posteriors, strict=True)
    ]
    correct = np.array([p == t for p, t in zip(sys_preds, y, strict=True)])
    auto = np.array([not d.escalated for d in decisions])
    routing = {
        "threshold": args.threshold,
        "signal": "margin",
        "auto_handled": int(auto.sum()),
        "escalated": int((~auto).sum()),
        "coverage": float(auto.mean()),
        "error_on_auto_handled": float(1 - correct[auto].mean()) if auto.any() else None,
        "error_on_auto_handled_ci": list(wilson(int((auto & ~correct).sum()), int(auto.sum()))),
        "error_overall": float(1 - correct.mean()),
        "errors_caught": int((~auto & ~correct).sum()),
        "errors_total": int((~correct).sum()),
        "reasons": dict(
            sorted(
                {
                    d.rule: sum(x.rule == d.rule for x in decisions if x.escalated)
                    for d in decisions
                    if d.escalated
                }.items()
            )
        ),
    }
    routing["brand_dm_proxy"] = brand_dm_proxy(gold, ~auto, posteriors)
    # A router that escalates the same share of traffic at random catches errors in
    # proportion; "catches 77 of 82" only means something against that number.
    routing["random_escalation_expected_caught"] = float((~auto).mean() * (~correct).sum())
    routing["held_out"] = held_out_threshold(gold, texts, sys_preds, posteriors, correct)
    # The same router over every CV repeat: how much of 24% / 0.094 is the fold seed.
    per_seed = []
    for rep in range(args.repeats):
        p_rep, post_rep = oof_predictions(X, y, args.folds, GLOBAL_SEED + rep)
        ok = np.array([a == b for a, b in zip(p_rep, y, strict=True)])
        auto_rep = np.array(
            [
                not router.decide(a, b, c).escalated
                for a, b, c in zip(texts, p_rep, post_rep, strict=True)
            ]
        )
        per_seed.append((float(auto_rep.mean()), float(1 - ok[auto_rep].mean())))
    routing["across_fold_seeds"] = {
        "coverage": [min(c for c, _ in per_seed), max(c for c, _ in per_seed)],
        "error_on_auto": [min(e for _, e in per_seed), max(e for _, e in per_seed)],
    }
    rand = np.array([r.get("stratum") == "random" for r in gold])
    routing["random_stratum"] = {
        "n": int(rand.sum()),
        "auto_handled": int((auto & rand).sum()),
        "errors_on_auto": int((auto & rand & ~correct).sum()),
        "errors_caught": int((~auto & rand & ~correct).sum()),
        "errors": int((rand & ~correct).sum()),
    }

    # --- drafting on a subsample -------------------------------------------
    drafting: dict = {}
    if args.draft_n:
        retriever = corpus_retriever(emb, args.retriever)
        _pick, sample, intents, hits = draft_sample(
            len(gold),
            args.draft_n,
            texts,
            lambda m: search_batch(retriever, m, k=args.k),
            sys_preds,
        )
        drafting["retrieval_top1_similarity"] = float(np.mean([h[0].score for h in hits if h]))
        if not args.no_judge:
            drafting.update(drafting_eval(sample, intents, hits, args.offline))

    manifest = {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "python": platform.python_version(),
        "brand": settings.brand,
        "global_seed": GLOBAL_SEED,
        "taxonomy_version": tax.version,
        "golden_n": len(gold),
        "golden_excluded": sum(bool(r.get("excluded")) for r in load_golden(settings.brand, True)),
        "embed_model": EMBED_MODEL,
        "role_models": {k: f"{v[0]}/{v[1]}" for k, v in ROLE_MODELS.items()},
        "folds": args.folds,
        "naive_mode": args.naive,
        "offline": args.offline,
        "elapsed_s": round(time.time() - started, 1),
        "repeats": args.repeats,
        "intent": {
            k: {
                "macro_f1_mean": spread[k][0],
                "macro_f1_sd": spread[k][1],
                "macro_f1_single_split": r.macro_f1,
                "macro_f1_ci": list(r.macro_f1_ci),
                "accuracy": r.accuracy,
                "n": r.n,
            }
            for k, r in reports.items()
        },
        "routing": routing,
        "routing_sweep": sweep,
        "drafting": drafting,
        "embed_api_calls": emb.api_calls,
        "embed_cache_hits": emb.cache_hits,
    }
    return {
        "manifest": manifest,
        "reports": reports,
        "decisions": decisions,
        "y": y,
        "preds": intent_preds,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument(
        "--threshold",
        type=float,
        default=OPERATING_THRESHOLD,
        help="margin operating point; 0.10 puts error on auto-handled traffic at a "
        "quarter of the base rate while catching 77 of the 82 classifier errors (#44)",
    )
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--draft-n", type=int, default=218)
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM-as-judge drafting table")
    ap.add_argument("--retriever", choices=["embed", "tfidf"], default="embed")
    ap.add_argument("--naive", action="store_true", help="reproduce the inflated configuration")
    ap.add_argument("--offline", action="store_true", help="replay caches, make no network call")
    ap.add_argument("--out", type=Path, default=Paths.runs / "latest.json")
    args = ap.parse_args(argv)

    res = run(args)
    m, reports = res["manifest"], res["reports"]

    print(
        f"brand={m['brand']}  golden n={m['golden_n']}  seed={m['global_seed']}  "
        f"git={m['git_sha']}  {'NAIVE' if args.naive else 'audited'}"
    )
    print(
        f"embeddings: {m['embed_model']}  ({m['embed_api_calls']} api calls, "
        f"{m['embed_cache_hits']} cache hits)\n"
    )

    print(
        f"{'intent system':<26}{'macro-F1 mean':>14}{'sd':>7}   {'one split: F1 [95% CI]':<26}{'acc':>6}"
    )
    print("-" * 82)
    for name, r in reports.items():
        mean, sd = m["intent"][name]["macro_f1_mean"], m["intent"][name]["macro_f1_sd"]
        one = f"{r.macro_f1:.3f} [{r.macro_f1_ci[0]:.3f}, {r.macro_f1_ci[1]:.3f}]"
        print(f"{name:<26}{mean:>14.3f}{sd:>7.3f}   {one:<26}{r.accuracy:>6.3f}")
    print(
        f"  mean and sd over {m['repeats']} repeats of {m['folds']}-fold CV; the one-split F1, its"
        " bootstrap CI and accuracy come from repeat 0 alone"
    )

    best = reports["system (embed+logreg)"]
    print(f"\nper class (system), n={best.n}:\n")
    print(best.table())

    print("\ntop confusions:")
    for t, p, n in top_confusions(best, 5):
        print(f"  {t:<22} -> {p:<22} {n}")

    r = m["routing"]
    print(f"\nrouting (signal={r['signal']}, threshold={r['threshold']}):")
    lo, hi = r["error_on_auto_handled_ci"]
    print(
        f"  auto-handles {r['auto_handled']}/{r['auto_handled'] + r['escalated']} ({r['coverage']:.0%}), "
        f"error there {r['error_on_auto_handled']:.3f} [Wilson {lo:.2f}, {hi:.2f}] "
        f"(overall {r['error_overall']:.3f})"
    )
    print(
        f"  catches {r['errors_caught']}/{r['errors_total']} of the classifier's errors "
        f"(random escalation at the same rate: {r['random_escalation_expected_caught']:.0f})"
    )
    for rule, n in r["reasons"].items():
        print(f"    {rule:<26}{n:>4}")
    dm = r["brand_dm_proxy"]
    print(
        f"  vs the brand's own DM handoffs (a proxy the router never sees): escalates "
        f"{dm['escalated_dm']}/{dm['dm_cases']} DM cases and {dm['escalated_non_dm']}/"
        f"{dm['non_dm_cases']} others, Fisher p={dm['fisher_p']:.3f}; "
        f"low margin predicts a DM at AUC {dm['low_margin_auc']:.2f}"
    )

    sr = r["across_fold_seeds"]
    print(
        f"  across {m['repeats']} fold seeds: auto-handles {sr['coverage'][0]:.0%}-{sr['coverage'][1]:.0%}, "
        f"error there {sr['error_on_auto'][0]:.3f}-{sr['error_on_auto'][1]:.3f}"
    )
    ho = r["held_out"]
    if ho.get("chosen") is not None:
        tlo, thi = ho["test_err_ci"] or (float("nan"), float("nan"))
        print(
            f"  held out: threshold {ho['chosen']:.2f} chosen on dev (n={ho['dev_n']}; "
            f"{ho['dev_auto']} auto-handled, {ho['dev_err']} wrong) -> test (n={ho['test_n']}): "
            f"{ho['test_auto']} auto-handled, {ho['test_err']} wrong [Wilson {tlo:.2f}, {thi:.2f}], "
            f"test base error {ho['test_base_error']:.3f}"
        )
    rs = r["random_stratum"]
    print(
        f"  random stratum only (n={rs['n']}, the closest thing to real traffic): "
        f"{rs['auto_handled']} auto-handled, {rs['errors_on_auto']} wrong; "
        f"{rs['errors_caught']}/{rs['errors']} errors escalated"
    )

    print("\n  operating-point sweep (margin threshold):")
    print(f"    {'thr':>5}{'coverage':>10}{'err@auto':>10}{'caught':>8}{'wasted':>8}")
    for row in m["routing_sweep"]:
        err = "n/a" if row["error_on_auto"] is None else f"{row['error_on_auto']:.3f}"
        print(
            f"    {row['threshold']:>5.2f}{row['coverage']:>10.0%}{err:>10}"
            f"{row['errors_caught']:>8}{row['wasted_escalations']:>8}"
        )

    if m["drafting"]:
        dr = m["drafting"]
        print(f"\nretrieval top-1 similarity: {dr['retrieval_top1_similarity']:.3f}")
    if m["drafting"].get("arms"):
        print(
            f"\ndrafting, n={dr['n']} messages; judge {dr['judge']} (rubric {dr['rubric']}), "
            "a different model family from the drafter"
        )
        print(
            f"  {'drafter':<16}{'valid':>6}{'judged':>8}{'total/8':>9}   sendable [Wilson 95%]"
            "      placeholders   sendable as-is"
        )
        for name, a in dr["arms"].items():
            if not a["judged"]:
                print(
                    f"  {name:<16}{a['valid']:>6}{0:>8}   not scored ({a['uncached']} not in cache)"
                )
                continue
            lo, hi = a["sendable_ci"]
            slo, shi = a["sendable_as_is_ci"]
            print(
                f"  {name:<16}{a['valid']:>6}{a['judged']:>8}{a['mean_total']:>9.2f}   "
                f"{a['sendable'] / a['judged']:.0%} [{lo:.2f}, {hi:.2f}]{a['with_placeholder']:>18}"
                f"   {a['sendable_as_is'] / a['judged']:.0%} [{slo:.2f}, {shi:.2f}]"
            )
        missing = sum(a["uncached"] for a in dr["arms"].values())
        if missing:
            print(
                f"  {missing} drafts or judgements not cached: run scripts/bakeoff_draft.py --n {dr['n']}"
            )
        for comp, c in dr["sendable_mcnemar"].items():
            print(
                f"  sendable, {comp}: {c['a_only']} vs {c['b_only']} discordant, n={c['n']}, "
                f"McNemar p={c['p']:.3f}; as-is {c['as_is_a_only']} vs {c['as_is_b_only']}, "
                f"p={c['as_is_p']:.3f}"
            )
        for rival in ("nearest", "ungrounded_llm"):
            dims = "; ".join(
                f"{d} {v['llm']:.2f} vs {v['rival']:.2f} (p={v['p']:.4f})"
                for d, v in dr[f"dimensions_llm_vs_{rival}"].items()
                if v["n"]
            )
            print(f"  per dimension, grounded_llm vs {rival} (Wilcoxon, Bonferroni 0.0125): {dims}")

    if args.naive:
        rows = naive_waterfall(args, Embedder(offline=args.offline), load_taxonomy().ids)
        m["naive_waterfall"] = rows
        print("\nHONESTY WATERFALL - each step undone on its own, naive config actually run")
        print(f"{'honesty step':<62}{'naive':>8}{'audited':>9}{'delta':>8}  n / notes")
        print("-" * 120)
        for row in rows:
            if row["naive"] is None:
                print(f"{row['step']:<62}{'':>25}  {row['n']}")
                continue
            print(
                f"{row['step']:<62}{row['naive']:>8.3f}{row['audited']:>9.3f}"
                f"{row['naive'] - row['audited']:>+8.3f}  {row['n']}"
            )
            print(f"  {'metric: ' + row['metric']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(m)
    payload["per_class"] = [asdict(c) for c in best.per_class]
    args.out.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nelapsed {m['elapsed_s']}s  ->  {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
