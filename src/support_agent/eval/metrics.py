"""Metrics, with the uncertainty attached.

Every function here returns the count it was computed over, because `0.82` on a
220-row golden set and `0.82` on a 20,000-row test set are not the same claim and
should not be printable in the same way.

Macro-F1 is the headline because the class distribution is skewed: the largest
intent is 18% of the golden set and the smallest is 4%. Accuracy on that
distribution rewards a model for being right about common intents and is nearly
insensitive to it being useless on rare ones.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    The normal approximation is wrong at the sample sizes in this project - with
    n=9 for the rarest intent it can produce bounds outside [0, 1]. Wilson does
    not.
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass
class ClassMetrics:
    label: str
    support: int
    precision: float
    recall: float
    f1: float
    ci_low: float
    ci_high: float


@dataclass
class IntentReport:
    n: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    macro_f1_ci: tuple[float, float]
    per_class: list[ClassMetrics] = field(default_factory=list)
    confusion: dict[tuple[str, str], int] = field(default_factory=dict)

    def table(self) -> str:
        w = max(len(c.label) for c in self.per_class) if self.per_class else 8
        lines = [f"{'intent':<{w}} {'n':>4} {'prec':>6} {'recall':>7} {'F1':>6}  95% CI on recall"]
        for c in sorted(self.per_class, key=lambda c: -c.support):
            lines.append(
                f"{c.label:<{w}} {c.support:>4} {c.precision:>6.2f} {c.recall:>7.2f} "
                f"{c.f1:>6.2f}  [{c.ci_low:.2f}, {c.ci_high:.2f}]"
            )
        lines.append(
            f"\nmacro-F1 {self.macro_f1:.3f} "
            f"[{self.macro_f1_ci[0]:.3f}, {self.macro_f1_ci[1]:.3f}]  "
            f"accuracy {self.accuracy:.3f}  n={self.n}"
        )
        return "\n".join(lines)


def _f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def macro_f1(y_true: list[str], y_pred: list[str], labels: list[str] | None = None) -> float:
    labels = labels or sorted(set(y_true) | set(y_pred))
    scores = []
    for lab in labels:
        tp = sum(t == lab and p == lab for t, p in zip(y_true, y_pred, strict=True))
        fp = sum(t != lab and p == lab for t, p in zip(y_true, y_pred, strict=True))
        fn = sum(t == lab and p != lab for t, p in zip(y_true, y_pred, strict=True))
        scores.append(_f1(tp, fp, fn)[2])
    return float(np.mean(scores)) if scores else 0.0


def bootstrap_ci(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap over examples.

    Resampling examples, not predictions, is the point: it answers "how much
    would this number move if I had drawn a different 220 messages", which is
    the question a reader of a small golden set actually has.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return (0.0, 0.0)
    labels = labels or sorted(set(y_true) | set(y_pred))
    yt, yp = np.array(y_true), np.array(y_pred)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        draws.append(macro_f1(list(yt[idx]), list(yp[idx]), labels))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return (float(lo), float(hi))


def intent_report(
    y_true: list[str], y_pred: list[str], labels: list[str] | None = None, seed: int = 0
) -> IntentReport:
    labels = labels or sorted(set(y_true) | set(y_pred))
    per_class = []
    for lab in labels:
        tp = sum(t == lab and p == lab for t, p in zip(y_true, y_pred, strict=True))
        fp = sum(t != lab and p == lab for t, p in zip(y_true, y_pred, strict=True))
        fn = sum(t == lab and p != lab for t, p in zip(y_true, y_pred, strict=True))
        prec, rec, f1 = _f1(tp, fp, fn)
        lo, hi = wilson(tp, tp + fn)
        per_class.append(ClassMetrics(lab, tp + fn, prec, rec, f1, lo, hi))

    acc = sum(t == p for t, p in zip(y_true, y_pred, strict=True)) / len(y_true)
    support = Counter(y_true)
    weighted = sum(c.f1 * support[c.label] for c in per_class) / max(1, len(y_true))
    return IntentReport(
        n=len(y_true),
        accuracy=acc,
        macro_f1=macro_f1(y_true, y_pred, labels),
        weighted_f1=weighted,
        macro_f1_ci=bootstrap_ci(y_true, y_pred, labels, seed=seed),
        per_class=per_class,
        confusion=Counter(zip(y_true, y_pred, strict=True)),
    )


def cohens_kappa(a: list[str], b: list[str]) -> tuple[float, int]:
    """Cohen's kappa between two label sequences, with n.

    Used for judge-human and annotator agreement. Landis & Koch
    bands: <0 poor, 0-.20 slight, .21-.40 fair, .41-.60 moderate, .61-.80
    substantial, .81-1 almost perfect.
    """
    n = len(a)
    if n == 0:
        return (0.0, 0)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum((ca[k] / n) * (cb[k] / n) for k in set(a) | set(b))
    if expected == 1.0:
        return (1.0 if observed == 1.0 else 0.0, n)
    return ((observed - expected) / (1 - expected), n)


def mcnemar(a: list[bool], b: list[bool]) -> tuple[int, int, int, int, float]:
    """Exact McNemar test on paired booleans: (both, a_only, b_only, neither, p).

    Valid for one set of paired items. Do not pool counts over CV repeats of the
    same items (DECISIONS #42).
    """
    from scipy.stats import binomtest

    both = sum(x and y for x, y in zip(a, b, strict=True))
    a_only = sum(x and not y for x, y in zip(a, b, strict=True))
    b_only = sum(y and not x for x, y in zip(a, b, strict=True))
    neither = len(a) - both - a_only - b_only
    disc = a_only + b_only
    p = float(binomtest(a_only, disc, 0.5).pvalue) if disc else 1.0
    return both, a_only, b_only, neither, p


def weighted_kappa(a: list[int], b: list[int], categories: tuple[int, ...] = (0, 1, 2)) -> float:
    """Quadratic-weighted Cohen's kappa for ordinal scores.

    Unweighted kappa counts a 2-vs-1 disagreement on a 0-2 rubric as fully as a
    2-vs-0 one. For ordinal scales the quadratic weighting is standard, and it
    equals the intraclass correlation under mild conditions.
    """
    n, k = len(a), len(categories)
    if n == 0:
        return 0.0
    pos = {c: i for i, c in enumerate(categories)}
    obs = np.zeros((k, k))
    for x, z in zip(a, b, strict=True):
        obs[pos[x], pos[z]] += 1
    obs /= n
    exp = np.outer(obs.sum(1), obs.sum(0))
    idx = np.arange(k)
    w = (idx[:, None] - idx[None, :]) ** 2 / (k - 1) ** 2
    denom = (w * exp).sum()
    return 1.0 if denom == 0 else float(1 - (w * obs).sum() / denom)


def gwet_ac1(a: list[str], b: list[str]) -> float:
    """Gwet's AC1: chance-corrected agreement that stays stable at skewed prevalence.

    Kappa collapses when one category dominates (the "kappa paradox"): two raters
    who agree on 38 of 40 drafts can still get a kappa near zero if nearly every
    draft is `safe=2`. AC1 is reported beside kappa, never instead of it.
    """
    n = len(a)
    cats = sorted(set(a) | set(b))
    if n == 0 or len(cats) < 2:
        return 1.0 if n else 0.0
    observed = sum(x == z for x, z in zip(a, b, strict=True)) / n
    pi = [(a.count(c) + b.count(c)) / (2 * n) for c in cats]
    expected = sum(p * (1 - p) for p in pi) / (len(cats) - 1)
    return float((observed - expected) / (1 - expected))


def corrected_resampled_ttest(
    diffs: np.ndarray, n_train: int, n_test: int
) -> tuple[float, float, float]:
    """Nadeau-Bengio corrected t-test for r repeats x k folds of paired scores.

    `diffs` holds one score difference per fold, over every repeat. Folds share
    training data and repeats re-use the same examples, so the naive t-test (or a
    McNemar test on counts pooled over repeats) treats correlated numbers as
    independent and reports p-values that are far too small. The correction
    inflates the variance by n_test/n_train (Nadeau & Bengio 2003; Bouckaert &
    Frank 2004). Returns (mean difference, t, two-sided p).
    """
    from scipy.stats import t as t_dist

    d = np.asarray(diffs, dtype=float).ravel()
    m = len(d)
    var = d.var(ddof=1) if m > 1 else 0.0
    if var == 0:
        return float(d.mean()), float("inf") if d.mean() else 0.0, 0.0 if d.mean() else 1.0
    t = d.mean() / np.sqrt((1 / m + n_test / n_train) * var)
    return float(d.mean()), float(t), float(2 * t_dist.sf(abs(t), m - 1))


def kappa_ci(
    a: list[str],
    b: list[str],
    n_boot: int = 2000,
    seed: int = 0,
    groups: list[str] | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap 95% interval for Cohen's kappa over paired items.

    At n~40 a point kappa alone hides an interval that is often 0.4 wide, which
    is the difference between "moderate" and "fair" agreement. With `groups`, whole
    groups are resampled: two drafts for the same customer message are not
    independent evidence.
    """
    n = len(a)
    if n < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    clusters: list[list[int]] = (
        [[i for i in range(n) if groups[i] == g] for g in dict.fromkeys(groups)]
        if groups
        else [[i] for i in range(n)]
    )
    ks = []
    for _ in range(n_boot):
        idx = [i for c in rng.integers(0, len(clusters), len(clusters)) for i in clusters[c]]
        ks.append(cohens_kappa([a[i] for i in idx], [b[i] for i in idx])[0])
    return (float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5)))


def kappa_band(kappa: float) -> str:
    for threshold, name in (
        (0.0, "poor"),
        (0.20, "slight"),
        (0.40, "fair"),
        (0.60, "moderate"),
        (0.80, "substantial"),
    ):
        if kappa <= threshold:
            return name
    return "almost perfect"


def top_confusions(report: IntentReport, k: int = 8) -> list[tuple[str, str, int]]:
    """Most frequent (true, predicted) mistakes, for the failure analysis."""
    wrong = [(t, p, n) for (t, p), n in report.confusion.items() if t != p]
    return sorted(wrong, key=lambda r: -r[2])[:k]
