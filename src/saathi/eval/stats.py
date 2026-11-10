"""Statistics for the report: confidence intervals, paired tests, multiple-comparison
correction, and judge agreement. numpy only, so every step can be checked by hand.

Clustering: the three translations of a question (same group_id) are not independent, so
bootstrap resampling draws whole groups. Treating them as independent would make the
confidence intervals too narrow.
"""

from math import comb

import numpy as np
from pydantic import BaseModel


class CI(BaseModel):
    mean: float
    lo: float
    hi: float
    n: int


class Paired(BaseModel):
    diff: float  # mean(a) - mean(b)
    lo: float
    hi: float
    p: float
    n: int


def _cluster_index(clusters: list[str]) -> list[np.ndarray]:
    """Row indices for each cluster, in sorted cluster order."""
    order: dict[str, list[int]] = {}
    for i, c in enumerate(clusters):
        order.setdefault(c, []).append(i)
    return [np.array(order[c]) for c in sorted(order)]


def _bootstrap_means(values: np.ndarray, clusters: list[str], n_boot: int, seed: int) -> np.ndarray:
    """Mean of `values` over n_boot cluster-bootstrap resamples."""
    groups = _cluster_index(clusters)
    sums = np.array([values[g].sum() for g in groups])
    sizes = np.array([len(g) for g in groups])
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(groups), size=(n_boot, len(groups)))
    return sums[picks].sum(axis=1) / sizes[picks].sum(axis=1)


def bootstrap_ci(
    values: list[float],
    clusters: list[str] | None = None,
    n_boot: int = 10_000,
    seed: int = 0,
    level: float = 0.95,
) -> CI:
    """Percentile bootstrap CI of the mean. Without clusters, each value is its own cluster."""
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        raise ValueError("no values")
    clusters = clusters if clusters is not None else [str(i) for i in range(len(x))]
    means = _bootstrap_means(x, clusters, n_boot, seed)
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(means, [tail, 100 - tail])
    return CI(mean=float(x.mean()), lo=float(lo), hi=float(hi), n=len(x))


def paired_bootstrap(
    a: list[float],
    b: list[float],
    clusters: list[str],
    n_boot: int = 10_000,
    seed: int = 0,
    level: float = 0.95,
) -> Paired:
    """Difference of means between two arms scored on the SAME items (same order).
    Two-sided p-value: twice the share of resampled differences on the far side of 0."""
    if len(a) != len(b) or len(a) != len(clusters):
        raise ValueError("a, b and clusters must align item by item")
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    means = _bootstrap_means(d, clusters, n_boot, seed)
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(means, [tail, 100 - tail])
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return Paired(diff=float(d.mean()), lo=float(lo), hi=float(hi), p=float(min(1.0, p)), n=len(d))


def mcnemar_exact(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    """Exact McNemar test on paired pass/fail outcomes.
    Returns (a right & b wrong, a wrong & b right, two-sided p). Ignores clustering, so it
    is reported alongside the cluster bootstrap, not instead of it."""
    if len(a) != len(b):
        raise ValueError("paired lists must have equal length")
    n10 = sum(x and not y for x, y in zip(a, b, strict=True))
    n01 = sum(y and not x for x, y in zip(a, b, strict=True))
    n = n10 + n01
    if n == 0:
        return 0, 0, 1.0
    k = min(n10, n01)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n
    return n10, n01, min(1.0, p)


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, returned in the input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def cohen_kappa(x: list[str], y: list[str]) -> float:
    """Agreement between two raters beyond chance: (p_observed - p_chance) / (1 - p_chance)."""
    if len(x) != len(y) or not x:
        raise ValueError("need two equal-length, non-empty label lists")
    n = len(x)
    labels = sorted(set(x) | set(y))
    p_observed = sum(a == b for a, b in zip(x, y, strict=True)) / n
    p_chance = sum((x.count(lab) / n) * (y.count(lab) / n) for lab in labels)
    if p_chance == 1:
        return 1.0  # both raters used one identical label throughout
    return (p_observed - p_chance) / (1 - p_chance)
