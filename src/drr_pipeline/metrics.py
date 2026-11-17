"""Metrics for Exp-1 (numpy + scipy only): AUROC, average precision, sensitivity at fixed specificity,
bootstrap confidence intervals and paired bootstrap of a difference between two scorers."""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy.stats import rankdata


def auroc(y, s) -> float:
    y, s = np.asarray(y).astype(int), np.asarray(s, dtype=float)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)                       # average ranks: ties count 1/2
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def average_precision(y, s) -> float:
    y, s = np.asarray(y).astype(int), np.asarray(s, dtype=float)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    yy = y[order]
    tp = np.cumsum(yy)
    prec = tp / np.arange(1, len(yy) + 1)
    return float((prec * yy).sum() / yy.sum())


def threshold_at_specificity(y, s, spec: float = 0.90) -> float:
    """Score threshold such that a fraction `spec` of the NEGATIVES scores at or below it."""
    neg = np.asarray(s, dtype=float)[np.asarray(y).astype(int) == 0]
    return float(np.quantile(neg, spec))


def sensitivity_at_threshold(y, s, thr: float) -> float:
    pos = np.asarray(s, dtype=float)[np.asarray(y).astype(int) == 1]
    return float((pos > thr).mean()) if len(pos) else float("nan")


def sens_at_spec(y, s, spec: float = 0.90) -> float:
    return sensitivity_at_threshold(y, s, threshold_at_specificity(y, s, spec))


def bootstrap_ci(metric: Callable, y, s, n: int = 2000, seed: int = 0, alpha: float = 0.05) -> Tuple[float, float]:
    y, s = np.asarray(y), np.asarray(s)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        v = metric(y[i], s[i])
        if not np.isnan(v):
            vals.append(v)
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def paired_bootstrap_diff(metric: Callable, y, s_a, s_b, n: int = 2000, seed: int = 0,
                          alpha: float = 0.05) -> Dict[str, float]:
    """metric(s_a) - metric(s_b) on the SAME resampled images; reports the point estimate and the CI."""
    y, a, b = np.asarray(y), np.asarray(s_a), np.asarray(s_b)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        va, vb = metric(y[i], a[i]), metric(y[i], b[i])
        if not (np.isnan(va) or np.isnan(vb)):
            d.append(va - vb)
    return {"diff": float(metric(y, a) - metric(y, b)), "lo": float(np.quantile(d, alpha / 2)),
            "hi": float(np.quantile(d, 1 - alpha / 2))}


def seed_mean_bootstrap(metric: Callable, y, S, n: int = 2000, seed: int = 0, alpha: float = 0.05) -> Dict[str, float]:
    """S: (n_runs, N) scores of several runs (seeds/folds) on the SAME N test images. Resamples the images (the same resample for
    every run) and averages the metric over runs, so the CI is for the run-averaged metric and reflects test-set sampling only
    (not the training seed: that is the SD over runs, reported as `sd_runs`)."""
    y, S = np.asarray(y), np.atleast_2d(np.asarray(S, dtype=float))
    per_run = np.array([metric(y, s) for s in S])
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        v = np.mean([metric(y[i], s[i]) for s in S])
        if not np.isnan(v):
            vals.append(v)
    return {"mean": float(per_run.mean()), "sd_runs": float(per_run.std(ddof=1)) if len(per_run) > 1 else float("nan"),
            "lo": float(np.quantile(vals, alpha / 2)), "hi": float(np.quantile(vals, 1 - alpha / 2)), "n_runs": int(len(per_run))}


def paired_seed_mean_diff(metric: Callable, y, S_a, S_b, n: int = 2000, seed: int = 0, alpha: float = 0.05) -> Dict[str, float]:
    """mean-over-runs metric(A) - mean-over-runs metric(B) with the same image resample for every run of both sides."""
    y, A, B = np.asarray(y), np.atleast_2d(np.asarray(S_a, dtype=float)), np.atleast_2d(np.asarray(S_b, dtype=float))
    d0 = float(np.mean([metric(y, s) for s in A]) - np.mean([metric(y, s) for s in B]))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        v = np.mean([metric(y[i], s[i]) for s in A]) - np.mean([metric(y[i], s[i]) for s in B])
        if not np.isnan(v):
            vals.append(v)
    return {"diff": d0, "lo": float(np.quantile(vals, alpha / 2)), "hi": float(np.quantile(vals, 1 - alpha / 2))}
