"""Segmentation metrics and location priors (E16; numpy only). Masks are boolean / {0,1} arrays (N, H, W)."""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np


def dice(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-image Dice of (N,H,W) masks. Both empty -> 1; one empty -> 0."""
    p, g = np.asarray(pred, bool), np.asarray(gt, bool)
    inter = (p & g).sum((1, 2)).astype(float)
    s = p.sum((1, 2)) + g.sum((1, 2))
    return np.where(s == 0, 1.0, 2 * inter / np.maximum(s, 1))


def iou(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    p, g = np.asarray(pred, bool), np.asarray(gt, bool)
    inter = (p & g).sum((1, 2)).astype(float)
    u = (p | g).sum((1, 2))
    return np.where(u == 0, 1.0, inter / np.maximum(u, 1))


def within(pred: np.ndarray, gt: np.ndarray, region: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(pred, gt) restricted to a region mask (N,H,W); used for 'Dice inside the lungs'."""
    r = np.asarray(region, bool)
    return np.asarray(pred, bool) & r, np.asarray(gt, bool) & r


def bootstrap_mean_ci(v: np.ndarray, n: int = 2000, seed: int = 0, alpha: float = 0.05) -> Tuple[float, float]:
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def extent(gt: np.ndarray, lung: Optional[np.ndarray] = None) -> np.ndarray:
    """Lesion area as a fraction of the lung (or of the image when no lung mask is given / the lung mask is empty)."""
    g = np.asarray(gt, bool).sum((1, 2)).astype(float)
    if lung is None:
        return g / (gt.shape[1] * gt.shape[2])
    l = np.asarray(lung, bool).sum((1, 2)).astype(float)
    return np.where(l > 0, g / np.maximum(l, 1), g / (gt.shape[1] * gt.shape[2]))


def by_quartile(values: np.ndarray, key: np.ndarray) -> Dict[str, float]:
    """Mean of `values` within the four quartiles of `key` (q1 = smallest lesions)."""
    qs = np.quantile(key, [0.25, 0.5, 0.75])
    b = np.digitize(key, qs, right=True)
    return {f"q{i + 1}": float(values[b == i].mean()) if (b == i).any() else float("nan") for i in range(4)}


def mean_prior(masks: np.ndarray) -> np.ndarray:
    """Pixel-wise frequency of the lesion mask over the training images: the 'where do lesions usually sit' map."""
    return np.asarray(masks, float).mean(0)


def threshold_prior(prior: np.ndarray, thr: float, n: int) -> np.ndarray:
    return np.repeat((prior >= thr)[None], n, 0)


def best_prior_threshold(prior: np.ndarray, gt_val: np.ndarray, grid: Optional[Sequence[float]] = None,
                         region: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Threshold of the mean-mask prior that maximises mean Dice on validation masks; `region` (N,H,W) restricts the prediction
    (P3: prior inside the lung). Returns (threshold, mean Dice)."""
    grid = list(grid) if grid is not None else list(np.linspace(0.02, 0.9, 45) * float(prior.max()))
    best = (grid[0], -1.0)
    for t in grid:
        p = threshold_prior(prior, t, len(gt_val))
        if region is not None:
            p = p & np.asarray(region, bool)
        d = float(dice(p, gt_val).mean())
        if d > best[1]:
            best = (float(t), d)
    return best


def best_prob_threshold(prob: np.ndarray, gt: np.ndarray, grid: Sequence[float] = tuple(np.linspace(0.1, 0.9, 17))) -> Tuple[float, float]:
    """Probability threshold with the best mean Dice (diagnostic; uses the masks it is given)."""
    best = (0.5, -1.0)
    for t in grid:
        d = float(dice(prob >= t, gt).mean())
        if d > best[1]:
            best = (float(t), d)
    return best


def permutation_pairs(n_pos: int, rng: np.random.Generator) -> np.ndarray:
    """pair[i] = index whose mask image i receives in the null control. A random cyclic order, so no index maps to itself
    (n_pos > 1) and every mask is used exactly once."""
    if n_pos < 2:
        return np.arange(n_pos)
    order = rng.permutation(n_pos)
    pair = np.empty(n_pos, int)
    pair[order] = np.roll(order, -1)
    return pair


def subsample_k(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """k distinct indices of 0..n-1 (all of them if k <= 0 or k >= n)."""
    return np.arange(n) if k <= 0 or k >= n else np.sort(rng.choice(n, k, replace=False))
