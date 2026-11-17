"""E15a: linear probes and zero-shot scoring on frozen features (numpy + scipy only, no torch).

Everything here is label-free with respect to the TARGET unless a function says it takes target labels:
  fit_logreg / choose_C_by_auroc : L2 logistic regression (Newton with backtracking), C picked on a validation set the caller supplies;
  inner_cv_C                     : C picked by stratified k-fold on the labelled samples themselves (used for the K-label probes);
  zero_shot_scores               : cosine(image, positive prompts) - cosine(image, negative prompts) (BiomedCLIP);
  auroc_by_age                   : AUROC inside age strata (pediatric target).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from . import metrics as Mx

CS = (0.003, 0.03, 0.3)
AGE_BINS = (("<6m", 0, 6), ("6-12m", 6, 12), ("12-24m", 12, 24), (">=24m", 24, 1e9))


def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return X.mean(0), X.std(0) + 1e-6


def standardize(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd


def _objective(X1, y, sw, w, lam):
    z = X1 @ w
    return float((sw * (np.logaddexp(0.0, z) - y * z)).sum() + 0.5 * lam * (w[:-1] ** 2).sum())


def fit_logreg(X: np.ndarray, y: np.ndarray, C: float, sw: Optional[np.ndarray] = None, max_iter: int = 40,
               tol: float = 1e-7) -> Tuple[np.ndarray, float]:
    """minimise sum_i sw_i * logloss_i + 0.5/C * ||w||^2 (intercept not penalised). Returns (w, b)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = X.shape
    sw = np.ones(n) if sw is None else np.asarray(sw, dtype=np.float64)
    lam = 1.0 / C
    X1 = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(d + 1)
    pen = np.full(d + 1, lam)
    pen[-1] = 1e-8
    f = _objective(X1, y, sw, w, lam)
    for _ in range(max_iter):
        p = sigmoid(X1 @ w)
        g = X1.T @ (sw * (p - y)) + pen * w
        if np.linalg.norm(g) < tol * max(1.0, n):
            break
        H = (X1 * (sw * p * (1 - p))[:, None]).T @ X1 + np.diag(pen)
        step = np.linalg.solve(H, g)
        t = 1.0
        while t > 1e-6:
            w_new = w - t * step
            f_new = _objective(X1, y, sw, w_new, lam)
            if f_new <= f - 1e-4 * t * float(g @ step):
                break
            t *= 0.5
        else:
            break
        converged = abs(f - f_new) < 1e-10 * max(1.0, abs(f))
        w, f = w_new, f_new
        if converged:
            break
    return w[:-1], float(w[-1])


def predict_logit(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return np.asarray(X, dtype=np.float64) @ w + b


def choose_C_by_auroc(Xtr, ytr, Xva, yva, Cs: Sequence[float] = CS, sw=None) -> Tuple[float, np.ndarray, float]:
    """C with the best validation AUROC (ties -> the smaller C). Returns (C, w, b)."""
    best = None
    for C in Cs:
        w, b = fit_logreg(Xtr, ytr, C, sw)
        a = Mx.auroc(yva, predict_logit(Xva, w, b))
        a = -1.0 if np.isnan(a) else a
        if best is None or a > best[0] + 1e-12:
            best = (a, C, w, b)
    return best[1], best[2], best[3]


def stratified_folds(y: np.ndarray, k: int, rng: np.random.Generator):
    y = np.asarray(y).astype(int)
    fold = np.zeros(len(y), int)
    for c in (0, 1):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % k
    return fold


def inner_cv_C(X, y, Cs: Sequence[float] = CS, k: int = 5, seed: int = 0, default: Optional[float] = None,
               prestandardized: bool = False) -> float:
    """C chosen by stratified k-fold AUROC on the labelled samples. Too few samples per class -> k shrinks; < 2 -> default.
    prestandardized=True: X is already standardized (e.g. with unlabeled-pool statistics) and is used as given in every fold."""
    y = np.asarray(y).astype(int)
    k = int(min(k, y.sum(), (1 - y).sum()))
    default = Cs[len(Cs) // 2] if default is None else default
    if k < 2:
        return default
    fold = stratified_folds(y, k, np.random.default_rng(seed))
    scores = []
    for C in Cs:
        pred = np.zeros(len(y))
        for f in range(k):
            tr, te = fold != f, fold == f
            mu, sd = (np.zeros(X.shape[1]), np.ones(X.shape[1])) if prestandardized else standardize_fit(X[tr])
            w, b = fit_logreg(standardize(X[tr], mu, sd), y[tr], C)
            pred[te] = predict_logit(standardize(X[te], mu, sd), w, b)
        a = Mx.auroc(y, pred)
        scores.append(-1.0 if np.isnan(a) else a)
    return Cs[int(np.argmax(scores))]


def subsample_stratified(y: np.ndarray, k: int, rng: np.random.Generator, min_per_class: int = 3) -> np.ndarray:
    """k indices keeping the prevalence of y, with at least `min_per_class` of each class."""
    y = np.asarray(y).astype(int)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    kp = int(np.clip(round(k * len(pos) / len(y)), min_per_class, min(len(pos), k - min_per_class)))
    return np.sort(np.concatenate([rng.choice(pos, kp, replace=False), rng.choice(neg, k - kp, replace=False)]))


def zero_shot_scores(img_emb: np.ndarray, t_pos: np.ndarray, t_neg: np.ndarray) -> np.ndarray:
    """cos(image, mean positive prompt) - cos(image, mean negative prompt); embeddings are L2-normalised first."""
    nz = lambda a: a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    tp, tn = nz(nz(t_pos).mean(0)), nz(nz(t_neg).mean(0))
    return nz(img_emb) @ (tp - tn)


def auroc_by_age(y, s, age_months, bins=AGE_BINS) -> Dict[str, Dict[str, float]]:
    y, s, age = np.asarray(y).astype(int), np.asarray(s, dtype=float), np.asarray(age_months, dtype=float)
    out = {}
    for name, lo, hi in bins:
        m = (age >= lo) & (age < hi)
        out[name] = {"n_pos": int(y[m].sum()), "n_neg": int((1 - y[m]).sum()), "auroc": Mx.auroc(y[m], s[m]) if m.any() else float("nan")}
    m = ~np.isfinite(age)
    out["unknown"] = {"n_pos": int(y[m].sum()), "n_neg": int((1 - y[m]).sum()), "auroc": Mx.auroc(y[m], s[m]) if m.any() else float("nan")}
    return out
