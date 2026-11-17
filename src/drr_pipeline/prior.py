"""Label-free estimate of the target class prior (Saerens, Latinne & Decaestecker 2002), numpy only.

A classifier trained with class prior pi_s gives posteriors p_s(y=1|x). On a target pool with prior pi_t the
posteriors are re-weighted by the prior ratio; the EM iteration below finds the pi_t that is consistent with the
pool's own predictions. It needs reasonably calibrated posteriors, so under a strong domain shift the estimate can
be off: it is reported (E7 in docs/experiment_ideas.md), never taken on trust.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def log_odds(pi: float) -> float:
    pi = min(max(float(pi), 1e-6), 1 - 1e-6)
    return float(np.log(pi / (1 - pi)))


def estimate_target_prior(p_src, pi_src: float, iters: int = 100, tol: float = 1e-5) -> Tuple[float, float]:
    """p_src: source-model P(y=1|x) on target images. Returns (pi_t, logit_shift) where adding logit_shift to the
    model's logits gives target-prior-adjusted posteriors."""
    z = logit(p_src)
    pi_t = float(pi_src)
    for _ in range(iters):
        shift = log_odds(pi_t) - log_odds(pi_src)
        new = float(np.mean(1.0 / (1.0 + np.exp(-(z + shift)))))
        if abs(new - pi_t) < tol:
            pi_t = new
            break
        pi_t = new
    return pi_t, log_odds(pi_t) - log_odds(pi_src)
