"""Preprocessing of real chest X-rays for Exp-1 (pure numpy/stdlib; DICOM reading is in the script).

Fixed recipe (docs/experiment_design.md section 3), the same for every arm:
  1. invert MONOCHROME1 (pixel low = bright) so that bone / soft tissue are bright, as in the DRRs;
  2. window to the 1-99 percentile of the image, scale to uint8;
  3. pad to a square (aspect ratio kept) and resize to 256 x 256.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# roles used by Exp-1 (docs/experiment_design.md section 2)
POSITIVE, NEGATIVE, AIRWAY, OTHER, UNLABELED = "positive", "negative", "airway_hard_negative", "other_excluded", "unlabeled_excluded"
MAIN_ROLES = (POSITIVE, NEGATIVE)


def to_uint8(arr: np.ndarray, monochrome1: bool, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 3:
        a = a[..., 0]
    if monochrome1:
        a = a.max() - a
    lo, hi = np.percentile(a, [lo_pct, hi_pct])
    if hi <= lo:
        return np.zeros(a.shape, np.uint8)
    return (np.clip((a - lo) / (hi - lo), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def pad_square(img: np.ndarray, value: int = 0) -> np.ndarray:
    h, w = img.shape
    s = max(h, w)
    out = np.full((s, s), value, dtype=img.dtype)
    y0, x0 = (s - h) // 2, (s - w) // 2
    out[y0:y0 + h, x0:x0 + w] = img
    return out


def assign_role(positive_labels: Iterable[str], groups: Dict[str, Sequence[str]], no_finding: str = "No finding") -> str:
    """Role of one image from its set of positive label names.

    positive: in the pneumonia group (even if also airway / other labels);
    negative: No finding; airway_hard_negative: airway group and not pneumonia group;
    other_excluded: any other disease label; unlabeled_excluded: no label at all.
    """
    labs = set(positive_labels)
    if labs & set(groups["pneumonia_group"]):
        return POSITIVE
    if no_finding in labs:
        return NEGATIVE
    if labs & set(groups["airway_group"]):
        return AIRWAY
    if labs:
        return OTHER
    return UNLABELED


def stratified_val_split(ids: Sequence[str], strata: Sequence[str], frac: float = 0.10, seed: int = 0) -> List[str]:
    """Ids of a validation subset: `frac` of every stratum (at least 1 if the stratum has >= 2 members)."""
    by = defaultdict(list)
    for i, s in zip(ids, strata):
        by[s].append(i)
    rng = random.Random(seed)
    val = []
    for s in sorted(by):
        m = sorted(by[s])
        rng.shuffle(m)
        k = int(round(frac * len(m)))
        if len(m) >= 2:
            k = max(k, 1)
        val += m[:k]
    return sorted(val)
