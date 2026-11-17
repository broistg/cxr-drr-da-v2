"""Cache of category projections and derivation of images/labels (no torch needed).

Layout (under --out-dir, default outputs/drr):
  cache/<profile>/<view>/<case>.npz   poses + projections for the 'normal' and 'positive' pose sets
  images/<profile>/<view>/<case>/...  exported by `generate_drr.py export`

A cache file holds, for each kind in {normal, positive}: rot_<kind>, trans_<kind> (n, 3) and
M_<kind>, R_<kind> (n, 3, H, W) float32 (category projections, see weighted.py).
"""
from __future__ import annotations

import zlib
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .weighted import weighted_from_projections

KINDS = ("normal", "positive")


def pose_seed(seed: int, case: str, profile: str, view: str, kind: str) -> list:
    """Stable seed material: the same (seed, case, profile, view, kind) always gives the same poses.

    Normal and positive images of one CT use DIFFERENT poses (kind is part of the seed), as in
    DRR4Covid where the two types are generated independently.
    """
    return [int(seed)] + [zlib.crc32(s.encode("utf8")) for s in (case, profile, view, kind)]


def cache_path(out_dir: Path, profile: str, view: str, case: str) -> Path:
    return Path(out_dir) / "cache" / profile / view / f"{case}.npz"


def save_cache(path: Path, data: Dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")   # written aside so an interrupted run leaves no half file
    np.savez_compressed(tmp, **data)
    tmp.replace(path)


def load_cache(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def to_uint8(p: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    """Per-image robust scaling of the weighted DRR to 8 bit. ASSUMPTION (the paper gives none).

    Percentiles are taken over pixels with p > 0 (rays that cross the body volume).
    """
    pos = p[p > 0]
    if pos.size == 0:
        return np.zeros(p.shape, np.uint8)
    lo, hi = np.percentile(pos, [lo_pct, hi_pct])
    return (np.clip((p - lo) / (hi - lo + 1e-12), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def derive_positive(M: np.ndarray, R: np.ndarray, w: Sequence[float], thresholds: Sequence[float],
                    T1: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """One pose, one weight set -> (image uint8 (H, W), labels uint8 (len(thresholds), H, W))."""
    img, labels = None, []
    for T2 in thresholds:
        p, _, _, lab = weighted_from_projections(M, R, tuple(w), T1, T2)
        if img is None:
            img = to_uint8(p)           # the image does not depend on T2
        labels.append(lab.astype(np.uint8))
    return img, np.stack(labels)


def derive_normal(M: np.ndarray, R: np.ndarray, w_normal: Sequence[float], T1: float = 0.0,
                  T2: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """Normal image with the 'normal' weights. Its label map is ALL ZERO by definition of the class
    (ASSUMPTION: the paper does not say how normal DRRs are labelled)."""
    p, _, _, _ = weighted_from_projections(M, R, tuple(w_normal), T1, T2)
    return to_uint8(p), np.zeros(p.shape, np.uint8)


def weight_tag(w: Sequence[float]) -> str:
    return "w" + "_".join(f"{x:g}" for x in w)
