"""Helpers to ask whether a class-activation map (CAM) points at the lesion (numpy only).

Two ground truths: DRR holdout pixel labels (known exactly) and VinDr-PCXR boxes (original-pixel coordinates; the PNGs were
padded to a square, centred, then resized: see cxr_prep.pad_square / prep_vindr.py).
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


def map_box_to_square(box: Sequence[float], rows: int, cols: int, size: int = 256) -> Tuple[int, int, int, int]:
    """(xmin, ymin, xmax, ymax) in the original image -> pixel box in the size x size padded image (half-open)."""
    s = max(rows, cols)
    x0, y0 = (s - cols) // 2, (s - rows) // 2
    k = size / s
    xmin, ymin, xmax, ymax = box
    out = [(xmin + x0) * k, (ymin + y0) * k, (xmax + x0) * k, (ymax + y0) * k]
    xa, ya = int(np.floor(max(0, min(size, out[0])))), int(np.floor(max(0, min(size, out[1]))))
    xb, yb = int(np.ceil(max(0, min(size, out[2])))), int(np.ceil(max(0, min(size, out[3]))))
    return xa, ya, xb, yb


def boxes_mask(boxes: Iterable[Tuple[int, int, int, int]], size: int = 256) -> np.ndarray:
    m = np.zeros((size, size), bool)
    for xa, ya, xb, yb in boxes:
        m[ya:yb, xa:xb] = True
    return m


def normalize_cam(cam: np.ndarray, region: Optional[np.ndarray] = None) -> np.ndarray:
    """Non-negative map summing to 1 (inside `region` if given; zero outside)."""
    c = np.clip(np.asarray(cam, dtype=np.float64), 0, None)
    if region is not None:
        c = c * region
    t = c.sum()
    return c / t if t > 0 else c


def pointing_hit(cam: np.ndarray, mask: np.ndarray, region: Optional[np.ndarray] = None) -> Optional[bool]:
    """True if the strongest CAM pixel (inside `region`) lies in the ground-truth mask; None if the mask is empty."""
    m = mask if region is None else (mask & region)
    if not m.any():
        return None
    c = np.asarray(cam, dtype=np.float64)
    if region is not None:
        c = np.where(region, c, -np.inf)
    return bool(m.flat[int(np.argmax(c))])


def enrichment(cam: np.ndarray, mask: np.ndarray, region: Optional[np.ndarray] = None) -> float:
    """(share of CAM mass inside the mask) / (share of the area that is mask). 1 = no better than uniform, >1 = concentrated
    on the lesion. Within `region` if given (e.g. the lung footprint). nan if the mask is empty or covers the whole region."""
    reg = np.ones(mask.shape, bool) if region is None else region
    m = mask & reg
    area = m.sum() / max(1, reg.sum())
    if m.sum() == 0 or area >= 1:
        return float("nan")
    c = normalize_cam(cam, reg)
    return float(c[m].sum() / area)


def center_prior(size: int = 256, sigma_frac: float = 0.25) -> np.ndarray:
    """Fixed centred Gaussian blob: the 'lesions are usually central' baseline a CAM has to beat."""
    y, x = np.mgrid[0:size, 0:size].astype(float)
    c = (size - 1) / 2
    return np.exp(-(((x - c) ** 2 + (y - c) ** 2) / (2 * (sigma_frac * size) ** 2)))
