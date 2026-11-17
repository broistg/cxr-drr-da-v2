"""Per-case statistics for CT + label volumes (pure numpy, no file I/O).

All functions take arrays in the file's own axis order plus the voxel spacing (mm), so they are
testable without nibabel. `scripts/inspect_dataset.py` does the file reading.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np


def looks_like_hu(ct: np.ndarray) -> bool:
    """True if the intensities look like Hounsfield units rather than a rescaled [0, 255] image.

    Air in a chest CT is about -1000 HU, so a volume whose minimum is far below 0 is HU.
    A volume with min >= 0 and max <= 255 is treated as normalised (unusable for physical DRRs).
    """
    lo, hi = float(ct.min()), float(ct.max())
    if lo >= 0 and hi <= 255:
        return False
    return lo <= -200


def hu_summary(ct: np.ndarray) -> Dict[str, float]:
    p = np.percentile(ct, [1, 50, 99])
    return {
        "hu_min": float(ct.min()),
        "hu_p01": float(p[0]),
        "hu_median": float(p[1]),
        "hu_p99": float(p[2]),
        "hu_max": float(ct.max()),
    }


def label_counts(lab: np.ndarray) -> Dict[int, int]:
    vals, cnt = np.unique(lab.astype(np.int64), return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, cnt)}


def volume_ml(n_voxels: int, spacing_mm: Sequence[float]) -> float:
    return n_voxels * float(np.prod(spacing_mm)) / 1000.0


def case_stats(
    ct: np.ndarray,
    spacing_mm: Sequence[float],
    lung: np.ndarray,
    infection: np.ndarray,
    separate_masks: bool,
) -> Dict[str, float]:
    """Core numbers for one case.

    lung, infection: boolean arrays, same shape as ct.
    separate_masks: True if they came from two different files (overlap is then meaningful);
    False if they were split from a single label volume (overlap is 0 by construction).
    """
    if not (ct.shape == lung.shape == infection.shape):
        raise ValueError(f"shape mismatch: ct {ct.shape}, lung {lung.shape}, inf {infection.shape}")
    n_lung, n_inf = int(lung.sum()), int(infection.sum())
    n_inf_in_lung = int((infection & lung).sum())
    out = {
        "shape": "x".join(map(str, ct.shape)),
        "spacing_mm": "x".join(f"{s:.2f}" for s in spacing_mm),
        "is_hu": looks_like_hu(ct),
        "lung_ml": volume_ml(n_lung, spacing_mm),
        "infection_ml": volume_ml(n_inf, spacing_mm),
        # percent infected volume (PIV): infection / (lung + infection) when combined,
        # infection / lung when the lung mask already contains the infection.
        "piv_percent": float("nan"),
        "inf_outside_lung_percent": float("nan"),
        "has_infection": n_inf > 0,
    }
    if separate_masks:
        out["piv_percent"] = 100.0 * n_inf_in_lung / max(n_lung, 1)
        out["inf_outside_lung_percent"] = 100.0 * (n_inf - n_inf_in_lung) / max(n_inf, 1)
    else:
        out["piv_percent"] = 100.0 * n_inf / max(n_lung + n_inf, 1)
        out["inf_outside_lung_percent"] = 0.0
    out.update(hu_summary(ct))
    return out


def usable(stats: Dict[str, float], min_infection_ml: float = 1.0) -> bool:
    """Case can be used for physical DRRs with infection labels."""
    return bool(stats["is_hu"] and stats["lung_ml"] > 0 and stats["infection_ml"] >= min_infection_ml)


def coronal_projection(vol: np.ndarray, ap_axis: int) -> np.ndarray:
    """Mean along the antero-posterior axis: a crude DRR-like view for eyeballing."""
    return vol.mean(axis=ap_axis)
