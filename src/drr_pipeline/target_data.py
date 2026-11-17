"""Target-domain (real CXR) data access shared by training, diagnostics and feature extraction.

Layout written by scripts/prep_vindr.py:
  <target_dir>/manifest.csv  (image_id, split, role, ... age_months when known)
  <target_dir>/<split>/<image_id>.png   with split in {train_target, target_val, test}
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import cxr_prep as P

MAIN = P.MAIN_ROLES                                           # ("positive", "negative")
ALL_ROLES = (P.POSITIVE, P.NEGATIVE, P.AIRWAY, P.OTHER, P.UNLABELED)
TEST_ROLES = MAIN + (P.AIRWAY,)                               # test also keeps the airway hard negatives (VinDr-PCXR)

# name -> directory relative to the repository root (used by --target of the feature scripts)
TARGET_DIRS: Dict[str, str] = {
    "vindr_pcxr": "data/processed/vindr_pcxr",
}


def read_manifest(target_dir) -> List[dict]:
    with open(Path(target_dir) / "manifest.csv", encoding="utf8") as f:
        return list(csv.DictReader(f))


def rows_of(rows: Sequence[dict], split: str, roles: Optional[Sequence[str]] = None) -> List[dict]:
    return [r for r in rows if r["split"] == split and (roles is None or r["role"] in roles)]


def load_png(target_dir, row: dict) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(Path(target_dir) / row["split"] / f"{row['image_id']}.png").convert("L"))


def load_png_split(target_dir, rows: Sequence[dict], roles: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """(images uint8 (N, H, W), labels 1 = positive, image ids, roles) of the rows whose role is in `roles`."""
    sel = [r for r in rows if r["role"] in roles]
    x = np.stack([load_png(target_dir, r) for r in sel])
    return x, np.array([1 if r["role"] == P.POSITIVE else 0 for r in sel]), [r["image_id"] for r in sel], [r["role"] for r in sel]


def age_months(rows: Sequence[dict]) -> np.ndarray:
    """Age in months per row, NaN when unknown (empty or missing column)."""
    out = []
    for r in rows:
        v = r.get("age_months", "")
        try:
            out.append(float(v) if v not in ("", None) else np.nan)
        except ValueError:
            out.append(np.nan)
    return np.array(out, dtype=float)
