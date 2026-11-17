"""Adult CXR target for E10 built from the COVID-19 Radiography Database (pure numpy/stdlib).

The database mixes sources (URL column of the metadata). Only the pairs below are used, because the label must not be
decided by the source: see docs/decisions.md D38.

  task "opacity": positive = Lung_Opacity (all RSNA)            negative = Normal from RSNA
  task "covid"  : positive = COVID (mostly BIMCV, other sites)   negative = Normal from RSNA   (source-confounded: report with caution)

Excluded in both tasks: Viral Pneumonia (all from the pediatric Kaggle chest-xray-pneumonia set) and the 1,341 Normal
images from that same set (pediatric as far as we know), so that age is not mixed back into an adult target.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import cxr_prep as P

SOURCE_PATTERNS = [("rsna", "rsna-pneumonia-detection-challenge"), ("kermany", "chest-xray-pneumonia"), ("bimcv", "bimcv"),
                   ("covid_cxnet", "covid-cxnet"), ("eurorad", "eurorad"), ("ml_workgroup", "ml-workgroup"),
                   ("ieee8023", "ieee8023"), ("sirm", "sirm.org")]
TASKS = {"opacity": "Lung_Opacity", "covid": "COVID"}
DATASET_CLASSES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]


def find_dataset_root(base, classes: Sequence[str] = DATASET_CLASSES) -> Optional[Path]:
    """Directory holding <class>/images for every class (the zip may add one wrapper folder), or None."""
    base = Path(base)
    for cand in [base, *sorted(p for p in base.rglob("*") if p.is_dir())]:
        if all((cand / c / "images").is_dir() for c in classes):
            return cand
    return None


def source_of(url) -> str:
    u = str(url).lower()
    for name, pat in SOURCE_PATTERNS:
        if pat in u:
            return name
    return "other"


def select_records(records: Sequence[Dict], task: str, n_pos: int, n_neg: int, seed: int = 0) -> List[Dict]:
    """records: dicts with cls ('COVID' | 'Lung_Opacity' | 'Normal' | ...), source, id. Returns records with a 'role' key.
    n_pos / n_neg = 0 keeps everything. The subsample is seeded and independent of the input order."""
    if task not in TASKS:
        raise ValueError(f"task must be one of {sorted(TASKS)}")
    pos = sorted((r for r in records if r["cls"] == TASKS[task]), key=lambda r: r["id"])
    neg = sorted((r for r in records if r["cls"] == "Normal" and r["source"] == "rsna"), key=lambda r: r["id"])
    rng = np.random.default_rng(seed)

    def take(lst, n):
        if n and n < len(lst):
            return [lst[i] for i in sorted(rng.choice(len(lst), n, replace=False))]
        return lst

    return [{**r, "role": P.POSITIVE} for r in take(pos, n_pos)] + [{**r, "role": P.NEGATIVE} for r in take(neg, n_neg)]


def assign_splits(recs: Sequence[Dict], test_frac: float = 0.20, val_frac: float = 0.10, seed: int = 0) -> Dict[str, str]:
    """id -> 'test' | 'target_val' | 'train_target', stratified by role. target_val is taken from the non-test part."""
    ids, roles = [r["id"] for r in recs], [r["role"] for r in recs]
    test = set(P.stratified_val_split(ids, roles, test_frac, seed))
    rest = [(i, ro) for i, ro in zip(ids, roles) if i not in test]
    val = set(P.stratified_val_split([i for i, _ in rest], [ro for _, ro in rest], val_frac, seed + 1))
    return {i: ("test" if i in test else "target_val" if i in val else "train_target") for i in ids}


def lung_bbox(mask: np.ndarray, margin: float = 0.05) -> Tuple[int, int, int, int]:
    """(y0, y1, x0, x1) of mask > 127 grown by `margin` of the box size; the whole image if the mask is empty."""
    m = np.asarray(mask) > 127
    h, w = m.shape
    if not m.any():
        return 0, h, 0, w
    ys, xs = np.where(m.any(1))[0], np.where(m.any(0))[0]
    y0, y1, x0, x1 = ys[0], ys[-1] + 1, xs[0], xs[-1] + 1
    dy, dx = int(round(margin * (y1 - y0))), int(round(margin * (x1 - x0)))
    return max(0, y0 - dy), min(h, y1 + dy), max(0, x0 - dx), min(w, x1 + dx)


def source_label_table(recs: Iterable[Dict]) -> Dict[Tuple[str, str], int]:
    """Counts per (role, source): shows how far the label is determined by the source."""
    t: Dict[Tuple[str, str], int] = defaultdict(int)
    for r in recs:
        t[(r["role"], r["source"])] += 1
    return dict(t)


FRAMES = ("full", "crop", "maskout", "center", "maskout_fixed")      # E17 (D47): how the lung box is used; the folder suffix is _<frame> (none for full)


def box_fractions(boxes_and_shapes, q: float = 0.5) -> Tuple[float, float]:
    """Quantile q (default median) of (height, width) fraction of the image covered by the lung box, from [((y0, y1, x0, x1), (h, w)), ...].
    Used to place a FIXED central box of the same typical size (the 'zoom without the mask' control)."""
    fh = [(b[1] - b[0]) / shape[0] for b, shape in boxes_and_shapes]
    fw = [(b[3] - b[2]) / shape[1] for b, shape in boxes_and_shapes]
    return float(np.quantile(fh, q)), float(np.quantile(fw, q))


def center_box(h: int, w: int, fh: float, fw: float) -> Tuple[int, int, int, int]:
    """(y0, y1, x0, x1) of the centred box covering fraction fh x fw of an h x w image (at least 1 pixel each way)."""
    bh, bw = max(1, int(round(fh * h))), max(1, int(round(fw * w)))
    y0, x0 = (h - bh) // 2, (w - bw) // 2
    return y0, y0 + bh, x0, x0 + bw


def apply_frame(img_u8: np.ndarray, box: Tuple[int, int, int, int], frame: str) -> np.ndarray:
    """img_u8 is the full image ALREADY windowed to uint8 from the pixels inside `box`.
    'crop' / 'center': the box region only (zoomed after padding/resizing); 'maskout' / 'maskout_fixed': full frame, zero outside the box
    (same scale as 'full', same intensities as 'crop'; maskout_fixed uses a FIXED centred box, so the box shape carries no per-image information); 'full': unchanged."""
    y0, y1, x0, x1 = box
    if frame in ("crop", "center"):
        return img_u8[y0:y1, x0:x1]
    if frame in ("maskout", "maskout_fixed"):
        out = np.zeros_like(img_u8)
        out[y0:y1, x0:x1] = img_u8[y0:y1, x0:x1]
        return out
    return img_u8


def box_geometry_auroc(boxes_and_shapes, labels) -> Dict[str, float]:
    """AUROC of the lung-box geometry (height, width, area as image fractions, centre y / x) for the positive label: how much the
    FRAMING ALONE says about the class. Crop changes the zoom per image by the box size, so a box-size AUROC far from 0.5 means a
    cropped frame can carry class information that is not in the lung content (E17 caveat). 0.5 = uninformative."""
    from .metrics import auroc
    y = np.asarray(labels, int)
    rows = {"height": [], "width": [], "area": [], "center_y": [], "center_x": []}
    for (y0, y1, x0, x1), (h, w) in boxes_and_shapes:
        rows["height"].append((y1 - y0) / h); rows["width"].append((x1 - x0) / w)
        rows["area"].append((y1 - y0) * (x1 - x0) / (h * w))
        rows["center_y"].append((y0 + y1) / 2 / h); rows["center_x"].append((x0 + x1) / 2 / w)
    return {k: float(auroc(y, np.asarray(v))) for k, v in rows.items()} if len(set(y.tolist())) == 2 else {}
