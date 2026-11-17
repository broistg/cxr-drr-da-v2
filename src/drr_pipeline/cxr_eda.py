"""Statistics for a real chest X-ray dataset given as CSV label files + DICOM (stdlib only here;
pydicom is imported only by scripts/inspect_cxr.py).

The exact column layout of VinDr-PCXR was NOT known when this was written, so nothing is hard-coded
beyond the guesses in `guess_*`; the script prints the columns it found and what it decided.
"""
from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        rows = list(r)
        return list(r.fieldnames or []), rows


def guess_id_column(cols: Sequence[str]) -> Optional[str]:
    for c in cols:
        if "image" in c.lower() and "id" in c.lower():
            return c
    for c in cols:
        if c.lower() in ("image", "image_id", "id", "study_id", "sop_instance_uid"):
            return c
    return cols[0] if cols else None


def _is_flag(values: Sequence[str]) -> bool:
    vals = {v.strip().lower() for v in values if v is not None and v.strip() != ""}
    return bool(vals) and vals <= {"0", "1", "0.0", "1.0", "true", "false"}


def _truthy(v: str) -> bool:
    return v.strip().lower() in ("1", "1.0", "true")


def summarize_labels(cols: Sequence[str], rows: Sequence[Dict[str, str]], id_col: Optional[str] = None) -> dict:
    """Image-level label file.

    Wide layout (one 0/1 column per label) -> positives per label column.
    Long layout (a text column naming the label) -> counts per value of the first column whose name
    contains 'label' or 'class' (and is not 0/1).
    Several rows per image (several readers) are reported via rows vs unique images.
    """
    id_col = id_col or guess_id_column(cols)
    ids = [r[id_col] for r in rows] if id_col else []
    out = {"rows": len(rows), "unique_images": len(set(ids)), "id_col": id_col}
    flag_cols = [c for c in cols if c != id_col and _is_flag([r.get(c, "") for r in rows])]
    if len(flag_cols) >= 2:
        out["layout"] = "wide"
        out["positives_rows"] = {c: sum(_truthy(r[c]) for r in rows if r.get(c, "") != "") for c in flag_cols}
        per_img = defaultdict(set)
        for r in rows:
            for c in flag_cols:
                if r.get(c, "") != "" and _truthy(r[c]):
                    per_img[r[id_col]].add(c)
        cnt = Counter()
        for labs in per_img.values():
            cnt.update(labs)
        out["positive_images"] = dict(cnt)
    else:
        name_cols = [c for c in cols if re.search(r"label|class|finding|diagnos", c, re.I)]
        out["layout"] = "long"
        out["label_col"] = name_cols[0] if name_cols else None
        if name_cols:
            c = name_cols[0]
            per_img = defaultdict(set)
            for r in rows:
                per_img[r[id_col]].add(r[c])
            cnt = Counter()
            for labs in per_img.values():
                cnt.update(labs)
            out["positive_images"] = dict(cnt)
            out["rows_per_label"] = dict(Counter(r[c] for r in rows))
    return out


def summarize_boxes(cols: Sequence[str], rows: Sequence[Dict[str, str]], no_finding: str = "No finding") -> dict:
    """Box file. VinDr-PCXR lists 'No finding' as a CLASS (one row per normal image, not a real box),
    so those rows are counted separately and excluded from the box statistics."""
    id_col = guess_id_column(cols)
    cls = next((c for c in cols if re.search(r"class|label|finding", c, re.I)), None)
    real = [r for r in rows if not cls or r[cls] != no_finding]
    out = {"rows": len(rows), "rows_no_finding": len(rows) - len(real), "real_boxes": len(real),
           "images_with_boxes": len({r[id_col] for r in real}) if id_col else None,
           "class_col": cls, "id_col": id_col}
    if cls:
        out["boxes_per_class"] = dict(Counter(r[cls] for r in real))
        imgs = defaultdict(set)
        for r in real:
            imgs[r[cls]].add(r[id_col])
        out["images_per_class"] = {k: len(v) for k, v in imgs.items()}
    return out


def label_groups(cols: Sequence[str], rows: Sequence[Dict[str, str]], groups: Dict[str, Sequence[str]],
                 no_finding: str = "No finding", id_col: Optional[str] = None) -> dict:
    """Image-level groups (union of label columns) for a wide label file.

    Reports, per group: images positive, how many of them are ALSO 'No finding' (contradiction), and
    how many are exclusive (no positive label outside the group). Also the distribution of the number
    of positive labels per image (excluding 'No finding') and images with no label at all.
    """
    id_col = id_col or guess_id_column(cols)
    labelcols = [c for c in cols if c not in (id_col, "rad_ID", "rad_id")]
    per = defaultdict(set)
    for r in rows:
        for c in labelcols:
            if r.get(c, "") != "" and _truthy(r[c]):
                per[r[id_col]].add(c)
    for r in rows:
        per.setdefault(r[id_col], set())
    n_lab = Counter(len(v - {no_finding}) for v in per.values())
    out = {"images": len(per), "no_label_at_all": sum(1 for v in per.values() if not v),
           "no_finding_and_other": sum(1 for v in per.values() if no_finding in v and len(v) > 1),
           "positive_labels_per_image": dict(sorted(n_lab.items())), "groups": {}}
    for name, gcols in groups.items():
        g = set(gcols)
        pos = [v for v in per.values() if v & g]
        out["groups"][name] = {"images": len(pos), "also_no_finding": sum(no_finding in v for v in pos),
                               "exclusive": sum(1 for v in pos if not (v - g - {no_finding}))}
    return out


def match_files(ids: Sequence[str], stems: Sequence[str]) -> dict:
    """Which label ids have no image file, and which files have no label."""
    a, b = set(ids), set(stems)
    return {"ids": len(a), "files": len(b), "ids_without_file": sorted(a - b)[:20],
            "n_ids_without_file": len(a - b), "files_without_id": sorted(b - a)[:20],
            "n_files_without_id": len(b - a)}


def parse_age_months(age: str) -> Optional[float]:
    """DICOM AS format: 'nnnD', 'nnnW', 'nnnM', 'nnnY' (e.g. '012M'). Returns months or None."""
    m = re.fullmatch(r"\s*(\d{1,3})\s*([DWMY])\s*", age or "")
    if not m:
        return None
    n, u = int(m.group(1)), m.group(2)
    return {"D": n / 30.4375, "W": n / 4.348, "M": float(n), "Y": n * 12.0}[u]


def summarize_numeric(vals: Sequence[float]) -> dict:
    v = sorted(x for x in vals if x is not None)
    if not v:
        return {"n": 0}
    pick = lambda q: v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))]
    return {"n": len(v), "min": v[0], "p25": pick(0.25), "median": pick(0.5), "p75": pick(0.75), "max": v[-1]}
