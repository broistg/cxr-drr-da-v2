"""Công cụ hộp tổn thương của VinDr (CSV annotations): đọc, nhóm lớp. stdlib."""
from __future__ import annotations

import csv
import re
from pathlib import Path

AIRSPACE = ("Consolidation", "Diffuse aveolar opacity", "Infiltration", "Other opacity")


def _col(cols, *patterns):
    for p in patterns:
        for c in cols:
            if re.fullmatch(p, c, re.I):
                return c
    return None


def load_boxes(path: Path, no_finding: str = "No finding"):
    """image_id -> list of (class_name, (xmin, ymin, xmax, ymax)) in original pixels; 'No finding' rows are not boxes."""
    with open(path, encoding="utf8") as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames or []
        idc = _col(cols, r"image_id", r"image.*id")
        clc = _col(cols, r"class_name", r"class.*", r"label.*")
        xs = [_col(cols, p) for p in (r"x_?min", r"y_?min", r"x_?max", r"y_?max")]
        if not idc or not clc or None in xs:
            raise SystemExit(f"cannot find id/class/box columns in {path}; columns are {cols}")
        out = {}
        for r in rd:
            if r[clc] == no_finding or not all(r[c] not in ("", None) for c in xs):
                continue
            out.setdefault(r[idc], []).append((r[clc], tuple(float(r[c]) for c in xs)))
    return out
