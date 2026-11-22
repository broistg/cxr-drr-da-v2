"""Đếm CT-RATE "bình thường" theo từng định nghĩa (chỉ đếm, không chọn ca; D60 cần nhóm duyệt định nghĩa trước khi chọn).

Tên 18 nhãn là hiểu biết từ tài liệu CT-RATE, CHƯA kiểm với file thật: `match_columns` so khớp không phân biệt hoa/thường và ký tự
đặc biệt, và báo nhãn nào không tìm thấy. Nhóm "liên quan X-quang" là bản NHÁP chờ nhóm duyệt.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Sequence

ALL_LABELS = ["Medical material", "Arterial wall calcification", "Cardiomegaly", "Pericardial effusion",
              "Coronary artery wall calcification", "Hiatal hernia", "Lymphadenopathy", "Emphysema", "Atelectasis",
              "Lung nodule", "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion", "Mosaic attenuation pattern",
              "Peribronchial thickening", "Consolidation", "Bronchiectasis", "Interlobular septal thickening"]
NOT_CXR = ["Medical material", "Arterial wall calcification", "Coronary artery wall calcification", "Hiatal hernia",
           "Lymphadenopathy", "Pericardial effusion"]
CXR_RELEVANT = [x for x in ALL_LABELS if x not in NOT_CXR]


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def match_columns(header: Sequence[str], wanted: Sequence[str]):
    """({nhãn: tên cột}, [nhãn không thấy])."""
    by = {norm(h): h for h in header}
    found = {w: by[norm(w)] for w in wanted if norm(w) in by}
    return found, [w for w in wanted if w not in found]


def _is_zero(v: str) -> bool:
    try:
        return float(v) == 0.0
    except (TypeError, ValueError):
        return False


def normal_volumes(rows: List[Dict[str, str]], columns: Sequence[str]) -> List[str]:
    """Tên volume có MỌI cột trong `columns` bằng 0 (ô trống/không đọc được thì KHÔNG tính là 0)."""
    return [r["VolumeName"] for r in rows if columns and all(_is_zero(r.get(c)) for c in columns)]


def count_tiers(rows: List[Dict[str, str]], header: Sequence[str]) -> Dict[str, object]:
    strict_cols, miss_all = match_columns(header, ALL_LABELS)
    rel_cols, miss_rel = match_columns(header, CXR_RELEVANT)
    # thiếu cột nhãn -> định nghĩa không còn nghĩa (sẽ lỏng hơn thật): không đếm, trả rỗng và báo cột thiếu
    return {"n_volumes": len(rows),
            "strict": [] if miss_all else normal_volumes(rows, list(strict_cols.values())), "strict_missing_cols": miss_all,
            "relevant": [] if miss_rel else normal_volumes(rows, list(rel_cols.values())), "relevant_missing_cols": miss_rel}


def read_csvs(folder) -> tuple:
    """(các hàng, tên cột) gộp mọi CSV trong thư mục (train + validation)."""
    rows, header = [], None
    for f in sorted(Path(folder).glob("*.csv")):
        with open(f, encoding="utf8", newline="") as h:
            r = csv.DictReader(h)
            header = header or r.fieldnames
            rows += list(r)
    return rows, header or []
