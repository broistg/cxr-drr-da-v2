"""Nhãn theo dấu hiệu cho X-quang đích từ hộp tổn thương (VinDr-PCXR annotations_*.csv). stdlib/numpy.

Vai trò mỗi ảnh với dấu hiệu s (lớp hộp ``classes``):
  positive            : có ít nhất một hộp thuộc ``classes``
  negative            : chỉ có dòng "No finding" (đã kiểm: 0 hộp, 0 nhãn bệnh khác; docs/data_cards/vindr_pcxr.md)
  other_finding       : có hộp lớp khác, không có dấu hiệu s (không dùng làm âm tính chính)
  unlabeled_excluded  : ảnh không có dòng nào trong annotations
Một bác sĩ đọc mỗi ảnh (VinDr-PCXR): nhãn không có đồng thuận.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

NO_FINDING = "No finding"
POSITIVE, NEGATIVE, OTHER, UNLABELED = "positive", "negative", "other_finding", "unlabeled_excluded"


def classes_by_image(path, id_col: str = "image_id", cls_col: str = "class_name") -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    with open(Path(path), encoding="utf8", newline="") as f:
        for r in csv.DictReader(f):
            out.setdefault(r[id_col], set()).add(r[cls_col].strip())
    return out


def sign_roles(image_ids: Iterable[str], by_image: Dict[str, Set[str]], classes: Sequence[str]) -> List[str]:
    want = {c.strip().lower() for c in classes}
    roles = []
    for i in image_ids:
        cs = by_image.get(i)
        if not cs:
            roles.append(UNLABELED)
        elif any(c.lower() in want for c in cs):
            roles.append(POSITIVE)
        elif cs == {NO_FINDING}:
            roles.append(NEGATIVE)
        else:
            roles.append(OTHER)
    return roles
