"""ReXGroundingCT: đọc ``dataset.json`` và đếm CT/tổn thương theo nhóm (không cần mask/CT).

GIẢ ĐỊNH về định dạng (chưa kiểm trên tệp thật trong repo này; sửa ``iter_entries`` nếu khác):
  - top-level là dict ``{split: {volume_name: entry}}`` hoặc ``{split: [entry,...]}`` (entry có ``name``);
  - entry có ``categories`` (danh sách mã như "2g" hoặc dict mã -> ...), ``pixels`` (số voxel, tuỳ chọn),
    ``findings`` (câu mô tả, tuỳ chọn). Mã nhóm trích bằng regex ``[12][a-h]``.
Số mong đợi (docs/DATA_ACCESS.md, đếm tay từ tệp thật): 2g 18 tổn thương/15 CT, 2e 241/186, 2b 1384/924 (train+val).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from .ctrate import parse_volume_name

_CAT = re.compile(r"[12][a-h]")


def _codes(cats) -> list[str]:
    if isinstance(cats, dict):
        cats = list(cats)
    if isinstance(cats, str):
        cats = [cats]
    return [m.group(0) for c in (cats or []) for m in [_CAT.search(str(c))] if m]


def iter_entries(data: dict, splits=("train", "val")):
    """Sinh (split, volume_name, entry) cho các split được chọn."""
    for sp, body in data.items():
        if splits is not None and sp not in splits:
            continue
        if isinstance(body, dict):
            for name, e in body.items():
                yield sp, name, e
        else:
            for e in body:
                yield sp, e["name"], e


def load(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf8"))


def count_by_category(data: dict, splits=("train", "val")) -> dict[str, dict]:
    """{mã: {'lesions': n, 'ct': n, 'patients': n}}; mỗi mục trong ``categories`` là một tổn thương."""
    les, cts, pts = defaultdict(int), defaultdict(set), defaultdict(set)
    for _, name, e in iter_entries(data, splits):
        for c in _codes(e.get("categories")):
            les[c] += 1
            cts[c].add(name)
            try:
                pts[c].add(parse_volume_name(name).patient)
            except ValueError:
                pts[c].add(name)
    return {c: {"lesions": les[c], "ct": len(cts[c]), "patients": len(pts[c])} for c in sorted(les)}


def patient_overlap(data: dict, a="train", b="val") -> set[str]:
    """Bệnh nhân (số, bỏ tiền tố) có mặt ở cả hai split: cảnh báo rò rỉ nếu chia theo tiền tố."""
    def ids(sp):
        return {parse_volume_name(n).patient.split("_", 1)[1] for _, n, _ in iter_entries(data, (sp,))}
    return ids(a) & ids(b)
