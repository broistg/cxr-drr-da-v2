"""Cross-fit trên ảnh X-quang đích (D59): mỗi ảnh được đánh giá đúng một lần, bởi mô hình/bộ thích nghi KHÔNG thấy ảnh đó.

Tập gộp = mọi ảnh vai trò positive/negative của train_target + target_val + test. Chia K fold; ở fold f:
thích nghi không nhãn (DA) trên các ảnh ngoài fold f, đánh giá trên fold f. Chia phân tầng theo (tổ hợp lớp hộp, nhóm tuổi) để
mọi dấu hiệu có ca dương ở mọi fold; nhóm theo ``patient_id`` nếu manifest có cột này (manifest hiện tại KHÔNG có: chia theo ảnh).
numpy/stdlib, xác định theo seed.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

AGE_EDGES = (6.0, 12.0, 24.0, 60.0)


def age_bin(v) -> str:
    try:
        a = float(v)
    except (TypeError, ValueError):
        return "na"
    if a != a:
        return "na"
    return str(sum(a >= e for e in AGE_EDGES))


def assign_folds(ids: Sequence[str], strata: Sequence[str], k: int = 5, seed: int = 0,
                 groups: Optional[Sequence[str]] = None) -> Dict[str, int]:
    """image_id -> fold. Trong mỗi tầng: xáo theo seed rồi chia vòng tròn; điểm bắt đầu quay vòng qua các tầng để cân bằng cỡ fold."""
    if len(ids) != len(strata) or (groups is not None and len(groups) != len(ids)):
        raise ValueError("độ dài ids/strata/groups không khớp")
    units = defaultdict(list)                      # (tầng) -> danh sách đơn vị (nhóm hoặc ảnh)
    by_group = defaultdict(list)
    for n, (i, s) in enumerate(zip(ids, strata)):
        by_group[groups[n] if groups is not None else i].append((i, s))
    for g, items in by_group.items():
        units[items[0][1]].append(items)           # nhóm lấy tầng của ảnh đầu tiên
    rng = random.Random(seed)
    out: Dict[str, int] = {}
    offset = 0
    for s in sorted(units):
        us = sorted(units[s], key=lambda it: it[0][0])
        rng.shuffle(us)
        for j, items in enumerate(us):
            f = (offset + j) % k
            for i, _ in items:
                out[i] = f
        offset = (offset + len(us)) % k
    return out


def fold_table(folds: Dict[str, int], labels: Dict[str, Sequence[int]], names: Sequence[str], k: int) -> List[dict]:
    """Số ảnh và số dương mỗi dấu hiệu ở mỗi fold (để in và kiểm)."""
    rows = []
    for f in range(k):
        ids = [i for i, g in folds.items() if g == f]
        r = {"fold": f, "n": len(ids)}
        for j, nm in enumerate(names):
            r[nm] = int(sum(labels[i][j] for i in ids))
        rows.append(r)
    return rows
