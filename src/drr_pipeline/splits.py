"""Chia fold nguồn CT theo BỆNH NHÂN (không theo ảnh/scan): numpy/stdlib, xác định theo seed.

Các volume cùng bệnh nhân (``train_1741_a_1``, ``train_1741_b_2``...) luôn ở cùng một fold. Mã bệnh nhân lấy từ
``ctrate.parse_volume_name`` (giữ tiền tố train/valid). Ghi YAML cùng định dạng ``train_data.load_splits`` đọc.
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from .ctrate import parse_volume_name


def patient_of(name: str) -> str:
    try:
        return parse_volume_name(name).patient
    except ValueError:
        return name


def make_folds(cases: Sequence[str], k: int = 4, seed: int = 0, group: Callable[[str], str] = patient_of,
               weight: Optional[Dict[str, float]] = None) -> Dict[int, List[str]]:
    """Gộp theo nhóm; xếp nhóm nặng nhất trước vào fold đang nhẹ nhất (cân bằng số ca, hoặc tổng ``weight``)."""
    groups = defaultdict(list)
    for c in sorted(set(cases)):
        groups[group(c)].append(c)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    load = lambda g: sum((weight or {}).get(c, 1.0) for c in groups[g])
    keys.sort(key=load, reverse=True)
    folds: Dict[int, List[str]] = {i: [] for i in range(k)}
    tot = [0.0] * k
    for g in keys:
        i = min(range(k), key=lambda j: (tot[j], len(folds[j]), j))
        folds[i] += groups[g]
        tot[i] += load(g)
    return {i: sorted(v) for i, v in folds.items()}


def write_yaml(path, folds: Dict[int, List[str]], note: str = "") -> None:
    import yaml
    Path(path).write_text(yaml.safe_dump({"seed_note": note, "folds": folds}, allow_unicode=True, sort_keys=False), encoding="utf8")
