"""Chia ảnh đích theo cross-fit (D59) cho một dấu hiệu và một fold (numpy/stdlib).

Với fold f: ``eval`` = ảnh của fold f (chấm), ``pool`` = mọi ảnh NGOÀI fold f (thích nghi không nhãn; gồm cả ảnh hộp lớp khác),
``train_labeled`` = ảnh có nhãn ngoài fold f trừ ``val`` (chỉ nhánh a2 dùng), ``val`` = một phần phân tầng của ảnh có nhãn ngoài fold f,
CHỈ dùng để ghi log và ngưỡng phụ (không chọn checkpoint: giao thức dùng source_val). Ảnh trong ``eval`` không bao giờ nằm trong
``pool``, ``val`` hay ``train_labeled``. ``role`` của mỗi dòng được đặt lại theo dấu hiệu: positive / negative / other_excluded.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from . import cxr_prep as P
from . import sign_labels as SL

_ROLE = {SL.POSITIVE: P.POSITIVE, SL.NEGATIVE: P.NEGATIVE, SL.OTHER: P.OTHER, SL.UNLABELED: P.UNLABELED}


def read_folds(path) -> Dict[str, int]:
    with open(Path(path), encoding="utf8", newline="") as f:
        return {r["image_id"]: int(r["fold"]) for r in csv.DictReader(f)}


def split_for_fold(rows: Sequence[dict], fold_of: Dict[str, int], by_image, classes: Sequence[str], fold: int,
                   val_frac: float = 0.1, seed: int = 0) -> Dict[str, List[dict]]:
    ids = [r["image_id"] for r in rows]
    roles = SL.sign_roles(ids, by_image, classes)
    tagged = [dict(r, role=_ROLE[ro], fold=fold_of[r["image_id"]]) for r, ro in zip(rows, roles) if r["image_id"] in fold_of]
    ev = [r for r in tagged if r["fold"] == fold and r["role"] in P.MAIN_ROLES]
    rest = [r for r in tagged if r["fold"] != fold]
    labeled = [r for r in rest if r["role"] in P.MAIN_ROLES]
    rng = np.random.default_rng(seed)
    val_ids = set()
    for role in P.MAIN_ROLES:                       # phân tầng theo nhãn
        pool = sorted(r["image_id"] for r in labeled if r["role"] == role)
        n = max(1, int(round(val_frac * len(pool)))) if pool else 0
        val_ids |= set(rng.choice(pool, n, replace=False).tolist()) if n else set()
    return {"eval": ev, "pool": rest, "val": [r for r in labeled if r["image_id"] in val_ids],
            "train_labeled": [r for r in labeled if r["image_id"] not in val_ids]}


def split_all(rows: Sequence[dict], by_image, classes: Sequence[str], val_frac: float = 0.1, seed: int = 0) -> Dict[str, List[dict]]:
    """Chấm TOÀN BỘ ảnh đích có vai trò dương/âm (``--target-fold all``), chỉ cho mô hình KHÔNG dùng nhãn hay ảnh đích (nhánh a0; M1, null).
    ``eval`` = mọi ảnh dương/âm; ``pool`` và ``train_labeled`` rỗng. ``val`` là một phần phân tầng NẰM TRONG ``eval``, chỉ để ghi log và
    ngưỡng phụ (độ nhạy tại độ đặc hiệu 90%): ngưỡng đó vì thế hơi lạc quan, AUROC/AP không bị ảnh hưởng. Checkpoint chọn bằng source_val."""
    ids = [r["image_id"] for r in rows]
    roles = SL.sign_roles(ids, by_image, classes)
    tagged = [dict(r, role=_ROLE[ro]) for r, ro in zip(rows, roles)]
    ev = [r for r in tagged if r["role"] in P.MAIN_ROLES]
    rng = np.random.default_rng(seed)
    val_ids = set()
    for role in P.MAIN_ROLES:
        pool = sorted(r["image_id"] for r in ev if r["role"] == role)
        n = max(1, int(round(val_frac * len(pool)))) if pool else 0
        val_ids |= set(rng.choice(pool, n, replace=False).tolist()) if n else set()
    return {"eval": ev, "pool": [], "val": [r for r in ev if r["image_id"] in val_ids], "train_labeled": []}
