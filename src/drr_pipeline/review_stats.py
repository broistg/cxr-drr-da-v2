"""Thống kê cho đợt gán nhãn tay báo cáo: khoảng tin cậy Wilson của tỉ lệ nhiễm và độ đồng thuận Cohen kappa giữa hai người."""
from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Khoảng tin cậy Wilson 95% của tỉ lệ k/n; (nan, nan) nếu n = 0."""
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Cohen kappa cho hai danh sách nhãn cùng độ dài; nan nếu không có biến thiên."""
    n = len(a)
    if n == 0 or n != len(b):
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    labels = set(a) | set(b)
    pe = sum((sum(x == l for x in a) / n) * (sum(y == l for y in b) / n) for l in labels)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def count_labels(labels: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for l in labels:
        out[l] = out.get(l, 0) + 1
    return out
