"""Mục 4: kiểm tra trùng lặp gần giữa tập huấn luyện và tập test của đích (VinDr-PCXR không có mã bệnh nhân, nên dùng ảnh làm đại diện).

Mỗi ảnh PNG (256x256) được thu nhỏ về s x s, trừ trung bình và chuẩn hóa độ dài; độ tương đồng = tương quan Pearson (cosin của vector đã chuẩn hóa).
Với mỗi ảnh test, lấy tương đồng cao nhất trong tập train_target (+ target_val). Đây là ĐẠI DIỆN: cùng bệnh nhân chụp lại / ảnh lặp cho tương quan rất cao,
nhưng tương quan cao cũng có thể do cùng tư thế (ảnh X-quang ngực giống nhau tự nhiên) nên cần một mốc so sánh: tương đồng cao nhất của ảnh test trong
CHÍNH tập test (loại trừ bản thân) và phân phối train-train. Không phải bằng chứng cùng bệnh nhân; chỉ cho thấy rò rỉ có thể có hay không.
"""
from __future__ import annotations

import numpy as np


def small_vectors(images: np.ndarray, size: int = 32) -> np.ndarray:
    """(N, H, W) uint8/float -> (N, size*size) zero-mean unit-norm rows (block-mean downsampling; H, W cropped to a multiple of size)."""
    x = np.asarray(images, dtype=np.float64)
    n, h, w = x.shape
    bh, bw = h // size, w // size
    if bh < 1 or bw < 1:
        raise ValueError("image smaller than the target size")
    x = x[:, :bh * size, :bw * size].reshape(n, size, bh, size, bw).mean((2, 4)).reshape(n, -1)
    x = x - x.mean(1, keepdims=True)
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(nrm == 0, 1.0, nrm)


def max_similarity(query: np.ndarray, ref: np.ndarray, exclude_self: bool = False, chunk: int = 512):
    """For each row of `query` (unit rows): (max similarity, index in ref). exclude_self: query is ref, ignore the diagonal."""
    best, arg = np.full(len(query), -np.inf), np.zeros(len(query), int)
    for s in range(0, len(query), chunk):
        sim = query[s:s + chunk] @ ref.T
        if exclude_self:
            sim[np.arange(sim.shape[0]), np.arange(s, s + sim.shape[0])] = -np.inf
        j = sim.argmax(1)
        best[s:s + chunk], arg[s:s + chunk] = sim[np.arange(sim.shape[0]), j], j
    return best, arg


def summarize(sims: np.ndarray, thresholds=(0.90, 0.95, 0.98, 0.99)) -> dict:
    sims = np.asarray(sims)
    out = {"n": int(len(sims)), "median": float(np.median(sims)), "p95": float(np.quantile(sims, 0.95)), "max": float(sims.max())}
    for t in thresholds:
        out[f">={t}"] = int((sims >= t).sum())
    return out
