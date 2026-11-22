"""B0: mức tham chiếu "yếu tố không phải tổn thương" (numpy + PIL; không cần torch).

Hồi quy logistic L2 trên thống kê ảnh và siêu dữ liệu (tuổi, MONOCHROME1, kích thước gốc, độ sáng, tương phản, độ sáng vùng giữa,
tỉ lệ pixel 0). Một AUROC của mô hình chỉ có nghĩa khi vượt B0 (docs/PROTOCOL.md). Hai cách đọc:
  strict  : khớp trên tập huấn luyện đích, đánh giá trên test (không dùng nhãn test để khớp);
  cv_test : kiểm chứng chéo 5 fold trên chính test (như Giai đoạn 0; lạc quan hơn, để so sánh với số cũ 0,684 / 0,797).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .metrics import auroc

FACTOR_NAMES = ("age_months", "monochrome1", "log_orig_pixels", "img_mean", "img_std", "center_mean", "zero_share")


def image_factors(png_path) -> Dict[str, float]:
    from PIL import Image
    a = np.asarray(Image.open(png_path).convert("L"), dtype=float)
    h, w = a.shape
    c = a[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    return {"img_mean": float(a.mean()), "img_std": float(a.std()), "center_mean": float(c.mean()), "zero_share": float((a == 0).mean())}


def _num(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x


def factor_matrix(rows: Sequence[dict], png_root) -> np.ndarray:
    """(n, 7) theo FACTOR_NAMES; thiếu = NaN. rows là dòng manifest (image_id, split, age_months, photometric, rows, cols)."""
    from pathlib import Path
    out = np.full((len(rows), len(FACTOR_NAMES)), np.nan)
    for i, m in enumerate(rows):
        out[i, 0] = _num(m.get("age_months"))
        out[i, 1] = 1.0 if str(m.get("photometric", "")).upper().endswith("1") else 0.0
        r, c = _num(m.get("rows")), _num(m.get("cols"))
        out[i, 2] = float(np.log(r * c)) if r > 0 and c > 0 else np.nan
        f = image_factors(Path(png_root) / m["split"] / f"{m['image_id']}.png")
        out[i, 3:] = [f["img_mean"], f["img_std"], f["center_mean"], f["zero_share"]]
    return out


def impute(Xtr: np.ndarray, *others: np.ndarray):
    """Trung vị của TẬP HUẤN LUYỆN cho NaN, cộng cột chỉ báo thiếu cho mỗi cột có NaN ở tập huấn luyện."""
    med = np.nanmedian(np.where(np.isfinite(Xtr), Xtr, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    miss_cols = [j for j in range(Xtr.shape[1]) if np.isnan(Xtr[:, j]).any()]

    def f(X):
        Z = np.where(np.isfinite(X), X, med)
        return np.c_[Z, np.isnan(X[:, miss_cols]).astype(float)] if miss_cols else Z
    return (f(Xtr), *[f(X) for X in others])


def _fit(X, y, l2=1.0, iters=300):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[(X - mu) / sd, np.ones(len(X))]
    w = np.zeros(Z.shape[1])
    pen = np.r_[np.ones(Z.shape[1] - 1), 0]
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
        g = Z.T @ (p - y) + l2 * pen * w
        H = (Z.T * (p * (1 - p))) @ Z + l2 * np.diag(pen) + 1e-6 * np.eye(Z.shape[1])
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-6:
            break
    return w, mu, sd


def _score(X, wmusd):
    w, mu, sd = wmusd
    return np.c_[(X - mu) / sd, np.ones(len(X))] @ w


def logistic_cv_auroc(X, y, folds=5, l2=1.0, iters=300, seed=0):
    """AUROC ngoài-fold của hồi quy logistic L2 (Newton toàn batch); X không có NaN."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    idx = np.random.default_rng(seed).permutation(len(y))
    oof = np.zeros(len(y))
    for k in range(folds):
        te = idx[k::folds]
        tr = np.setdiff1d(idx, te)
        oof[te] = _score(X[te], _fit(X[tr], y[tr], l2, iters))
    return auroc(y, oof)


def b0_strict(Xtr, ytr, Xte, l2=1.0) -> np.ndarray:
    """Điểm B0 trên test, khớp CHỈ trên huấn luyện (NaN được điền bằng trung vị huấn luyện + cột chỉ báo thiếu)."""
    Ztr, Zte = impute(np.asarray(Xtr, float), np.asarray(Xte, float))
    return _score(Zte, _fit(Ztr, np.asarray(ytr, float), l2))


def b0_cv_on_test(Xte, yte, folds=5, seed=0, l2=1.0) -> float:
    Z, = impute(np.asarray(Xte, float))
    return logistic_cv_auroc(Z, yte, folds, l2, seed=seed)


def single_factor_auroc(X, y, names: Sequence[str] = FACTOR_NAMES) -> Dict[str, float]:
    """AUROC thô của từng yếu tố (bỏ NaN); < 0,5 nghĩa là yếu tố cao hơn ở âm tính."""
    y = np.asarray(y)
    out = {}
    for j, n in enumerate(names):
        ok = np.isfinite(X[:, j])
        out[n] = auroc(y[ok], X[ok, j]) if ok.sum() > 1 and len(set(y[ok])) == 2 else float("nan")
    return out


def b0_crossfit(X, y, fold_ids, l2=1.0) -> np.ndarray:
    """Điểm B0 ngoài-fold trên tập gộp: ở mỗi fold, khớp trên các fold còn lại (trung vị điền thiếu cũng từ đó), chấm fold này.
    Dùng cùng ``fold_ids`` với cross-fit của mô hình (docs/DECISIONS.md D59)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    f = np.asarray(fold_ids)
    oof = np.zeros(len(y))
    for k in np.unique(f):
        te, tr = f == k, f != k
        oof[te] = b0_strict(X[tr], y[tr], X[te], l2)
    return oof
