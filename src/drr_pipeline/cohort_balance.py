"""Cân bằng metadata CT-RATE giữa các nhóm CT (dương, âm nhánh R, bình thường nhánh N): đo nguy cơ "lối tắt" tuổi/giới/máy.

Đặc trưng chỉ từ metadata kỹ thuật (không dùng nhãn hay báo cáo): tuổi, giới nữ, đường kính tái tạo, XY/Z spacing, số lát, hãng máy (one-hot).
`shortcut_auroc` = AUROC ngoài-fold của hồi quy logistic phân biệt hai nhóm CHỈ bằng các đặc trưng này: gần 0,5 là cân bằng; cao nghĩa là
mô hình có thể học nhóm từ tuổi/máy thay vì từ tổn thương. Cột Manufacturer có nhiều biến thể ('Siemens Healthineers' và 'SIEMENS'): giữ nguyên, không gộp.
"""
from __future__ import annotations

import ast
from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import b0

NUMERIC = ["age_years", "female", "recon_diameter", "xy_spacing", "z_spacing", "n_slices"]


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def age_years(v) -> float:
    s = str(v or "").strip().upper()
    return _f(s[:-1]) if s.endswith("Y") else (_f(s) if s else np.nan)


def numeric_features(row: Dict[str, str]) -> List[float]:
    try:
        xy = float(ast.literal_eval(row.get("XYSpacing", "[]"))[0])
    except (ValueError, SyntaxError, IndexError, TypeError):
        xy = np.nan
    sex = str(row.get("PatientSex", "")).strip().upper()
    return [age_years(row.get("PatientAge")), 1.0 if sex == "F" else (0.0 if sex == "M" else np.nan),
            _f(row.get("ReconstructionDiameter")), xy, _f(row.get("ZSpacing")), _f(row.get("NumberofSlices"))]


def design(groups: Dict[str, List[Dict[str, str]]], top_manufacturers: int = 6):
    """(tên cột, {nhóm: ma trận}) với one-hot hãng máy lấy từ các hãng phổ biến nhất trên tất cả nhóm."""
    from collections import Counter
    cnt = Counter(r.get("Manufacturer", "") for rows in groups.values() for r in rows)
    mans = [m for m, _ in cnt.most_common(top_manufacturers)]
    names = NUMERIC + [f"mfr={m}" for m in mans]
    out = {}
    for g, rows in groups.items():
        out[g] = np.array([numeric_features(r) + [1.0 if r.get("Manufacturer", "") == m else 0.0 for m in mans] for r in rows], dtype=float)
    return names, out


def smd(a: np.ndarray, b: np.ndarray) -> float:
    """Chênh lệch trung bình chuẩn hoá (|SMD| > 0,1 thường coi là lệch); NaN bị bỏ."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / sd) if sd > 0 else 0.0


def shortcut_auroc(Xa: np.ndarray, Xb: np.ndarray, max_per_group: int = 2000, seed: int = 0, folds: int = 5) -> float:
    """AUROC phân biệt nhóm a (nhãn 1) với nhóm b chỉ bằng đặc trưng metadata; chọn ngẫu nhiên tối đa max_per_group mỗi nhóm."""
    rng = np.random.default_rng(seed)
    pick = lambda X: X[rng.permutation(len(X))[:max_per_group]]
    Xa, Xb = pick(Xa), pick(Xb)
    X = np.vstack([Xa, Xb])
    y = np.r_[np.ones(len(Xa)), np.zeros(len(Xb))]
    Z, = b0.impute(X)
    sd = Z.std(axis=0)
    Z = (Z - Z.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    return float(b0.logistic_cv_auroc(Z, y, folds, seed=seed))


def match_negatives(pos_meta: Dict[str, Dict[str, str]], cand_meta: Dict[str, Dict[str, str]], seed: int = 0,
                    age_caliper: float | None = None, use_sex: bool = False) -> Tuple[Dict[str, str], List[str]]:
    """Ghép 1:1 không hoàn lại: mỗi ca dương nhận ca âm gần nhất theo (tuổi, XY spacing, Z spacing, hãng máy khác nhau cộng 1), mỗi khoảng
    cách chuẩn hoá bằng độ lệch chuẩn của nhóm ứng viên. Tham lam theo thứ tự ngẫu nhiên có hạt giống. Chỉ dùng metadata kỹ thuật (không nhãn,
    không ảnh). Trả ({ca dương: ca âm}, [ca dương không ghép được trong age_caliper năm])."""
    pos_ids, cand_ids = sorted(pos_meta), sorted(cand_meta)

    def vec(rows):
        f = np.array([[numeric_features(r)[0], numeric_features(r)[3], numeric_features(r)[4]] for r in rows], dtype=float)
        return f, [r.get("Manufacturer", "") for r in rows]

    Fp, Mp = vec([pos_meta[c] for c in pos_ids])
    Fc, Mc = vec([cand_meta[c] for c in cand_ids])
    sex = lambda rows: np.array([1.0 if str(r.get("PatientSex", "")).strip().upper() == "F" else (0.0 if str(r.get("PatientSex", "")).strip().upper() == "M" else -1.0) for r in rows])
    Sp, Sc = sex([pos_meta[c] for c in pos_ids]), sex([cand_meta[c] for c in cand_ids])
    med = np.nanmedian(Fc, axis=0)
    Fp, Fc = np.where(np.isfinite(Fp), Fp, med), np.where(np.isfinite(Fc), Fc, med)
    sd = Fc.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    free = np.ones(len(cand_ids), dtype=bool)
    out, unmatched = {}, []
    for i in np.random.default_rng(seed).permutation(len(pos_ids)):
        d = (np.abs(Fc - Fp[i]) / sd).sum(axis=1) + np.array([m != Mp[i] for m in Mc], dtype=float)
        if use_sex:
            d = d + (Sc != Sp[i]).astype(float)
        d = np.where(free, d, np.inf)
        if age_caliper is not None:
            d = np.where(np.abs(Fc[:, 0] - Fp[i, 0]) <= age_caliper, d, np.inf)
        j = int(np.argmin(d))
        if not np.isfinite(d[j]):
            unmatched.append(pos_ids[i])
            continue
        free[j] = False
        out[pos_ids[i]] = cand_ids[j]
    return out, sorted(unmatched)
