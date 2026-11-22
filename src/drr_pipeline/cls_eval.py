"""Đánh giá và ghi kết quả của phân loại nhị phân (numpy; không cần torch).

``evaluate_scores`` nhận điểm trên target_val và test; ngưỡng đặc hiệu 90% chọn CHỈ trên target_val. Vai trò test:
positive/negative là chính; ``airway_hard_negative`` chỉ để đo dương tính giả.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from . import metrics as Mx
from .target_data import MAIN


def run_name(a) -> str:
    """Tên run xác định bởi các cờ (a: argparse.Namespace). Chỉ thêm hậu tố khi cờ khác mặc định."""
    s = f"{a.arm}_{a.profile}_{'+'.join(a.views)}_f{a.val_fold}_s{a.seed}" + (f"_{a.sign}" if getattr(a, "sign", None) else "") \
        + (f"_cf{a.target_fold}" if getattr(a, "crossfit", None) else "")
    if a.arm == "a1":
        s += f"_g{a.gamma:g}_w{a.warmup_epochs}"
        if a.seed_labels:
            s += f"_k{a.seed_labels}" + ("_sup" if a.seed_supervised else "") + (f"_sf{round(100 * a.seed_frac)}" if a.seed_frac else "")
    if a.max_train_cases:
        s += f"_n{a.max_train_cases}"
    if a.pixel_adapt != "none":
        s += f"_pa{a.pixel_adapt}" + (f"{round(100 * a.fda_beta)}" if a.pixel_adapt == "fda" else "")
    if a.drr_blur > 0:
        s += f"_bl{a.drr_blur:g}"
    if a.drr_noise > 0:
        s += f"_nz{a.drr_noise:g}"
    return s + ("_adabn" if a.adabn else "") + ("_shuffled" if a.shuffle_labels else "") + ("_selsrc" if a.select == "source_val" else "")


def evaluate_scores(yv, tv, yt, st, roles: Sequence[str], n_boot: int = 2000, seed: int = 0) -> dict:
    roles = np.asarray(roles)
    thr = Mx.threshold_at_specificity(yv, tv, 0.90)
    mm = np.isin(roles, MAIN)
    y, s = yt[mm], st[mm]
    lo, hi = Mx.bootstrap_ci(Mx.auroc, y, s, n_boot, seed)
    alo, ahi = Mx.bootstrap_ci(Mx.average_precision, y, s, n_boot, seed)
    al, pos = st[roles == "airway_hard_negative"], st[roles == "positive"]
    return {"n_positive": int(y.sum()), "n_negative": int((1 - y).sum()),
            "auroc": Mx.auroc(y, s), "auroc_ci95": [lo, hi],
            "average_precision": Mx.average_precision(y, s), "ap_ci95": [alo, ahi], "ap_chance": float(y.mean()),
            "target_val_auroc": Mx.auroc(yv, tv),
            "threshold_spec90_from_target_val": thr,
            "sensitivity_at_that_threshold": Mx.sensitivity_at_threshold(y, s, thr),
            "n_airway_hard_negative": int(len(al)),
            "airway_false_positive_rate_at_that_threshold": float((al > thr).mean()) if len(al) else None,
            "auroc_positive_vs_airway": Mx.auroc(np.r_[np.ones(len(pos)), np.zeros(len(al))], np.r_[pos, al]) if len(al) else None}


def write_outputs(out: Path, result: dict, ids, roles, y, scores) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=1), encoding="utf8")
    with open(out / "predictions_test.csv", "w", newline="", encoding="utf8") as f:
        wr = csv.writer(f)
        wr.writerow(["image_id", "role", "label", "score"])
        for i, r, yy, ss in zip(ids, roles, y, scores):
            wr.writerow([i, r, int(yy), f"{ss:.6f}"])
