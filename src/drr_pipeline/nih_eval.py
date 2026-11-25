"""Đánh giá trên đích NIH nhi (D62, P1-T1/T3) [numpy]: AUROC gộp, có trọng số theo tư thế, riêng PA/AP; CI bootstrap theo BỆNH NHÂN.

Manifest do ``scripts/prep_nih_peds.py`` ghi (cột image, patient_id, age, sex, view, labels, pneumothorax, pleural_effusion,
consolidation, fold). Nhóm theo dấu hiệu: positive / negative (chỉ "No Finding") / other (bỏ qua).
Trọng số tư thế: ca dương trọng số 1; ca âm trọng số ``share_pos(view) / share_neg(view)`` để tỉ lệ AP:PA của ca âm bằng ca dương
(tính lại trong từng lần bootstrap).
"""
from __future__ import annotations

import csv
from typing import Dict, List, Optional, Sequence

import numpy as np

MIN_POS_PATIENTS = 30          # quy tắc đặt trước (P1-T1/T3): tầng có ít hơn thì "không kết luận"


def read_manifest(path) -> List[dict]:
    with open(path, encoding="utf8", newline="") as f:
        return [dict(r, age=float(r["age"]), fold=int(r["fold"])) for r in csv.DictReader(f)]


def select(rows: Sequence[dict], sign: str, max_age: Optional[float] = None) -> List[dict]:
    """Dòng positive/negative của dấu hiệu (có tuổi < max_age nếu cho)."""
    return [r for r in rows if r[sign] in ("positive", "negative") and (max_age is None or r["age"] < max_age)]


def weighted_auroc(y, s, w=None) -> float:
    """AUROC có trọng số: sum_{i dương, j âm} w_i w_j [s_i > s_j (+ 0,5 nếu bằng)] / (W_dương W_âm). NaN nếu thiếu một lớp."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
    wp, wn = w[y == 1].sum(), w[y == 0].sum()
    if wp <= 0 or wn <= 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    s, y, w = s[order], y[order], w[order]
    # gom theo giá trị điểm (hoà): với mỗi nhóm, dương thắng mọi âm có điểm thấp hơn và hoà 0,5 với âm cùng điểm
    uniq, start = np.unique(s, return_index=True)
    neg_below = 0.0
    total = 0.0
    bounds = list(start) + [len(s)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        gp = w[a:b][y[a:b] == 1].sum()
        gn = w[a:b][y[a:b] == 0].sum()
        total += gp * (neg_below + 0.5 * gn)
        neg_below += gn
    return float(total / (wp * wn))


def view_weights(views: Sequence[str], y: Sequence[int]) -> np.ndarray:
    """Trọng số ca âm theo tư thế (dương = 1). Tư thế không có ca âm hoặc dương thì trọng số ca âm 0 hoặc 1 (không chia cho 0)."""
    views = np.asarray(views)
    y = np.asarray(y).astype(int)
    w = np.ones(len(y))
    npos, nneg = max(int((y == 1).sum()), 1), max(int((y == 0).sum()), 1)
    for v in np.unique(views):
        p = ((views == v) & (y == 1)).sum() / npos
        n = ((views == v) & (y == 0)).sum() / nneg
        w[(views == v) & (y == 0)] = (p / n) if n > 0 else 0.0
    return w


def _metrics(y, s, views) -> Dict[str, float]:
    y = np.asarray(y); s = np.asarray(s); views = np.asarray(views)
    out = {"auroc": weighted_auroc(y, s), "auroc_weighted": weighted_auroc(y, s, view_weights(views, y))}
    for v in ("PA", "AP"):
        m = views == v
        out[f"auroc_{v}"] = weighted_auroc(y[m], s[m]) if m.any() else float("nan")
    return out


def evaluate(rows: Sequence[dict], scores: Sequence[float], sign: str, n_boot: int = 1000, seed: int = 0) -> Dict[str, object]:
    """rows đã qua ``select``; ``scores`` cùng thứ tự. Trả về điểm, CI 95% (bootstrap theo bệnh nhân) và số ca theo tầng."""
    y = np.array([1 if r[sign] == "positive" else 0 for r in rows])
    s = np.asarray(scores, dtype=float)
    views = np.array([r["view"] for r in rows])
    pids = np.array([r["patient_id"] for r in rows])
    point = _metrics(y, s, views)
    rng = np.random.default_rng(seed)
    uniq = np.unique(pids)
    idx = {p: np.where(pids == p)[0] for p in uniq}
    boots = {k: [] for k in point}
    for _ in range(n_boot):
        pick = np.concatenate([idx[p] for p in rng.choice(uniq, len(uniq))])
        m = _metrics(y[pick], s[pick], views[pick])
        for k, v in m.items():
            if not np.isnan(v):
                boots[k].append(v)
    ci = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) > 20 else (float("nan"), float("nan"))
          for k, v in boots.items()}
    counts = {}
    for name, mask in (("all", np.ones(len(y), bool)), ("PA", views == "PA"), ("AP", views == "AP")):
        pos, neg = mask & (y == 1), mask & (y == 0)
        counts[name] = {"pos_images": int(pos.sum()), "neg_images": int(neg.sum()),
                        "pos_patients": len(set(pids[pos])), "neg_patients": len(set(pids[neg]))}
    flags = [f"tầng {v}: {counts[v]['pos_patients']} bệnh nhân dương < {MIN_POS_PATIENTS}: KHÔNG KẾT LUẬN"
             for v in ("PA", "AP") if counts[v]["pos_patients"] < MIN_POS_PATIENTS]
    return {"sign": sign, "point": point, "ci95": ci, "counts": counts, "flags": flags, "n_boot": n_boot}


def to_markdown(res: Dict[str, object], title: str) -> str:
    p, ci, c = res["point"], res["ci95"], res["counts"]
    L = [f"# {title}", "", f"dấu hiệu {res['sign']}; CI 95% bootstrap theo bệnh nhân ({res['n_boot']} lần)", "",
         "| tầng | AUROC [CI95] | dương (ảnh / bn) | âm (ảnh / bn) |", "|---|---|---|---|"]
    names = {"auroc": ("gộp (không trọng số)", "all"), "auroc_weighted": ("gộp, trọng số tư thế", "all"),
             "auroc_PA": ("chỉ PA", "PA"), "auroc_AP": ("chỉ AP", "AP")}
    for k, (label, st) in names.items():
        L.append(f"| {label} | {p[k]:.3f} [{ci[k][0]:.3f}, {ci[k][1]:.3f}] | {c[st]['pos_images']} / {c[st]['pos_patients']} | "
                 f"{c[st]['neg_images']} / {c[st]['neg_patients']} |")
    if res["flags"]:
        L += [""] + [f"- {f}" for f in res["flags"]]
    return "\n".join(L) + "\n"
