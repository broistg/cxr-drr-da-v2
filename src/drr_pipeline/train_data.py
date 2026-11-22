"""Source-domain (DRR) training data for Exp-1, built from the projection cache (numpy only).

A pose of a CT is the sampling unit. Each positive pose offers a set of KEPT (weight set, T2) cells
(docs/experiment_design.md section 3, rule from configs/drr.yaml); at every draw one kept cell is chosen at
random, so the many near-duplicate (w, T2) variants of one pose act like a data augmentation instead of
outnumbering the 20 normal poses. With 20 normal + 20 positive poses per CT and view the classes are balanced.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from . import dataset as D
from .weighted import weighted_from_projections


def load_splits(path: Path) -> Dict[int, List[str]]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf8"))
    return {int(k): list(v) for k, v in cfg["folds"].items()}


def load_roles(path: Path) -> Optional[Dict[str, str]]:
    """Vai trò CT theo ca ({ca: positive|negative}) nếu tệp chia fold có khoá ``roles`` (scripts/make_source_folds.py), nếu không None."""
    r = yaml.safe_load(Path(path).read_text(encoding="utf8")).get("roles")
    if r is not None:
        bad = {v for v in r.values()} - {"positive", "negative"}
        assert not bad, f"vai trò không hợp lệ trong roles: {bad}"
    return r


def cases_for(folds: Dict[int, List[str]], keep_folds: Sequence[int]) -> List[str]:
    return [c for f in keep_folds for c in folds[f]]


def min_pct_from(cfg: dict, override: Optional[float] = None) -> float:
    """positive_min_inf_pct_of_foot of configs/drr.yaml unless overridden."""
    return cfg.get("positive_min_inf_pct_of_foot", 0.1) if override is None else override


def source_kwargs(cfg: dict, min_pct: float) -> dict:
    """Keyword arguments of SourceDRR from configs/drr.yaml. SourceDRR's default eval cell is (weight index 6, T2 index 5):
    fail loudly if the config no longer means w = (1, 1, 3), T2 = 0.2."""
    wts, thr = cfg["weights"]["positive"], cfg["thresholds_T2"]
    assert list(wts[6]) == [1, 1, 3] and abs(thr[5] - 0.2) < 1e-9, "configs/drr.yaml order changed: update eval_cell"
    return dict(weights=wts, thresholds=thr, normal_w=cfg["weights"]["normal"], T1=cfg["threshold_T1"], min_pct=min_pct)


def sign_source_kwargs(sign_cfg: dict) -> dict:
    """Keyword arguments của SourceDRR theo configs/signs/<dấu hiệu>.yaml (D57): một bộ trọng số, một ngưỡng T2, luật nhãn theo số pixel.
    Ảnh thường (plain) hay có trọng số (weighted) đều đi qua đây; eval_cell = (0, 0) vì chỉ có một ô. Không dùng luật phần trăm vết chiếu phổi."""
    w, lr = [float(x) for x in sign_cfg["drr"]["weights"]], sign_cfg["label_rule"]
    return dict(weights=[w], thresholds=[float(lr.get("T2", 0.0))], normal_w=w, T1=0.0,
                min_px=max(int(lr.get("min_mask_px", 1)), 0), min_pct=0.0, eval_cell=(0, 0))


def split_cases(splits_path, val_fold: int, max_train_cases: int = 0, seed: int = 0) -> Tuple[List[str], List[str]]:
    """(validation CT cases, training CT cases); max_train_cases > 0 draws a seeded subset of the training cases (E9)."""
    folds = load_splits(splits_path)
    val_cases = cases_for(folds, [val_fold])
    train_cases = cases_for(folds, [f for f in folds if f != val_fold])
    if max_train_cases:
        assert 0 < max_train_cases <= len(train_cases), f"only {len(train_cases)} training cases"
        train_cases = sorted(np.random.default_rng(1000 + seed).choice(train_cases, max_train_cases, replace=False).tolist())
    return val_cases, train_cases


def kept_cells(M: np.ndarray, R: np.ndarray, weights, thresholds, T1: float = 0.0,
               min_px: int = 1, min_pct: float = 0.1) -> np.ndarray:
    """(n_w, n_T) bool for ONE positive pose: infection label has >= min_px pixels and >= min_pct % of the
    lung footprint (rays crossing lung or infection)."""
    foot = int(((M[1] + M[2]) > 1e-3).sum())
    out = np.zeros((len(weights), len(thresholds)), bool)
    if foot == 0:
        return out
    for wi, w in enumerate(weights):
        _, _, pi2, _ = weighted_from_projections(M, R, tuple(w), T1, 0.0)
        for ti, T2 in enumerate(thresholds):
            n = int((pi2 > T2).sum())
            out[wi, ti] = n >= min_px and 100.0 * n / foot >= min_pct
    return out


class SourceDRR:
    """Items are (case, view, kind, pose). `get(k, rng)` returns (uint8 image, label 0/1).

    Training (eval_mode False): a positive pose is kept if ANY cell passes the rule and each draw (`rng` given)
    picks one kept cell uniformly, so every kept (w, T2) set is equally likely, as the 63 equal training sets of
    DRR4Covid. The image depends on w only (T2 only changes the label), so cells with more passing T2 values
    give their w more weight.
    Validation (eval_mode True): a positive pose is kept only if `eval_cell` = (weight_index, T2_index) passes the
    rule, and `get(k)` without rng uses that weight set, so the evaluation set is fixed and deterministic.
    Positive poses that fail are dropped (counted in `.dropped`).
    """

    def __init__(self, cache_root: Path, profile: str, views: Sequence[str], cases: Sequence[str], weights,
                 thresholds, normal_w, T1: float = 0.0, min_px: int = 1, min_pct: float = 0.1,
                 eval_cell: Tuple[int, int] = (6, 5), eval_mode: bool = False,
                 case_roles: Optional[Dict[str, str]] = None):
        self.weights, self.thresholds, self.normal_w, self.T1 = weights, list(thresholds), normal_w, T1
        self.eval_cell, self.eval_mode = eval_cell, eval_mode
        self.arr: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
        self.items: List[Tuple[str, str, str, int]] = []
        self.kept: Dict[Tuple[str, str, int], np.ndarray] = {}
        self.dropped = 0
        for case in cases:
            role = case_roles.get(case) if case_roles else None
            assert case_roles is None or role is not None, f"ca {case} không có trong roles"
            for view in views:
                z = D.load_cache(D.cache_path(cache_root, profile, view, case))
                self.arr[(case, view)] = z
                # case_roles (D60): CT dương chỉ cho tư thế dương (không dùng ảnh "thường" tạo bằng trọng số), CT âm chỉ cho tư thế
                # "thường" render bằng đúng trọng số của dấu hiệu (plain: cùng trọng số với dương).
                for i in range(len(z["M_normal"]) if role != "positive" else 0):
                    self.items.append((case, view, "normal", i))
                for i in range(len(z["M_positive"]) if role != "negative" else 0):
                    k = kept_cells(z["M_positive"][i], z["R_positive"][i], weights, self.thresholds, T1, min_px, min_pct)
                    if (k[eval_cell] if eval_mode else k.any()):
                        self.kept[(case, view, i)] = k
                        self.items.append((case, view, "positive", i))
                    else:
                        self.dropped += 1

    def __len__(self):
        return len(self.items)

    def labels(self) -> np.ndarray:
        return np.array([1 if it[2] == "positive" else 0 for it in self.items])

    def get(self, k: int, rng: Optional[np.random.Generator] = None, with_mask: bool = False):
        """(uint8 image, label) or, with `with_mask`, (image, label, uint8 lesion mask pi2 > T2 of the SAME (w, T2) cell; zeros for
        normal poses). The random draw (one call of rng.integers per positive pose) is the same with and without the mask."""
        case, view, kind, i = self.items[k]
        z = self.arr[(case, view)]
        if kind == "normal":
            p, _, _, _ = weighted_from_projections(z["M_normal"][i], z["R_normal"][i], tuple(self.normal_w), self.T1, 0.2)
            img = D.to_uint8(p)
            return (img, 0, np.zeros(img.shape, np.uint8)) if with_mask else (img, 0)
        if rng is None:
            wi, ti = self.eval_cell
        else:
            cells = np.argwhere(self.kept[(case, view, i)])
            c = cells[rng.integers(len(cells))]
            wi, ti = int(c[0]), int(c[1])
        p, _, pi2, _ = weighted_from_projections(z["M_positive"][i], z["R_positive"][i], tuple(self.weights[wi]), self.T1, 0.2)
        img = D.to_uint8(p)
        return (img, 1, (pi2 > self.thresholds[ti]).astype(np.uint8)) if with_mask else (img, 1)
