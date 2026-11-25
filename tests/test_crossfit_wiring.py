import numpy as np
import pytest

from drr_pipeline import cxr_prep as P
from drr_pipeline import sign_labels as SL
from drr_pipeline import target_crossfit as TC
from drr_pipeline import train_data as T


def _rows(n=200):
    return [{"image_id": f"i{k}", "split": ["train_target", "target_val", "test"][k % 3], "role": "x"} for k in range(n)]


def test_split_for_fold_disjoint_and_roles():
    rows = _rows()
    by = {f"i{k}": ({"Consolidation"} if k % 8 == 0 else ({"Lung Opacity"} if k % 8 == 1 else {"No finding"})) for k in range(200)}
    fold_of = {r["image_id"]: k % 5 for k, r in enumerate(rows)}
    cf = TC.split_for_fold(rows, fold_of, by, ["Consolidation"], fold=2, val_frac=0.1, seed=0)
    ev, pool = {r["image_id"] for r in cf["eval"]}, {r["image_id"] for r in cf["pool"]}
    assert ev and not (ev & pool)                                         # chấm fold 2 không có trong pool/val/train
    assert not ({r["image_id"] for r in cf["val"]} & ev) and not ({r["image_id"] for r in cf["train_labeled"]} & ev)
    assert {r["role"] for r in cf["eval"]} <= set(P.MAIN_ROLES)
    assert any(r["role"] == P.OTHER for r in cf["pool"])                  # ảnh hộp lớp khác chỉ nằm trong pool không nhãn
    assert {r["image_id"] for r in cf["val"]} | {r["image_id"] for r in cf["train_labeled"]} <= pool
    assert all(r["fold"] == 2 for r in cf["eval"]) and all(r["fold"] != 2 for r in cf["pool"])
    assert {r["role"] for r in cf["val"]} == set(P.MAIN_ROLES)            # val phân tầng: có cả dương và âm
    again = TC.split_for_fold(rows, fold_of, by, ["Consolidation"], fold=2, val_frac=0.1, seed=0)
    assert [r["image_id"] for r in again["val"]] == [r["image_id"] for r in cf["val"]]


def test_source_drr_case_roles(tmp_path):
    from drr_pipeline import dataset as D
    M = np.zeros((3, 8, 8)); M[1] = 1.0; M[2, :4, :4] = 1.0
    R = M.copy()
    z = {"M_normal": np.stack([M] * 2), "R_normal": np.stack([R] * 2), "M_positive": np.stack([M] * 3), "R_positive": np.stack([R] * 3)}
    for c in ("pos1", "neg1"):
        p = D.cache_path(tmp_path, "moderate", "ap", c)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, **z)
    kw = dict(weights=[[1.0, 1.0, 1.0]], thresholds=[0.0], normal_w=[1.0, 1.0, 1.0], T1=0.0, min_px=1, min_pct=0.0, eval_cell=(0, 0))
    roles = {"pos1": "positive", "neg1": "negative"}
    s = T.SourceDRR(tmp_path, "moderate", ["ap"], ["pos1", "neg1"], case_roles=roles, **kw)
    kinds = [(it[0], it[2]) for it in s.items]
    assert ("pos1", "normal") not in kinds and ("neg1", "positive") not in kinds        # CT dương chỉ tư thế dương, CT âm chỉ tư thế thường
    assert sum(k == ("pos1", "positive") for k in kinds) == 3 and sum(k == ("neg1", "normal") for k in kinds) == 2
    assert s.labels().tolist().count(1) == 3
    with pytest.raises(AssertionError):
        T.SourceDRR(tmp_path, "moderate", ["ap"], ["pos1", "neg1"], case_roles={"pos1": "positive"}, **kw)
    old = T.SourceDRR(tmp_path, "moderate", ["ap"], ["pos1"], **kw)                     # không có roles: hành vi cũ
    assert len(old) == 5


def test_make_source_folds(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("msf", "scripts/make_source_folds.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    plan = {"jobs": [{"volume_name": f"train_{i}_a_1.nii.gz", "patient_id": f"train_{i}", "channels": [{"sign": "consolidation"}]} for i in range(1, 9)]
                    + [{"volume_name": "train_50_a_1.nii.gz", "patient_id": "train_50", "channels": [{"sign": "pleural_effusion"}]}]}
    negs = [{"volume_name": f"train_{i}_a_1.nii.gz", "patient_id": f"train_{i}"} for i in (3, 50, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109)]
    pos, neg, info = m.select_cases(plan, negs, "consolidation", 1.0, 0)
    assert len(pos) == 8 and len(neg) == 8                                               # ca 3 (cùng bệnh nhân với dương) và 50 (dương khác) bị loại
    assert "train_3_a_1" not in neg and "train_50_a_1" not in neg and info["n_neg_candidates"] == 10
    assert m.select_cases(plan, negs, "consolidation", 1.0, 0)[1] == neg                  # xác định theo seed


def test_select_cases_chooser():
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location("msf", pathlib.Path(__file__).resolve().parents[1] / "scripts/make_source_folds.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    plan = {"jobs": [{"volume_name": "train_1_a_1.nii.gz", "patient_id": "train_1", "channels": [{"sign": "x"}]}]}
    negs = [{"volume_name": f"train_{i}_a_1.nii.gz", "patient_id": f"train_{i}"} for i in (2, 3, 4)]
    pos, neg, info = m.select_cases(plan, negs, "x", 1.0, 0, True, lambda p, c: ([c[-1]], {"ghép cặp": 1}))
    assert (pos, neg, info["ghép cặp"]) == (["train_1_a_1"], ["train_4_a_1"], 1)


def test_split_all_scores_every_labelled_image_and_uses_no_pool():
    rows = _rows()
    by = {f"i{k}": ({"Consolidation"} if k % 8 == 0 else ({"Lung Opacity"} if k % 8 == 1 else {"No finding"})) for k in range(200)}
    cf = TC.split_all(rows, by, ["Consolidation"], val_frac=0.1, seed=0)
    assert cf["pool"] == [] and cf["train_labeled"] == []
    assert {r["role"] for r in cf["eval"]} == set(P.MAIN_ROLES)
    assert len(cf["eval"]) == sum(1 for k in range(200) if k % 8 != 1)                         # ảnh "other" bị loại
    assert {r["image_id"] for r in cf["val"]} <= {r["image_id"] for r in cf["eval"]} and cf["val"]
