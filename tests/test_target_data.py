import csv

import numpy as np
from PIL import Image

from drr_pipeline import target_data as TD


def _make(tmp_path):
    rows = []
    for i, (sp, role, age) in enumerate([("train_target", "positive", "5"), ("train_target", "unlabeled_excluded", ""), ("target_val", "negative", "30"),
                                         ("test", "positive", "x"), ("test", "airway_hard_negative", "12"), ("test", "negative", "")]):
        (tmp_path / sp).mkdir(exist_ok=True)
        Image.fromarray(np.full((8, 8), 10 * i, np.uint8)).save(tmp_path / sp / f"im{i}.png")
        rows.append({"image_id": f"im{i}", "split": sp, "role": role, "age_months": age})
    with open(tmp_path / "manifest.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return rows


def test_manifest_rows_and_png_loading(tmp_path):
    rows = _make(tmp_path)
    man = TD.read_manifest(tmp_path)
    assert man == rows
    test = TD.rows_of(man, "test")
    assert [r["role"] for r in test] == ["positive", "airway_hard_negative", "negative"]
    assert [r["image_id"] for r in TD.rows_of(man, "test", TD.MAIN)] == ["im3", "im5"]
    x, y, ids, roles = TD.load_png_split(tmp_path, test, TD.TEST_ROLES)
    assert x.shape == (3, 8, 8) and y.tolist() == [1, 0, 0] and ids == ["im3", "im4", "im5"]
    assert x[2].max() == 50
    x2, y2, _, r2 = TD.load_png_split(tmp_path, TD.rows_of(man, "train_target"), TD.ALL_ROLES)
    assert len(x2) == 2 and r2 == ["positive", "unlabeled_excluded"] and y2.tolist() == [1, 0]


def test_age_months_handles_missing_and_bad_values(tmp_path):
    man = TD.read_manifest(tmp_path) if _make(tmp_path) else None
    a = TD.age_months(man)
    assert np.allclose(a, [5, np.nan, 30, np.nan, 12, np.nan], equal_nan=True)
    assert TD.age_months([{"image_id": "z"}]).tolist()[0] != TD.age_months([{"image_id": "z"}]).tolist()[0]   # NaN


def test_roles_and_registry():
    assert TD.MAIN == ("positive", "negative") and TD.TEST_ROLES[-1] == "airway_hard_negative"
    assert set(TD.MAIN) < set(TD.ALL_ROLES) and "vindr_pcxr" in TD.TARGET_DIRS
