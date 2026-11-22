import csv
import numpy as np
from PIL import Image
from drr_pipeline import b0 as B, sign_labels as SL


def test_sign_roles(tmp_path):
    p = tmp_path / "a.csv"
    rows = [("a", "Consolidation"), ("b", "No finding"), ("c", "Lung Opacity"), ("d", "No finding"), ("d", "Consolidation")]
    with open(p, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image_id", "class_name"]); w.writerows(rows)
    by = SL.classes_by_image(p)
    assert SL.sign_roles(list("abcde"), by, ["consolidation"]) == [SL.POSITIVE, SL.NEGATIVE, SL.OTHER, SL.POSITIVE, SL.UNLABELED]


def test_factor_matrix_and_b0(tmp_path):
    rng = np.random.default_rng(0)
    (tmp_path / "test").mkdir()
    rows, y = [], []
    for i in range(60):
        pos = i % 2
        a = np.clip(rng.normal(100 + 40 * pos, 10, (32, 32)), 0, 255).astype(np.uint8)
        Image.fromarray(a).save(tmp_path / "test" / f"i{i}.png")
        rows.append({"image_id": f"i{i}", "split": "test", "age_months": "" if i % 5 == 0 else 12 + pos, "photometric": "MONOCHROME2", "rows": 100, "cols": 100})
        y.append(pos)
    X = B.factor_matrix(rows, tmp_path)
    assert X.shape == (60, 7) and np.isnan(X[0, 0]) and np.isfinite(X[1, 3])
    y = np.array(y)
    assert B.b0_cv_on_test(X, y) > 0.9          # độ sáng tách được nhãn
    s = B.b0_strict(X[:40], y[:40], X[40:])
    from drr_pipeline.metrics import auroc
    assert auroc(y[40:], s) > 0.9
    sf = B.single_factor_auroc(X, y)
    assert sf["img_mean"] > 0.9


def test_crossfit_folds_and_b0(tmp_path):
    from drr_pipeline import crossfit as CF
    ids = [f"i{n}" for n in range(200)]
    strata = ["1|0" if n % 10 == 0 else "0|1" for n in range(200)]
    f = CF.assign_folds(ids, strata, k=5, seed=0)
    assert sorted(set(f.values())) == [0, 1, 2, 3, 4]
    assert f == CF.assign_folds(ids, strata, k=5, seed=0)                    # xác định theo seed
    for k in range(5):                                                       # tầng hiếm có mặt đều ở mọi fold
        assert sum(1 for i in ids[::10] if f[i] == k) == 4
    g = CF.assign_folds(ids, strata, k=5, seed=0, groups=[i[:-1] for i in ids])   # nhóm cùng fold
    assert all(len({g[i] for i in ids if i[:-1] == p}) == 1 for p in {i[:-1] for i in ids})
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    X = rng.normal(size=(200, 7)); X[:, 3] += 1.5 * y
    s = B.b0_crossfit(X, y, [f[i] for i in ids])
    from drr_pipeline.metrics import auroc
    assert auroc(y, s) > 0.85
