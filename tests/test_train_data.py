import numpy as np

from drr_pipeline import dataset as D
from drr_pipeline import train_data as T
from drr_pipeline.weighted import category_projections

WEIGHTS = [[1, 1, 1], [1, 1, 3], [12, 12, 1]]
THR = [0.0, 0.2]


def _pose(rng, infection=True):
    m = np.where(rng.random((6, 16, 16)) < 0.6, 1, 0).astype(np.uint8)
    if infection:
        m[rng.random(m.shape) < 0.15] = 2
    rho = rng.uniform(0.5, 1.5, m.shape)
    return category_projections(rho, m)


def _cache(tmp, case, n=3, seed=0):
    rng = np.random.default_rng(seed)
    Mn, Rn, Mp, Rp = [], [], [], []
    for _ in range(n):
        M, R = _pose(rng, False); Mn.append(M); Rn.append(R)
        M, R = _pose(rng, True); Mp.append(M); Rp.append(R)
    D.save_cache(D.cache_path(tmp, "moderate", "ap", case),
                 dict(M_normal=np.stack(Mn).astype("f4"), R_normal=np.stack(Rn).astype("f4"),
                      M_positive=np.stack(Mp).astype("f4"), R_positive=np.stack(Rp).astype("f4")))


def test_splits_and_cases(tmp_path):
    y = tmp_path / "s.yaml"
    y.write_text("folds:\n  0: [a, b]\n  1: [c]\n  2: [d]\n")
    f = T.load_splits(y)
    assert T.cases_for(f, [1, 2]) == ["c", "d"]


def test_kept_cells_rule():
    M, R = _pose(np.random.default_rng(0), True)
    k = T.kept_cells(M, R, WEIGHTS, THR, min_px=1, min_pct=0.0)
    assert k.shape == (3, 2) and k[:, 0].all()                 # T2 = 0: every pose with infection passes
    assert not T.kept_cells(M, R, WEIGHTS, THR, min_px=10 ** 6).any()
    # a higher w2 can only add pixels, so kept-ness is monotone in w2 at fixed T2
    assert k[1, 1] >= k[0, 1]
    assert not T.kept_cells(np.zeros_like(M), np.zeros_like(R), WEIGHTS, THR).any()


def test_source_dataset_balance_and_labels(tmp_path):
    _cache(tmp_path, "c1"); _cache(tmp_path, "c2", seed=1)
    ds = T.SourceDRR(tmp_path, "moderate", ["ap"], ["c1", "c2"], WEIGHTS, THR, [24, 24, 1], min_pct=0.0)
    y = ds.labels()
    assert len(ds) == 12 and y.sum() == 6                      # 3 normal + 3 positive poses per case
    rng = np.random.default_rng(0)
    for k in range(len(ds)):
        img, lab = ds.get(k, rng)
        assert img.shape == (16, 16) and img.dtype == np.uint8 and lab == y[k]


def test_training_draw_varies_eval_is_fixed(tmp_path):
    _cache(tmp_path, "c1")
    ds = T.SourceDRR(tmp_path, "moderate", ["ap"], ["c1"], WEIGHTS, THR, [24, 24, 1], min_pct=0.0)
    k = int(np.where(ds.labels() == 1)[0][0])
    rng = np.random.default_rng(0)
    imgs = {ds.get(k, rng)[0].tobytes() for _ in range(30)}
    assert len(imgs) > 1                                        # different weight sets give different images
    ev = T.SourceDRR(tmp_path, "moderate", ["ap"], ["c1"], WEIGHTS, THR, [24, 24, 1], min_pct=0.0,
                     eval_cell=(1, 1), eval_mode=True)
    a = ev.get(int(np.where(ev.labels() == 1)[0][0]))[0]
    b = ev.get(int(np.where(ev.labels() == 1)[0][0]))[0]
    assert np.array_equal(a, b)


def test_strict_rule_drops_positives(tmp_path):
    _cache(tmp_path, "c1")
    ds = T.SourceDRR(tmp_path, "moderate", ["ap"], ["c1"], WEIGHTS, THR, [24, 24, 1], min_px=10 ** 6)
    assert ds.dropped == 3 and ds.labels().sum() == 0


def test_mask_matches_cell_and_rng_stream_unchanged(tmp_path):
    _cache(tmp_path, "c1", n=4)
    ds = T.SourceDRR(tmp_path, "moderate", ["ap"], ["c1"], WEIGHTS, THR, [24, 24, 1], min_pct=0.0)
    r1, r2 = np.random.default_rng(5), np.random.default_rng(5)
    for k in range(len(ds)):
        img_a, lab_a = ds.get(k, r1)
        img_b, lab_b, m = ds.get(k, r2, with_mask=True)
        assert np.array_equal(img_a, img_b) and lab_a == lab_b          # same draws, same image
        assert m.shape == img_b.shape and m.dtype == np.uint8
        if lab_b == 0:
            assert m.sum() == 0
        else:
            assert m.sum() > 0                                          # min_pct 0 and T2 <= 0.2 keep a lesion pixel
    # deterministic eval mask at the validation cell
    ev = T.SourceDRR(tmp_path, "moderate", ["ap"], ["c1"], WEIGHTS, THR, [24, 24, 1], min_pct=0.0, eval_cell=(1, 1), eval_mode=True)
    k = int(np.where(ev.labels() == 1)[0][0])
    assert np.array_equal(ev.get(k, None, with_mask=True)[2], ev.get(k, None, with_mask=True)[2])


def test_source_kwargs_and_split_cases(tmp_path):
    import numpy as np
    import pytest
    import yaml
    from drr_pipeline import train_data as T
    cfg = yaml.safe_load(open("configs/drr.yaml", encoding="utf8"))
    mk = T.source_kwargs(cfg, 0.1)
    assert set(mk) == {"weights", "thresholds", "normal_w", "T1", "min_pct"} and mk["min_pct"] == 0.1
    assert T.min_pct_from(cfg) == cfg.get("positive_min_inf_pct_of_foot", 0.1) and T.min_pct_from(cfg, 0.5) == 0.5
    bad = dict(cfg, thresholds_T2=list(reversed(cfg["thresholds_T2"])))
    with pytest.raises(AssertionError):
        T.source_kwargs(bad, 0.1)
    from drr_pipeline import splits as SP
    cases = [f"train_{i}_a_1" for i in range(10)]
    sp = tmp_path / "splits.yaml"
    SP.write_yaml(sp, SP.make_folds(cases, 4, 0))
    val, tr = T.split_cases(sp, 0)
    assert not set(val) & set(tr) and len(val) + len(tr) == 10
    _, sub = T.split_cases(sp, 0, max_train_cases=5, seed=2)
    _, sub2 = T.split_cases(sp, 0, max_train_cases=5, seed=2)
    assert sub == sub2 and len(sub) == 5 and set(sub) <= set(tr) and sub == sorted(sub)
    with pytest.raises(AssertionError):
        T.split_cases(sp, 0, max_train_cases=99)


def test_sign_kwargs_plain_source_and_four_categories(tmp_path):
    from drr_pipeline import signs as SG
    cfg = SG.load_sign("consolidation")
    mk = T.sign_source_kwargs(cfg)
    assert mk["weights"] == [[1.0, 1.0, 1.0]] and mk["eval_cell"] == (0, 0) and mk["min_pct"] == 0.0
    _cache(tmp_path, "c1")
    ds = T.SourceDRR(tmp_path, "moderate", ["ap"], ["c1"], **mk)
    assert ds.labels().sum() >= 1 and len(ds) == 3 + int(ds.labels().sum())
    ev = T.SourceDRR(tmp_path, "moderate", ["ap"], ["c1"], eval_mode=True, **mk)
    img, lab, mask = ev.get(int(np.argmax(ev.labels())), with_mask=True)
    assert lab == 1 and mask.sum() >= mk["min_px"]
    # 4 lớp: tim (lớp 3) ở phép chiếu; trọng số 4 phần tử cho DRR thường
    from drr_pipeline.weighted import weighted_from_projections
    rng = np.random.default_rng(3)
    m = rng.integers(0, 4, (5, 8, 8)); rho = rng.uniform(0.5, 1.5, m.shape)
    M, R = category_projections(rho, m, n_cat=4)
    assert M.shape == (4, 8, 8)
    p, _, pi2, _ = weighted_from_projections(M, R, (1, 1, 1, 1), 0.0, 0.0)
    assert np.allclose(p, R.sum(0)) and np.allclose(pi2, M[2] / M.sum(0))


def test_subset_train_cases_by_positives_with_pairs():
    roles = {f"p{i}": "positive" for i in range(20)} | {f"n{i}": "negative" for i in range(25)}
    train = list(roles)
    pairs = {f"p{i}": f"n{i}" for i in range(20)} | {"p3": "n_val_only"}      # n_val_only không nằm trong train
    a = T.subset_train_cases(train, roles, 5, seed=0, pairs=pairs)
    assert sum(roles[c] == "positive" for c in a) == 5 and sum(roles[c] == "negative" for c in a) == 5
    pos = [c for c in a if roles[c] == "positive"]
    for c in pos:
        if c != "p3":
            assert pairs[c] in a                                                     # ca âm ghép cặp được giữ
    assert "n_val_only" not in a
    b = T.subset_train_cases(train, roles, 10, seed=0, pairs=pairs)
    assert set(c for c in a if roles[c] == "positive") <= set(c for c in b if roles[c] == "positive")   # lồng nhau
    assert T.subset_train_cases(train, roles, 5, seed=0, pairs=pairs) == a            # tất định
    assert T.subset_train_cases(train, roles, 5, seed=1, pairs=pairs) != a
