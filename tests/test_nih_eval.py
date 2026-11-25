import numpy as np

from drr_pipeline import nih_eval as N


def _rows(n_pat=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_pat):
        pos = p % 3 == 0
        for i in range(3):
            rows.append({"image": f"{p}_{i}", "patient_id": str(p), "age": 12.0, "view": "AP" if rng.random() < (0.8 if pos else 0.4) else "PA",
                         "pleural_effusion": "positive" if pos else "negative", "fold": p % 5})
    return rows


def test_weighted_auroc_matches_unweighted_and_ties():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200); s = rng.normal(size=200) + y
    from drr_pipeline.metrics import auroc
    assert abs(N.weighted_auroc(y, s) - auroc(y, s)) < 1e-9
    assert N.weighted_auroc([1, 0], [0.5, 0.5]) == 0.5
    assert N.weighted_auroc([1, 1], [1, 2]) != N.weighted_auroc([1, 1], [1, 2])      # NaN khi thiếu lớp âm
    w = np.ones(200); w[y == 0] = 3.0                                                # nhân đều trọng số ca âm: không đổi
    assert abs(N.weighted_auroc(y, s, w) - N.weighted_auroc(y, s)) < 1e-9


def test_view_weights_equalise_view_mix():
    views = np.array(["AP"] * 8 + ["PA"] * 2 + ["AP"] * 4 + ["PA"] * 6)
    y = np.array([1] * 10 + [0] * 10)
    w = N.view_weights(views, y)
    ap_neg = w[(views == "AP") & (y == 0)].sum() / w[y == 0].sum()
    assert abs(ap_neg - 0.8) < 1e-9 and (w[y == 1] == 1).all()


def test_evaluate_shortcut_scores_flag_view_only_model():
    rows = _rows()
    sel = N.select(rows, "pleural_effusion", max_age=18)
    assert len(sel) == len(rows) and N.select(rows, "pleural_effusion", max_age=10) == []
    s_view = np.array([1.0 if r["view"] == "AP" else 0.0 for r in sel])             # mô hình chỉ biết tư thế
    res = N.evaluate(sel, s_view, "pleural_effusion", n_boot=100)
    assert res["point"]["auroc"] > 0.6                                             # tư thế một mình có tín hiệu khi chưa cân tư thế
    assert abs(res["point"]["auroc_weighted"] - 0.5) < 0.05                        # trọng số tư thế khử nó
    assert abs(res["point"]["auroc_PA"] - 0.5) < 1e-9 and abs(res["point"]["auroc_AP"] - 0.5) < 1e-9
    assert res["counts"]["all"]["pos_patients"] == 20 and "AP" in N.to_markdown(res, "t")
    assert any("KHÔNG KẾT LUẬN" in f for f in res["flags"])                         # tầng PA < 30 bệnh nhân dương
