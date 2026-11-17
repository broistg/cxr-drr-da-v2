import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_train_seg_test_metrics():
    ts = _load("train_seg")
    gt = np.zeros((6, 16, 16), bool); gt[:, 4:12, 4:12] = True
    lung = np.zeros((6, 16, 16), np.uint8); lung[:, 2:14, 2:14] = 1
    perfect = np.where(gt, 0.9, 0.1)
    r = ts.test_metrics(perfect, gt, lung, 0.5, 100, 0)
    assert r["dice"] == 1.0 and r["iou"] == 1.0 and r["dice_in_lung"] == 1.0 and abs(r["predicted_area_over_true_area"] - 1) < 1e-9
    half = np.where(gt & (np.arange(16)[None, None, :] < 8), 0.9, 0.1)
    assert 0.5 < ts.test_metrics(half, gt, lung, 0.5, 100, 0)["dice"] < 0.8
    assert set(r["dice_by_extent_quartile"]) == {"q1", "q2", "q3", "q4"}


def test_summarize_variant_name():
    sm = _load("summarize_seg")
    assert sm.variant("seg_drr_moderate_f0_s2") == "seg_drr_moderate_f0"
    assert sm.variant("seg_real_k50_s1_init") == "seg_real_k50_init"
    assert sm.variant("seg_drr_moderate_f0_s0_pafda5_nullmask") == "seg_drr_moderate_f0_pafda5_nullmask"


def test_summarize_runs(tmp_path, capsys=None):
    sm = _load("summarize_seg")
    import sys
    for seed, d in ((0, 0.40), (1, 0.42)):
        for run, dd in ((f"seg_drr_moderate_f0_s{seed}", d), (f"seg_real_k50_s{seed}", d + 0.1)):
            (tmp_path / run).mkdir()
            t = {"dice": dd, "iou": dd / 2, "dice_in_lung": dd, "predicted_area_over_true_area": 1.0}
            (tmp_path / run / "metrics.json").write_text(json.dumps({"run": run, "args": {"seed": seed}, "test": t, "test_last_epoch": t}))
    argv = sys.argv
    sys.argv = ["x", "--dir", str(tmp_path)]
    try:
        sm.main()
    finally:
        sys.argv = argv
