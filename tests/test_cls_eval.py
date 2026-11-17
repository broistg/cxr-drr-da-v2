import argparse

import numpy as np

from drr_pipeline import cls_eval as CE


def _ns(**kw):
    d = dict(arm="a0", profile="moderate", views=["ap"], val_fold=0, seed=1, gamma=1.0, warmup_epochs=10, seed_labels=0,
             seed_supervised=False, seed_frac=0.0, max_train_cases=0, pixel_adapt="none", fda_beta=0.05, drr_blur=0.0,
             drr_noise=0.0, adabn=False, shuffle_labels=False, select="source_val")
    d.update(kw)
    return argparse.Namespace(**d)


def test_run_name_suffixes():
    assert CE.run_name(_ns()) == "a0_moderate_ap_f0_s1_selsrc"
    n = CE.run_name(_ns(arm="a1", seed_labels=50, seed_supervised=True, pixel_adapt="fda", adabn=True, select="target_val"))
    assert n == "a1_moderate_ap_f0_s1_g1_w10_k50_sup_pafda5_adabn"


def test_evaluate_scores_threshold_from_val_only(tmp_path):
    rng = np.random.default_rng(0)
    yv = np.r_[np.ones(50), np.zeros(50)]; tv = np.r_[rng.uniform(.4, 1, 50), rng.uniform(0, .6, 50)]
    roles = ["positive"] * 40 + ["negative"] * 40 + ["airway_hard_negative"] * 10
    yt = np.r_[np.ones(40), np.zeros(50)]; st = np.r_[rng.uniform(.5, 1, 40), rng.uniform(0, .5, 40), rng.uniform(.3, .7, 10)]
    d = CE.evaluate_scores(yv, tv, yt, st, roles, n_boot=50, seed=0)
    assert d["n_positive"] == 40 and d["n_negative"] == 40 and d["n_airway_hard_negative"] == 10
    assert d["auroc"] > 0.9 and d["auroc_ci95"][0] <= d["auroc"] <= d["auroc_ci95"][1]
    assert 0 <= d["airway_false_positive_rate_at_that_threshold"] <= 1
    CE.write_outputs(tmp_path / "r", {"x": 1}, [f"i{k}" for k in range(90)], roles, yt, st)
    assert (tmp_path / "r" / "metrics.json").exists() and len((tmp_path / "r" / "predictions_test.csv").read_text().splitlines()) == 91


def test_load_boxes_reads_vindr_csv(tmp_path):
    from drr_pipeline import boxes as B
    p = tmp_path / "ann.csv"
    p.write_text("image_id,rad_ID,class_name,x_min,y_min,x_max,y_max\n"
                 "a,R1,Consolidation,1,2,30,40\na,R1,No finding,,,,\nb,R2,No finding,,,,\n", encoding="utf8")
    out = B.load_boxes(p)
    assert out == {"a": [("Consolidation", (1.0, 2.0, 30.0, 40.0))]} and "Consolidation" in B.AIRSPACE
