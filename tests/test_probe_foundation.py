import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import probe_foundation as PF  # noqa: E402


def _feat(n, d, rng, shift=0.0, signal=1.5, prev=0.25):
    y = (rng.random(n) < prev).astype(int)
    X = rng.normal(size=(n, d)) + signal * y[:, None] * (np.arange(d) < 6) + shift
    return X, y


def _features(with_text=True, ages=True):
    r = np.random.default_rng(0)
    d = 24
    Xs, ys = _feat(300, d, r, signal=1.5, prev=0.5)
    Xsv, ysv = _feat(120, d, r, signal=1.5, prev=0.5)
    S = {"X_tr": Xs, "y_tr": ys, "X_val": Xsv, "y_val": ysv}
    Xp, yp = _feat(600, d, r, shift=2.0, signal=1.0)
    Xv, yv = _feat(150, d, r, shift=2.0, signal=1.0)
    Xt, yt = _feat(300, d, r, shift=2.0, signal=1.0)
    role = np.where(yp == 1, "positive", "negative").astype(object)
    role[:30] = "unlabeled_excluded"
    D = {"X_pool": Xp, "y_pool": yp, "role_pool": role.astype(str), "X_val": Xv, "y_val": yv, "X_te": Xt, "y_te": yt,
         "age_te": r.choice([3.0, 8.0, 18.0, 40.0], 300) if ages else np.full(300, np.nan)}
    if with_text:
        tp, tn = np.zeros((2, d)), np.zeros((2, d))
        tp[:, :6], tn[:, :6] = 1.0, -1.0
        for k in ("pneumonia", "opacity", "ensemble"):
            D[f"T_{k}_pos"], D[f"T_{k}_neg"] = tp + 0.01 * r.normal(size=tp.shape), tn + 0.01 * r.normal(size=tn.shape)
    return S, D


def test_run_probe_arms_and_ordering():
    S, D = _features()
    r = PF.run_probe(S, D, ks=(20, 100), drr_ks=(20,), reps=3, n_boot=50)
    for arm in ("zs_pneumonia", "zs_opacity", "zs_ensemble", "lp_drr", "lp_drr_domstd", "lp_k20", "lp_k100", "lp_all", "lp_drr_k20", "_meta"):
        assert arm in r, arm
    assert r["lp_all"]["auroc"] > 0.8 and r["zs_ensemble"]["auroc"] > 0.6
    assert r["lp_k100"]["auroc"] >= r["lp_k20"]["auroc"] - 0.05
    # the target is shifted by +2 in every feature: aligning with the unlabeled target statistics helps
    assert r["lp_drr_domstd"]["auroc"] >= r["lp_drr"]["auroc"] - 0.02
    assert r["lp_drr"]["source_val_auroc"] > 0.8
    assert r["_meta"]["n_real_labels"] == 570 and r["_meta"]["n_src"] == 300          # unlabeled_excluded rows are not used as labels
    assert set(r["lp_drr"]["by_age"]) >= {"<6m", "6-12m", "12-24m", ">=24m", "unknown"}


def test_run_probe_without_text_or_age_and_report():
    S, D = _features(with_text=False, ages=False)
    r = PF.run_probe(S, D, ks=(20,), drr_ks=(20,), reps=2, n_boot=20)
    assert not any(k.startswith("zs_") for k in r) and "by_age" not in r["lp_drr"]
    txt = PF.format_report({("rad_dino", "vindr_pcxr"): r, ("resnet18_in", "vindr_pcxr"): r})
    assert "TARGET vindr_pcxr" in txt and "lp_drr" in txt and "reference: A0" in txt and "rad_dino" in txt


def test_run_probe_is_deterministic():
    S, D = _features()
    a = PF.run_probe(S, D, ks=(20,), drr_ks=(20,), reps=2, n_boot=0)
    b = PF.run_probe(S, D, ks=(20,), drr_ks=(20,), reps=2, n_boot=0)
    assert a["lp_k20"]["values"] == b["lp_k20"]["values"] and a["lp_all"]["auroc"] == b["lp_all"]["auroc"]
