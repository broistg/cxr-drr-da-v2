import numpy as np

from drr_pipeline import metrics as Mx


def test_auroc_perfect_inverse_ties():
    y = np.array([0, 0, 1, 1])
    assert Mx.auroc(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert Mx.auroc(y, [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert Mx.auroc(y, [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert np.isnan(Mx.auroc([1, 1], [0.1, 0.2]))


def test_auroc_matches_pair_counting():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200); s = rng.normal(size=200) + y
    pos, neg = s[y == 1], s[y == 0]
    brute = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
    assert abs(Mx.auroc(y, s) - brute) < 1e-12


def test_average_precision():
    assert Mx.average_precision([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    # one positive ranked second of three
    assert abs(Mx.average_precision([1, 0, 0], [0.5, 0.9, 0.1]) - 0.5) < 1e-12


def test_sens_at_spec():
    y = np.array([0] * 10 + [1] * 4)
    s = np.concatenate([np.arange(10) / 10.0, [0.95, 0.5, 1.2, 1.5]])
    thr = Mx.threshold_at_specificity(y, s, 0.9)          # 90th percentile of negatives
    assert abs(thr - 0.81) < 1e-9
    assert Mx.sensitivity_at_threshold(y, s, thr) == 0.75


def test_bootstrap_ci_and_paired_diff():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    good, bad = rng.normal(size=400) + 2 * y, rng.normal(size=400) + 0.3 * y
    lo, hi = Mx.bootstrap_ci(Mx.auroc, y, good, n=300, seed=0)
    assert lo < Mx.auroc(y, good) < hi
    d = Mx.paired_bootstrap_diff(Mx.auroc, y, good, bad, n=300, seed=0)
    assert d["lo"] > 0 and d["diff"] > 0
