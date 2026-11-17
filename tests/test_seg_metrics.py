import numpy as np

from drr_pipeline import seg_metrics as M


def test_dice_iou():
    g = np.zeros((2, 8, 8), bool); g[:, :4] = True
    p = np.zeros_like(g); p[0, :4] = True; p[1, :2] = True
    d = M.dice(p, g)
    assert d[0] == 1.0 and abs(d[1] - 2 * 16 / (16 + 32)) < 1e-9
    assert abs(M.iou(p, g)[1] - 0.5) < 1e-9
    e = np.zeros((1, 4, 4), bool)
    assert M.dice(e, e)[0] == 1.0 and M.dice(np.ones_like(e), e)[0] == 0.0


def test_within_extent_quartile():
    g = np.zeros((4, 8, 8), bool); g[:, :2] = True
    lung = np.zeros_like(g); lung[:, :, :4] = True
    p, q = M.within(np.ones_like(g), g, lung)
    assert p.sum() == 4 * 8 * 4 and q.sum() == 4 * 2 * 4
    assert np.allclose(M.extent(g, lung), 0.5) and np.allclose(M.extent(g), 0.25)
    v = np.array([0.0, 1, 2, 3, 4, 5, 6, 7]); key = np.arange(8.0)
    qd = M.by_quartile(v, key)
    assert qd["q1"] < qd["q2"] < qd["q3"] < qd["q4"]


def test_prior_threshold_beats_uniform():
    rng = np.random.default_rng(0)
    gt = np.zeros((30, 16, 16), bool); gt[:, 4:12, 2:8] = True          # lesions always in the same place
    prior = M.mean_prior(gt)
    t, d = M.best_prior_threshold(prior, gt)
    assert d > 0.95 and prior.max() == 1.0
    p3 = M.best_prior_threshold(prior, gt, region=np.ones_like(gt))[1]
    assert abs(p3 - d) < 1e-9


def test_bootstrap_and_permutation_and_subsample():
    lo, hi = M.bootstrap_mean_ci(np.linspace(0, 1, 100), 500, 0)
    assert lo < 0.5 < hi
    for n in (2, 5, 17):
        p = M.permutation_pairs(n, np.random.default_rng(n))
        assert sorted(p) == list(range(n)) and (p != np.arange(n)).all()
    assert M.permutation_pairs(1, np.random.default_rng(0)).tolist() == [0]
    r = np.random.default_rng(0)
    k = M.subsample_k(100, 10, r)
    assert len(set(k)) == 10 and len(M.subsample_k(5, 0, r)) == 5 and len(M.subsample_k(5, 9, r)) == 5


def test_best_prob_threshold():
    gt = np.zeros((4, 8, 8), bool); gt[:, :4] = True
    prob = np.where(gt, 0.8, 0.2)
    t, d = M.best_prob_threshold(prob, gt)
    assert 0.2 < t <= 0.8 and d == 1.0
