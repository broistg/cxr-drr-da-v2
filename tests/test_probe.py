import numpy as np

from drr_pipeline import probe as P


def _data(n=400, d=20, shift=1.0, seed=0):
    r = np.random.default_rng(seed)
    y = (r.random(n) < 0.3).astype(int)
    X = r.normal(size=(n, d)) + shift * y[:, None] * (np.arange(d) < 5)
    return X, y


def test_fit_logreg_gradient_zero_and_discriminates():
    X, y = _data()
    w, b = P.fit_logreg(X, y, C=1.0)
    z = X @ w + b
    g = np.append(X.T @ (P.sigmoid(z) - y) + w, (P.sigmoid(z) - y).sum())       # penalty 1/C = 1, intercept free
    assert np.abs(g).max() < 1e-3
    Xt, yt = _data(seed=1)
    from drr_pipeline import metrics as Mx
    assert Mx.auroc(yt, P.predict_logit(Xt, w, b)) > 0.85


def test_fit_logreg_weights_and_regularisation():
    X, y = _data()
    w1, _ = P.fit_logreg(X, y, C=0.003)
    w2, _ = P.fit_logreg(X, y, C=0.3)
    assert np.linalg.norm(w1) < np.linalg.norm(w2)                              # smaller C = stronger shrinkage
    wa, ba = P.fit_logreg(X, y, 1.0, sw=np.ones(len(y)))
    wb, bb = P.fit_logreg(X, y, 1.0)
    assert np.allclose(wa, wb) and abs(ba - bb) < 1e-9
    # doubling the weight of the positives raises the intercept (more positive predictions)
    _, b3 = P.fit_logreg(X, y, 1.0, sw=np.where(y == 1, 3.0, 1.0))
    assert b3 > bb


def test_separable_data_does_not_blow_up():
    X = np.vstack([np.ones((20, 3)), -np.ones((20, 3))]); y = np.r_[np.ones(20), np.zeros(20)]
    w, b = P.fit_logreg(X, y, 0.03)
    assert np.all(np.isfinite(w)) and (P.predict_logit(X, w, b) > 0).tolist() == (y == 1).tolist()


def test_choose_C_and_inner_cv_in_grid():
    X, y = _data(); Xv, yv = _data(seed=2)
    C, w, b = P.choose_C_by_auroc(X, y, Xv, yv)
    assert C in P.CS and w.shape == (20,)
    assert P.inner_cv_C(X[:80], y[:80]) in P.CS
    assert P.inner_cv_C(X[:10], np.zeros(10, int)) == P.CS[1]                     # no positives: default


def test_subsample_stratified():
    y = (np.random.default_rng(0).random(1000) < 0.15).astype(int)
    for k in (20, 50, 200):
        i = P.subsample_stratified(y, k, np.random.default_rng(1))
        assert len(i) == k == len(set(i.tolist())) and y[i].sum() >= 3 and (1 - y[i]).sum() >= 3
    i = P.subsample_stratified(y, 200, np.random.default_rng(1))
    assert abs(y[i].mean() - y.mean()) < 0.03


def test_zero_shot_scores_prefers_positive_direction():
    r = np.random.default_rng(0)
    tp, tn = r.normal(size=(3, 16)) + 3.0, r.normal(size=(3, 16)) - 3.0
    img = np.vstack([np.full((5, 16), 3.0), np.full((5, 16), -3.0)]) + r.normal(size=(10, 16)) * 0.1
    s = P.zero_shot_scores(img, tp, tn)
    assert s[:5].min() > s[5:].max()


def test_auroc_by_age():
    age = np.r_[np.full(40, 3.0), np.full(40, 18.0), np.full(10, np.nan)]
    y = np.tile([0, 1], 45); s = y + np.random.default_rng(0).normal(size=90) * 0.3
    r = P.auroc_by_age(y, s, age)
    assert r["<6m"]["n_pos"] + r["<6m"]["n_neg"] == 40 and r["12-24m"]["auroc"] > 0.8
    assert np.isnan(r["6-12m"]["auroc"]) and r["unknown"]["n_pos"] + r["unknown"]["n_neg"] == 10
