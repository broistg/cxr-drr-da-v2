import numpy as np

from drr_pipeline import metrics as Mx
from drr_pipeline import overlap as O


def test_small_vectors_unit_zero_mean():
    rng = np.random.default_rng(0)
    v = O.small_vectors(rng.integers(0, 255, (5, 64, 64)), 8)
    assert v.shape == (5, 64) and np.allclose(np.linalg.norm(v, axis=1), 1) and np.allclose(v.mean(1), 0, atol=1e-9)


def test_max_similarity_finds_duplicate_and_excludes_self():
    rng = np.random.default_rng(1)
    imgs = rng.integers(0, 255, (6, 64, 64)).astype(float)
    imgs[5] = imgs[2] * 0.8 + 10                       # brightness/contrast change of image 2: correlation 1
    v = O.small_vectors(imgs, 16)
    s, j = O.max_similarity(v, v, exclude_self=True)
    assert j[2] == 5 and j[5] == 2 and s[2] > 0.999 and s[0] < 0.5
    s2, _ = O.max_similarity(v[:2], v, exclude_self=False)
    assert np.allclose(s2[:2], 1.0)
    out = O.summarize(s)
    assert out["n"] == 6 and out[">=0.99"] == 2


def test_seed_mean_bootstrap_and_paired():
    rng = np.random.default_rng(2)
    y = np.r_[np.ones(60), np.zeros(140)].astype(int)
    S = np.vstack([y * 1.0 + rng.normal(0, 1, 200) for _ in range(3)])
    r = Mx.seed_mean_bootstrap(Mx.auroc, y, S, 200)
    assert r["n_runs"] == 3 and r["lo"] < r["mean"] < r["hi"] and 0.7 < r["mean"] < 0.95
    d = Mx.paired_seed_mean_diff(Mx.auroc, y, S, S, 100)
    assert d["diff"] == 0 and d["lo"] == 0 and d["hi"] == 0
    d2 = Mx.paired_seed_mean_diff(Mx.auroc, y, S, rng.normal(size=S.shape), 200)
    assert d2["lo"] > 0
