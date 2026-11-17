import numpy as np

from drr_pipeline import dataset as D
from drr_pipeline.weighted import category_projections


def _toy(h=8, w=8, depth=6, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.integers(0, 3, size=(depth, h, w)).astype(np.uint8)
    rho = rng.uniform(0.1, 1.5, size=m.shape)
    return category_projections(rho, m)


def test_pose_seed_stable_and_kind_dependent():
    a = D.pose_seed(0, "c1", "moderate", "ap", "normal")
    assert a == D.pose_seed(0, "c1", "moderate", "ap", "normal")
    assert a != D.pose_seed(0, "c1", "moderate", "ap", "positive")
    assert a != D.pose_seed(0, "c2", "moderate", "ap", "normal")
    assert a != D.pose_seed(1, "c1", "moderate", "ap", "normal")
    x = np.random.default_rng(a).uniform(size=3)
    y = np.random.default_rng(D.pose_seed(0, "c1", "moderate", "ap", "normal")).uniform(size=3)
    assert np.allclose(x, y)


def test_cache_roundtrip(tmp_path):
    M, R = _toy()
    p = D.cache_path(tmp_path, "moderate", "ap", "case")
    D.save_cache(p, {"M_normal": M[None], "R_normal": R[None]})
    z = D.load_cache(p)
    assert np.array_equal(z["M_normal"][0], M) and np.array_equal(z["R_normal"][0], R)
    assert not list(p.parent.glob("*.tmp.npz"))


def test_to_uint8_range_and_empty():
    p = np.linspace(0, 5, 100).reshape(10, 10)
    u = D.to_uint8(p)
    assert u.dtype == np.uint8 and u.max() == 255 and u.min() == 0
    assert D.to_uint8(np.zeros((4, 4))).sum() == 0


def test_derive_positive_shapes_and_monotonic_labels():
    M, R = _toy()
    thr = [0.0, 0.1, 0.4]
    img, labs = D.derive_positive(M, R, [1, 1, 3], thr)
    assert img.shape == (8, 8) and labs.shape == (3, 8, 8)
    # a higher threshold can only remove infection pixels
    inf = [(labs[k] == 2).sum() for k in range(3)]
    assert inf[0] >= inf[1] >= inf[2]


def test_derive_normal_label_zero():
    M, R = _toy()
    img, lab = D.derive_normal(M, R, [24, 24, 1])
    assert img.shape == lab.shape == (8, 8) and lab.sum() == 0


def test_image_independent_of_threshold():
    M, R = _toy()
    a, _ = D.derive_positive(M, R, [1, 1, 3], [0.0])
    b, _ = D.derive_positive(M, R, [1, 1, 3], [0.4])
    assert np.array_equal(a, b)
