import numpy as np
import pytest

from drr_pipeline.weighted import (
    category_projections,
    weighted_from_projections,
    min_fraction_for_label2,
    pi2_closed_form,
    weighted_drr,
)


def ray(n_bg, n_lung, n_inf, rho=None):
    """One ray as a (n, 1, 1) volume; project() sums along axis 0."""
    m = np.array([0] * n_bg + [1] * n_lung + [2] * n_inf).reshape(-1, 1, 1)
    r = np.ones_like(m, dtype=float) if rho is None else rho.reshape(-1, 1, 1)
    return r, m


def test_equal_weights_give_standard_drr():
    rng = np.random.default_rng(0)
    m = rng.integers(0, 3, size=(20, 4, 4))
    rho = rng.random((20, 4, 4))
    p, _, _, _ = weighted_drr(rho, m, (1, 1, 1))
    assert np.allclose(p, rho.sum(axis=0))


def test_ten_voxel_example_from_slides():
    # 5 background, 4 lung, 1 infected voxel; T2 = 0.20
    rho, m = ray(5, 4, 1)
    _, _, pi2_a, lab_a = weighted_drr(rho, m, (1, 1, 1), T2=0.2)
    _, _, pi2_b, lab_b = weighted_drr(rho, m, (1, 1, 3), T2=0.2)
    assert pi2_a[0, 0] == pytest.approx(0.10)
    assert lab_a[0, 0] == 1
    assert pi2_b[0, 0] == pytest.approx(0.25)
    assert lab_b[0, 0] == 2


def test_total_contribution_is_10_and_12():
    rho, m = ray(5, 4, 1)
    for w2, total in [(1, 10), (3, 12)]:
        _, _, pi2, _ = weighted_drr(rho, m, (1, 1, w2))
        assert pi2[0, 0] == pytest.approx(w2 / total)


@pytest.mark.parametrize("f,w2", [(0.1, 3), (0.25, 6), (0.4, 1.5)])
def test_closed_form(f, w2):
    n = 20
    n_inf = round(f * n)
    rho, m = ray(n // 2, n - n // 2 - n_inf, n_inf)
    _, _, pi2, _ = weighted_drr(rho, m, (1, 1, w2))
    assert pi2[0, 0] == pytest.approx(pi2_closed_form(n_inf / n, w2))


def test_threshold_boundary_w2_3_T2_02():
    # f* = 0.2 / (3 * 0.8 + 0.2) = 1/13 ; label 2 needs f strictly above it
    assert min_fraction_for_label2(0.2, 3) == pytest.approx(1 / 13)
    for n_total, expected in [(14, 1), (13, 1), (12, 2)]:   # 1 infected voxel
        rho, m = ray(n_total // 2, n_total - n_total // 2 - 1, 1)
        _, _, _, lab = weighted_drr(rho, m, (1, 1, 3), T2=0.2)
        assert lab[0, 0] == expected, n_total


def test_raising_w2_never_removes_infection_pixels():
    rng = np.random.default_rng(1)
    m = rng.choice([0, 1, 2], size=(30, 16, 16), p=[0.4, 0.5, 0.1])
    rho = rng.random(m.shape)
    prev = None
    for w2 in [0.5, 1, 1.5, 3, 6, 12]:
        _, _, _, lab = weighted_drr(rho, m, (1, 1, w2), T2=0.2)
        cur = lab == 2
        if prev is not None:
            assert (cur >= prev).all()
        prev = cur


def test_small_w2_removes_labels_like_normal_drr():
    # DRR4Covid normal images use (24, 24, 1): infection contribution becomes tiny
    rho, m = ray(5, 4, 1)
    _, _, pi2, lab = weighted_drr(rho, m, (24, 24, 1), T2=0.2)
    assert pi2[0, 0] < 0.01 and lab[0, 0] == 1


def test_empty_ray_is_safe():
    m = np.zeros((5, 2, 2), dtype=int)
    p, pi1, pi2, lab = weighted_drr(np.ones((5, 2, 2)), m, (0, 0, 0))
    assert np.all(p == 0) and np.all(lab == 0) and np.all(np.isfinite(pi2))


def test_six_projections_reproduce_every_weight_configuration():
    rng = np.random.default_rng(2)
    m = rng.choice([0, 1, 2], size=(25, 8, 8), p=[0.4, 0.45, 0.15])
    rho = rng.random(m.shape)
    M, R = category_projections(rho, m)
    for w in [(1, 1, 1), (1, 1, 3), (12, 12, 1), (24, 24, 1), (1, 1, 12)]:
        for T2 in (0.0, 0.2, 0.4):
            ref = weighted_drr(rho, m, w, T2=T2)
            out = weighted_from_projections(M, R, w, T2=T2)
            for x, y in zip(ref, out):
                assert np.allclose(x, y)
    # and against a brute-force evaluation of Eq. 1 / Eq. 4 on one ray
    d = (slice(None), 3, 4)
    wv = np.where(m[d] == 2, 3.0, np.where(m[d] == 1, 1.0, 1.0))
    p_ref = (rho[d] * wv).sum() / (wv.sum() / len(wv))
    pi2_ref = (wv * (m[d] == 2)).sum() / wv.sum()
    p, _, pi2, _ = weighted_from_projections(M, R, (1, 1, 3))
    assert p[3, 4] == pytest.approx(p_ref)
    assert pi2[3, 4] == pytest.approx(pi2_ref)
