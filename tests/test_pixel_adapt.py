import numpy as np

from drr_pipeline import pixel_adapt as PA


def _img(seed, lo=0, hi=255, n=64):
    return np.random.default_rng(seed).integers(lo, hi, (n, n), dtype=np.uint8)


def test_hist_match_moves_distribution_to_reference():
    src, ref = _img(0, 0, 100), _img(1, 100, 250)
    out = PA.match_histogram(src, ref)
    assert out.dtype == np.uint8 and out.shape == src.shape
    assert abs(out.mean() - ref.mean()) < 3 and out.min() >= 100 - 2
    # monotone: ordering of source pixels is preserved
    o = np.argsort(src.ravel(), kind="stable")
    assert np.all(np.diff(out.ravel()[o].astype(int)) >= 0)
    assert np.array_equal(PA.match_histogram(ref, ref), ref)


def test_fda_identity_when_same_or_beta_zero_and_low_freq_swapped():
    a, b = _img(0), _img(1)
    assert np.array_equal(PA.fda(a, b, beta=0.0), a)
    assert np.abs(PA.fda(a, a, 0.1).astype(int) - a).max() <= 1
    out = PA.fda(a, b, 0.1).astype(float)
    # low-frequency content (DC term) moves to the reference's, high-frequency stays the source's
    assert abs(out.mean() - b.mean()) < abs(out.mean() - a.mean()) + 1e-9 or abs(a.mean() - b.mean()) < 1
    sub = lambda x: np.abs(np.fft.fftshift(np.fft.fft2(x))[20:44, 20:44])
    assert np.abs(sub(out) - sub(b.astype(float))).mean() < np.abs(sub(a.astype(float)) - sub(b.astype(float))).mean()


def test_adapt_batch_noop_and_rng_untouched():
    x = np.stack([_img(i) for i in range(4)]); pool = np.stack([_img(10 + i) for i in range(6)])
    r1, r2 = np.random.default_rng(5), np.random.default_rng(5)
    assert PA.adapt_batch(x, pool, r1, "none") is x
    assert r1.integers(0, 10**9) == r2.integers(0, 10**9)                      # "none" draws nothing
    for mode in ("hist", "fda"):
        y = PA.adapt_batch(x, pool, np.random.default_rng(1), mode)
        assert y.shape == x.shape and y.dtype == np.uint8
        assert np.array_equal(y, PA.adapt_batch(x, pool, np.random.default_rng(1), mode))


def test_blur_reduces_variance_keeps_mean_and_noop():
    a = _img(0).astype(float)
    b = PA.gaussian_blur(a, 2.0)
    assert b.shape == a.shape and b.std() < 0.5 * a.std() and abs(b.mean() - a.mean()) < 2
    assert PA.gaussian_blur(a, 0.0) is a


def test_degrade_batch():
    x = np.stack([_img(i) for i in range(5)])
    r1, r2 = np.random.default_rng(3), np.random.default_rng(3)
    assert PA.degrade_batch(x, r1, 0, 0) is x and r1.integers(0, 10**9) == r2.integers(0, 10**9)
    y = PA.degrade_batch(x, np.random.default_rng(3), 3.0, 0.0)
    assert y.dtype == np.uint8 and y.std() < x.std()
    flat = np.full((3, 64, 64), 128, np.uint8)
    z = PA.degrade_batch(flat, np.random.default_rng(3), 0.0, 0.1)
    assert z.std() > 1 and abs(float(z.mean()) - 128) < 3
    assert np.array_equal(z, PA.degrade_batch(flat, np.random.default_rng(3), 0.0, 0.1))
