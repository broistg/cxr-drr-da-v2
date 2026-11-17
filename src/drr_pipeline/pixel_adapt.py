"""Label-free pixel-level adaptation (E11), DRR degradation (E1) and helpers: pure numpy, uint8 images (B, H, W).

E11  hist: classical histogram matching of each DRR to a random real CXR from the (unlabeled) target pool;
     fda : Fourier Domain Adaptation (Yang & Soatto 2020): replace the low-frequency amplitude (central window of half-size
           floor(beta * min(H, W))) of the DRR spectrum by that of a random target image, keep the DRR phase.
E1   degrade: random Gaussian blur (sigma ~ U(0, blur_max) px) and Gaussian noise (std ~ U(0, noise_max), in [0, 1] units).
Every function is a no-op (and draws nothing from the rng) when its mode is off, so existing runs keep their random streams.
"""
from __future__ import annotations

import numpy as np

MODES = ("none", "hist", "fda")


def match_histogram(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    s_vals, s_inv, s_cnt = np.unique(src.ravel(), return_inverse=True, return_counts=True)
    r_vals, r_cnt = np.unique(ref.ravel(), return_counts=True)
    s_q = np.cumsum(s_cnt) / src.size
    r_q = np.cumsum(r_cnt) / ref.size
    mapped = np.interp(s_q, r_q, r_vals)
    return np.clip(np.rint(mapped[s_inv]).reshape(src.shape), 0, 255).astype(np.uint8)


def fda(src: np.ndarray, ref: np.ndarray, beta: float = 0.05) -> np.ndarray:
    assert src.shape == ref.shape and src.ndim == 2
    h, w = src.shape
    b = int(np.floor(min(h, w) * beta))
    if b < 1:
        return src.copy()
    fs = np.fft.fftshift(np.fft.fft2(src.astype(np.float64)))
    fr = np.fft.fftshift(np.fft.fft2(ref.astype(np.float64)))
    amp, pha = np.abs(fs), np.angle(fs)
    cy, cx = h // 2, w // 2
    amp[cy - b:cy + b + 1, cx - b:cx + b + 1] = np.abs(fr)[cy - b:cy + b + 1, cx - b:cx + b + 1]
    out = np.fft.ifft2(np.fft.ifftshift(amp * np.exp(1j * pha))).real
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def adapt_batch(x: np.ndarray, pool: np.ndarray, rng: np.random.Generator, mode: str = "none", beta: float = 0.05) -> np.ndarray:
    if mode == "none":
        return x
    assert mode in MODES, mode
    ref = pool[rng.integers(0, len(pool), len(x))]
    f = (lambda s, r: match_histogram(s, r)) if mode == "hist" else (lambda s, r: fda(s, r, beta))
    return np.stack([f(s, r) for s, r in zip(x, ref)])


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur of a float 2-D array with reflect padding."""
    if sigma < 0.05:
        return img
    from numpy.lib.stride_tricks import sliding_window_view as swv
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    out = swv(np.pad(img, ((0, 0), (r, r)), mode="reflect"), 2 * r + 1, axis=1) @ k
    return swv(np.pad(out, ((r, r), (0, 0)), mode="reflect"), 2 * r + 1, axis=0) @ k


def degrade_batch(x: np.ndarray, rng: np.random.Generator, blur_max: float = 0.0, noise_max: float = 0.0) -> np.ndarray:
    if blur_max <= 0 and noise_max <= 0:
        return x
    out = np.empty_like(x)
    for i, img in enumerate(x):
        f = img.astype(np.float64)
        if blur_max > 0:
            f = gaussian_blur(f, rng.uniform(0.0, blur_max))
        if noise_max > 0:
            f = f + rng.normal(0.0, rng.uniform(0.0, noise_max) * 255.0, f.shape)
        out[i] = np.clip(np.rint(f), 0, 255).astype(np.uint8)
    return out
