"""Infection-aware weighted DRR (DRR4Covid, Eq. 1-4), independent of the renderer.

Every quantity is a line integral along a ray, hence linear in the voxel weights:

    A  = sum l * rho * w          (numerator of Eq. 1)
    B  = sum l * w                (denominator of Eq. 4)
    C2 = sum l * w * [m == 2]     (infected voxels)
    C1 = sum l * w * [m == 1]     (lung voxels)
    L  = sum l                    (ray length, stands in for |Omega_d|)

and all of them follow from six weight-independent projections (category_projections).

`project` is any callable mapping a 3D array to the 2D projection. The default
sums along axis 0 (parallel rays, l = 1), which is exact for the unit tests.
To use DiffDRR, pass a function that renders the given volume with the same
geometry (or, more efficiently, call category_projections' six renders once per pose and then
weighted_from_projections for every weight configuration). Check first that the renderer does not apply its own HU->attenuation
conversion or normalisation, otherwise Eq. 1 and 4 are no longer reproduced.

Convention (paper): m = 0 background, 1 lung, 2 infection.
Assumption (ours): when l is not constant, |Omega_d| is replaced by the ray length,
so mean(w) = B / L. With l = 1 this is exactly Eq. 1.
"""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

Project = Callable[[np.ndarray], np.ndarray]


def project_sum(vol: np.ndarray) -> np.ndarray:
    """Parallel-ray projection along axis 0 with unit step (l = 1)."""
    return vol.sum(axis=0)


def voxel_weights(m: np.ndarray, w: Tuple[float, float, float]) -> np.ndarray:
    """Eq. 2: weight of each voxel from its category m in {0, 1, 2}."""
    w = np.asarray(w, dtype=np.float64)
    return w[np.asarray(m, dtype=np.int64)]


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a, dtype=np.float64)
    np.divide(a, b, out=out, where=b > 0)
    return out


def category_projections(rho: np.ndarray, m: np.ndarray, project: Project = project_sum, n_cat: int = 3):
    """2 * n_cat weight-independent projections (n_cat = 3: nền, phổi, tổn thương; thêm lớp 3, 4... cho tim, khí ngoài phổi).

    M[c] = sum l * [m == c]         (path length through category c)
    R[c] = sum l * rho * [m == c]   (attenuation integral through category c)

    Eq. 1-4 are linear in the weights, so for ANY (w0, w1, w2) they follow from these six arrays
    (see weighted_from_projections). One pose therefore needs 6 renders, however many weight
    configurations and thresholds are tried afterwards (DRR4Covid uses 9 weights x 7 thresholds).
    """
    M = np.stack([project((m == c).astype(np.float64)) for c in range(n_cat)])
    R = np.stack([project(rho * (m == c)) for c in range(n_cat)])
    return M, R


def weighted_from_projections(M, R, w: Tuple[float, float, float], T1: float = 0.0, T2: float = 0.2):
    """Eq. 1-4 from the six projections. M and R have shape (3, ...) with the image dims last.

    Returns (p, pi1, pi2, label) with the same meaning as weighted_drr.
    """
    M = np.asarray(M, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    wv = np.asarray(w, dtype=np.float64).reshape((len(w),) + (1,) * (M.ndim - 1))
    A = (wv * R).sum(axis=0)             # sum l rho w
    B = (wv * M).sum(axis=0)             # sum l w
    L = M.sum(axis=0)                    # sum l
    mean_w = _safe_div(B, L)             # (sum w) / |Omega|
    p = _safe_div(A, mean_w)             # Eq. 1
    pi1 = _safe_div(wv[1] * M[1], B)     # Eq. 4, c = 1
    pi2 = _safe_div(wv[2] * M[2], B)     # Eq. 4, c = 2
    label = np.where(pi2 > T2, 2, np.where(pi1 > T1, 1, 0)).astype(np.uint8)  # Eq. 3
    return p, pi1, pi2, label


def weighted_drr(
    rho: np.ndarray,
    m: np.ndarray,
    w: Tuple[float, float, float],
    T1: float = 0.0,
    T2: float = 0.2,
    project: Project = project_sum,
):
    """Return (p, pi1, pi2, label) for one weight configuration.

    p     : Eq. 1, the weighted DRR
    pi1/2 : Eq. 4, contribution rate of lung / infection voxels
    label : Eq. 3, pixel label in {0, 1, 2}
    """
    if rho.shape != m.shape:
        raise ValueError("rho and m must have the same shape")
    M, R = category_projections(rho, m, project, n_cat=len(w))
    return weighted_from_projections(M, R, w, T1, T2)


def pi2_closed_form(f: float, w2: float) -> float:
    """pi^2 for a ray with infected fraction f when w0 = w1 = 1 and l is constant."""
    return f * w2 / (1.0 - f + f * w2)


def min_fraction_for_label2(T2: float, w2: float) -> float:
    """Smallest infected fraction f with pi^2 > T2 (w0 = w1 = 1)."""
    return T2 / (w2 * (1.0 - T2) + T2)
