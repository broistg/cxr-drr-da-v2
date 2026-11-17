import numpy as np
import pytest

from drr_pipeline import eda


def test_looks_like_hu():
    hu = np.array([-1000, -800, 40, 300], dtype=float)
    norm = np.array([0, 40, 128, 255], dtype=float)
    assert eda.looks_like_hu(hu) is True
    assert eda.looks_like_hu(norm) is False


def test_volume_ml():
    assert eda.volume_ml(1000, (1, 1, 1)) == pytest.approx(1.0)
    assert eda.volume_ml(100, (0.7, 0.7, 5.0)) == pytest.approx(100 * 2.45 / 1000)


def test_case_stats_combined_masks():
    ct = np.full((10, 10, 10), -800.0)
    lung = np.zeros(ct.shape, bool); lung[2:8, 2:8, 2:8] = True      # 216 voxels
    inf = np.zeros(ct.shape, bool); inf[4:6, 4:6, 4:6] = True        # 8 voxels (inside lung here)
    st = eda.case_stats(ct, (10, 10, 10), lung & ~inf, inf, separate_masks=False)
    assert st["piv_percent"] == pytest.approx(100 * 8 / 216)
    assert st["inf_outside_lung_percent"] == 0
    assert eda.usable(st) is True


def test_case_stats_separate_masks_reports_overlap():
    ct = np.full((10, 10, 10), -800.0)
    lung = np.zeros(ct.shape, bool); lung[0:5] = True                # 500 voxels
    inf = np.zeros(ct.shape, bool); inf[4:7] = True                  # 300 voxels, 100 inside lung
    st = eda.case_stats(ct, (1, 1, 1), lung, inf, separate_masks=True)
    assert st["piv_percent"] == pytest.approx(100 * 100 / 500)
    assert st["inf_outside_lung_percent"] == pytest.approx(100 * 200 / 300)


def test_normalised_ct_is_not_usable():
    ct = np.random.default_rng(0).integers(0, 256, (6, 6, 6)).astype(float)
    m = np.ones(ct.shape, bool)
    st = eda.case_stats(ct, (1, 1, 1), m, m, separate_masks=True)
    assert eda.usable(st, min_infection_ml=0.0) is False


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        eda.case_stats(np.zeros((4, 4, 4)), (1, 1, 1), np.zeros((4, 4, 5), bool), np.zeros((4, 4, 4), bool), True)


def test_label_counts():
    assert eda.label_counts(np.array([0, 0, 1, 3, 3, 3])) == {0: 2, 1: 1, 3: 3}
