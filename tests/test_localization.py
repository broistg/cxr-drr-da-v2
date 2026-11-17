import numpy as np

from drr_pipeline import localization as L


def test_box_mapping_landscape_and_portrait():
    # 100 rows x 200 cols: padded to 200x200 with 50 rows of padding top and bottom, then scaled 256/200
    xa, ya, xb, yb = L.map_box_to_square((0, 0, 200, 100), rows=100, cols=200, size=256)
    assert (xa, xb) == (0, 256)
    assert abs(ya - 64) <= 1 and abs(yb - 192) <= 1
    # portrait: 200 rows x 100 cols -> 50 columns of padding left/right
    xa, ya, xb, yb = L.map_box_to_square((0, 0, 100, 200), rows=200, cols=100, size=256)
    assert (ya, yb) == (0, 256) and abs(xa - 64) <= 1 and abs(xb - 192) <= 1
    # clipping
    assert L.map_box_to_square((-10, -10, 1e6, 1e6), 100, 100, 256) == (0, 0, 256, 256)


def test_enrichment_and_pointing():
    m = np.zeros((10, 10), bool); m[2:4, 2:4] = True            # 4% of the image
    uniform = np.ones((10, 10))
    peak = np.zeros((10, 10)); peak[2, 2] = 1.0
    assert abs(L.enrichment(uniform, m) - 1.0) < 1e-9
    assert abs(L.enrichment(peak, m) - 25.0) < 1e-9
    assert L.pointing_hit(peak, m) is True
    assert L.pointing_hit(np.flipud(peak), m) is False
    assert L.pointing_hit(peak, np.zeros_like(m)) is None
    assert np.isnan(L.enrichment(peak, np.zeros_like(m)))


def test_region_restriction():
    m = np.zeros((10, 10), bool); m[0:2, 0:2] = True
    region = np.zeros((10, 10), bool); region[0:5, 0:5] = True   # lesion is 4 of 25 region pixels
    cam = np.ones((10, 10))
    assert abs(L.enrichment(cam, m, region) - 1.0) < 1e-9        # uniform inside the region is neutral
    cam2 = cam.copy(); cam2[~region] = 100                       # mass outside the region is ignored
    assert abs(L.enrichment(cam2, m, region) - 1.0) < 1e-9


def test_center_prior_peaks_in_middle():
    c = L.center_prior(64)
    assert c[32, 32] > c[0, 0] and c.max() <= 1.0
