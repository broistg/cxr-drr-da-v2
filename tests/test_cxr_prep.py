import numpy as np

from drr_pipeline import cxr_prep as P

G = {"pneumonia_group": ["Pneumonia", "Brocho-pneumonia", "Pleuro-pneumonia"],
     "airway_group": ["Bronchitis", "Bronchiolitis"]}


def test_monochrome1_inverted():
    a = np.tile(np.arange(100, dtype=float), (10, 1))      # bright to the right
    b = P.to_uint8(a, monochrome1=False)
    c = P.to_uint8(a, monochrome1=True)
    assert b[0, 99] > b[0, 0] and c[0, 99] < c[0, 0]
    assert np.array_equal(b, 255 - c) or abs(int(b[0, 50]) + int(c[0, 50]) - 255) <= 2


def test_constant_image_is_zero():
    assert P.to_uint8(np.full((5, 5), 7.0), False).sum() == 0


def test_pad_square_keeps_content_centered():
    img = np.ones((4, 8), np.uint8)
    out = P.pad_square(img)
    assert out.shape == (8, 8) and out.sum() == 32 and out[2:6].all() and not out[:2].any()


def test_roles():
    assert P.assign_role({"Pneumonia"}, G) == P.POSITIVE
    assert P.assign_role({"Bronchitis", "Brocho-pneumonia"}, G) == P.POSITIVE
    assert P.assign_role({"No finding"}, G) == P.NEGATIVE
    assert P.assign_role({"Bronchiolitis"}, G) == P.AIRWAY
    assert P.assign_role({"Bronchiolitis", "Other disease"}, G) == P.AIRWAY
    assert P.assign_role({"Tuberculosis"}, G) == P.OTHER
    assert P.assign_role(set(), G) == P.UNLABELED


def test_val_split_stratified_deterministic():
    ids = [f"i{k}" for k in range(100)]
    strata = ["a"] * 80 + ["b"] * 20
    v1 = P.stratified_val_split(ids, strata, 0.1, 0)
    v2 = P.stratified_val_split(ids, strata, 0.1, 0)
    assert v1 == v2 and len(v1) == 10
    assert sum(i in v1 for i in ids[:80]) == 8 and sum(i in v1 for i in ids[80:]) == 2
    assert P.stratified_val_split(ids, strata, 0.1, 1) != v1
