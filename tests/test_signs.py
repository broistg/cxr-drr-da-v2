import numpy as np
import pytest

from drr_pipeline import signs as S
from drr_pipeline import weighted as W


def test_shipped_signs_load_and_are_plain():
    names = S.list_signs()
    assert {"consolidation", "pleural_effusion", "pneumothorax", "cardiomegaly"} <= set(names)
    for n in names:
        cfg = S.load_sign(n)
        assert cfg["drr"]["kind"] == "plain" and len(set(S.drr_weights(cfg))) == 1


def test_validate_rejects_unequal_plain_and_bad_rule():
    base = S.load_sign("consolidation")
    bad = dict(base, drr={"kind": "plain", "weights": [1, 1, 3]})
    with pytest.raises(ValueError):
        S.validate(bad)
    with pytest.raises(ValueError):
        S.validate(dict(base, label_rule={"min_mask_px": 1, "T2": 1.5}))
    S.validate(dict(base, drr={"kind": "weighted", "weights": [1, 1, 3]}))


def test_plain_weights_give_unweighted_projection_and_length_fraction():
    rng = np.random.default_rng(0)
    rho = rng.uniform(0, 1, (6, 6, 6)); m = rng.integers(0, 3, (6, 6, 6))
    M, R = W.category_projections(rho, m)
    p, _, pi2, _ = W.weighted_from_projections(M, R, (1.0, 1.0, 1.0), 0.0, 0.0)
    assert np.allclose(p, R.sum(0))                              # DRR thường = tích phân hấp thụ
    assert np.allclose(pi2, M[2] / M.sum(0))                     # pi2 = phần chiều dài tia qua tổn thương
