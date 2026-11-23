import math

from drr_pipeline import review_stats as R


def test_wilson_bounds():
    lo, hi = R.wilson(0, 100)
    assert lo == 0 or lo < 1e-9
    assert 0.03 < hi < 0.05          # 0/100: cận trên ~3,7%
    lo, hi = R.wilson(50, 100)
    assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60
    assert math.isnan(R.wilson(0, 0)[0])


def test_kappa():
    a = ["yes", "no", "no", "yes", "no"]
    assert R.cohen_kappa(a, a) == 1.0
    assert R.cohen_kappa(["yes", "yes", "no", "no"], ["yes", "no", "yes", "no"]) == 0.0
    assert math.isnan(R.cohen_kappa(["no", "no"], ["no", "no"]))
