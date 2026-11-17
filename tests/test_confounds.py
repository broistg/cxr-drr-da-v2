import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("analyze_confounds", Path(__file__).resolve().parents[1] / "scripts" / "diag" / "confounds.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)


def test_spearman_monotone_and_ties():
    a = np.array([1.0, 2, 3, 4, 5])
    assert abs(C.spearman(a, a ** 3) - 1) < 1e-9
    assert abs(C.spearman(a, -a) + 1) < 1e-9
    assert np.isnan(C.spearman(a, np.ones(5)))


def test_logistic_cv_finds_signal_and_noise():
    rng = np.random.default_rng(0)
    y = (rng.random(600) < 0.3).astype(int)
    X = np.c_[rng.normal(size=600) + 2.0 * y, rng.normal(size=600)]
    assert C.logistic_cv_auroc(X, y) > 0.85
    assert abs(C.logistic_cv_auroc(rng.normal(size=(600, 2)), y) - 0.5) < 0.1
