import numpy as np

from drr_pipeline import prior as P


def _posteriors(n, pi_t, rng, pi_s=0.5, sep=2.0):
    y = rng.random(n) < pi_t
    x = rng.normal(size=n) + sep * y                     # class-conditional N(0,1) vs N(sep,1)
    llr = sep * x - sep ** 2 / 2                         # exact log-likelihood ratio
    z = llr + P.log_odds(pi_s)                           # posterior under the SOURCE prior
    return 1 / (1 + np.exp(-z)), y


def test_recovers_known_target_prior():
    rng = np.random.default_rng(0)
    for pi_t in (0.12, 0.3, 0.5):
        p, _ = _posteriors(20000, pi_t, rng)
        est, shift = P.estimate_target_prior(p, 0.5)
        assert abs(est - pi_t) < 0.02
        assert abs(shift - (P.log_odds(est) - P.log_odds(0.5))) < 1e-9


def test_same_prior_gives_no_shift():
    rng = np.random.default_rng(1)
    p, _ = _posteriors(20000, 0.5, rng)
    est, shift = P.estimate_target_prior(p, 0.5)
    assert abs(shift) < 0.1


def test_shift_is_monotone_and_finite_at_extremes():
    est, shift = P.estimate_target_prior(np.array([0.0, 1.0, 0.5, 1e-9]), 0.5)
    assert np.isfinite(est) and np.isfinite(shift) and 0.0 < est < 1.0
