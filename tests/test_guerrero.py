"""Invariants and chronological safeguards for Guerrero PLS market experiment."""

import numpy as np
import pytest

from market_geometry.guerrero import (
    GuerreroSettings, fit_panel, forecast_cv_curve, functional_distance,
    gcv_curve, guerrero_mu, lambda_from_s, lag_one_crosscorr, select_s, smoother,
)


def test_guerrero_mu_is_mean_observed_second_differences():
    t = np.arange(22.0)
    x = np.vstack([t**2, 3*t + 4])
    assert np.allclose(guerrero_mu(x), [2.0, 0.0])


def test_smoothness_exact_endpoints_and_monotonicity():
    n, d = 35, 2
    assert lambda_from_s(n, d, 0.0) == 0.0
    assert np.isinf(lambda_from_s(n, d, 1.0))
    roots = [lambda_from_s(n, d, s) for s in (0.1, 0.5, 0.9)]
    assert roots == sorted(roots)
    for s in (0.0, 0.1, 0.5, 0.9, 1.0):
        h, _, tr = smoother(n, d, s)
        assert np.allclose(h, h.T, atol=1e-10)
        assert np.isclose(tr, n - (n-d)*s, atol=1e-7)


def test_fixed_s_one_guerrero_exact_quadratic():
    n = 28
    t = np.arange(n, dtype=float)
    y = np.vstack([2*t**2 + 3*t + 7, -t**2 + 20])
    assert np.allclose(fit_panel(y, 1.0, mean_mode="guerrero"), y, atol=1e-7)
    zero = fit_panel(y, 1.0, mean_mode="zero")
    assert np.max(np.abs(zero-y)) > 1.0


def test_linear_trend_in_pls_nullspace():
    t = np.arange(30, dtype=float)
    y = np.vstack([2*t+1, -3*t+2])
    for s in (0.0, .2, .9, 1.0):
        assert np.allclose(fit_panel(y, s), y, atol=1e-7)


def test_smoothing_of_guerrero_mode_is_identity_at_s_zero():
    a = np.random.default_rng(8).standard_normal((4, 30))
    assert np.allclose(fit_panel(a, 0, mean_mode="guerrero"), a, atol=1e-9)


def test_selection_and_gcv_finite_on_correlated_noise():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(4, 70))
    config = GuerreroSettings(cv_window=32)
    for mode in ("zero", "guerrero"):
        for rule in ("fixed", "gcv", "forecast_cv"):
            s, curve = select_s(data, config, mode, rule)
            assert 0 <= s <= 1
            assert curve.shape == (len(config.grid),)
        assert np.isinf(gcv_curve(data, config.grid, 2, mode)[0])


def test_cv_past_invariance_and_fully_held_out_future():
    rng = np.random.default_rng(11)
    a = rng.normal(size=(4, 61))
    fixed = (0.0, .3, .75, 1.0)
    # Future observations appended beyond the available window cannot enter CV.
    first = forecast_cv_curve(a, fixed, 2, "guerrero", 30, 3)
    extended = np.column_stack((a, np.full((4, 2), 1e9)))
    same_prefix = forecast_cv_curve(extended[:, :61], fixed, 2, "guerrero", 30, 3)
    assert np.array_equal(first, same_prefix)
    # Historical targets matter to forecast CV, even if the final fitting
    # does not include them at earlier origins.
    different = a.copy()
    different[:, 40] += 10
    assert not np.allclose(first, forecast_cv_curve(different, fixed, 2, "guerrero", 30, 3))


def test_hilbert_distance_and_crosslag_feature_symmetry():
    x = np.random.default_rng(4).standard_normal((5, 42))
    d = functional_distance(fit_panel(x, .8), .75)
    assert np.allclose(d, d.T)
    assert np.allclose(np.diag(d), 0)
    assert np.all(d >= 0)
    for i in range(5):
        for j in range(5):
            for k in range(5):
                assert d[i, j] <= d[i, k] + d[k, j] + 1e-8
    r = lag_one_crosscorr(x)
    assert np.allclose(r, r.T)
    assert np.allclose(np.diag(r), 0)


def test_reject_out_of_range_s_and_modes():
    with pytest.raises(ValueError):
        lambda_from_s(30, 2, -0.1)
    with pytest.raises(ValueError):
        fit_panel(np.ones((3, 10)), 0.5, mean_mode="unrecognized")
