"""EXP-003: independent masked-refit tests and Gaussian profile-ML checks."""

import numpy as np
import pytest

from market_geometry.guerrero import fit_panel, lambda_from_s, smoother
from market_geometry.order_selection import (
    OrderStudyConfig, differences, gaussian_profile_ml_lambda, hat_matrix,
    inner_gcv_scores, inner_loo_scores, nested_loo_comparison,
    press_residuals, two_delete_residuals,
)


def test_orders_1_to_4_and_guerrero_mean_identity():
    t = np.arange(22.0)
    for d in (1, 2, 3, 4):
        p = np.array([__import__('math').comb(i, d) if i >= d else 0 for i in range(len(t))], dtype=float)
        assert np.allclose(differences(len(t), d) @ p, 1.0)
        assert np.allclose(fit_panel(p[None, :], 1.0, d=d, mean_mode="guerrero"), p[None, :], atol=1e-6)
        a, _, edf = smoother(len(t), d, .75)
        assert np.allclose(a, a.T)
        assert np.isclose(edf, len(t) - (len(t)-d)*.75, atol=1e-7)
        assert np.isfinite(lambda_from_s(len(t), d, .99))


def test_loo_and_two_delete_match_explicit_weighted_refit():
    rng = np.random.default_rng(13)
    y = rng.normal(size=20)
    for d in (1, 2, 3, 4):
        s = .67
        hat = hat_matrix(len(y), d, s)
        lam = lambda_from_s(len(y), d, s)
        q = differences(len(y), d).T @ differences(len(y), d)
        for outer in (0, 6, 19):
            w = np.ones(len(y))
            w[outer] = 0
            direct = np.linalg.solve(np.diag(w) + lam*q, w*y)
            assert np.isclose(y[outer]-direct[outer], press_residuals(y, hat)[outer], atol=1e-6)
            two = two_delete_residuals(y, hat, outer)
            for inner in (1, 8, 18):
                if outer == inner:
                    continue
                ww = w.copy()
                ww[inner] = 0
                direct2 = np.linalg.solve(np.diag(ww) + lam*q, ww*y)
                assert np.isclose(two[inner], y[inner]-direct2[inner], atol=1e-6)


def test_inner_scores_depend_on_train_only_using_direct_refits():
    y = np.random.default_rng(9).normal(size=23)
    h = hat_matrix(len(y), 2, .55)
    pair = inner_loo_scores(y, h)
    gcv = inner_gcv_scores(y, h)
    assert np.all(np.isfinite(pair)) and np.all(np.isfinite(gcv))
    for outer in (0, 8, 22):
        altered = y.copy()
        altered[outer] += 500
        assert np.isclose(inner_loo_scores(altered, h)[outer], pair[outer], atol=2e-8)
        assert np.isclose(inner_gcv_scores(altered, h)[outer], gcv[outer], atol=2e-8)


def test_profile_gaussian_ml_is_finite_and_ignores_outer_target():
    y = np.random.default_rng(8).standard_normal(25)
    for d in (1, 2, 3, 4):
        lam, boundary, val = gaussian_profile_ml_lambda(y, d, missing=7)
        assert np.isfinite(lam) and lam > 0 and np.isfinite(val)
        assert isinstance(boundary, bool)
        edited = y.copy()
        edited[7] = 1e9
        lam2, boundary2, val2 = gaussian_profile_ml_lambda(edited, d, missing=7)
        assert np.isclose(lam, lam2, rtol=1e-8)
        assert boundary == boundary2
        assert np.isclose(val, val2, rtol=1e-8)
    with pytest.raises(ValueError):
        gaussian_profile_ml_lambda(y, 25)


def test_nested_cv_smoke_and_boundary_tracking():
    y = np.sin(np.arange(23)/5) + np.random.default_rng(6).normal(0, .12, 23)
    r = nested_loo_comparison(y, OrderStudyConfig(orders=(1, 2, 3, 4)))
    assert len(r) == 16
    for key, v in r.items():
        assert len(v["outer_loo_prediction"]) == len(y)
        assert 0 <= v["rmse_observed"] < 2
        assert np.all((v["selected_s"] >= 0) & (v["selected_s"] <= 1))
    assert r[(2, 'inner_loocv')]['rmse_observed'] > 0
